#include <stdlib.h>
#include <math.h>
#include "memcached.h"
#include "storage.h"
#include "embeddings.h"

static pthread_mutex_t emb_lock;

void emb_init() {
	pthread_mutexattr_t attr;
	pthread_mutexattr_init(&attr);
	pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
	pthread_mutex_init(&emb_lock, &attr);
	pthread_mutexattr_destroy(&attr);
}

#define EMB_HISTORY 50
// define circular buffer
embedding emb_ring_buffer[EMB_HISTORY];
embedding rolling_avg;
uint32_t rolling_avg_write_ptr;



const bool EMB_ERR_PRINT = false;
const bool EMB_API_PRINT = false;



// ------- HASHMAP
#define EMB_MAP_SIZE (1 << 20)
typedef struct embedding_map_slot embedding_map_slot;
struct embedding_map_slot {
	embedding emb;
	item* it;
	uint32_t sample_pool_idx;
	embedding_map_slot* next;
	bool present;
};

// define hashmap
// each slot entry will be
embedding_map_slot* emb_hashmap[EMB_MAP_SIZE];

// sampling pool: big bucket of item* pointers
// array that we can pull from in O(1)
// track size as we add/remove
item* emb_valid_items[EMB_MAP_SIZE];
volatile uint32_t emb_valid_items_size = 0;

embedding_map_slot* emb_map_lookup(item* it, uint32_t hv);
embedding* get_obj_emb(item* it, uint32_t hv);
uint32_t get_sample_pool_idx(item* it, uint32_t hv);
bool emb_map_make_entry(item* it, uint32_t hv);
void emb_map_delete_entry(item* it, uint32_t hv);

void make_random_emb(embedding* obj_emb);
void emb_normalize(embedding* obj_emb);
void shift_to_rolling_avg(embedding* obj_emb);
void emb_update_rolling_avg(embedding* obj_emb);
float emb_compute_obj_similarity(embedding* obj_emb);

uint32_t add_valid_item(item* it);
void emb_remove_item_nolock(item* it, uint32_t hv);
void verify_pool_and_map(void);

void verify_pool_and_map() {
	return;
	if (rand() % 50 != 0) {
		//return;
	}
	for (uint32_t i = 0; i < emb_valid_items_size; i++) {
		item* curr_it = emb_valid_items[i];
		uint32_t curr_hv = hash(ITEM_key(curr_it), curr_it->nkey);
		embedding_map_slot* itslot = emb_map_lookup(curr_it, curr_hv);
		if (itslot == NULL) {
			fprintf(stderr, "EMB_ERR: item hash %x in pool slot %u has no entry in hashmap\n", curr_hv, i);
			abort();
		}

		if (itslot->sample_pool_idx != i) {
			fprintf(stderr, "EMB_ERR: item hash %x in pool slot %u has WRONG entry in hashmap, it says index is %u\n", curr_hv, i, itslot->sample_pool_idx);
			abort();
		}
	}
}


embedding_map_slot* emb_map_lookup(item* it, uint32_t hv) {
	embedding_map_slot* curr_slot = emb_hashmap[hv % EMB_MAP_SIZE];
	while (curr_slot != NULL && curr_slot->it != it) {
		curr_slot = curr_slot->next;
	}
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMB_DEBUG] looking up item ptr %p hv %x result %p\n", (void*) it, hv, (void*) curr_slot);
	}
	return curr_slot;
}

// get the pointer to embedding associated with an item
embedding* get_obj_emb(item* it, uint32_t hv) {
	// lookup the slot
	embedding_map_slot* slot = emb_map_lookup(it, hv);
	// if null, return null
	if (slot == NULL) {
		return NULL;
	}
	return &(slot->emb);
}

uint32_t get_sample_pool_idx(item* it, uint32_t hv) {
	// lookup the slot
	embedding_map_slot* slot = emb_map_lookup(it, hv);
	// if null, return null
	if (slot == NULL) {
		return (uint32_t) (-1);
	}

	return slot->sample_pool_idx;
}

bool emb_map_make_entry(item* it, uint32_t hv) {
	if (emb_map_lookup(it, hv) != NULL) {
		// it must not be present
		embedding_map_slot* slot = emb_map_lookup(it, hv);
		if (slot->present) {
			if (EMB_DEBUG_PRINT || EMB_ERR_PRINT) {
				fprintf(stderr, "EMB_ERROR: duplicate entry for object in hashmap\n");
			}
			abort();
		}
		slot->present = true;
		return true;
	}

	// add to the map
	embedding_map_slot* curr_slot = emb_hashmap[hv % EMB_MAP_SIZE];
	if (curr_slot == NULL) {
		curr_slot = (embedding_map_slot*) malloc(sizeof(embedding_map_slot));
		emb_hashmap[hv % EMB_MAP_SIZE] = curr_slot;

		curr_slot->it = it;
		curr_slot->sample_pool_idx = (uint32_t) -1;
		curr_slot->next = NULL;
		curr_slot->present = true;
		return false;
	}

	while (curr_slot->next != NULL) {
		curr_slot = curr_slot->next;
	}

	curr_slot->next = malloc(sizeof(embedding_map_slot));
	curr_slot = curr_slot->next;

	curr_slot->it = it;
	curr_slot->sample_pool_idx = (uint32_t) -1;
	curr_slot->next = NULL;
	curr_slot->present = true;
	return false;
}

void emb_map_delete_entry(item* it, uint32_t hv) {
	// remove from the map
	embedding_map_slot* curr_slot = emb_hashmap[hv % EMB_MAP_SIZE];
	if (curr_slot != NULL && curr_slot->it == it) {
		emb_hashmap[hv % EMB_MAP_SIZE] = curr_slot->next;
		free(curr_slot);
		return;
	}

	if (curr_slot == NULL) {
		// problem, we are trying to delete something that's not there
		if (EMB_DEBUG_PRINT || EMB_ERR_PRINT) {
			fprintf(stderr, "EMB_ERROR: trying to delete nonexistent item\n");
		}
		abort();
		return;
	}

	while (curr_slot->next != NULL && curr_slot->next->it != it) {
		curr_slot = curr_slot->next;
	}

	// at this point, if next is null then it DNE
	if (curr_slot->next == NULL) {
		// problem, we are trying to delete something that's not there
		if (EMB_DEBUG_PRINT || EMB_ERR_PRINT) {
			fprintf(stderr, "EMB_ERROR: trying to delete nonexistent item\n");
		}
		abort();
		return;
	}

	if (EMB_ERR_PRINT && curr_slot->next->it != it) {
		fprintf(stderr, "EMB_ERROR: curr_slot->next->it != it\n");
	}
	// replace curr.next with curr.next.next
	embedding_map_slot* next_slot = curr_slot->next->next;
	free(curr_slot->next);
	curr_slot->next = next_slot;
}






void make_random_emb(embedding* obj) {
	for (int i = 0; i < EMBEDDING_DIM; i++) {
		// make something random in [0, 1]
		float r = ((float)rand() / (float)RAND_MAX);
		// shift it to [-1, 1]
		obj->vec[i] = r * 2.0f - 1.0f;
	}
}

// normalize a vector
void emb_normalize(embedding* obj_vec) {
	// find mag, divide
	float mag = 0;
	for (int i = 0; i < EMBEDDING_DIM; i++) {
		mag += obj_vec->vec[i] * obj_vec->vec[i];
	}
	mag = sqrtf(mag);
	for (int i = 0; i < EMBEDDING_DIM; i++) {
		obj_vec->vec[i] /= mag;
	}
}

#define EMB_LEARNING_RATE 0.1
// shift a vector closer to the rolling avg
void shift_to_rolling_avg(embedding* obj_emb) {
	// update each entry in obj_emb to be itself times alpha + rolling avg times alpha + noise
	// TODO: fast noise
	for (int i = 0; i < EMBEDDING_DIM; i++) {
		obj_emb->vec[i] += EMB_LEARNING_RATE * rolling_avg.vec[i];
	}
}

// add a vector to rolling avg, remove old one
void emb_update_rolling_avg(embedding* obj_emb) {
	for (int i = 0; i < EMBEDDING_DIM; i++) {
		// add obj_emb, remove the old one
		rolling_avg.vec[i] -= emb_ring_buffer[rolling_avg_write_ptr].vec[i];
		emb_ring_buffer[rolling_avg_write_ptr].vec[i] = obj_emb->vec[i] / EMB_HISTORY;
		rolling_avg.vec[i] += emb_ring_buffer[rolling_avg_write_ptr].vec[i];
	}

	rolling_avg_write_ptr++;
	rolling_avg_write_ptr %= EMB_HISTORY;
}

float emb_compute_obj_similarity(embedding* obj_emb) {
	// return dot product of rolling avg and obj emb
	float sim = 0.0f;
	for (int i = 0; i < EMBEDDING_DIM; i++) {
		sim += obj_emb->vec[i] * rolling_avg.vec[i];
	}
	return sim;
}


// called from user command path when objects are accessed
void emb_update_object(item* it) {
	pthread_mutex_lock(&emb_lock);

	if ((it->it_flags & ITEM_LINKED) == 0) {
		pthread_mutex_unlock(&emb_lock);
		return;
	}
	if (EMB_ERR_PRINT) verify_pool_and_map();

	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] updating key=%.*s\n", it->nkey, ITEM_key(it));
	}
	uint32_t hv = hash(ITEM_key(it), it->nkey);
	if (EMB_API_PRINT) {
		fprintf(stderr, "[%lu] UPDATE ptr %p hash %x\n", (unsigned long) pthread_self(), (void*) it, hv);
	}
	
	// if there is no entry
	embedding* obj_emb = get_obj_emb(it, hv);
	if (obj_emb == NULL) {
		if (EMB_DEBUG_PRINT || EMB_API_PRINT) {
			fprintf(stderr, "[EMBDEBUG] new object\n");
		}
		// install it in hashmap
		bool had_previous_emb = emb_map_make_entry(it, hv);
		obj_emb = get_obj_emb(it, hv);
		if (!had_previous_emb) {
			if (EMB_DEBUG_PRINT) {
				fprintf(stderr, "[EMBDEBUG] did not have previous emb\n");
			}
			// initialize vector
			make_random_emb(obj_emb);
		}
		// update the sample pool idx
		embedding_map_slot* slot = emb_map_lookup(it, hv);
		slot->sample_pool_idx = add_valid_item(it);
		if (EMB_API_PRINT) {
			fprintf(stderr, "[%lu] ADDPOOL item %p hash %x slot %u\n", (unsigned long) pthread_self(), (void*) it, hv, slot->sample_pool_idx);
		}
	} else {
		if (EMB_DEBUG_PRINT || EMB_API_PRINT) {
			fprintf(stderr, "[EMBDEBUG] old object\n");
		}
	}

	embedding_map_slot* slot = emb_map_lookup(it, hv);
	if (EMB_ERR_PRINT && emb_valid_items[slot->sample_pool_idx] != it) {
		fprintf(stderr, "[EMB_ERR] inserting item with hash %x ptr %p, obj->slot = %u but valid_items[slot] has ptr %p\n", hv, (void*) it, slot->sample_pool_idx, (void*) emb_valid_items[slot->sample_pool_idx]);
		abort();
	}

	// shift it towards the rolling avg
	shift_to_rolling_avg(obj_emb);
	emb_normalize(obj_emb);

	// update ring buffer and rolling avg
	emb_update_rolling_avg(obj_emb);

	if (EMB_ERR_PRINT) verify_pool_and_map();
	if (EMB_API_PRINT) {
		fprintf(stderr, "[%lu] FINISHED UPDATE\n", (unsigned long) pthread_self());
	}
	pthread_mutex_unlock(&emb_lock);
}

void emb_query_embedding(item* it) {
	// TODO
	uint32_t hv = hash(ITEM_key(it), it->nkey);
	embedding* emb = get_obj_emb(it, hv);
	if (emb == NULL) {
		// print item key -> null
	} else {
		// print item key -> emb
	}
}


uint32_t add_valid_item(item* it) {
	// increase size by one
	if (emb_valid_items_size == EMB_MAP_SIZE) {
		if (EMB_DEBUG_PRINT || EMB_ERR_PRINT) {
			fprintf(stderr, "EMB_ERROR: valid items pool ran out of slots!\n");
		}
		abort();
		return (uint32_t) -1;
	}

	emb_valid_items_size++;
	if (EMB_ERR_PRINT && emb_valid_items[emb_valid_items_size - 1] != NULL) {
		fprintf(stderr, "EMB_ERR: trying to add to tail at slot %u but ptr %p is in there\n", emb_valid_items_size - 1, (void*) emb_valid_items[emb_valid_items_size - 1]);
	}
	emb_valid_items[emb_valid_items_size - 1] = it;
	return emb_valid_items_size - 1;
}

/*bool emb_evict_candidate() {
	pthread_mutex_lock(&emb_lock);
	if (EMB_API_PRINT) {
		fprintf(stderr, "[%lu] FIND_EVICT\n", (unsigned long) pthread_self());
	}
	if (EMB_ERR_PRINT) verify_pool_and_map();
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] searching for eviction candidate\n");
	}
	if (emb_valid_items_size == 0) {
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "[EMBDEBUG] no valid items\n");
		}
		pthread_mutex_unlock(&emb_lock);
		return false;
	}
	// randomly sample objects from the sampling pool
	item* least_similar = NULL;
	uint32_t least_similar_hash = (uint32_t) -1;
	float least_similar_sim = 100;

	for (int i = 0; i < 32; i++) {
		// get a random item
		uint32_t obj_index = rand() % emb_valid_items_size;
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "[EMBDEBUG] sampling index %d\n", obj_index);
		}
		item* it = emb_valid_items[obj_index];
		// if it is in the sample pool, it must be in the hashmap
		uint32_t hv = hash(ITEM_key(it), it->nkey);
		if (EMB_ERR_PRINT && emb_map_lookup(it, hv) == NULL) {
			fprintf(stderr, "[EMBERR] trying to sample something that is not in hashmap num_valid=%u, sampled_index=%u, obj_hash=%x, item_ptr=%p\n", emb_valid_items_size, obj_index, hv, (void*)it);
			abort();
		}
		if (EMB_ERR_PRINT && (emb_map_lookup(it, hv)->sample_pool_idx != obj_index)) {
			fprintf(stderr, "[EMBERR] obj with hash %x object sample pool idx is %u but it's actually in %u\n", hv, emb_map_lookup(it,hv)->sample_pool_idx, obj_index);
			abort();
		}

		// look it up in the table
		embedding* obj_emb = get_obj_emb(it, hv);
		// get its similarity to rolling avg
		float obj_sim = emb_compute_obj_similarity(obj_emb);

		if (least_similar == NULL || obj_sim < least_similar_sim) {
			least_similar = it;
			least_similar_hash = hv;
			least_similar_sim = obj_sim;
		}
	}


	if (EMB_API_PRINT) {
		fprintf(stderr, "[%lu] EVICT ptr %p hv %x\n", (unsigned long) pthread_self(), (void*) least_similar, least_similar_hash);
	}

	// remove the least similar one: do_item_unlink()
	// LOGGER_LOG(NULL, LOG_EVICTIONS, LOGGER_EVICTION, least_similar);
	// STORAGE_delete(ext_storage, least_similar);
	// emb_remove_item_nolock(least_similar, least_similar_hash);
	if (EMB_ERR_PRINT) verify_pool_and_map();
	//emb_remove_item_nolock(least_similar, least_similar_hash);
	//do_item_unlink(least_similar, least_similar_hash);

	pthread_mutex_unlock(&emb_lock);
	// ^this will call emb_remove_item, but it will not see

	// we evicted something
	return true;
}*/
bool emb_evict_candidate() {
    /* ---------- 1. pick a victim under emb_lock ---------- */
    pthread_mutex_lock(&emb_lock);

    if (emb_valid_items_size == 0) {
        pthread_mutex_unlock(&emb_lock);
        return false;
    }

    item        *victim = NULL;
    uint32_t     victim_hv = 0;
    float        worst_sim = 999.0f;

    for (int i = 0; i < 32; i++) {
        uint32_t idx = rand() % emb_valid_items_size;
        item    *it  = emb_valid_items[idx];

        uint32_t hv  = hash(ITEM_key(it), it->nkey);
        embedding *e = get_obj_emb(it, hv);

        float sim = emb_compute_obj_similarity(e);
        if (sim < worst_sim) {
            victim     = it;
            victim_hv  = hv;
            worst_sim  = sim;
        }
    }

    if (victim == NULL) {                /* should never happen */
        pthread_mutex_unlock(&emb_lock);
        return false;
    }

    /* Hold an extra reference so the slab chunk stays valid
       until we finish unlinking.                            */
    refcount_incr(victim);

    /* ---------- 2. outside emb_lock:  unlink safely ---------- */

    /* Grab the per‑bucket item lock the same way lru_pull_tail does. */
    void *bucket_lock = NULL;
    if ((bucket_lock = item_trylock(victim_hv)) == NULL) {
		pthread_mutex_unlock(&emb_lock);
		return false;
    }

    /* Unlink from the cache; this will call emb_remove_item()
       which updates the hashmap + sampling pool once.        */
    do_item_unlink_nolock(victim, victim_hv);

    /* Release bucket lock and our extra reference */
    item_trylock_unlock(bucket_lock);
    do_item_remove(victim);        /* drops the ref we added above */
    pthread_mutex_unlock(&emb_lock);

    return true;                   /* we evicted something */
}

void emb_remove_item_nolock(item* it, uint32_t hv) {
	if (EMB_API_PRINT) {
		fprintf(stderr, "[%lu] REMOVE ptr %p hash %x\n", (unsigned long) pthread_self(), (void*) it, hv);
	}
	// verify the state of hashmap/sample pool
	if (EMB_ERR_PRINT) verify_pool_and_map();

	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] removing item key=%.*s\n", it->nkey, ITEM_key(it));
	}
	if (EMB_ERR_PRINT) {
		uint32_t confirm_hv = hash(ITEM_key(it), it->nkey);
		if (confirm_hv != hv) {
			fprintf(stderr, "EMB_ERR: for pointer %p passed in hv=%x but recomputed=%x\n", (void*) it, hv, confirm_hv);
			hv = confirm_hv;
		}
	}
	// delete this item from the hashmap: look up its pos in the sampling pool
	embedding_map_slot* slot = emb_map_lookup(it, hv);
	if (slot == NULL || !slot->present) {
		// we don't know about this item, so it's okay
		// check through valid items
		if (EMB_ERR_PRINT) {
			for (int i = 0; i < emb_valid_items_size; i++) {
				if (emb_valid_items[i] == it) {
					fprintf(stderr, "EMB_ERR: called remove on obj ptr %p and it's not in map, but found it in slot %u\n", (void*) it, i);
				}
			}
		}
		if (EMB_ERR_PRINT) verify_pool_and_map();

		return;
	}
	uint32_t sample_pool_idx = slot->sample_pool_idx;
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] found sample pool idx=%u\n", sample_pool_idx);
	}

	// swap whatever is in the back with this thing

	/*if (emb_valid_items_size == 1) {
		if (emb_valid_items[0] != it) {
			if (EMB_DEBUG_PRINT || EMB_ERR_PRINT) fprintf(stderr, "EMB_ERROR: only one valid item left! how did this happen\n");
			abort();
		}
	}*/

	if (EMB_ERR_PRINT && emb_valid_items[emb_valid_items_size - 1] == NULL) {
		fprintf(stderr, "[EMB_ERR] trying to remove obj from slot %u and the tail is %u, has nullptr\n", sample_pool_idx, emb_valid_items_size-1);
	}

	// write the tail into the index of the removed item
	if (EMB_ERR_PRINT && emb_valid_items[sample_pool_idx] != it) {
		fprintf(stderr, "[EMB_ERR] trying to remove obj ptr %p and obj->slot = %u but emb_items has ptr %p\n", (void*) it, sample_pool_idx, (void*) emb_valid_items[sample_pool_idx]);
	}
	emb_valid_items[sample_pool_idx] = emb_valid_items[emb_valid_items_size - 1];
	emb_valid_items[emb_valid_items_size - 1] = NULL;
	emb_valid_items_size--;
	slot->sample_pool_idx = (uint32_t) -1;

	if (sample_pool_idx != emb_valid_items_size) {
		// update the object's position in the map
		item* shifted_it = emb_valid_items[sample_pool_idx];
		uint32_t shifted_hv = hash(ITEM_key(shifted_it), shifted_it->nkey);
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "[EMBDEBUG] swap with=%.*s\n", shifted_it->nkey, ITEM_key(shifted_it));
		}
		if (EMB_API_PRINT) {
			fprintf(stderr, "[%lu] MOVE ptr %p hash %x from slot %u into %u\n", (unsigned long) pthread_self(), (void*) shifted_it, shifted_hv, emb_valid_items_size, sample_pool_idx);
		}

		embedding_map_slot* shifted_slot = emb_map_lookup(shifted_it, shifted_hv);
		if (EMB_ERR_PRINT && shifted_slot == NULL) {
			fprintf(stderr, "[EMB_ERR] removing item with hash %x from slot %u and putting item with hash %x from slot %u but tail item is not in map\n", hv, sample_pool_idx, shifted_hv, emb_valid_items_size - 1);
			abort();
		}
		shifted_slot->sample_pool_idx = sample_pool_idx;
	}

	// now let's remove it from the hashmap
	emb_map_delete_entry(it, hv);
	if (EMB_ERR_PRINT) verify_pool_and_map();
}

void emb_remove_item(item* it, uint32_t hv) {
	pthread_mutex_lock(&emb_lock);
	emb_remove_item_nolock(it, hv);
	pthread_mutex_unlock(&emb_lock);
}
