#include <stdlib.h>
#include <math.h>
#include "memcached.h"
#include "storage.h"
#include "embeddings.h"

pthread_mutex_t emb_lock = PTHREAD_MUTEX_INITIALIZER;

#define EMBEDDING_DIM 16
typedef struct {
	float vec[EMBEDDING_DIM];
} embedding;

#define EMB_HISTORY 50
// define circular buffer
embedding emb_ring_buffer[EMB_HISTORY];
embedding rolling_avg;
uint32_t rolling_avg_write_ptr;





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
	if (slot == NULL || !slot->present) {
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
			if (EMB_DEBUG_PRINT) {
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
		if (EMB_DEBUG_PRINT) {
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
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "EMB_ERROR: trying to delete nonexistent item\n");
		}
		abort();
		return;
	}

	assert(curr_slot->next->it == it);
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
	refcount_incr(it);

	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] updating key=%.*s\n", it->nkey, ITEM_key(it));
	}
	uint32_t hv = hash(ITEM_key(it), it->nkey);
	
	// if there is no entry
	embedding* obj_emb = get_obj_emb(it, hv);
	if (obj_emb == NULL) {
		if (EMB_DEBUG_PRINT) {
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
	}

	// shift it towards the rolling avg
	shift_to_rolling_avg(obj_emb);
	emb_normalize(obj_emb);

	// update ring buffer and rolling avg
	emb_update_rolling_avg(obj_emb);

	refcount_decr(it);
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

// sampling pool: big bucket of item* pointers
// array that we can pull from in O(1)
// track size as we add/remove
item* emb_valid_items[EMB_MAP_SIZE];
uint32_t emb_valid_items_size = 0;

uint32_t add_valid_item(item* it) {
	refcount_incr(it);
	// increase size by one
	if (emb_valid_items_size == EMB_MAP_SIZE) {
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "EMB_ERROR: valid items pool ran out of slots!\n");
		}
		abort();
		return (uint32_t) -1;
	}

	emb_valid_items_size++;
	emb_valid_items[emb_valid_items_size - 1] = it;
	return emb_valid_items_size - 1;
}

bool emb_evict_candidate() {
	pthread_mutex_lock(&emb_lock);
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
		int obj_index = rand() % emb_valid_items_size;
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "[EMBDEBUG] sampling index %d\n", obj_index);
		}
		item* it = emb_valid_items[obj_index];
		uint32_t hv = hash(ITEM_key(it), it->nkey);

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


	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] evicting key=%.*s, freeing %lu bytes\n", least_similar->nkey, ITEM_key(least_similar), ITEM_ntotal(least_similar));
	}

	pthread_mutex_unlock(&emb_lock);
	// remove the least similar one: do_item_unlink()
	// LOGGER_LOG(NULL, LOG_EVICTIONS, LOGGER_EVICTION, least_similar);
	// STORAGE_delete(ext_storage, least_similar);
	do_item_unlink(least_similar, least_similar_hash);
	// ^this will call emb_remove_item

	// we evicted something
	return true;
}

void emb_remove_item(item* it, uint32_t hv) {
	pthread_mutex_lock(&emb_lock);
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] removing item key=%.*s\n", it->nkey, ITEM_key(it));
	}
	// delete this item from the hashmap: look up its pos in the sampling pool
	embedding_map_slot* slot = emb_map_lookup(it, hv);
	if (slot == NULL || !slot->present) {
		// we don't know about this item, so it's okay
		pthread_mutex_unlock(&emb_lock);
		return;
	}
	uint32_t sample_pool_idx = slot->sample_pool_idx;
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] found sample pool idx=%u\n", sample_pool_idx);
	}

	// swap whatever is in the back with this thing

	if (emb_valid_items_size == 1) {
		// what to do in this case???
	}

	emb_valid_items[sample_pool_idx] = emb_valid_items[emb_valid_items_size - 1];
	// update the object's position in the map
	item* shifted_it = emb_valid_items[sample_pool_idx];
	uint32_t shifted_hv = hash(ITEM_key(shifted_it), shifted_it->nkey);
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] swap with=%.*s\n", shifted_it->nkey, ITEM_key(shifted_it));
	}

	embedding_map_slot* shifted_slot = emb_map_lookup(shifted_it, shifted_hv);
	shifted_slot->sample_pool_idx = sample_pool_idx;
	emb_valid_items_size--;

	// now let's remove it from the hashmap
	emb_map_delete_entry(it, hv);
	refcount_decr(it);
	pthread_mutex_unlock(&emb_lock);
}
