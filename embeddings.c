#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <math.h>
#include "memcached.h"
#include "storage.h"
#include "embeddings.h"

// one lock for each row of hashmap if traversing/modifying the row
// one lock for each item hash % shardnum if modifying the item
// one lock for the ring buffer and rolling avg
#define EMB_LOCK_SHARD 128

pthread_spinlock_t emb_map_locks[EMB_LOCK_SHARD];
pthread_spinlock_t emb_obj_locks[EMB_LOCK_SHARD];
pthread_spinlock_t emb_pool_locks[EMB_LOCK_SHARD];
pthread_mutex_t emb_poolsize_lock = PTHREAD_MUTEX_INITIALIZER;
pthread_spinlock_t emb_ringbuffer_rollingavg_lock;

void embeddings_init() {
	// init all the spinlocks
	for (int i = 0; i < EMB_LOCK_SHARD; i++) {
		pthread_spin_init(&emb_map_locks[i], PTHREAD_PROCESS_PRIVATE);
		pthread_spin_init(&emb_obj_locks[i], PTHREAD_PROCESS_PRIVATE);
		pthread_spin_init(&emb_pool_locks[i], PTHREAD_PROCESS_PRIVATE);
	}

	// pthread_spin_init(&emb_poolsize_lock, PTHREAD_PROCESS_PRIVATE);
	pthread_spin_init(&emb_ringbuffer_rollingavg_lock, PTHREAD_PROCESS_PRIVATE);
}

const bool EMB_API_DEBUG = false;

void emb_lock_object(uint32_t hv);
void emb_unlock_object(uint32_t hv);

const bool EMB_LOCK_DEBUG = false;
void emb_lock_object(uint32_t hv) {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] locking obj lock %u\n", syscall(SYS_gettid), hv % EMB_LOCK_SHARD);
	pthread_spin_lock(&emb_obj_locks[hv % EMB_LOCK_SHARD]); }
void emb_unlock_object(uint32_t hv) {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] unlocking obj lock %u\n", syscall(SYS_gettid), hv % EMB_LOCK_SHARD);
	pthread_spin_unlock(&emb_obj_locks[hv % EMB_LOCK_SHARD]);
}

void emb_lock_mapslot(uint32_t hv);
void emb_unlock_mapslot(uint32_t hv);

void emb_lock_mapslot(uint32_t hv) {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] locking map lock %u\n", syscall(SYS_gettid), hv % EMB_LOCK_SHARD);
	pthread_spin_lock(&emb_map_locks[hv % EMB_LOCK_SHARD]);
}
void emb_unlock_mapslot(uint32_t hv) {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] unlocking map lock %u\n", syscall(SYS_gettid), hv % EMB_LOCK_SHARD);
	pthread_spin_unlock(&emb_map_locks[hv % EMB_LOCK_SHARD]);
}

void emb_lock_poolslot(uint32_t pool_idx);
void emb_unlock_poolslot(uint32_t pool_idx);

void emb_lock_poolslot(uint32_t pool_idx) {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] locking pool lock %u\n", syscall(SYS_gettid), pool_idx % EMB_LOCK_SHARD);
	pthread_spin_lock(&emb_pool_locks[pool_idx % EMB_LOCK_SHARD]);
}
void emb_unlock_poolslot(uint32_t pool_idx) {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] unlocking pool lock %u\n", syscall(SYS_gettid), pool_idx % EMB_LOCK_SHARD);
	pthread_spin_unlock(&emb_pool_locks[pool_idx % EMB_LOCK_SHARD]);
}

void emb_lock_poolsize(void);
void emb_unlock_poolsize(void);
void emb_lock_poolsize() {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] locking pool size lock\n", syscall(SYS_gettid));
	pthread_mutex_lock(&emb_poolsize_lock);
}
void emb_unlock_poolsize() {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] unlocking pool size lock\n", syscall(SYS_gettid));
	pthread_mutex_unlock(&emb_poolsize_lock);
}

void emb_lock_ringbuf(void);
void emb_unlock_ringbuf(void);
void emb_lock_ringbuf() {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] locking ringbuf\n", syscall(SYS_gettid));
	pthread_spin_lock(&emb_ringbuffer_rollingavg_lock);
}
void emb_unlock_ringbuf() {
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] unlocking ringbuf\n", syscall(SYS_gettid));
	pthread_spin_unlock(&emb_ringbuffer_rollingavg_lock);
}

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
embedding_map_slot* emb_hashmap[EMB_MAP_SIZE];

// need obj lock, will acquire slot lock
embedding_map_slot* emb_map_lookup(item* it, uint32_t hv);

// need obj lock
embedding* get_obj_emb(item* it, uint32_t hv);

// need obj lock
uint32_t get_sample_pool_idx(item* it, uint32_t hv);

// need obj lock, will acquire map lock inside
bool emb_map_make_entry(item* it, uint32_t hv);

// need obj lock, will acquire map lock inside
void emb_map_delete_entry(item* it, uint32_t hv);

// need to hold obj lock
void make_random_emb(embedding* obj_emb);

// need to hold obj lock
void emb_normalize(embedding* obj_emb);

// need to hold obj lock
void shift_to_rolling_avg(embedding* obj_emb);

// need to hold rolling avg lock and obj lock
void emb_update_rolling_avg(embedding* obj_emb);

// need to hold obj lock
float emb_compute_obj_similarity(embedding* obj_emb);

// need to hold obj lock, will acquire pool lock
uint32_t add_valid_item(item* it);

embedding_map_slot* emb_map_lookup(item* it, uint32_t hv) {
	emb_lock_mapslot(hv % EMB_MAP_SIZE);
	embedding_map_slot* curr_slot = emb_hashmap[hv % EMB_MAP_SIZE];
	while (curr_slot != NULL && curr_slot->it != it) {
		curr_slot = curr_slot->next;
	}
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMB_DEBUG] looking up item ptr %p hv %x result %p\n", (void*) it, hv, (void*) curr_slot);
	}
	emb_unlock_mapslot(hv % EMB_MAP_SIZE);
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
	emb_lock_mapslot(hv % EMB_MAP_SIZE);
	embedding_map_slot* curr_slot = emb_hashmap[hv % EMB_MAP_SIZE];
	if (curr_slot == NULL) {
		curr_slot = (embedding_map_slot*) malloc(sizeof(embedding_map_slot));
		emb_hashmap[hv % EMB_MAP_SIZE] = curr_slot;

		curr_slot->it = it;
		curr_slot->sample_pool_idx = (uint32_t) -1;
		curr_slot->next = NULL;
		curr_slot->present = true;
		emb_unlock_mapslot(hv % EMB_MAP_SIZE);
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
	emb_unlock_mapslot(hv % EMB_MAP_SIZE);
	return false;
}

void emb_map_delete_entry(item* it, uint32_t hv) {
	// remove from the map
	emb_lock_mapslot(hv % EMB_MAP_SIZE);
	embedding_map_slot* curr_slot = emb_hashmap[hv % EMB_MAP_SIZE];
	if (curr_slot != NULL && curr_slot->it == it) {
		emb_hashmap[hv % EMB_MAP_SIZE] = curr_slot->next;
		free(curr_slot);
		emb_unlock_mapslot(hv % EMB_MAP_SIZE);
		return;
	}

	if (curr_slot == NULL) {
		// problem, we are trying to delete something that's not there
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "EMB_ERROR: trying to delete nonexistent item\n");
		}
		emb_unlock_mapslot(hv % EMB_MAP_SIZE);
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
		emb_unlock_mapslot(hv % EMB_MAP_SIZE);
		abort();
		return;
	}

	assert(curr_slot->next->it == it);
	// replace curr.next with curr.next.next
	embedding_map_slot* next_slot = curr_slot->next->next;
	free(curr_slot->next);
	curr_slot->next = next_slot;
	emb_unlock_mapslot(hv % EMB_MAP_SIZE);
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
	emb_lock_ringbuf();
	for (int i = 0; i < EMBEDDING_DIM; i++) {
		// add obj_emb, remove the old one
		rolling_avg.vec[i] -= emb_ring_buffer[rolling_avg_write_ptr].vec[i];
		emb_ring_buffer[rolling_avg_write_ptr].vec[i] = obj_emb->vec[i] / EMB_HISTORY;
		rolling_avg.vec[i] += emb_ring_buffer[rolling_avg_write_ptr].vec[i];
	}

	rolling_avg_write_ptr++;
	rolling_avg_write_ptr %= EMB_HISTORY;
	emb_unlock_ringbuf();
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
	refcount_incr(it);
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] STARTING UPDATE OBJECT\n", syscall(SYS_gettid));
	if (EMB_API_DEBUG) fprintf(stderr, "[%ld] STARTING UPDATE OBJECT\n", syscall(SYS_gettid));

	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] updating key=%.*s\n", it->nkey, ITEM_key(it));
	}
	uint32_t hv = hash(ITEM_key(it), it->nkey);
	emb_lock_poolsize();
	// acquire the lock on the object
	emb_lock_object(hv);
	
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
		// this will acquire the pool lock while we already have the object lock
		slot->sample_pool_idx = add_valid_item(it);
	}
	emb_unlock_poolsize();

	// shift it towards the rolling avg
	shift_to_rolling_avg(obj_emb);
	emb_normalize(obj_emb);

	// update ring buffer and rolling avg
	emb_update_rolling_avg(obj_emb);

	emb_unlock_object(hv);
	refcount_decr(it);
	if (EMB_API_DEBUG) fprintf(stderr, "[%ld] ENDING UPDATE OBJECT\n", syscall(SYS_gettid));
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
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] STARTING FIND EVICTION ITEM\n", syscall(SYS_gettid));
	if (EMB_API_DEBUG) fprintf(stderr, "[%ld] STARTING FIND EVICTION ITEM\n", syscall(SYS_gettid));
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] searching for eviction candidate\n");
	}
	if (emb_valid_items_size == 0) {
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "[EMBDEBUG] no valid items\n");
		}
		if (EMB_API_DEBUG) fprintf(stderr, "[%ld] ENDING FIND EVICTION ITEM\n", syscall(SYS_gettid));
		return false;
	}


	// randomly sample objects from the sampling pool
	item* least_similar = NULL;
	uint32_t least_similar_hash = (uint32_t) -1;
	float least_similar_sim = 100;

	for (int i = 0; i < 32; i++) {
		// get a random item
		// lock the pool size
		emb_lock_poolsize();
		int obj_index = rand() % emb_valid_items_size;
		if (EMB_API_DEBUG) fprintf(stderr, "[%ld] evict - sampling index %d/%u\n", syscall(SYS_gettid), obj_index, emb_valid_items_size);
		if (EMB_DEBUG_PRINT) {
			fprintf(stderr, "[EMBDEBUG] sampling index %d\n", obj_index);
		}

		item* it = emb_valid_items[obj_index];
		refcount_incr(it);
		emb_unlock_poolsize();

		uint32_t hv = hash(ITEM_key(it), it->nkey);
		if (EMB_API_DEBUG) fprintf(stderr, "[%ld] evict - sampling index %d dereferenced safely, hv %x\n", syscall(SYS_gettid), obj_index, hv);
		emb_lock_object(hv);

		// look it up in the table
		embedding* obj_emb = get_obj_emb(it, hv);
		if (obj_emb == NULL) {
			refcount_decr(it);
			emb_unlock_object(hv);
			// this item was deleted
			if (EMB_API_DEBUG) fprintf(stderr, "[%ld] evict - ITEM idx %d hash %x WAS DELETED!!! BAD !!!!!\n", syscall(SYS_gettid), obj_index, hv);
			continue;
		}
		// get its similarity to rolling avg
		float obj_sim = emb_compute_obj_similarity(obj_emb);

		if (least_similar == NULL || (least_similar != it && obj_sim < least_similar_sim)) {
			if (least_similar != NULL) {
				refcount_decr(least_similar);
			}
			refcount_incr(it);
			least_similar = it;
			least_similar_hash = hv;
			least_similar_sim = obj_sim;
		}

		refcount_decr(it);
		emb_unlock_object(hv);
	}


	if (EMB_API_DEBUG) fprintf(stderr, "[%ld] evict - decided on obj hash %x\n", syscall(SYS_gettid), least_similar_hash);
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] evicting key=%.*s, freeing %lu bytes\n", least_similar->nkey, ITEM_key(least_similar), ITEM_ntotal(least_similar));
	}

	// remove the least similar one: do_item_unlink()
	// LOGGER_LOG(NULL, LOG_EVICTIONS, LOGGER_EVICTION, least_similar);
	// STORAGE_delete(ext_storage, least_similar);
	do_item_unlink(least_similar, least_similar_hash);
	// ^this will call emb_remove_item and decr our ref that we started

	// we evicted something
	if (EMB_API_DEBUG) fprintf(stderr, "[%ld] ENDING EVICT ITEM\n", syscall(SYS_gettid));
	return true;
}

void emb_remove_item(item* it, uint32_t hv) {
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] removing item key=%.*s\n", it->nkey, ITEM_key(it));
	}
	if (EMB_LOCK_DEBUG) fprintf(stderr, "[%ld] STARTING REMOVE ITEM\n", syscall(SYS_gettid));
	if (EMB_API_DEBUG) fprintf(stderr, "[%ld] STARTING REMOVE ITEM\n", syscall(SYS_gettid));
	// delete this item from the hashmap: look up its pos in the sampling pool
	// lock the pool so we can find something from the back
	emb_lock_poolsize();
	uint32_t sample_pool_size = emb_valid_items_size;

	item* shifted_it = emb_valid_items[sample_pool_size - 1];
	refcount_incr(shifted_it);
	uint32_t shifted_hv = hash(ITEM_key(shifted_it), shifted_it->nkey);

	uint32_t mylockid = hv % EMB_LOCK_SHARD;
	uint32_t shiftlockid = shifted_hv % EMB_LOCK_SHARD;
	// lock both the objects, min then max
	if (mylockid == shiftlockid) {
		// they share a lock
		emb_lock_object(mylockid);
	} else if (mylockid < shiftlockid) {
		emb_lock_object(mylockid);
		emb_lock_object(shiftlockid);
	} else {
		emb_lock_object(mylockid);
		emb_lock_object(shiftlockid);
	}

	embedding_map_slot* slot = emb_map_lookup(it, hv);
	if (slot == NULL || !slot->present) {
		refcount_decr(shifted_it);
		// we don't know about this item, so it's okay
		emb_unlock_poolsize();
		if (mylockid == shiftlockid) {
			emb_unlock_object(mylockid);
		} else {
			emb_unlock_object(mylockid);
			emb_unlock_object(shiftlockid);
		}
		if (EMB_API_DEBUG) fprintf(stderr, "[%ld] ENDING REMOVE ITEM\n", syscall(SYS_gettid));
		return;
	}

	uint32_t sample_pool_idx = slot->sample_pool_idx;
	if (EMB_API_DEBUG) fprintf(stderr, "[%ld] REMOVING ITEM AT SLOT %u\n", syscall(SYS_gettid), sample_pool_idx);
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] found sample pool idx=%u\n", sample_pool_idx);
	}

	// swap whatever is in the back with this thing
	// read whats in the back -> replace our slot index with that -> reduce the total size by one
	// for now lets lock the whole pool!

	emb_valid_items[sample_pool_idx] = emb_valid_items[emb_valid_items_size - 1];
	// update the object's position in the map
	if (EMB_DEBUG_PRINT) {
		fprintf(stderr, "[EMBDEBUG] swap with=%.*s\n", shifted_it->nkey, ITEM_key(shifted_it));
	}

	embedding_map_slot* shifted_slot = emb_map_lookup(shifted_it, shifted_hv);
	shifted_slot->sample_pool_idx = sample_pool_idx;
	slot->sample_pool_idx = (uint32_t) -1;
	emb_valid_items_size--;

	// now let's remove it from the hashmap
	emb_map_delete_entry(it, hv);

	emb_unlock_poolsize();
	if (mylockid != shiftlockid) {
		emb_unlock_object(shiftlockid);
	}

	emb_unlock_object(mylockid);
	refcount_decr(it);
	refcount_decr(shifted_it);
	if (EMB_API_DEBUG) fprintf(stderr, "[%ld] ENDING REMOVE ITEM\n", syscall(SYS_gettid));
}
