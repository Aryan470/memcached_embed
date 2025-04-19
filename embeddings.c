#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <float.h>
#include "memcached.h"
#include "storage.h"
#include "embeddings.h"

/*
 * CONFIGURATION
 */
#define EMB_MAP_BITS       20
#define EMB_MAP_SIZE       (1u << EMB_MAP_BITS)   /* must be power of two */
#define EMB_LOCK_SHARDS    16                     /* power of two, <= number of cores */
#define EMB_SAMPLE_SIZE    8                      /* how many random buckets to sample on eviction */
#define EMB_ALPHA          0.1f                   /* smoothing factor for rolling average */
#define EMBED_DIM          16          /* from embeddings.h */

/*
 * PER‑BUCKET ENTRY
 */
typedef struct emb_entry {
    item*                it;
    float                vec[EMBED_DIM];
    struct emb_entry*    next;
} emb_entry_t;

/*
 * GLOBAL STATE
 */
static emb_entry_t*      emb_map[EMB_MAP_SIZE];
static pthread_mutex_t   bucket_locks[EMB_LOCK_SHARDS];
static float             rolling_avg[EMBED_DIM];
static pthread_mutex_t   avg_lock;

/* thread‑local seed for rand_r */
static __thread unsigned int rand_seed;

/*
 * ABI FUNCTIONS (called from items.c)
 */
void emb_init(void) {
    /* init per‑bucket locks */
    for (int i = 0; i < EMB_LOCK_SHARDS; i++) {
        pthread_mutex_init(&bucket_locks[i], NULL);
    }
    /* init avg lock */
    pthread_mutex_init(&avg_lock, NULL);
    /* zero out the rolling average */
    memset(rolling_avg, 0, sizeof(rolling_avg));
    /* init thread‑local RNG */
    rand_seed = (unsigned int)time(NULL) ^ (unsigned int)(uintptr_t)pthread_self();
}

embedding* get_obj_emb(item* it, uint32_t hv);
embedding* get_obj_emb(item* it, uint32_t hv) {
    uint32_t idx      = hv & (EMB_MAP_SIZE - 1);
    uint32_t lock_id  = idx & (EMB_LOCK_SHARDS - 1);
    pthread_mutex_lock(&bucket_locks[lock_id]);
    emb_entry_t* e = emb_map[idx];
    while (e) {
        if (e->it == it) {
            pthread_mutex_unlock(&bucket_locks[lock_id]);
            return (embedding*)e->vec;
        }
        e = e->next;
    }
    pthread_mutex_unlock(&bucket_locks[lock_id]);
    return NULL;
}

/*
 * If entry already exists, returns true.
 * Otherwise allocates+links a new one (filled with a random unit vector) and returns false.
 */
bool emb_map_make_entry(item* it, uint32_t hv);
bool emb_map_make_entry(item* it, uint32_t hv) {
    uint32_t idx     = hv & (EMB_MAP_SIZE - 1);
    uint32_t lock_id = idx & (EMB_LOCK_SHARDS - 1);
    pthread_mutex_lock(&bucket_locks[lock_id]);

    emb_entry_t* e = emb_map[idx];
    while (e) {
        if (e->it == it) {
            pthread_mutex_unlock(&bucket_locks[lock_id]);
            return true;
        }
        e = e->next;
    }

    /* not found: create it */
    emb_entry_t* ne = calloc(1, sizeof(*ne));
    ne->it = it;
    /* random vector in [-1,1]^d then normalize */
    float norm = 0.0f;
    for (int i = 0; i < EMBED_DIM; i++) {
        float v = (rand_r(&rand_seed)/(float)RAND_MAX)*2.0f - 1.0f;
        ne->vec[i] = v;
        norm += v*v;
    }
    norm = sqrtf(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < EMBED_DIM; i++) {
            ne->vec[i] /= norm;
        }
    }

    ne->next = emb_map[idx];
    emb_map[idx] = ne;
    pthread_mutex_unlock(&bucket_locks[lock_id]);
    return false;
}

void emb_update_object(item* it) {
    uint32_t hv = hash(ITEM_key(it), it->nkey);
    embedding* e = get_obj_emb(it, hv);
    if (!e) return;

    pthread_mutex_lock(&avg_lock);
    /* exponential moving average */
    for (int i = 0; i < EMBED_DIM; i++) {
        rolling_avg[i] = EMB_ALPHA * rolling_avg[i]
                       + (1.0f - EMB_ALPHA) * ((float*)e)[i];
    }
    pthread_mutex_unlock(&avg_lock);
}

void emb_remove_item(item* it, uint32_t hv) {
    uint32_t idx     = hv & (EMB_MAP_SIZE - 1);
    uint32_t lock_id = idx & (EMB_LOCK_SHARDS - 1);
    pthread_mutex_lock(&bucket_locks[lock_id]);

    emb_entry_t** pp = &emb_map[idx];
    while (*pp) {
        if ((*pp)->it == it) {
            emb_entry_t* todel = *pp;
            *pp = todel->next;
            free(todel);
            break;
        }
        pp = &(*pp)->next;
    }

    pthread_mutex_unlock(&bucket_locks[lock_id]);
}

/*
 * Called from do_item_alloc_pull when we need to free one object.
 * Samples EMB_SAMPLE_SIZE random buckets, picks the one with minimum
 * cosine similarity against the rolling_avg, unlinks it from map+cache,
 * and returns true if we evicted someone.
 */
bool emb_evict_candidate(void) {
    float        worst_sim = FLT_MAX;
    emb_entry_t* victim    = NULL;

    /* 1) Sample a handful of buckets & pick worst head entry */
    for (int s = 0; s < EMB_SAMPLE_SIZE; s++) {
        uint32_t bidx    = rand_r(&rand_seed) & (EMB_MAP_SIZE - 1);
        uint32_t lock_id = bidx & (EMB_LOCK_SHARDS - 1);
        pthread_mutex_lock(&bucket_locks[lock_id]);
        emb_entry_t* e = emb_map[bidx];
        if (e) {
            float dot = 0.0f;
            pthread_mutex_lock(&avg_lock);
            for (int i = 0; i < EMBED_DIM; i++) {
                dot += rolling_avg[i] * e->vec[i];
            }
            pthread_mutex_unlock(&avg_lock);

            if (dot < worst_sim) {
                worst_sim = dot;
                victim    = e;
            }
        }
        pthread_mutex_unlock(&bucket_locks[lock_id]);
    }

    if (!victim) return false;

    /* 2) Compute its hash & try to lock the item */
    uint32_t vhv = hash(ITEM_key(victim->it), victim->it->nkey);
    void *hold_lock = item_trylock(vhv);
    if (hold_lock == NULL) {
        /* someone else is touching it right now—skip this eviction */
        return false;
    }

    /* 3) Unlink from our embedding map, then evict from the cache */
    emb_remove_item(victim->it, vhv);
    do_item_unlink(victim->it, vhv);

    /* 4) And finally release the per‑item lock */
    item_trylock_unlock(hold_lock);
    return true;
}
