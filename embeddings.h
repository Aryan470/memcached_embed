#ifndef EMBEDDING_H
#define EMBEDDING_H

// indicate that an object was accessed
void emb_update_object(item* it);
void emb_query_embedding(item* it);
bool emb_evict_candidate(void);
void emb_remove_item(item* it, uint32_t hv);

#define EMB_DEBUG_PRINT 0
#define USE_EMBEDDING_EVICT 1

#endif
