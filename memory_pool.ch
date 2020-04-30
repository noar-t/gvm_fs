#include <inttypes.h>

#define MEM_POOL_SIZE 1024*1024*10 // 10mb
#define MIN_ALLOCATION 256 // 256b
#define NUM_POOL_PAGES (MEM_POOL_SIZE/MIN_ALLOCATION)

__host__
void init_memory_pool(void);

__host__
void * allocate_from_memory_pool(size_t amount);

__host__
void free_from_memory_pool(void * mem, size_t amount);
