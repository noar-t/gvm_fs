#include <inttypes.h>

#define MEM_POOL_SIZE 1024*1024*10 // 10mb
#define MIN_ALLOCATION 256 // 256b
#define NUM_POOL_PAGES (MEM_POOL_SIZE/MIN_ALLOCATION)

__host__
void init_memory_pool(void);

__host__
void * allocate_memory(size_t amount);
