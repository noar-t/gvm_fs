#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "types.ch"
#include "memory_pool.ch"

#define FREE 0
#define ALLOCATED 1

#define MEMORY_FULL -1

void * memory_pool;

// false is free, true is allocated
static bool memory_pool_usage_map[NUM_POOL_PAGES];

__host__
void init_memory_pool() {
  printf("mem pool size %d\n", MEM_POOL_SIZE);
  CUDA_CALL(cudaMallocManaged((void **) &memory_pool, MEM_POOL_SIZE));
  memset(memory_pool, 0, MEM_POOL_SIZE);
}

__host__
void toggle_range(int start, int length, bool val) {
  for (int i = start; i <= length; i++) {
    memory_pool_usage_map[i] = val;
  }
}

__host__
int find_free_slot(size_t num_pages_requested) {
  int run_start = 0;
  int current_run = 0; 

  for (int i = 0; i < NUM_POOL_PAGES; i++) {
    if (memory_pool_usage_map[i] == FREE) {
      current_run++;
      if (current_run >= num_pages_requested) {
        toggle_range(run_start, current_run, ALLOCATED);
        return run_start;
      }
    } else {
      run_start = i;
      current_run = 0;
    }
  }

  return MEMORY_FULL;
}

__host__
void * allocate_from_memory_pool(size_t amount) {
  int num_pages_requested = (amount / MIN_ALLOCATION) + ((amount % MIN_ALLOCATION) > 0);
  int index = find_free_slot(num_pages_requested);

  if (index == MEMORY_FULL)
    return NULL;

  char * ret_ptr = (char *) memory_pool;
  ret_ptr += (index * MIN_ALLOCATION);
  memset(ret_ptr, 0, amount);
  
  return (void *) ret_ptr;
}

__host__
void free_from_memory_pool(void * ptr, size_t size) {
  // TODO
  return;
}
