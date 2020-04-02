#include <stdio.h>

#include "request.ch"
#include "ringbuf.ch"
#include "util.ch"

// TODO add fd array for each process

__host__
ringbuf_t * init_ringbuf() {
  ringbuf_t * ringbuf = NULL;
  CUDA_CALL(cudaMallocManaged(&ringbuf, sizeof(ringbuf_t)));

  int dev_id;
  CUDA_CALL(cudaGetDevice(&dev_id));
  // TODO required higher cuda version
  //CUDA_CALL(cudaMemAdvise(ringbuf, sizeof(ringbuf_t), cudaMemAdviseSetAccessedBy, dev_id));
  memset(ringbuf, 0, sizeof(ringbuf_t));

  return ringbuf;
}

__host__
void free_ringbuf(ringbuf_t * ringbuf) {
  CUDA_CALL(cudaFree(ringbuf));
}


__host__
bool cpu_dequeue(ringbuf_t * ringbuf, request_t * ret_request) {
  bool success = true;

  CPU_SPINLOCK_LOCK(&ringbuf->cpu_spin_lock);
  unsigned int read_index = ringbuf->read_index;
  if (read_index == ringbuf->write_index)
    return false;

  if (read_index >= (RINGBUF_SIZE-1)) {
    ringbuf->read_index = 0;
    read_index = 0;
  } // HANDLE wrap around at the end
  CPU_SPINLOCK_UNLOCK(&ringbuf->cpu_spin_lock);

  request_t * cur_request = &(ringbuf->requests[ringbuf->read_index]);
 
  while (!cur_request->ready_to_read) {
    *ret_request = *cur_request;
    memset(cur_request, 0, sizeof(request_t));
  }

  __sync_synchronize();
  return success;
}

/* Get a valid write_index
   write into the index
   */
__device__
bool gpu_enqueue(ringbuf_t * ringbuf, request_t * new_request) {
  BEGIN_SINGLE_THREAD;

  GPU_SPINLOCK_LOCK(&ringbuf->gpu_spin_lock);
  unsigned int write_index = ringbuf->write_index;

  /* wrap index around */
  if (write_index >= (RINGBUF_SIZE - 1)) {
    ringbuf->write_index = 0;
    write_index = 0;
  }

  /* buffer is full */
  if (write_index == ringbuf->read_index && !(write_index == 0 && ringbuf->read_index == 0)) {
    return false; // TODO come up with more graceful error

  } else { /* take write slot */
    ringbuf->write_index++;
  }
  
  GPU_SPINLOCK_UNLOCK(&ringbuf->gpu_spin_lock);

  printf("thread id:%d index:%d:%d\n", blockIdx.x, write_index, ringbuf->write_index);
  ringbuf->requests[write_index] = *new_request;
  
  __threadfence_system();
  ringbuf->requests[write_index].ready_to_read = true;

  __threadfence_system();
  END_SINGLE_THREAD;
  
  return true;
}

