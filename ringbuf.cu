#include <stdio.h>

#include "request.ch"
#include "ringbuf.ch"
#include "util.ch"

// TODO add fd array for each process

__host__
ringbuf_t * init_ringbuf() {
  ringbuf_t * ringbuf = NULL;
  CUDA_CALL(cudaMallocManaged(&ringbuf, sizeof(ringbuf_t)));
  CUDA_CALL(cudaMemAdvise(ringbuf, sizeof(ringbuf_t), cudaMemAdviseSetAccessedBy, 0));
  memset(ringbuf, 0, sizeof(ringbuf_t));

  return ringbuf;
}

__host__
void free_ringbuf(ringbuf_t * ringbuf) {
  CUDA_CALL(cudaFree(ringbuf));
}


/*XXX I think the loop condition is write_index != read_index */
/* XXX cpu operations are not concurrency safe */
__host__
bool cpu_dequeue(ringbuf_t * ringbuf, request_t * ret_request) {
  bool success = true;

  CPU_SPINLOCK_LOCK(&ringbuf->cpu_spin_lock);
  unsigned int read_index = ringbuf->read_index;
  //if (!read_index == 

  if (read_index >= RINGBUF_SIZE) {;} // HANDLE wrap around at the end
  CPU_SPINLOCK_UNLOCK(&ringbuf->cpu_spin_lock);

  request_t * cur_request = &(ringbuf->requests[ringbuf->read_index]);
  
  if (cur_request->ready_to_read) {
    *ret_request = *cur_request;
  } else {
    success = false;
  }

  return success;
}

/* Get a valid write_index
   write into the index
   incriment read_index (only if read_index = write_index -1
   */
__device__
bool gpu_enqueue(ringbuf_t * ringbuf, request_t * new_request) {
  BEGIN_SINGLE_THREAD;

  unsigned int atomic_result = 0;
  unsigned int write_index = ringbuf->write_index;

  GPU_SPINLOCK_LOCK(&ringbuf->gpu_spin_lock);

  /* Reserve the current write_index */
  // TODO race condition if we wrap around too fast 
  // must verify wedont pass read index
  //while (true) {
  //  if (write_index < (RINGBUF_SIZE-1)) {
  //    atomic_result = atomicInc(&write_index, write_index);
  //    if (atomic_result == write_index)
  //      continue;
  //  
  //  } else { /* Wrap around write_index if at end of ringbuf */
  //    atomic_result = atomicCAS(&write_index, write_index, 0);
  //    if (atomic_result != 0)
  //      continue;
  //  }

  //  break;
  //}
  GPU_SPINLOCK_UNLOCK(&ringbuf->gpu_spin_lock);

  // XXX TODO write_index is index to write to
    
  
  __threadfence_system();
  // TODO Set rdy flag
  __threadfence_system();
  END_SINGLE_THREAD;
  
  return true;
}

