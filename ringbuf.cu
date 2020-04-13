#include <pthread.h>
#include <stdio.h>

#include "request.ch"
#include "ringbuf.ch"
#include "util.ch"

// TODO add fd array for each process
void * request_handler_thread_func(void *);

__host__
ringbuf_t * init_ringbuf() {
  ringbuf_t * ringbuf = NULL;
  CUDA_CALL(cudaMallocManaged(&ringbuf, sizeof(ringbuf_t)));

  int dev_id;
  CUDA_CALL(cudaGetDevice(&dev_id));
  CUDA_CALL(cudaMemAdvise(ringbuf, sizeof(ringbuf_t), 
                          cudaMemAdviseSetAccessedBy, dev_id));
  CUDA_CALL(cudaMemset(ringbuf, 0, sizeof(ringbuf_t)));

  ringbuf->cpu_mutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
  CUDA_CALL(cudaMalloc(&ringbuf->gpu_mutex, sizeof(gpu_mutex_t)));
  CUDA_CALL(cudaMemset(ringbuf->gpu_mutex, 0, sizeof(gpu_mutex_t)));

  if (pthread_create(&ringbuf->request_handler,
                     NULL, request_handler_thread_func, NULL) != 0)
    PRINT_ERR("pthread_create failed");

  return ringbuf;
}

__host__
void free_ringbuf(ringbuf_t * ringbuf) {
  free(ringbuf->cpu_mutex);
  CUDA_CALL(cudaFree(ringbuf));
}


__host__
void * request_handler_thread_func(void *) {
  printf("REPLACE ME WITH THREAD LOGIC\n");
  pthread_exit(NULL);
}

__host__
bool cpu_dequeue(ringbuf_t * ringbuf, request_t * ret_request) {
  bool success = true;

  //CPU_SPINLOCK_LOCK(ringbuf->cpu_mutex);
  unsigned int read_index = ringbuf->read_index;
  if (read_index == ringbuf->write_index)
    return false;

  if (read_index >= (RINGBUF_SIZE-1)) {
    ringbuf->read_index = 0;
    read_index = 0;
  } // HANDLE wrap around at the end
  //CPU_SPINLOCK_UNLOCK(ringbuf->cpu_mutex);

  request_t * cur_request = NULL;//&(ringbuf->requests[ringbuf->read_index]);
 
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
  GPU_SPINLOCK_LOCK(ringbuf->gpu_mutex);

  //int write_index = 0;

  ringbuf->requests[blockIdx.x].request_type = new_request->request_type;
  ringbuf->requests[blockIdx.x].placeholder = new_request->placeholder;

  __threadfence_system();

  /* Enable read flag */
  ringbuf->requests[blockIdx.x].ready_to_read = true;
  __threadfence_system();

  GPU_SPINLOCK_UNLOCK(ringbuf->gpu_mutex);
  END_SINGLE_THREAD;
  
  return true;
}

