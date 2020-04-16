#include <pthread.h>
#include <stdio.h>

#include "request.ch"
#include "ringbuf.ch"
#include "util.ch"

#define QUEUE_EMPTY -1

static bool run_host_thread = true;

__host__ void * host_thread_func(void * void_ringbuf);
__host__ int poll_queue(ringbuf_t * ringbuf);
__host__ void handle_request(ringbuf_t * ringbuf, int index);

/* Init all ringbuf memory and create the request handler thread */
__host__
ringbuf_t * init_ringbuf() {
  ringbuf_t * ringbuf = NULL;
  CUDA_CALL(cudaMallocManaged(&ringbuf, sizeof(ringbuf_t)));

  int dev_id;
  CUDA_CALL(cudaGetDevice(&dev_id));
  CUDA_CALL(cudaMemAdvise(ringbuf, sizeof(ringbuf_t), 
                          cudaMemAdviseSetAccessedBy, dev_id));
  CUDA_CALL(cudaMemset(ringbuf, 0, sizeof(ringbuf_t)));

  //ringbuf->cpu_mutex = (pthread_mutex_t *) malloc(sizeof(pthread_mutex_t));
  CUDA_CALL(cudaMalloc(&ringbuf->gpu_mutex, sizeof(gpu_mutex_t)));
  CUDA_CALL(cudaMemset(ringbuf->gpu_mutex, 0, sizeof(gpu_mutex_t)));

  if (pthread_create(&ringbuf->request_handler,
                     NULL, host_thread_func, ringbuf) != 0)
    PRINT_ERR("pthread_create failed");

  return ringbuf;
}

/* Free all/join resources associated with the ringbuf */
__host__
void free_ringbuf(ringbuf_t * ringbuf) {
  run_host_thread = false;
  pthread_join(ringbuf->request_handler, NULL);

  //free(ringbuf->cpu_mutex);
  CUDA_CALL(cudaFree(ringbuf->gpu_mutex));
  CUDA_CALL(cudaFree(ringbuf));
}


__host__
void * host_thread_func(void * void_ringbuf) {
  ringbuf_t * ringbuf = (ringbuf_t *) void_ringbuf;
  // TODO validate argument passed correctly
  printf("REPLACE ME WITH THREAD LOGIC\n");

  while (run_host_thread) {
    printf("LOOP\n");
    int queue_index = poll_queue(ringbuf);
    if (queue_index != QUEUE_EMPTY) {
      printf("FOUND ONE %d\n", queue_index);
     // handle_request(ringbuf, queue_index);
    }
  }

  pthread_exit(NULL);
}

__host__
int poll_queue(ringbuf_t * ringbuf) {
  for (int i = 0; i < RINGBUF_SIZE; i++) {
    if (ringbuf->requests[i].ready_to_read) {
      return i;
    }
  }

  return QUEUE_EMPTY;
}

__host__
void handle_request(ringbuf_t * ringbuf, int index) {
  request_t * cur_request = &(ringbuf->requests[index]);

  char * file_data;
  bool success = false;
  int fd = 0;

  /* Handle CPU side of request */
  switch (cur_request->request_type) {
    case OPEN_REQUEST:
      file_data = handle_gpu_file_open(cur_request->file_name, 
                                         cur_request->permissions,
                                         &fd);
      break;
    case CLOSE_REQUEST:
      success = handle_gpu_file_close(cur_request->host_fd);
      break;
    case GROW_REQUEST:
      file_data = handle_gpu_file_grow(cur_request->host_fd,
                                         cur_request->new_size);
      break;
    default:
      PRINT_ERR("Bad request type\n");
      break;
  }


  /* Fill out request */

  ringbuf->responses[index].host_fd     = fd;
  //ringbuf->responses[index].file_size   = file_size; TODO file size
  ringbuf->responses[index].permissions = RWX_; // TODO fix
  ringbuf->responses[index].file_data   = file_data;

  /* Clear out request once it is filled */
  memset(&(ringbuf->requests[index]), 0, sizeof(request_t));

  __sync_synchronize();
  ringbuf->responses[index].ready_to_read = true;
  __sync_synchronize();
}

/* Write a request into the ringbuf,
   XXX currently not a ringbuf, just a array
   big enough for each Java thread to have its
   own unique entry into the array. Can be modified
   later to be circular, but it has to be polled
   either way so doesn't seem more efficient. */
__device__
bool gpu_enqueue(ringbuf_t * ringbuf, request_t * new_request) {
  BEGIN_SINGLE_THREAD;
  GPU_SPINLOCK_LOCK(ringbuf->gpu_mutex);

  ringbuf->requests[blockIdx.x].request_type = new_request->request_type;
  //ringbuf->requests[blockIdx.x].file_name    = new_request->file_name; TODO fix
  ringbuf->requests[blockIdx.x].permissions  = new_request->permissions;
  ringbuf->requests[blockIdx.x].host_fd      = new_request->host_fd;
  ringbuf->requests[blockIdx.x].new_size     = new_request->new_size;

  // TODO clear out response slot
  ringbuf->responses[blockIdx.x].ready_to_read = false;
  ringbuf->responses[blockIdx.x].host_fd       = 0;
  ringbuf->responses[blockIdx.x].file_size     = 0;
  ringbuf->responses[blockIdx.x].permissions   = RWX_; // TODO fix
  ringbuf->responses[blockIdx.x].file_data     = NULL;

  __threadfence_system();
  /* Enable read flag */
  ringbuf->requests[blockIdx.x].ready_to_read = true;
  __threadfence_system();

  GPU_SPINLOCK_UNLOCK(ringbuf->gpu_mutex);
  END_SINGLE_THREAD;
  
  return true;
}

