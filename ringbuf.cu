#include <pthread.h>
#include <stdio.h>

#include "gpu_file.ch"
#include "ringbuf.ch"
#include "types.ch"
#include "util.ch"

// TODO remove
#include <unistd.h>

#define QUEUE_EMPTY -1

static bool run_host_thread = true;
__device__ __constant__ ringbuf_t * gpu_ringbuf_ref;
ringbuf_t * cpu_ringbuf_ref;

__host__ void * host_thread_func(void * unused);
__host__ int poll_queue();
__host__ void handle_request(int index);

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

  cpu_ringbuf_ref = ringbuf;
  CUDA_CALL(cudaMemcpyToSymbol(gpu_ringbuf_ref, &ringbuf, sizeof(ringbuf_t *)));

  CUDA_CALL(cudaMalloc(&ringbuf->gpu_mutex, sizeof(gpu_mutex_t)));
  CUDA_CALL(cudaMemset(ringbuf->gpu_mutex, 0, sizeof(gpu_mutex_t)));

  if (pthread_create(&ringbuf->request_handler,
                     NULL, host_thread_func, NULL) != 0)
    PRINT_ERR("pthread_create failed");

  return ringbuf;
}

/* Free all/join resources associated with the ringbuf */
__host__
void free_ringbuf(void) {
  run_host_thread = false;
  pthread_join(cpu_ringbuf_ref->request_handler, NULL);

  CUDA_CALL(cudaFree((void *) cpu_ringbuf_ref->gpu_mutex));
  CUDA_CALL(cudaFree((void *) cpu_ringbuf_ref));
}


__host__
void * host_thread_func(void * unused) {
  printf("REPLACE ME WITH THREAD LOGIC\n");

  while (run_host_thread) {
    //printf("LOOP\n");
    int queue_index = poll_queue();
    if (queue_index != QUEUE_EMPTY) {
      printf("FOUND ONE %d\n", queue_index);
      request_t * request = &(cpu_ringbuf_ref->requests[queue_index]);
      printf("Request: {.type = %d, .file_name = %s, .permissions = %d, .host_fd = %d, .new_size = %u } \n",
             request->request_type, request->file_name, request->permissions,
             request->host_fd, request->new_size);
      //sleep(10);
      __sync_synchronize();
      //sleep(10);
      handle_request(queue_index);
      break;
    }
  }

  pthread_exit(NULL);
}

__host__
int poll_queue(void) {
  for (int i = 0; i < RINGBUF_SIZE; i++) {
    if (cpu_ringbuf_ref->requests[i].ready_to_read) {
      return i;
    }
  }

  return QUEUE_EMPTY;
}

__host__
void handle_request(int index) {
  request_t * cur_request = &(cpu_ringbuf_ref->requests[index]);
  response_t * ret_response = &(cpu_ringbuf_ref->responses[index]);

  /* Handle CPU side of request */
  /* Fill out request */
  switch (cur_request->request_type) {
    case OPEN_REQUEST:
      handle_gpu_file_open(cur_request, ret_response);
      break;
    case CLOSE_REQUEST:
      handle_gpu_file_close(cur_request, ret_response);
      break;
    case GROW_REQUEST:
      handle_gpu_file_grow(cur_request, ret_response);
      break;
    default:
      PRINT_ERR("Bad request type\n");
      break;
  }

  /* Clear out request once it is filled */
  memset(&(cpu_ringbuf_ref->requests[index]), 0, sizeof(request_t));
  printf("resp ptr %x\n", cpu_ringbuf_ref->responses[index].file_data);

  //cpu_ringbuf_ref->responses[index].file_size = (off_t) cpu_ringbuf_ref->responses[index].file_data;
  __sync_synchronize();
  cpu_ringbuf_ref->responses[index].ready_to_read = true;
  __sync_synchronize();
}

/* Write a request into the ringbuf,
   XXX currently not a ringbuf, just a array
   big enough for each Java thread to have its
   own unique entry into the array. Can be modified
   later to be circular, but it has to be polled
   either way so doesn't seem more efficient. */
/* TODO could make more efficient with 0 copy, right now
   the gpu needs to fill int he struct which gets copied
   to this shared memory buffer, instead we could potentially
   just have it place data directly into the buffer */
// TODO might be better to split somehow or rename
__device__
void gpu_enqueue(request_t * new_request, response_t * ret_response) { 
  BEGIN_SINGLE_THREAD;
  GPU_SPINLOCK_LOCK(gpu_ringbuf_ref->gpu_mutex);

  /* Copy the request into the request buffer */
  request_t * cur_request = &(gpu_ringbuf_ref->requests[blockIdx.x]);
  cur_request->request_type = new_request->request_type;
  cur_request->permissions  = new_request->permissions;
  cur_request->host_fd      = new_request->host_fd;
  cur_request->new_size     = new_request->new_size;
  gpu_str_cpy(new_request->file_name, cur_request->file_name, MAX_PATH_SIZE);

  /* Clear out response such that it can be used for our new request */
  response_t * cur_response = &(gpu_ringbuf_ref->responses[blockIdx.x]);
  cur_response->ready_to_read = false;
  cur_response->host_fd       = 0;
  cur_response->file_size     = 0;
  cur_response->permissions   = RW__; // TODO fix
  cur_response->file_data     = NULL;

  __threadfence_system();
  /* Enable read flag */
  cur_request->ready_to_read = true;
  __threadfence_system();

  while (cur_response->ready_to_read != true) {
    ;/* XXX wait for CPU to respond to request */
  }
  __threadfence_system();
  printf("response received %d:%d:%u:%x\n", cur_response->ready_to_read, 
      cur_response->host_fd, cur_response->file_size, cur_response->file_data);
  //printf("response received %d:%d:%s\n", cur_response->ready_to_read, 
  //    cur_response->host_fd, cur_response->file_data);
  ret_response->host_fd     = cur_response->host_fd;
  ret_response->file_size   = cur_response->file_size;
  ret_response->permissions = cur_response->permissions;
  ret_response->file_data   = cur_response->file_data;
  __threadfence_system();


  GPU_SPINLOCK_UNLOCK(gpu_ringbuf_ref->gpu_mutex);
  END_SINGLE_THREAD;
}

