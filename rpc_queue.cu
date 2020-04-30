#include <pthread.h>
#include <stdio.h>

#include "gpu_file.ch"
#include "rpc_queue.ch"
#include "types.ch"
#include "util.ch"

// TODO remove
#include <unistd.h>
#include <inttypes.h>

#define QUEUE_EMPTY -1

static bool run_host_thread = true;
__device__ __constant__ rpc_queue_t * gpu_rpc_queue_ref;
rpc_queue_t * cpu_rpc_queue_ref;

__host__ static void * host_thread_func(void * unused);
__host__ static int poll_queue();
__host__ static void handle_request(int index);

/* Init all rpc_queue memory and create the request handler thread */
__host__
rpc_queue_t * init_rpc_queue() {
  rpc_queue_t * rpc_queue = NULL;
  CUDA_CALL(cudaMallocManaged(&rpc_queue, sizeof(rpc_queue_t)));

  int dev_id;
  CUDA_CALL(cudaGetDevice(&dev_id));
  CUDA_CALL(cudaMemAdvise(rpc_queue, sizeof(rpc_queue_t), 
                          cudaMemAdviseSetAccessedBy, dev_id));
  CUDA_CALL(cudaMemset(rpc_queue, 0, sizeof(rpc_queue_t)));

  cpu_rpc_queue_ref = rpc_queue;
  CUDA_CALL(cudaMemcpyToSymbol(gpu_rpc_queue_ref, &rpc_queue, sizeof(rpc_queue_t *)));

  CUDA_CALL(cudaMalloc(&rpc_queue->gpu_mutex, sizeof(gpu_mutex_t)));
  CUDA_CALL(cudaMemset(rpc_queue->gpu_mutex, 0, sizeof(gpu_mutex_t)));

  if (pthread_create(&rpc_queue->request_handler,
                     NULL, host_thread_func, NULL) != 0)
    PRINT_ERR("pthread_create failed");

  return rpc_queue;
}

/* Free all/join resources associated with the rpc_queue */
__host__
void free_rpc_queue(void) {
  run_host_thread = false;
  pthread_join(cpu_rpc_queue_ref->request_handler, NULL);

  CUDA_CALL(cudaFree((void *) cpu_rpc_queue_ref->gpu_mutex));
  CUDA_CALL(cudaFree((void *) cpu_rpc_queue_ref));
}


__host__
static void * host_thread_func(void * unused) {
  while (run_host_thread) {
    int queue_index = poll_queue();
    if (queue_index != QUEUE_EMPTY) {
      printf("FOUND ONE %d\n", queue_index);
      request_t * request = &(cpu_rpc_queue_ref->requests[queue_index]);
      printf("CPU Request: {.type = %d, .file_name = %s, .permissions = %d, .host_fd = %d,"
             " .file_mem = %p, .new_size = %u, .current_size = %u } \n",
             request->request_type, request->file_name, request->permissions,
             request->host_fd, request->file_mem, request->new_size, request->current_size);
      __sync_synchronize();
      handle_request(queue_index);
    }
  }

  pthread_exit(NULL);
}

__host__
static int poll_queue(void) {
  for (int i = 0; i < RPC_QUEUE_SIZE; i++) {
    if (cpu_rpc_queue_ref->requests[i].ready_to_read) {
      return i;
    }
  }

  return QUEUE_EMPTY;
}

__host__
static void handle_request(int index) {
  volatile request_t * cur_request = &(cpu_rpc_queue_ref->requests[index]);
  volatile response_t * ret_response = &(cpu_rpc_queue_ref->responses[index]);

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
  memset(&(cpu_rpc_queue_ref->requests[index]), 0, sizeof(request_t));
  printf("CPU Response: {.host_fd = %d, .file_size = %ld, .permissions = %d, .file_data = %p }\n",
          ret_response->host_fd, ret_response->file_size,
          ret_response->permissions, ret_response->file_data);


  //printf("test: %x\n", test);
  __sync_synchronize();
  cpu_rpc_queue_ref->responses[index].ready_to_read = true;
  __sync_synchronize();
}

/* Write a request into the rpc_queue,
   XXX currently not a rpc_queue, just a array
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
  GPU_SPINLOCK_LOCK(gpu_rpc_queue_ref->gpu_mutex);

  /* Copy the request into the request buffer */
  volatile request_t * cur_request = &(gpu_rpc_queue_ref->requests[blockIdx.x]);
  cur_request->request_type = new_request->request_type;
  cur_request->permissions  = new_request->permissions;
  cur_request->host_fd      = new_request->host_fd;
  cur_request->file_mem     = new_request->file_mem;
  cur_request->new_size     = new_request->new_size;
  cur_request->current_size = new_request->current_size;
  gpu_str_cpy((char *) new_request->file_name, (char *) cur_request->file_name, MAX_PATH_SIZE);

  /* Clear out response such that it can be used for our new request */
  volatile response_t * cur_response = &(gpu_rpc_queue_ref->responses[blockIdx.x]);
  cur_response->ready_to_read = false;
  cur_response->host_fd       = 0;
  cur_response->file_size     = 0;
  cur_response->permissions   = RW__; // TODO fix
  cur_response->file_data     = NULL;

  printf("GPU Request (%d): {.type = %d, .file_name = %s, .permissions = %d, .host_fd = %d,"
         " .file_mem = %p, .new_size = %d, .current_size = %d } \n",
         blockIdx.x,
         cur_request->request_type, cur_request->file_name, cur_request->permissions,
         cur_request->host_fd, cur_request->file_mem, (int) cur_request->new_size, (int) cur_request->current_size);

  __threadfence_system();
  /* Enable read flag */
  cur_request->ready_to_read = true;
  __threadfence_system();

  while (cur_response->ready_to_read != true) {
    ;/* XXX wait for CPU to respond to request */
  }
  printf("request made\n");

  __threadfence_system();
  printf("GPU Response (%d): {.host_fd = %d, .file_size = %ld, .permissions = %d, .file_data = %p }\n",
          blockIdx.x,
          cur_response->host_fd, cur_response->file_size,
          cur_response->permissions, cur_response->file_data);

  ret_response->host_fd     = cur_response->host_fd;
  ret_response->file_size   = cur_response->file_size;
  ret_response->permissions = cur_response->permissions;
  ret_response->file_data   = cur_response->file_data;
  __threadfence_system();
  printf("What error\n");


  GPU_SPINLOCK_UNLOCK(gpu_rpc_queue_ref->gpu_mutex);
  END_SINGLE_THREAD;
}

