#ifndef RINGBUF_CH
#define RINGBUF_CH

#include <pthread.h>

#include "request.ch"
#include "util.ch"

#define RINGBUF_SIZE 100

typedef struct ringbuf_t {
  /* Synchronization Variables */
  gpu_mutex_t * gpu_mutex;

  /* Request Handling Thread */
  pthread_t request_handler;

  /* Request Buffer Data */
  //volatile unsigned int tmp_counter;
  //volatile unsigned int write_index;
  //volatile unsigned int read_index;
  request_t requests[RINGBUF_SIZE]; /* XXX struct elements will be volatile */
  response_t responses[RINGBUF_SIZE];
} ringbuf_t;

__host__ ringbuf_t * init_ringbuf();
__host__ void free_ringbuf(ringbuf_t * ringbuf);
__device__ bool gpu_enqueue(ringbuf_t * ringbuf, request_t * new_request);

#endif
