#ifndef RINGBUF_CH
#define RINGBUF_CH

#include "request.ch"
#include "util.ch"

#define RINGBUF_SIZE 100

typedef struct ringbuf_t {
  cpu_mutex_t * cpu_mutex;
  gpu_mutex_t * gpu_mutex;

  unsigned int tmp_counter;
  volatile unsigned int write_index;
  volatile unsigned int read_index;
  request_t requests[RINGBUF_SIZE]; // TODO replace with special request datatype
} ringbuf_t;

__host__ ringbuf_t * init_ringbuf();
__host__ void free_ringbuf(ringbuf_t * ringbuf);
__host__ bool cpu_dequeue(ringbuf_t * ringbuf, request_t * ret_request);
__device__ bool gpu_enqueue(ringbuf_t * ringbuf, request_t * new_request);

#endif
