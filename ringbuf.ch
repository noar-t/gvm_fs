#ifndef RINGBUF_CH
#define RINGBUF_CH

#include "request.ch"

#define RINGBUF_SIZE 100

typedef struct ringbuf_t {
  bool cpu_spin_lock; 
  unsigned int gpu_spin_lock;
  volatile unsigned int write_index;
  volatile unsigned int read_index;
  request_t requests[RINGBUF_SIZE]; // TODO replace with special request datatype
} ringbuf_t;

__host__ ringbuf_t * init_ringbuf();
__host__ void free_ringbuf(ringbuf_t * ringbuf);
__host__ bool cpu_dequeue(ringbuf_t * ringbuf, request_t * ret_request);
__device__ bool gpu_enqueue(ringbuf_t * ringbuf, request_t * new_request);

#endif
