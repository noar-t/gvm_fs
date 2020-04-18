#ifndef RINGBUF_CH
#define RINGBUF_CH

#include <pthread.h>

#include "types.ch"
#include "util.ch"

__host__ ringbuf_t * init_ringbuf();
__host__ void free_ringbuf();
__device__ void gpu_enqueue(request_t * new_request, response_t * ret_response);

#endif
