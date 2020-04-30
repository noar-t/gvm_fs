#ifndef RPC_QUEUE_CH
#define RPC_QUEUE_CH

#include <pthread.h>

#include "types.ch"
#include "util.ch"

__host__ rpc_queue_t * init_rpc_queue();
__host__ void free_rpc_queue();
__device__ void gpu_enqueue(request_t * new_request, response_t * ret_response);

#endif
