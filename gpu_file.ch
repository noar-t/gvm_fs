#ifndef GPU_FILE_CH
#define GPU_FILE_CH

#include "types.ch"


__host__ void init_gpu_file();
__device__ gpu_fd gpu_file_open(char * file_name, permissions_t permissions);
__device__ void gpu_file_grow(void);
__device__ void gpu_file_close(void);
__host__ void handle_gpu_file_open(volatile request_t * request, volatile response_t * ret_response);
__host__ void handle_gpu_file_grow(volatile request_t * request, volatile response_t * ret_response);
__host__ void handle_gpu_file_close(volatile request_t * request, volatile response_t * ret_response);

#endif
