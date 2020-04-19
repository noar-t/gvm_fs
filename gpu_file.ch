#ifndef GPU_FILE_CH
#define GPU_FILE_CH

#include "types.ch"

#define MAX_FILES 128

#define NUM_BLOCKS 100

__host__ void init_gpu_file();
__device__ void gpu_file_open(char * file_name, permissions_t permissions);
__device__ void gpu_file_grow(void);
__device__ void gpu_file_close(void);
__host__ void handle_gpu_file_open(request_t * request, response_t * ret_response);
__host__ void handle_gpu_file_grow(request_t * request, response_t * ret_response);
__host__ void handle_gpu_file_close(request_t * request, response_t * ret_response);

#endif
