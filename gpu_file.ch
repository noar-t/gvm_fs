#ifndef GPU_FILE_CH
#define GPU_FILE_CH

#include "request.ch"

#define MAX_FILES 128

#define NUM_BLOCKS 100

// TODO file struct with offset, size, read position, etc 
typedef enum permissions_t {
  R___ = 0b100,
  _W__ = 0b010,
  __X_ = 0b001,
  RW__ = 0b110,
  R_X_ = 0b101,
  _WX_ = 0b011,
  RWX_ = 0b111,
} permissions_t;

/* GPU File Struct */
typedef struct file_t {
  int host_fd;
  size_t current_size;
  
  //int local_fd;
  permissions_t permissions;
  size_t offset;
} file_t;

typedef file_t ** global_file_table_t;
typedef file_t * file_table_t;

__host__ global_file_table_t * init_gpu_file();
__device__ void gpu_file_open(file_table_t * file_table);
__device__ void gpu_file_grow(file_table_t * file_table);
__device__ void gpu_file_close(file_table_t * file_table);
__host__ void handle_gpu_file_open(request_t * request, response_t * ret_response);
__host__ void handle_gpu_file_grow(request_t * request, response_t * ret_response);
__host__ void handle_gpu_file_close(request_t * request, response_t * ret_response);

#endif
