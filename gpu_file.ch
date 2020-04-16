#ifndef GPU_FILE_CH
#define GPU_FILE_CH

#define MAX_FILES 128

#define NUM_BLOCKS 100

// TODO file struct with offset, size, read position, etc 
typedef enum {
  R_,
  W_,
  X_,
  RW_,
  RX_,
  WX_,
  RWX_,
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
__host__ char * handle_gpu_file_open(char * file_name, permissions_t permissions, int * host_fd);
__host__ char * handle_gpu_file_grow(int host_fd, size_t new_size);
__host__ bool handle_gpu_file_close(int host_fd);

#endif
