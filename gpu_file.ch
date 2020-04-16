#ifndef GPU_FILE_CH
#define GPU_FILE_CH

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

__host__ char * handle_gpu_file_open(char * file_name, permissions_t permissions, int * host_fd);
__host__ char * handle_gpu_file_grow(int host_fd, size_t new_size);
__host__ bool handle_gpu_file_close(int host_fd);

#endif
