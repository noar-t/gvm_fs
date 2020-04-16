#include <stdio.h>

#include "gpu_file.ch"
#include "util.ch"

/* Allocate the global file table */
__host__
global_file_table_t * init_gpu_file() {
  global_file_table_t * dev_ptr;
  CUDA_CALL(cudaMalloc((void **) &dev_ptr, NUM_BLOCKS * MAX_FILES * sizeof(file_t)));
  return dev_ptr;
}

__device__
void gpu_file_open(file_table_t * file_table) {
  ;
}

__device__
void gpu_file_grow(file_table_t * file_table) {
  ;
}

__device__
void gpu_file_close(file_table_t * file_table) {
  ;
}

__host__
char * handle_gpu_file_open(char * file_name, permissions_t permissions, int * host_fd) {
  // TODO
  return NULL;
}

__host__
char * handle_gpu_file_grow(int host_fd, size_t new_size) {
  // TODO might be best to close the file and flush then grow and reopen
  return NULL;
}

__host__
bool handle_gpu_file_close(int host_fd) {
  // TODO
  return true;
}
