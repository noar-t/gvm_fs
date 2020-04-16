#include <stdio.h>

#include "gpu_file.ch"

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
