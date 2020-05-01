#ifndef GVM_FS_H
#define GVM_FS_H

#include <inttypes.h>

#include "gpu_file.ch"
#include "memory_pool.ch"
#include "rpc_queue.ch"
#include "types.ch"

__host__
void init_gvm_fs() {
  init_rpc_queue();
  init_gpu_file();
  init_memory_pool();
}

__host__
void cleanup_gvm_fs() {
  free_rpc_queue();
  //TODO free gpu file mem
  //TODO free memory pool
}

__device__ int gvm_fs_open(char * file_name) { 
  return gpu_file_open(file_name, RWX_);
}

__device__ size_t gvm_fs_read(int fd, size_t size, char ** ret_ptr) {
  return gpu_file_read(fd, size, ret_ptr);
}

__device__ void gvm_fs_grow(int fd, size_t size) {
  return gpu_file_grow(fd, size);
}

__device__ void gvm_fs_close(int fd) {
  return gpu_file_close(fd);
}

#endif
