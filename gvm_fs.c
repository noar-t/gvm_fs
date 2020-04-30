
#include "gvm_fs.h"

#include "gpu_file.ch"
#include "rpc_queue.ch"
#include "memory_pool.ch"

void init_gvm_fs() {
  init_rpc_queue();
  init_gpu_file();
  init_memory_pool();
}

void cleanup_gvm_fs() {
  free_rpc_queue();
  //TODO free gpu file mem
  //TODO free memory pool
}
