#include <stdio.h>

#include "gvm_fs.ch"
#include "gpu_file.ch"
#include "rpc_queue.ch"
#include "types.ch"
#include "util.ch"

#include <unistd.h>

__global__
void fill_queue(void) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
  //if ((blockIdx.x == 0 && threadIdx.x == 0) || (blockIdx.x == 1 && threadIdx.x == 0)) {
  //if (threadIdx.x == 0) {
    //gpu_fd fd = gpu_file_open("/home/noah/School/gvm_fs/files/sm_test_file.txt", RW__);
    //char * file_buf = NULL;
    //size_t bytes_read = gpu_file_read(fd, 16, &file_buf);
    //printf("Bytes read: %d\n", bytes_read);
    //printf("File: %s\n", file_buf);
    //file_buf[blockIdx.x] = '*';
    //printf("File: %s\n", file_buf);
    ////gpu_file_grow(fd, 32);
    //gpu_file_close(fd);
    int gpu_fd = gvm_fs_open((char *) "/home/noah/School/Thesis/gvm_fs/files/sm_test_file.txt");
    char * file_buf = NULL;
    gvm_fs_read(gpu_fd, 16, &file_buf);
    file_buf[5] = '\0';
    printf("Filebuf: %s\n", file_buf);
  }
}


int main() {
  printf("MAIN\n");
  init_gvm_fs();
  printf("DONE init_rpc_queue\n");

  //fill_queue<<<rpc_queue_SIZE, 32>>>();
  fill_queue<<<10, 32>>>();

  CUDA_CALL(cudaDeviceSynchronize());
  //sleep(10);
  printf("Kernel finished\n");

  cleanup_gvm_fs();
}



