#include <stdio.h>

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
    gpu_fd fd = gpu_file_open("/home/noah/School/Thesis/gvm_fs/files/sm_test_file.txt", RW__);
    char * file_buf = NULL;
    gpu_file_read(fd, 16, &file_buf);
    //printf("Filebuf: %c%c%c%c", file_buf[0], file_buf[1], file_buf[2], file_buf[3]);
  }
}


int main() {
  printf("MAIN\n");
  init_rpc_queue();
  init_gpu_file();
  printf("DONE init_rpc_queue\n");

  //fill_queue<<<rpc_queue_SIZE, 32>>>();
  fill_queue<<<10, 32>>>();

  CUDA_CALL(cudaDeviceSynchronize());
  //sleep(10);
  printf("Kernel finished\n");

  free_rpc_queue();
}



