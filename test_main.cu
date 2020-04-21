#include <stdio.h>

#include "gpu_file.ch"
#include "ringbuf.ch"
#include "types.ch"
#include "util.ch"

#include <unistd.h>

__global__
void fill_queue(void) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
  //if ((blockIdx.x == 0 && threadIdx.x == 0) || (blockIdx.x == 1 && threadIdx.x == 0)) {
  //if (threadIdx.x == 0) {
    for (int i = 0; i <= MAX_FILES; i++ ) {
      gpu_fd fd = gpu_file_open("/home/noah/School/gvm_fs/files/sm_test_file.txt", RW__);
      printf("fd%d\n", fd);
    }
    //char * file_buf = NULL;
    //size_t bytes_read = gpu_file_read(fd, 16, &file_buf);
    //printf("Bytes read: %d\n", bytes_read);
    //printf("File: %s\n", file_buf);
    //file_buf[blockIdx.x] = '*';
    //printf("File: %s\n", file_buf);
    ////gpu_file_grow(fd, 32);
    //gpu_file_close(fd);
  }
}


int main() {
  printf("MAIN\n");
  init_ringbuf();
  init_gpu_file();
  printf("DONE init_ringbuf\n");

  //fill_queue<<<RINGBUF_SIZE, 32>>>();
  fill_queue<<<10, 32>>>();

  CUDA_CALL(cudaDeviceSynchronize());
  //sleep(10);
  printf("Kernel finished\n");

  free_ringbuf();
}



