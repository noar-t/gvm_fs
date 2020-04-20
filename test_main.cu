#include <stdio.h>

#include "gpu_file.ch"
#include "ringbuf.ch"
#include "types.ch"
#include "util.ch"

#include <unistd.h>

__global__
void fill_queue(void) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    gpu_fd fd = gpu_file_open("/home/noah/School/gvm_fs/files/sm_test_file.txt", RW__);
    char * file_buf = NULL;
    size_t bytes_read = gpu_file_read(fd, 8, &file_buf);
    printf("Bytes read: %d\n", bytes_read);
    file_buf[7] = '\0';
    printf("File: %s\n", file_buf);
    file_buf[0] = '*';
    file_buf[1] = '*';
    printf("File: %s\n", file_buf);
    gpu_file_grow(fd, 32);
    gpu_file_close(fd);
  }
}


int main() {
  printf("MAIN\n");
  init_ringbuf();
  init_gpu_file();
  printf("DONE init_ringbuf\n");

  fill_queue<<<RINGBUF_SIZE, 32>>>();

  CUDA_CALL(cudaDeviceSynchronize());
  //sleep(10);
  printf("Kernel finished\n");

  free_ringbuf();
}



