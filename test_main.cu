#include <stdio.h>

#include "gpu_file.ch"
#include "ringbuf.ch"
#include "types.ch"
#include "util.ch"

#include <unistd.h>

__global__
void fill_queue(void) {
  if (blockIdx.x == 0 && threadIdx.x == 0)
    gpu_file_open("/home/noah/School/gvm_fs/files/testFile.txt", RW__);
}


int main() {
  printf("MAIN\n");
  init_ringbuf();
  printf("DONE init_ringbuf\n");

  fill_queue<<<RINGBUF_SIZE, 32>>>();

  CUDA_CALL(cudaDeviceSynchronize());
  sleep(10);

  free_ringbuf();
}



