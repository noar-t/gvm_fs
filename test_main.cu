#include <stdio.h>

#include "ringbuf.ch"
#include "util.ch"

__global__
void fill_queue(ringbuf_t * ringbuf) {
  request_t request = { .request_type = test, .placeholder = 77};
  while (!gpu_enqueue(ringbuf, &request)) {;}
}


int main() {
  printf("MAIN\n");
  ringbuf_t * ringbuf = init_ringbuf();

  fill_queue<<<RINGBUF_SIZE * 2, 32>>>(ringbuf);

  CUDA_CALL(cudaDeviceSynchronize());
  __sync_synchronize();
  for (int i = 0; i < RINGBUF_SIZE; i++) {
    printf("ringbuf[%d] = {.ready_to_read = %d, .request_type = %d, .placeholder = %d}\n",
        i,
        ringbuf->requests[i].ready_to_read,
        ringbuf->requests[i].request_type,
        ringbuf->requests[i].placeholder);
  }

  free_ringbuf(ringbuf);
}



