#include <stdio.h>

#include "ringbuf.ch"
#include "util.ch"

__global__
void fill_queue(ringbuf_t * ringbuf) {
  request_t request = { .request_type = OPEN_REQUEST, .host_fd = blockIdx.x};
  while (!gpu_enqueue(ringbuf, &request)) {;}
}


int main() {
  printf("MAIN\n");
  ringbuf_t * ringbuf = init_ringbuf();
  printf("DONE init_ringbuf\n");

  fill_queue<<<RINGBUF_SIZE, 32>>>(ringbuf);

  CUDA_CALL(cudaDeviceSynchronize());
  __sync_synchronize();
  //printf("counter %d\n", ringbuf->tmp_counter2);
  //while (1) {
    for (int i = 0; i < RINGBUF_SIZE; i++) {
      printf("ringbuf[%d] = {.ready_to_read = %d, .request_type = %d}\n",
          i,
          ringbuf->requests[i].ready_to_read,
          ringbuf->requests[i].request_type);
    }
  //}

  free_ringbuf(ringbuf);
}



