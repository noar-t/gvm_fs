#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gpu_file.ch"
#include "ringbuf.ch"
#include "types.ch"
#include "util.ch"

__device__ __constant__ global_file_table_t  * global_file_table;

/* Allocate the global file table */
__host__
void init_gpu_file() {
  global_file_table_t * dev_ptr;
  CUDA_CALL(cudaMalloc((void **) &dev_ptr, NUM_BLOCKS * MAX_FILES * sizeof(file_t)));
  CUDA_CALL(cudaMemcpyToSymbol(global_file_table, &dev_ptr,
                               sizeof(global_file_table_t *)));
}

__device__
void gpu_file_open(char * file_name, permissions_t permissions) {
  request_t open_request = {0};
  open_request.request_type = OPEN_REQUEST;
  open_request.permissions  = permissions;
  gpu_str_cpy(file_name, open_request.file_name, MAX_PATH_SIZE);

  response_t response = {0};
  gpu_enqueue(&open_request, &response);


  printf("debug placeholder %s\n", response.file_data);
  // TODO do something with file data
}

__device__
void gpu_file_grow(void) { ; }

__device__
void gpu_file_close(void) { ; }

__host__
void handle_gpu_file_open(request_t * request, response_t * ret_response) {
  permissions_t permissions = request->permissions;
  char * file_name = request->file_name;

  int oflag = 0;//O_CREAT;
  if (permissions == R___)
    oflag |= O_RDONLY;
  else if (permissions == _W__)
    oflag |= O_WRONLY;
  else if (permissions == RW__)
    oflag |= O_RDWR;

  //if (permissions & __X_)
    //oflag |= O_EXEC; TODO undefined on this system for some reason

  int fd = open(file_name, oflag);
  if (fd == -1)
    perror("handle_gpu_file_open open() failed\n");

  struct stat file_stat;
  int err = fstat(fd, &file_stat);
  if (err == -1)
    perror("handle_gpu_file_open fstat() failed\n");

  off_t file_size = file_stat.st_size;
  // TODO should be able to create files, but cant currently
  assert(file_size > 0); 
 
  char * file_mem = NULL; 
  CUDA_CALL(cudaMallocManaged(&file_mem, file_size));
  ssize_t bytes_read = read(fd, file_mem, file_size);
  if (bytes_read != file_size)
    perror("handle_gpu_file_open error reading file\n");

  ret_response->host_fd     = fd;
  ret_response->file_size   = file_size;
  ret_response->permissions = permissions;
  ret_response->file_data   = file_mem;
  printf("file data %x\n", ret_response->file_data);
}

__host__
void handle_gpu_file_grow(request_t * request, response_t * ret_response) {
  // TODO might be best to close the file and flush then grow and reopen
  ;
}

__host__
void handle_gpu_file_close(request_t * request, response_t * ret_response) {
  // TODO
  ;
}
