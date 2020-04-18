#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gpu_file.ch"
#include "request.ch"
#include "util.ch"

/* Allocate the global file table */
__host__
global_file_table_t * init_gpu_file() {
  global_file_table_t * dev_ptr;
  CUDA_CALL(cudaMalloc((void **) &dev_ptr, NUM_BLOCKS * MAX_FILES * sizeof(file_t)));
  return dev_ptr;
}

__device__
void gpu_file_open(file_table_t * file_table) {
  ;
}

__device__
void gpu_file_grow(file_table_t * file_table) {
  ;
}

__device__
void gpu_file_close(file_table_t * file_table) {
  ;
}

__host__
void handle_gpu_file_open(request_t * request, response_t * ret_response) {
  permissions_t permissions = request->permissions;
  char * file_name = request->file_name;



  int oflag = O_CREAT;
  if (permissions & R___)
    oflag |= O_RDONLY;
    
  if (permissions & _W__)
    oflag |= O_WRONLY;

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
