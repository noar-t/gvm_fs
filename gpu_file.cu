#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gpu_file.ch"
#include "ringbuf.ch"
#include "types.ch"
#include "util.ch"

__device__ __constant__ global_file_meta_table_t  * global_file_meta_table;

/* Allocate the global file table */
__host__
void init_gpu_file() {
  global_file_meta_table_t * dev_ptr;
  CUDA_CALL(cudaMalloc((void **) &dev_ptr, NUM_BLOCKS * sizeof(file_meta_table_t)));
  CUDA_CALL(cudaMemset(dev_ptr, 0, NUM_BLOCKS * sizeof(file_meta_table_t)));
  CUDA_CALL(cudaMemcpyToSymbol(global_file_meta_table, &dev_ptr,
                               sizeof(global_file_meta_table_t *)));
}

__device__
gpu_fd gpu_file_open(char * file_name, permissions_t permissions) {
  // TODO may need to make gpu_file_* into a single thread function
  request_t open_request = {0};
  open_request.request_type = OPEN_REQUEST;
  open_request.permissions  = permissions;
  gpu_str_cpy(file_name, open_request.file_name, MAX_PATH_SIZE);

  response_t response = {0};
  gpu_enqueue(&open_request, &response);


  printf("debug placeholder %s\n", response.file_data);
  // TODO do something with file data
  file_meta_table_t * file_meta_table = (file_meta_table_t *) &(global_file_meta_table[blockIdx.x *MAX_FILES]);

  for (int i = 0; i < MAX_FILES; i++) {
    if (!file_meta_table->files[i].in_use) {
      file_meta_table->files[i] = {
        .in_use = true,
        .host_fd = response.host_fd,
        .current_size = (size_t) response.file_size,
        .permissions = response.permissions,
        .offset = 0,
      };

      return i;
    }
  }

  return FILE_TABLE_FULL;
}

__device__
void gpu_file_grow(void) {
  ; 
}

__device__
void gpu_file_close(void) { ; }

__host__
void handle_gpu_file_open(volatile request_t * request, volatile response_t * ret_response) {
  permissions_t permissions = request->permissions;
  char * file_name = (char *) request->file_name;

  int oflag = 0;//O_CREAT;
  if (permissions == R___)
    oflag |= O_RDONLY;
  else if (permissions == _W__)
    oflag |= O_WRONLY;
  else if (permissions == RW__)
    oflag |= O_RDWR;

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
}

__host__
void handle_gpu_file_grow(volatile request_t * request, volatile response_t * ret_response) {
  // TODO might be best to close the file and flush then grow and reopen
  ;
}

__host__
void handle_gpu_file_close(volatile request_t * request, volatile response_t * ret_response) {
  // TODO
  ;
}
