#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

#include "gpu_file.ch"
#include "memory_pool.ch"
#include "rpc_queue.ch"
#include "types.ch"
#include "util.ch"

// XXX REMOVE
#include <inttypes.h>

__device__ __constant__ global_file_meta_table_t  * global_file_meta_table;

/* Allocate the global file table */
__host__
void init_gpu_file() {
  global_file_meta_table_t * dev_ptr;
  size_t table_size = NUM_BLOCKS * MAX_FILES * sizeof(file_meta_table_t);
  printf("init_gpu_file: size %zu\n", table_size);
  CUDA_CALL(cudaMallocManaged((void **) &dev_ptr, table_size));
  CUDA_CALL(cudaMemset(dev_ptr, 0, table_size));
  CUDA_CALL(cudaMemcpyToSymbol(global_file_meta_table, &dev_ptr, 
                               sizeof(global_file_meta_table_t *)));
}

__device__
static file_t * get_file_from_gpu_fd(gpu_fd fd) {
  file_meta_table_t * file_meta_table =
        (file_meta_table_t *) &(global_file_meta_table[blockIdx.x * MAX_FILES]);
  file_t * cur_file = &(file_meta_table->files[fd]);

  if (!cur_file->in_use) {
    printf("Bad gpu_fd\n");
    return NULL;
  } else {
    return cur_file;
  }
}

__device__
gpu_fd gpu_file_open(char * file_name, permissions_t permissions) {
  // TODO may need to make gpu_file_* into a single thread function
  request_t open_request    = {0};
  open_request.request_type = OPEN_REQUEST;
  open_request.permissions  = permissions;
  gpu_str_cpy(file_name, open_request.file_name, MAX_PATH_SIZE);

  response_t response = {0};
  gpu_enqueue(&open_request, &response);

  //printf("debug placeholder %s\n", response.file_data);
  file_meta_table_t * file_meta_table =
        (file_meta_table_t *) &(global_file_meta_table[blockIdx.x * MAX_FILES]);

  /* Fill in slot in file descriptor table */
  for (int i = 0; i < MAX_FILES; i++) {
    if (!file_meta_table->files[i].in_use) {
      file_meta_table->files[i] = (file_t) {
        .in_use        = true,
        .host_fd       = response.host_fd,
        .current_size  = (size_t) response.file_size,
        .original_size = (size_t) response.file_size,
        .data          = response.file_data,
        .permissions   = response.permissions,
        .offset        = 0,
      };

      return i;
    }
  }

  return FILE_TABLE_FULL;
}

/* Read will return a pointer to memory starting at the 
   read offset. This buffer can also be used to write,
   as the memory in the buffer will be copied back to
   the file upon close. */
__device__
size_t gpu_file_read(gpu_fd fd, size_t size, char ** data_ptr) {
  file_t * cur_file = get_file_from_gpu_fd(fd);
  // TODO check file permissions

  size_t read_size = 0;
  if ((cur_file->offset + size) > cur_file->current_size) {
    read_size = cur_file->current_size - cur_file->offset;
  } else {
    read_size = size;
  }

  *data_ptr = (cur_file->data + cur_file->offset);
  cur_file->offset += read_size;

  return read_size;
}

__device__
off_t gpu_file_seek(gpu_fd fd, off_t offset, int whence) {
  file_t * cur_file = get_file_from_gpu_fd(fd);

  off_t new_offset = -1;
  switch (whence) {
    case SEEK_SET:
      if (offset >= cur_file->current_size) {
        new_offset = offset;
      }
      break;
    case SEEK_CUR:
      // TODO
      break;
    case SEEK_END:
      // TODO
      break;
  }

  cur_file->offset = new_offset;
  return new_offset;
}

__device__
void gpu_file_grow(gpu_fd fd, size_t size) {
  file_t * cur_file = get_file_from_gpu_fd(fd);
  assert(size > cur_file->current_size);
  request_t grow_request     = {0};
  grow_request.request_type  = GROW_REQUEST;
  grow_request.file_mem      = cur_file->data;
  grow_request.new_size      = size;
  grow_request.current_size  = cur_file->current_size;

  //printf("debug placeholder %p:%d:%s:%s\n", cur_file->data, __LINE__, __FILE__, __func__);
  //printf("debug placeholder %p:%d:%s:%s\n", grow_request.file_mem, __LINE__, __FILE__, __func__);

  response_t response = {0};
  gpu_enqueue(&grow_request, &response);

  cur_file->data         = response.file_data;
  cur_file->current_size = response.file_size;
  assert(response.file_size == size);
}

__device__
void gpu_file_close(gpu_fd fd) { 
  file_t * cur_file = get_file_from_gpu_fd(fd);
  request_t close_request     = (request_t) {0};
  close_request.request_type  = CLOSE_REQUEST;
  close_request.host_fd       = cur_file->host_fd;
  close_request.file_mem      = cur_file->data;
  close_request.actual_size   = cur_file->current_size;
  close_request.original_size = cur_file->original_size;
  //printf("debug placeholder %p:%d:%s\n", cur_file->data, __LINE__, __FILE__);
  //printf("debug placeholder %p:%d:%s\n", close_request.file_mem, __LINE__, __FILE__);

  response_t response = {0};
  gpu_enqueue(&close_request, &response);


  //printf("debug placeholder %s\n", response.file_data);
  // TODO if make this multithread may need to single thread this or
  // something because the file table is per block, not per thread
  /* Free up gpu file descriptor */
  *cur_file = (file_t) {0};
}

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

  /* Read data into CPU */ 
  //char * cpu_file_mem = (char *) malloc(file_size);
  //if (cpu_file_mem == NULL) 
  //  perror("failure in malloc withink file open\n");

  /* Memory pool should be unified */
  char * file_mem = (char *) allocate_from_memory_pool(file_size); 
  assert(file_mem != NULL);
  ssize_t bytes_read = read(fd, file_mem, file_size);
  if (bytes_read != file_size)
    perror("handle_gpu_file_open error reading file\n");

  fprintf(stderr, "pre cuda malloced %zu\n", file_size);
  //char * gpu_file_mem = (char *) allocate_from_memory_pool(file_size); 
  //assert(gpu_file_mem != NULL);
  fprintf(stderr, "cuda malloced\n");

  ret_response->host_fd     = fd;
  ret_response->file_size   = file_size;
  ret_response->permissions = permissions;
  ret_response->file_data   = file_mem;
}

__host__
void handle_gpu_file_grow(volatile request_t * request, volatile response_t * ret_response) {
  // XXX could also save ftruncate until the file is closed but this is
  // what we got now
  // TODO maybe round up allocations to be page size multiple
  char * old_file_mem  = request->file_mem;
  size_t new_size      = request->new_size;
  size_t current_size  = request->current_size;

  if (new_size <= current_size) {
    printf("Error grow size bad %zu:%zu\n", new_size, current_size);
  }


  char * new_file_mem = (char *) allocate_from_memory_pool(new_size);
  assert(new_file_mem != NULL);
  memcpy(new_file_mem, old_file_mem, current_size);
  free_from_memory_pool(old_file_mem, current_size);

  ret_response->file_data = new_file_mem;
  ret_response->file_size = new_size;
}

__host__
void handle_gpu_file_close(volatile request_t * request, volatile response_t * ret_response) {
  int host_fd          = request->host_fd;
  char * file_mem      = request->file_mem;
  size_t actual_size   = request->actual_size;
  size_t original_size = request->original_size;

  /* If the file has been appended to the actual file needs to grow */
  if (actual_size != original_size) {
    if (actual_size < original_size) {
      printf("Error close size bad %zu:%zu\n", actual_size, original_size);
    }

    int err = ftruncate(host_fd, actual_size);
    if (err == -1) {
      perror("Ftruncate failed to grow\n");
    }
  }

  printf("CPU File: %s\n", file_mem);

  ssize_t bytes_written = pwrite(host_fd, file_mem, actual_size, 0);
  if (bytes_written !=  actual_size) {
    perror("Failed to write file in close\n");
  }


  //CUDA_CALL(cudaFree(file_mem));
  // XXX since cudaFree is synchonous this call will never return
  printf("Done handle file close\n");
}
