#ifndef GPU_FILE_CH
#define GPU_FILE_CH

// TODO file struct with offset, size, read position, etc 
typedef enum {
  r,
  w,
  x,
  rw,
  rx,
  wx,
  rwx,
} permissions_t;

#endif
