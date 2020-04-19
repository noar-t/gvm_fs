#ifndef TYPES_CH
#define TYPES_CH

#include "util.ch"

#define MAX_FILES 128
#define NUM_BLOCKS 100
#define FILE_TABLE_FULL -1

// TODO might be best to split these back out

/***********************
 * File Metastructures *
 ***********************/

typedef enum permissions_t {
  R___ = 0b100,
  _W__ = 0b010,
  __X_ = 0b001,
  RW__ = 0b110,
  R_X_ = 0b101,
  _WX_ = 0b011,
  RWX_ = 0b111,
} permissions_t;

/* GPU File Struct */
typedef struct file_t {
  bool in_use;
  int host_fd;
  size_t current_size;
  
  //int local_fd;
  permissions_t permissions;
  size_t offset;
} file_t;

typedef struct file_meta_table_t {
  file_t files[MAX_FILES];
} file_meta_table_t;

typedef file_meta_table_t * global_file_meta_table_t;
typedef int gpu_fd;

/**************************
 * Ringbuf Datastructures *
 **************************/

#define RINGBUF_SIZE 100
#define MAX_PATH_SIZE 64

/* Type of fs request */
typedef enum request_type_t {
  OPEN_REQUEST = 1,
  CLOSE_REQUEST,
  GROW_REQUEST,
  // TODO need to add support for appending/growing files and allocations
} request_type_t;

/* All data needed for a given request */
typedef struct request_t {
  volatile bool ready_to_read;
  request_type_t request_type;

  /* Request variables */
  char file_name[MAX_PATH_SIZE]; /* Needs to be in shared memory (makes char * hard) */
  permissions_t permissions; /* Unix style permissions (rwx) */
  int host_fd;               /* Used for growing/closing */
  size_t new_size;           /*   ^    */
} request_t;

typedef struct response_t {
  volatile bool ready_to_read;
  int host_fd;       /* if open request */
  off_t file_size;
  permissions_t permissions;
  char * file_data;
} response_t;

typedef struct ringbuf_t {
  /* Synchronization Variables */
  gpu_mutex_t * gpu_mutex;

  /* Request Handling Thread */
  pthread_t request_handler;

  /* Request Buffer Data */
  request_t requests[RINGBUF_SIZE]; /* XXX struct elements will be volatile */
  volatile response_t responses[RINGBUF_SIZE];
} ringbuf_t;

#endif
