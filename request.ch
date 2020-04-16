#ifndef REQUEST_CH
#define REQUEST_CH

#include "gpu_file.ch"

/* Type of fs request */
typedef enum {
  OPEN_REQUEST,
  CLOSE_REQUEST,
  GROW_REQUEST,
  // TODO need to add support for appending/growing files and allocations
} request_type_t;

/* All data needed for a given request */
typedef struct request_t {
  volatile bool ready_to_read;
  request_type_t request_type;

  /* Request variables */
  char file_name[64]; /* Needs to be in shared memory (makes char * hard) */
  permissions_t permissions; /* Unix style permissions (rwx) */
  int host_fd;
  size_t new_size;
} request_t;

typedef struct response_t {
  volatile bool ready_to_read;
  int host_fd;       /* if open request */
  size_t file_size;
  permissions_t permissions;
  char * file_data;
} response_t;

#endif
