#ifndef REQUEST_CH
#define REQUEST_CH

#include "gpu_file.ch"

typedef enum {
  open,
  close,
  // TODO need to add support for appending/growing files and allocations
} request_type_t;



typedef struct request_t {
  volatile bool ready_to_read;
  request_type_t request_type;

  /* Open request variables */
  char * file_name;
  permissions_t permissions; /* Unix style permissions (rwx) */

  /* Close request variables */
  int fd;
 
  // XXX GET RID OF 
  int placeholder;
} request_t;

typedef void * response_t;

#endif
