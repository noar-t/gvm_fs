#ifndef REQUEST_CH
#define REQUEST_CH

typedef enum { // TODO might have better naming scheme
  read
} request_type_t;

typedef struct request_t {
  bool ready_to_read;
  request_type_t request_type;
  int placeholder; // TODO add more info for request details
} request_t;

#endif
