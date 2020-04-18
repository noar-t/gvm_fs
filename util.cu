#include "util.ch"

__device__
void gpu_str_cpy(char * src, char * dst, size_t max_size) {
  for (int i = 0; i < max_size; i++) {
    dst[i] = src[i];

    if (src[i] == '\0')
      break; /* End upon string end */
  }
}
