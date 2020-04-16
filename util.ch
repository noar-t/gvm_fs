#ifndef UTIL_CH
#define UTIL_CH

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d:Error:%d\n",__FILE__,__LINE__,x); \
      exit(EXIT_FAILURE);}} while(0)

#define PRINT_ERR(x) do { \
      printf("Error at %s:%d:Error:%s\n",__FILE__,__LINE__,x); \
      exit(EXIT_FAILURE);} while(0)

#define BEGIN_SINGLE_THREAD __syncthreads(); if (threadIdx.x == 0) { 
                                         /* TODO add id check for one thread per warp,
                                            because GVM will use one thread per warp as
                                            I understand */

#define END_SINGLE_THREAD } __syncthreads(); 

typedef unsigned int gpu_mutex_t;

/* locked is 1; free is 0 */
//#define GPU_SPINLOCK_LOCK(x) while (atomicCAS((gpu_mutex_t *) x, 0, 1) == 1) {;}
#define GPU_SPINLOCK_LOCK(x) while (atomicExch((gpu_mutex_t *) x, 1) == 1) {;}
#define GPU_SPINLOCK_UNLOCK(x) do { atomicExch((gpu_mutex_t *) x, 0); } while(0)


#endif
