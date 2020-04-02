#ifndef UTIL_CH
#define UTIL_CH

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
      printf("Error at %s:%d:Error:%d\n",__FILE__,__LINE__,x); \
      exit(EXIT_FAILURE);}} while(0)

#define BEGIN_SINGLE_THREAD __syncthreads(); { /* TODO add id check for one thread per warp,
                                            because GVM will use one thread per warp as
                                            I understand */

#define END_SINGLE_THREAD } __syncthreads(); 

/* locked is 1; free is 0 */
/* x must be a pointer to a bool used as a lock */
#define CPU_SPINLOCK_LOCK(x) while (!__sync_bool_compare_and_swap(x, false, true)) {;}
#define CPU_SPINLOCK_UNLOCK(x) while (!__sync_bool_compare_and_swap(x, true, false)) {;}

#define GPU_SPINLOCK_LOCK(x) while (!atomicCAS(x, false, true)) {;}
#define GPU_SPINLOCK_UNLOCK(x) while (atomicCAS(x, true, false)) {;}


#endif
