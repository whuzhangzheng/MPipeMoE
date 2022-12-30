#ifndef FUSED_COMPUTE_HH
#define FUSED_COMPUTE_HH
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#define SMGR_N_STREAMS 16

template<typename scalar_t>
__global__ 
void relu_kernel(scalar_t* a, size_t n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    scalar_t v;
    for (; i < n; i += stride) {
        v = a[i];
        if (v < 0) {
            a[i] = 0;
        }
    }
}


template<typename scalar_t>
__global__ 
void relu_backward_kernel(const scalar_t* a, scalar_t* grad_o, size_t n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        if (a[i] <= 0) {
            grad_o[i] = 0;
        }
    }
}

inline int count_in_current_rank(std::vector<int> counts, int rank, int num_per_rank){
    int c=0;
    for(int i=rank*num_per_rank; i< (rank+1)*num_per_rank; i++){
        c += counts[i];
    }
    return c;
}

inline int count_before_current_rank(std::vector<int> counts, int rank, int num_per_rank){
     int c=0;
    for(int i=0; i< rank*num_per_rank; i++){
         c += counts[i];
     }
     return c;
 }
inline int count_offset(std::vector<int> counts, int start_idx, int steps){
    int c=0;
    for(int i=start_idx; i< start_idx+steps; i++){
        c += counts[i];
    }
    return c;
}

inline int max(std::vector<int> counts, int n_gran=1){
    int c = 0;
    int tmp_c = 0;
    for(int g=0; g<counts.size()/n_gran; g++){
        tmp_c = 0;
        for (int i=0; i<n_gran; i++){
            tmp_c += counts[g*n_gran+i];
        }
        c = tmp_c>c?tmp_c:c;
    }
    return c;
}

inline int min(std::vector<int> counts, int start_idx, int steps){
    int c=9999;
    for(int i=start_idx; i< start_idx+steps; i++){
        c = c < counts[i] ? c : counts[i] ;
    }
    return c;
}

#define NTH 512
#endif 
