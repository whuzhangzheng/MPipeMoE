#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include "utils/helper_cuda.h"
#include <map> 
#include <string>
#ifdef USE_NCCL
#include <nccl.h>
#include <c10d/ProcessGroupNCCL.hpp>

#define NCCL_SAFE_CALL(__fn__) { \
    auto __res__ = __fn__; \
    if (__res__ != ncclSuccess) { \
        fprintf(stderr, "NCCL Error at %s:%d value %d\n", __FILE__, __LINE__, __res__); \
        exit(-1); \
    } \
}

#endif

#define SMGR_N_STREAMS 16
#define MAX_POOL_SIZE 200
class CudaStreamManager {
public:
    int device;
    // float *dispatched_input_ptr=0;
    // float *middle_ptr=0;
    int max_total_recv_tokens = 0;

    int index=0;
    // std::map<std::string, int> prealloc_idxs;
    std::map<std::string, int>  max_total_recv_tokens_map;
    std::map<std::string, float*>  dispatched_input_ptr_map;
    std::map<std::string, float*>  middle_ptr_map;
    
    cublasHandle_t* handles;
    cudaStream_t* streams;
    c10d::ProcessGroupNCCL *group;
#ifdef USE_NCCL
    char ncclgood;
    ncclComm_t ncclcomm;
    // ncclComm_t ncclcomm2=0;  // error
    ncclComm_t ncclcomm2;   
    std::vector<std::vector<int>> topos;
#endif

public:
    CudaStreamManager(int device_, int batch, int d_model, int d_hidden): device(device_) {
        this->setup(device, batch, d_model, d_hidden);
    }

    void setup(int device, int batch, int d_model, int d_hidden);
    void sync(int=0);
    void destroy();
    void preAllocate(int total_recv_tokens, int d_hidden, int d_model, std::string name="moe", bool prealloc_di=true, bool prealloc_mi=true);
    float* get_dispatched_input_ptr(std::string name="moe");
    float* get_middle_ptr(std::string name="moe");
    void deallocate();

    cudaStream_t stream(size_t=0);
    cublasHandle_t handle(size_t=0);

    ~CudaStreamManager() {
        this->destroy();
    }
}; 

CudaStreamManager* getCudaStreamManager(const int device, int batch, int d_model, int d_hidden);
CudaStreamManager* getCudaStreamManager(const int device);

#endif  // CUDA_STREAM_MANAGER 
