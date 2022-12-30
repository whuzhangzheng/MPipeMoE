#include <unordered_map>
#include <mutex>
#include <cassert>
#include <thread>
#include <iostream>
#include <nccl.h>
#include <vector>
#include "stream_manager.h"

//#define SMGR_N_STREAMS 16

cudaStream_t CudaStreamManager::stream(size_t idx) {
    return this->streams[idx % SMGR_N_STREAMS];
}

cublasHandle_t CudaStreamManager::handle(size_t idx) {
    return this->handles[idx % SMGR_N_STREAMS];
}


void CudaStreamManager::sync(int idx) {
    for (int i = 0; i < idx && i < SMGR_N_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }
}

void CudaStreamManager::setup(const int device, int batch, int d_model, int d_hidden) {
#ifdef USE_NCCL
    this->ncclgood = 0;
#endif
    this->device = device;
    checkCudaErrors(cudaSetDevice(device));
    streams = new cudaStream_t[SMGR_N_STREAMS];
    handles = new cublasHandle_t[SMGR_N_STREAMS];
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
        checkCudaErrors(cudaStreamCreate(streams + i));
        checkCudaErrors(cublasCreate(handles + i));
        cublasSetStream(handles[i], streams[i]);
    }
}

void CudaStreamManager::preAllocate(int total_recv_tokens, int d_hidden, int d_model,  std::string name, bool prealloc_di, bool prealloc_mi){
    std::map<std::string, int >::iterator l_it = max_total_recv_tokens_map.find(name);
    if(l_it == max_total_recv_tokens_map.end()){
        max_total_recv_tokens_map[name] = total_recv_tokens-10;
    }
    int max_total_recv_tokens = max_total_recv_tokens_map[name];
    float factor=1;
    std::map<std::string, float* >::iterator it;
    if(total_recv_tokens > max_total_recv_tokens * factor){
        max_total_recv_tokens_map[name] = total_recv_tokens;
        std::cout << "max_total_recv_tokens: "<< max_total_recv_tokens << ", total_recv_tokens: " << total_recv_tokens << std::endl;
    
        max_total_recv_tokens = total_recv_tokens;
        int pre_alloc_tokens = int(factor * total_recv_tokens);

        float* dispatched_input_ptr; float* middle_ptr;
        
        if (prealloc_di){
            std::cout << "prealloc dispatched input" << std::endl;
            it = dispatched_input_ptr_map.find(name);
            if (it == dispatched_input_ptr_map.end()){
                dispatched_input_ptr_map[name] = dispatched_input_ptr;
            }else{
                checkCudaErrors(cudaFreeHost(dispatched_input_ptr));
            }

            // checkCudaErrors(cudaMallocHost((void**)&dispatched_input_ptr, pre_alloc_tokens * d_model * sizeof(int))); 
            checkCudaErrors(cudaMallocHost((void**)&dispatched_input_ptr_map[name], pre_alloc_tokens * d_model * sizeof(int))); 

        }
        if (prealloc_mi){
            std::cout << "prealloc middle" << std::endl;

            it = middle_ptr_map.find(name);
            if (it == middle_ptr_map.end()){
                middle_ptr_map[name] = middle_ptr;
            }else{
                checkCudaErrors(cudaFreeHost(middle_ptr));
            }
            // checkCudaErrors(cudaMallocHost((void**)&middle_ptr, pre_alloc_tokens * d_hidden * sizeof(int)));
            checkCudaErrors(cudaMallocHost((void**)&middle_ptr_map[name], pre_alloc_tokens * d_hidden * sizeof(int)));

        }
    }
}

float* CudaStreamManager::get_middle_ptr(std::string name){
    std::map<std::string, float* >::iterator it;
    it = middle_ptr_map.find(name);
    if(it != middle_ptr_map.end()){
        return middle_ptr_map[name];
    }else{
        return nullptr;
    }
}

float* CudaStreamManager::get_dispatched_input_ptr(std::string name){
    std::map<std::string, float* >::iterator it;
    it = dispatched_input_ptr_map.find(name);
    if(it != dispatched_input_ptr_map.end()){
        return dispatched_input_ptr_map[name];
    }else{
        return nullptr;
    }
}

void CudaStreamManager::deallocate(){
    // checkCudaErrors(cudaFreeHost(middle_ptr));
    // checkCudaErrors(cudaFreeHost(dispatched_input_ptr));
}



void CudaStreamManager::destroy() {
    for (size_t i = 0; i < SMGR_N_STREAMS; ++i) {
        checkCudaErrors(cudaStreamDestroy(streams[i]));
        checkCudaErrors(cublasDestroy(handles[i]));

        std::map <std::string, float*> ::iterator iter;  
        for(iter = dispatched_input_ptr_map.begin(); iter != dispatched_input_ptr_map.end(); iter++){
            checkCudaErrors(cudaFreeHost(iter->second));
        }
	    for(iter = middle_ptr_map.begin(); iter != middle_ptr_map.end(); iter++){
            checkCudaErrors(cudaFreeHost(iter->second));
        }
    }
    delete[] streams;
    delete[] handles;
}

std::unordered_map<int, CudaStreamManager*> smgrs;
std::mutex smgr_mtx;

CudaStreamManager* getCudaStreamManager(const int device, int batch, int d_model, int d_hidden) {
    auto it = smgrs.find(device);
    if (it == smgrs.end()) {
        smgr_mtx.lock();
        it = smgrs.find(device);
        if (it == smgrs.end()) {
            auto smgr = new CudaStreamManager(device, batch, d_model, d_hidden);
            smgrs.insert(std::pair<int, CudaStreamManager*>(device, smgr));
            smgr_mtx.unlock();
            return smgr;
        } else {
            smgr_mtx.unlock();
        }
    }
    return it->second;
}

CudaStreamManager* getCudaStreamManager(const int device) {
    auto it = smgrs.find(device);
    if (it == smgrs.end()) {
        throw "Manager hasn't been initialized";
    }
    return it->second;
}
