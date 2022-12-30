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
void alltoall(
    scalar_t* ouput,
    scalar_t* inp,
    int length,
    int world_size, 
    ncclComm_t comm, 
    cudaStream_t stream, bool inplace=false, 
    std::vector<int> *send_topo=0, std::vector<int> *recv_topo=0){
    /*
    warp all send/recv into one pair of ncclGroupStart and ncclGroupEnd
    */
    if (inplace){
        ouput = inp;
    }
    int len_per_node = length / world_size;
    int rank_send, rank_recv;
    NCCL_SAFE_CALL(ncclGroupStart());
    for (int r=0; r < world_size; r++) {
        // if(send_topo && recv_topo){
        //     rank_send = (*send_topo)[r];
        //     rank_recv = (*recv_topo)[r];
        // }else{
        rank_send = rank_recv = r;
        // }
        NCCL_SAFE_CALL(ncclSend(inp + rank_send * len_per_node, len_per_node * sizeof(float), ncclChar, rank_send, comm, stream));
        NCCL_SAFE_CALL(ncclRecv(ouput + rank_recv * len_per_node, len_per_node * sizeof(float), ncclChar, rank_recv, comm, stream));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
    return;
}

template<typename scalar_t>
void stridedGemm(
    cublasHandle_t handle,
    scalar_t  * output,
    scalar_t  * input,
    scalar_t  * weight,
    int batch_per_expert, int in_feat,  int out_feat,  // m, k, n
    int num_local_expert,
    int type, int core_op
){
    
    cublasOperation_t transa, transb;
    int m, n, k;
    scalar_t alpha, beta;
    scalar_t  *A, *B, *C;
    int lda, ldb, ldc;
    long long int strideA, strideB, strideC;
    if (type==0){
        // forward: w @ input -> output  eoi, ebi -> ebo
        transa = CUBLAS_OP_T;       transb = CUBLAS_OP_N;
        m = out_feat;               n=batch_per_expert;         k = in_feat;
        alpha=1, beta=0;
        A = weight;                 B = input;                  C=output;
        lda = in_feat;              ldb = in_feat;              ldc = out_feat;
        strideA = out_feat * in_feat;   strideB = batch_per_expert * in_feat;  strideC = batch_per_expert * out_feat;
    }else if(type==1){
        // backeard for act: w, d_output -> d_input     eoi, ebo -> ebi
        transa = CUBLAS_OP_N;       transb = CUBLAS_OP_N;
        m = in_feat;                n=batch_per_expert;         k = out_feat;
        alpha=1, beta=0;
        A = weight;                 B = output;                 C=input;
        lda = in_feat;              ldb = out_feat;             ldc = in_feat;
        strideA = out_feat * in_feat;   strideB = batch_per_expert * out_feat;  strideC = batch_per_expert * in_feat;
    }else if(type==2){
        // backeard for w: input, d_output -> d)w      ebi, ebo ->  eoi
        transa = CUBLAS_OP_N;       transb = CUBLAS_OP_T;   
        m = in_feat;                n=out_feat;                 k = batch_per_expert;
        alpha=1, beta=1;
        A = input;                  B = output;                 C=weight;
        lda = in_feat;              ldb = out_feat;             ldc = in_feat;
        strideA = batch_per_expert * in_feat;   strideB = batch_per_expert * out_feat;  strideC = out_feat * in_feat;
    }

    if(core_op){
        // cudaDataType_t computeType = CUBLAS_COMPUTE_32F;
        cublasGemmAlgo_t algo = CUBLAS_GEMM_ALGO0_TENSOR_OP; // CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        checkCudaErrors(cublasGemmStridedBatchedEx( handle,
                                transa, transb,
                                m, n, k, 
                                &alpha,
                                A, CUDA_R_32F, lda, strideA,
                                B, CUDA_R_32F, ldb, strideB,
                                &beta,
                                C, CUDA_R_32F, ldc, strideC,
                                num_local_expert,
                                CUBLAS_COMPUTE_32F_FAST_16F, // CUBLAS_COMPUTE_32F, CUBLAS_COMPUTE_32F_FAST_16F
                                algo));
    }else{
        checkCudaErrors(cublasXgemmStridedBatched(
                        handle,
                        transa, transb,
                        m, n, k,
                        &alpha,
                        A, lda, strideA,
                        B, ldb, strideB,
                        &beta,
                        C, ldc, strideC,
                        num_local_expert
                        ));
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
