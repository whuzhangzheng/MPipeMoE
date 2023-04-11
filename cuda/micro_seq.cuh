
#include <vector>
#include <torch/extension.h>
#include "utils/cublas_wrapper.h"
#include "utils/fmoe_utils.h"
#include "stream_manager.h"
#include "fused_compute.h"

#include <cstdio>
#include <iostream>
#include <vector>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvToolsExt.h>

#include "utils/cublas_wrapper.h"
#include "utils/fmoe_utils.h"
#include "stream_manager.h"

#include "utils/pmoe_utils.h"

#define LOOP_RANK

template<typename scalar_t>
void experts_ffn_forward_impl(
        scalar_t* inp, // dispatched_input
        std::vector<torch::Tensor> experts_params,
        scalar_t* middle,
        scalar_t* output, // dispatched_output
        std::vector<int> recv_eles_per_split,
        int d_model, int d_hidden, int num_local_expert, int rank, int world_size,
        int num_split, CudaStreamManager* smgr, int core_op) {
    
    // TimersMap timers({""});
    int recv_offset_per_split = sum(recv_eles_per_split);
    int batch_per_expert = recv_offset_per_split / d_model;
    int recv_offset=0;
    int recv_offset_middle = 0;

    // const scalar_t alpha = 1, beta = 0;

    scalar_t *weight1, *weight2;
    weight1 = experts_params[0].data_ptr<scalar_t>();
    weight2 = experts_params[1].data_ptr<scalar_t>();


    int in_feat, out_feat;
    int comp_stream = 0;

    for (int g=0; g < num_split; g++) {
        // printf("[%d, %d] comp start\n", rank, g);
        recv_offset_middle = recv_offset / d_model * d_hidden;
        in_feat = d_model; out_feat = d_hidden;
        stridedGemm<scalar_t>(
            smgr->handle(comp_stream),
            middle + recv_offset_middle,
            inp  + recv_offset,
            weight1,
            batch_per_expert, in_feat,  out_feat,  // m, k, n
            num_local_expert,
            0, core_op
        );
        relu_kernel<<<CEIL(batch_per_expert * d_hidden, NTH), NTH, 0, smgr->stream(comp_stream)>>>(middle, batch_per_expert * d_hidden);

        in_feat = d_hidden; out_feat = d_model;
        stridedGemm<scalar_t>(
            smgr->handle(comp_stream),
            output + recv_offset,
            middle + recv_offset_middle,
            weight2,
            batch_per_expert, in_feat,  out_feat, 
            num_local_expert,
            0, core_op
        );
        recv_offset += recv_offset_per_split;
    }
    // printf("cal\n");
    // smgr->sync(3);
    // checkCudaErrors(cudaGetLastError());

    // printf("comm2\n");
    // std::cout << "comm end" << std::endl;

    return;
}

template<typename scalar_t>
void experts_ffn_backward_impl(
        scalar_t* grad_outs, // grad_dispatched_output
        scalar_t* inputs,  // dispatched_input
        std::vector<torch::Tensor> experts_params,
        std::vector<int> recv_eles_per_split,

        scalar_t* middle,

        scalar_t* grad_middle,
        scalar_t* grad_in, // grad_dispatched_input

        int d_model, int d_hidden, int num_local_expert, int rank, int world_size, 
        int num_split,
        CudaStreamManager* smgr, int core_op) {
    

    const scalar_t alpha = 1, beta = 0;
    // const scalar_t beta2 = 1;
    
    scalar_t *weight1, *weight2;
    scalar_t *grad_weight1, *grad_weight2;
    weight1 = experts_params[0].data_ptr<scalar_t>();
    weight2 = experts_params[1].data_ptr<scalar_t>();
    grad_weight1 = experts_params[0].mutable_grad().data_ptr<scalar_t>();
    grad_weight2 = experts_params[1].mutable_grad().data_ptr<scalar_t>();
    
    int in_feat, out_feat;
    

    int recv_offset=0;
    int recv_offset_middle = 0;
    int recv_offset_per_split = sum(recv_eles_per_split);
    int batch_per_expert = recv_offset_per_split / d_model;

    // smgr->sync(3);
    // checkCudaErrors(cudaGetLastError());
    int comp_stream = 0;
    for (int g=0; g < num_split; g++) {
        recv_offset_middle = recv_offset / d_model * d_hidden;
        // grad_middle
        in_feat = d_hidden; out_feat = d_model;
        stridedGemm<scalar_t>(
            smgr->handle(comp_stream),
            grad_outs + recv_offset,
            grad_middle + recv_offset_middle,
            weight2,
            batch_per_expert, in_feat,  out_feat,  // m, k, n
            num_local_expert,
            1, core_op
        );
        
        relu_backward_kernel<<<CEIL(batch_per_expert * d_hidden, NTH), NTH, 0, smgr->stream(comp_stream)>>>(
                middle +  recv_offset_middle, grad_middle +  recv_offset_middle, batch_per_expert * d_hidden);
        // grad_w2
        //(row) grad_w2(e, m, h) <- grad_out(g, e, [b, m](T)) middle(g, e, b, h)
        //(col) grad_w2(h, m, e) <- middle(h, b, e, g) grad_out([m, b](T), e, g)
        in_feat = d_hidden; out_feat = d_model;
        stridedGemm<scalar_t>(
            smgr->handle(comp_stream),
            grad_outs +  recv_offset,
            middle +  recv_offset_middle,
            grad_weight2,
            batch_per_expert, in_feat,  out_feat,  // m, k, n
            num_local_expert,
            2, core_op
        );
        // grad_w1
        in_feat = d_model; out_feat = d_hidden;
        stridedGemm<scalar_t>(
            smgr->handle(comp_stream),
            grad_middle +  recv_offset_middle,
            inputs +  recv_offset,
            grad_weight1,
            batch_per_expert, in_feat,  out_feat,  // m, k, n
            num_local_expert,
            2, core_op
        );
        // grad_dispatched_input
        in_feat = d_model; out_feat = d_hidden;
        stridedGemm<scalar_t>(
            smgr->handle(comp_stream),
            grad_middle +  recv_offset_middle,
            grad_in + recv_offset,
            weight1,
            batch_per_expert, in_feat,  out_feat,  // m, k, n
            num_local_expert,
            1, core_op
        );
        // smgr->sync(nstream);
        // checkCudaErrors(cudaGetLastError());
        recv_offset += recv_offset_per_split;
    }
    // smgr->sync(nstream);
    // checkCudaErrors(cudaGetLastError());
}



// _deprecated
// _deprecated
// _deprecated
template<typename scalar_t>
void experts_ffn_forward_deprecated_impl(
        scalar_t* inp, // dispatched_input
        std::vector<torch::Tensor> experts_params,
        scalar_t* middle,
        scalar_t* output, // dispatched_output
        int d_model, int d_hidden, int num_local_expert, int rank, int world_size,
        int token_per_split,  int num_split, CudaStreamManager* smgr, int core_op) {
    
    // TimersMap timers({""});
    int n_per_split1 = token_per_split * d_model, n_per_split2 = token_per_split * d_hidden;
    int batch_per_expert = token_per_split / world_size / num_local_expert;
    int n_per_node_split1 = n_per_split1 / world_size;
    int n_per_node_split2 = n_per_split2 / world_size;

    // const scalar_t alpha = 1, beta = 0;

    scalar_t *weight1, *weight2;
    weight1 = experts_params[0].data_ptr<scalar_t>();
    weight2 = experts_params[1].data_ptr<scalar_t>();


    int in_feat, out_feat;
    int compute_ws = num_local_expert == 1 ? 1 : world_size;
    batch_per_expert = num_local_expert == 1 ? batch_per_expert * world_size: batch_per_expert;


    int comp_stream = 0;

    for (int g=0; g < num_split; g++) {
        // printf("[%d, %d] comp start\n", rank, g);
        in_feat = d_model; out_feat = d_hidden;
        for (int j = 0; j < compute_ws; ++j) {
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                middle + n_per_split2 * g + n_per_node_split2 * j,
                inp  + n_per_split1 * g  + n_per_node_split1 * j,
                weight1,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                0, core_op
            );
        }
        relu_kernel<<<CEIL(n_per_split2, NTH), NTH, 0, smgr->stream(comp_stream)>>>(middle, n_per_split2);

        in_feat = d_hidden; out_feat = d_model;
        for (int j = 0; j < compute_ws; ++j) {
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                output + n_per_split1 * g + n_per_node_split1 * j,
                middle + n_per_split2 * g + n_per_node_split2 * j,
                weight2,
                batch_per_expert, in_feat,  out_feat, 
                num_local_expert,
                0, core_op
            );
        }

    }
    // printf("cal\n");
    // smgr->sync(3);
    // checkCudaErrors(cudaGetLastError());

    // printf("comm2\n");
    // std::cout << "comm end" << std::endl;

    return;
}

template<typename scalar_t>
void experts_ffn_backward_deprecated_impl(
        scalar_t* grad_outs, // grad_dispatched_output
        scalar_t* inputs,  // dispatched_input
        std::vector<torch::Tensor> experts_params,

        scalar_t* middle,

        scalar_t* grad_middle,
        scalar_t* grad_in, // grad_dispatched_input

        int d_model, int d_hidden, int num_local_expert, int rank, int world_size, 
        int token_per_split,  int num_split,
        CudaStreamManager* smgr, int core_op) {
    
    int n_per_split1 = token_per_split * d_model, n_per_split2 = token_per_split * d_hidden;
    int batch_per_expert = token_per_split / world_size / num_local_expert;
    int n_per_node_split1 = n_per_split1 / world_size;
    int n_per_node_split2 = n_per_split2 / world_size;

    const scalar_t alpha = 1, beta = 0;
    // const scalar_t beta2 = 1;
    
    scalar_t *weight1, *weight2;
    scalar_t *grad_weight1, *grad_weight2;
    weight1 = experts_params[0].data_ptr<scalar_t>();
    weight2 = experts_params[1].data_ptr<scalar_t>();
    grad_weight1 = experts_params[0].mutable_grad().data_ptr<scalar_t>();
    grad_weight2 = experts_params[1].mutable_grad().data_ptr<scalar_t>();
    
    int in_feat, out_feat;
    

    int compute_ws = num_local_expert == 1 ? 1 : world_size;
    batch_per_expert = num_local_expert == 1 ? batch_per_expert * world_size: batch_per_expert;

    // smgr->sync(3);
    // checkCudaErrors(cudaGetLastError());
    int comp_stream = 0;
    for (int g=0; g < num_split; g++) {

        // grad_middle
        in_feat = d_hidden; out_feat = d_model;
        for (int j = 0; j < compute_ws; ++j) {
            // grad_middle_buf
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_outs + n_per_split1 * g + n_per_node_split1 * j,
                grad_middle +  n_per_split2 * g + n_per_node_split2 * j,
                weight2,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                1, core_op
            );
        }

        
        relu_backward_kernel<<<CEIL(n_per_split2, NTH), NTH, 0, smgr->stream(comp_stream)>>>(
                middle +  n_per_split2 * g, grad_middle +  n_per_split2 * g, n_per_split2);

        // grad_w2
        //(row) grad_w2(e, m, h) <- grad_out(g, e, [b, m](T)) middle(g, e, b, h)
        //(col) grad_w2(h, m, e) <- middle(h, b, e, g) grad_out([m, b](T), e, g)
        in_feat = d_hidden; out_feat = d_model;
        for(int j=0; j<compute_ws; j++){
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_outs +  n_per_split1 * g + n_per_node_split1 * j,
                middle +  n_per_split2 * g + n_per_node_split2 * j,
                grad_weight2,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                2, core_op
            );
        }

        // grad_w1
        in_feat = d_model; out_feat = d_hidden;
        for(int j=0; j<compute_ws; j++){
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_middle +  n_per_split2 * g + n_per_node_split2 * j,
                inputs +  n_per_split1 * g + n_per_node_split1 * j,
                grad_weight1,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                2, core_op
            );
        }

        // grad_dispatched_input
        in_feat = d_model; out_feat = d_hidden;
        for (int j = 0; j < compute_ws; ++j) {
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_middle +  n_per_split2 * g + n_per_node_split2 * j,
                grad_in + n_per_split1 * g + n_per_node_split1 * j,
                weight1,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                1, core_op
            );

        }
        // smgr->sync(nstream);
        // checkCudaErrors(cudaGetLastError());
    }
    // smgr->sync(nstream);
    // checkCudaErrors(cudaGetLastError());
}



template<typename scalar_t>
void alltoall_deprecated_impl(
    scalar_t* out,
    scalar_t* inp,
    int num_split,
    std::vector<int> recv_counts,  std::vector<int> send_counts, int flag, 
    // flag==0: 全send，部分recv；flag==1 相反
    int max_send_count, int max_recv_count,
    int feat, 
    int world_size, int rank, CudaStreamManager* smgr){
    /*
    warp all send/recv into one pair of ncclGroupStart and ncclGroupEnd
    */

    int send_count, recv_count, send_offset, recv_offset;
    send_count = max_send_count/num_split * feat;
    recv_count = max_recv_count/num_split * feat;
    int rank_send, rank_recv;
    for(int i=0; i<num_split; i++){
        for (int r=0; r < world_size; r++) {
            rank_send = r;
            rank_recv = r;
            if (flag==0){
                // printf("flag=0 == %d, recv_counts: [%d, %d, %d, %d]\n", flag, recv_counts[0], recv_counts[1], recv_counts[2], recv_counts[3]);
                recv_count = recv_counts[r]/num_split * feat;
                send_offset = (r*num_split + i) * send_count;
                recv_offset = max_recv_count * r * feat + recv_count * i;
            }else{
                // printf("flag=1 == %d, send_counts: [%d, %d, %d, %d]\n", flag, send_counts[0], send_counts[1], send_counts[2], send_counts[3]);
                send_count = send_counts[r]/num_split * feat;
                send_offset = max_send_count * r * feat + send_count * i;
                recv_offset = (r*num_split + i) * recv_count;
            }
            // printf("[%d](%d, %d): %d+%d, %d+%d\n", rank, i, r, send_offset/feat, send_count/feat, recv_offset/feat, recv_count/feat);
            NCCL_SAFE_CALL(ncclGroupStart());
                NCCL_SAFE_CALL(ncclSend(inp + send_offset, send_count * sizeof(scalar_t), ncclChar, rank_send, (smgr->ncclcomm), smgr->stream(0)));
                NCCL_SAFE_CALL(ncclRecv(out + recv_offset, recv_count * sizeof(scalar_t), ncclChar, rank_recv, (smgr->ncclcomm), smgr->stream(0)));
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }

    // std::cout << "comm end" << std::endl;
    // smgr->sync(1);

    return;
}


template<typename scalar_t>
void alltoall_deprecated_impl(
    scalar_t* out,
    scalar_t* inp,
    int split,
    std::vector<int> recv_counts, std::vector<int> send_counts, 
    int feat, 
    int world_size, int rank, CudaStreamManager* smgr){
    /*
    warp all send/recv into one pair of ncclGroupStart and ncclGroupEnd
    */

    int send_count, recv_count, send_offset, recv_offset;
    int rank_send, rank_recv;
    int send_offset_base = 0, recv_offset_base = 0;
    for(int i=0; i<split; i++){
        for (int r=0; r < world_size; r++) {
            rank_send = r;
            rank_recv = r;
            send_count = send_counts[r]/split * feat;
            recv_count = recv_counts[r]/split * feat;
            send_offset = send_offset_base + i * send_count;
            recv_offset = recv_offset_base + i * recv_count;
            NCCL_SAFE_CALL(ncclGroupStart());
                NCCL_SAFE_CALL(ncclSend(inp + send_offset, send_count * sizeof(scalar_t), ncclChar, rank_send, (smgr->ncclcomm), smgr->stream(0)));
                NCCL_SAFE_CALL(ncclRecv(out + recv_offset, recv_count * sizeof(scalar_t), ncclChar, rank_recv, (smgr->ncclcomm), smgr->stream(0)));
            NCCL_SAFE_CALL(ncclGroupEnd());
            send_offset_base += send_counts[r] * feat;
            recv_offset_base += recv_counts[r] * feat;
        }
        send_offset_base = 0;
        recv_offset_base = 0;
    }

    // std::cout << "comm end" << std::endl;
    smgr->sync(1);

    return;
}
