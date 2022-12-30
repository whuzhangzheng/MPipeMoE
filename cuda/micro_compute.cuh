
#ifndef FUSED_COMPUTE_H
#define FUSED_COMPUTE_H


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
#include "fused_compute.h"

#define LOOP_RANK

template<typename scalar_t>
void pipe_moe_cuda_micro_forward_impl(
        scalar_t* inp,
        std::vector<torch::Tensor> experts_params,
        scalar_t* dispatched_input,
        scalar_t* middle,
        scalar_t* dispatched_output,
        scalar_t* output,
        int d_model, int d_hidden, int num_local_expert, int rank, int world_size,
        int token_per_split,  int num_split, CudaStreamManager* smgr, int core_op) {
    
    // TimersMap timers({""});
    int n_per_split1 = token_per_split * d_model, n_per_split2 = token_per_split * d_hidden;
    int batch_per_expert = token_per_split / world_size / num_local_expert;
    int n_per_node_split1 = n_per_split1 / world_size;
    int n_per_node_split2 = n_per_split2 / world_size;

    const scalar_t alpha = 1, beta = 0;

    scalar_t *weight1, *weight2;
    weight1 = experts_params[0].data_ptr<scalar_t>();
    weight2 = experts_params[1].data_ptr<scalar_t>();

    cudaEvent_t *input_ready = new cudaEvent_t[num_split];
    cudaEvent_t *output_ready = new cudaEvent_t[num_split];
    for (int i = 0; i < num_split; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
    }

    std::vector<std::vector<int>> send_topos = smgr -> topos;
    std::vector<std::vector<int>> recv_topos;
    if(send_topos.size() == world_size){
        recv_topos = send_topos;
    }else{
        std::vector<std::vector<int>>::const_iterator first=send_topos.begin()+world_size;
        std::vector<std::vector<int>>::const_iterator second=send_topos.begin()+world_size*2;
        recv_topos.assign(first,second);
    }

    // int group_recv, group_send; // the first rank in recv group, send group
    int comm1_stream=0, comp_stream=1, comm2_stream=2, nstream=3;
    int in_feat, out_feat;
    int compute_ws = num_local_expert == 1 ? 1 : world_size;
    batch_per_expert = num_local_expert == 1 ? batch_per_expert * world_size: batch_per_expert;


    for (int g=0; g < num_split; g++) {
        alltoall(dispatched_input + g * n_per_split1, inp + g * n_per_split1, n_per_split1, world_size, smgr->ncclcomm, smgr->stream(comm1_stream), false);
        cudaEventRecord(input_ready[g], smgr->stream(comm1_stream));
        // printf("[%d, %d] ->%d, %d->, comm1 end\n", rank, g, rank_send, rank_recv);
    }

    // smgr->sync(3);
    // checkCudaErrors(cudaGetLastError());

    // printf("comm1\n");

    for (int g=0; g < num_split; g++) {
        cudaStreamWaitEvent(smgr->stream(comp_stream), input_ready[g], 0);
        // printf("[%d, %d] comp start\n", rank, g);
        in_feat = d_model; out_feat = d_hidden;
        for (int j = 0; j < compute_ws; ++j) {
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                middle + n_per_split2 * g + n_per_node_split2 * j,
                dispatched_input  + n_per_split1 * g  + n_per_node_split1 * j,
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
                dispatched_output + n_per_split1 * g + n_per_node_split1 * j,
                middle + n_per_split2 * g + n_per_node_split2 * j,
                weight2,
                batch_per_expert, in_feat,  out_feat, 
                num_local_expert,
                0, core_op
            );
        }

        cudaEventRecord(output_ready[g], smgr->stream(comp_stream));
    }
    // printf("cal\n");
    // smgr->sync(3);
    // checkCudaErrors(cudaGetLastError());

    for (int g=0; g < num_split; g++) {
        cudaStreamWaitEvent(smgr->stream(comm2_stream), output_ready[g], 0);
        alltoall(output + g * n_per_split1, dispatched_output + + g * n_per_split1, n_per_split1, world_size, smgr->ncclcomm2, smgr->stream(comm2_stream), false);
    }

    // printf("comm2\n");
    // std::cout << "comm end" << std::endl;
    smgr->sync(nstream);
    checkCudaErrors(cudaGetLastError());

    for (int i = 0; i < num_split; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
    }
    delete [] input_ready;
    delete [] output_ready;

    return;
}

template<typename scalar_t>
void pipe_moe_cuda_micro_backward_impl(
        scalar_t* grad_outs,
        scalar_t* inputs,
        std::vector<torch::Tensor> experts_params,

        scalar_t* dispatched_input,
        scalar_t* middle,
        // const scalar_t* dispatched_output,

        scalar_t* grad_dispatched_output,
        scalar_t* grad_middle,
        scalar_t* grad_dispatched_input,
        scalar_t* grad_in,

        int d_model, int d_hidden, int num_local_expert, int rank, int world_size, 
        int token_per_split,  int num_split,
        CudaStreamManager* smgr, int core_op) {
    
    int n_per_split1 = token_per_split * d_model, n_per_split2 = token_per_split * d_hidden;
    int batch_per_expert = token_per_split / world_size / num_local_expert;
    int n_per_node_split1 = n_per_split1 / world_size;
    int n_per_node_split2 = n_per_split2 / world_size;

    const scalar_t alpha = 1, beta = 0;
    const scalar_t beta2 = 1;
    
    cudaEvent_t *input_ready = new cudaEvent_t[num_split];
    cudaEvent_t *output_ready = new cudaEvent_t[num_split];
    for (int i = 0; i < num_split; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
    }

    std::vector<std::vector<int>> send_topos = smgr -> topos;
    std::vector<std::vector<int>> recv_topos;
    if(send_topos.size() == world_size){
        recv_topos = send_topos;
    }else{
        std::vector<std::vector<int>>::const_iterator first=send_topos.begin()+world_size;
        std::vector<std::vector<int>>::const_iterator second=send_topos.begin()+world_size*2;
        recv_topos.assign(first,second);
    }
    scalar_t *weight1, *weight2;
    scalar_t *grad_weight1, *grad_weight2;
    weight1 = experts_params[0].data_ptr<scalar_t>();
    weight2 = experts_params[1].data_ptr<scalar_t>();
    grad_weight1 = experts_params[0].mutable_grad().data_ptr<scalar_t>();
    grad_weight2 = experts_params[1].mutable_grad().data_ptr<scalar_t>();
    
    int in_feat, out_feat;
    
    int comm1_stream=0, comp_stream=1, comm2_stream=0, nstream=2;

    for (int g=0; g < num_split; g++) {
	    alltoall(grad_dispatched_output +  n_per_split1 * g, grad_outs + g * n_per_split1, n_per_split1, world_size, smgr->ncclcomm, smgr->stream(comm1_stream), false);
        cudaEventRecord(input_ready[g], smgr->stream(comm1_stream));
    }
    int compute_ws = num_local_expert == 1 ? 1 : world_size;
    batch_per_expert = num_local_expert == 1 ? batch_per_expert * world_size: batch_per_expert;

    // smgr->sync(3);
    // checkCudaErrors(cudaGetLastError());

    for (int g=0; g < num_split; g++) {
        cudaStreamWaitEvent(smgr->stream(comp_stream), input_ready[g], 0);

        // grad_middle
        in_feat = d_hidden; out_feat = d_model;
        for (int j = 0; j < compute_ws; ++j) {
            // grad_middle_buf
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_dispatched_output + n_per_split1 * g + n_per_node_split1 * j,
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
                grad_dispatched_output +  n_per_split1 * g + n_per_node_split1 * j,
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
                dispatched_input +  n_per_split1 * g + n_per_node_split1 * j,
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
                grad_dispatched_input + n_per_split1 * g + n_per_node_split1 * j,
                weight1,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                1, core_op
            );

        }
        // smgr->sync(nstream);
        // checkCudaErrors(cudaGetLastError());
        cudaEventRecord(output_ready[g], smgr->stream(comp_stream));
    }
    // smgr->sync(nstream);
    // checkCudaErrors(cudaGetLastError());


    for (int g=0; g < num_split; g++) {
        cudaStreamWaitEvent(smgr->stream(comm2_stream), output_ready[g], 0);
        alltoall(grad_in + n_per_split1 * g, grad_dispatched_input + n_per_split1 * g, n_per_split1, world_size, smgr->ncclcomm2, smgr->stream(comm2_stream), false);
    }

    smgr->sync(nstream);
    checkCudaErrors(cudaGetLastError());
    for (int i = 0; i < num_split; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
}


#endif  // FUSED_COMPUTE_H
