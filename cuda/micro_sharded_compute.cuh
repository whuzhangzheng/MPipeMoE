

#ifndef SHARDED_FUSED_COMPUTE_H
#define SHARDED_FUSED_COMPUTE_H

#include <cstdio>
#include <iostream>
#include <vector>
#include <thread>
#include <assert.h>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvToolsExt.h>

#include "utils/cublas_wrapper.h"
#include "utils/fmoe_utils.h"
#include "stream_manager.h"
#include "fused_compute.h"




            
template<typename scalar_t>
void pmoe_cuda_micro_forward_sharded_impl(
        scalar_t* inp,  // num_split, ws, num_local_experts, batch_per_expert, d_model
        scalar_t* outputs,
        std::vector<torch::Tensor> &experts_params,
        float* dispatched_input,
        float* middle,
        int d_model, int d_hidden, int num_local_expert, int rank, int world_size,
        int token_per_split,  int num_split, 
        CudaStreamManager* smgr,
        scalar_t* dispatched_input_buf1,
        scalar_t* dispatched_input_buf2,
        scalar_t* middle_buf1,
        scalar_t* middle_buf2,
        scalar_t* dispatched_output_buf1,
        scalar_t* dispatched_output_buf2,
        bool recompute_di,
        bool recompute_mi,
        int inplace, int core_op) {
    
    // TimersMap timers({""});
    // printf("forward\n");
    // NCCL_SAFE_CALL(ncclCommUserRank((smgr->ncclcomm), &rank));
    int n_per_split1 = token_per_split * d_model, n_per_split2 = token_per_split * d_hidden;
    int batch_per_expert = token_per_split / world_size / num_local_expert;
    int n_per_node_split1 = n_per_split1 / world_size;
    int n_per_node_split2 = n_per_split2 / world_size;

    const scalar_t alpha = 1, beta = 0;
    
    int dl_dispatched_input = dispatched_input_buf2? 2:1 ;
    int dl_middle = middle_buf2? 2:1 ;
    int dl_dispatched_output = dispatched_output_buf2? 2:1 ;
    if (!dispatched_input_buf2){
        dispatched_input_buf2 = dispatched_input_buf1;
    }
    if (!middle_buf2){
        middle_buf2 = middle_buf1;
    }
    if (!dispatched_output_buf2){
        dispatched_output_buf2 = dispatched_output_buf1;
    }
    scalar_t *weight1, *weight2;
    weight1 = experts_params[0].data_ptr<scalar_t>();
    weight2 = experts_params[1].data_ptr<scalar_t>();

    ncclComm_t ncclcomm2 = smgr->ncclcomm2; //? smgr->ncclcomm2 : smgr->ncclcomm;


    scalar_t *dispatched_input_buf, *middle_buf, *dispatched_output_buf;

    cudaEvent_t *input_ready = new cudaEvent_t[num_split];
    cudaEvent_t *output_ready = new cudaEvent_t[num_split];
    cudaEvent_t *input_to_mid_ready = new cudaEvent_t[num_split];
    cudaEvent_t *mid_ready = new cudaEvent_t[num_split];
    cudaEvent_t *input_copy_ready = new cudaEvent_t[num_split];
    cudaEvent_t *middle_copy_ready = new cudaEvent_t[num_split];
    cudaEvent_t *output_send_ready = new cudaEvent_t[num_split];
    
    for (int i = 0; i < num_split; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
        cudaEventCreate(input_to_mid_ready + i);

        // cudaEventCreate(mid_ready + i);
        cudaEventCreate(input_copy_ready + i);
        cudaEventCreate(middle_copy_ready + i);
        cudaEventCreate(output_send_ready + i);
    }

    int d2h_stream = 0, comm1_stream = 1, comp_stream=2, comm2_stream=3, nstreams = 4;
    int in_feat, out_feat;
    int compute_ws = num_local_expert == 1 ? 1 : world_size;
    batch_per_expert = num_local_expert == 1 ? batch_per_expert * world_size: batch_per_expert;

    for (int g=0; g < num_split; g++) {
        if(g%2==0){
            dispatched_input_buf = dispatched_input_buf1;
            middle_buf = middle_buf1;
            dispatched_output_buf = dispatched_output_buf1;
        }else{
            dispatched_input_buf = dispatched_input_buf2;
            middle_buf = middle_buf2;
            dispatched_output_buf = dispatched_output_buf2;
        }
        if(inplace){
            dispatched_input_buf = inp + g * n_per_split1;
            dispatched_output_buf = outputs + g * n_per_split1;
        }

        // 1. comm1
        // assert(n_gran == 1);
        if(!inplace && g>=dl_dispatched_input){
            if (!recompute_di)
                cudaStreamWaitEvent(smgr->stream(comm1_stream), input_copy_ready[g-dl_dispatched_input], 0);
            if (!recompute_mi)
                cudaStreamWaitEvent(smgr->stream(comm1_stream), input_to_mid_ready[g-dl_dispatched_input], 0);
        }
        // printf("[%d, %d, %d] comm1 start\n", rank, g, topos[rank][g]);
        alltoall<scalar_t>(dispatched_input_buf, inp + g * n_per_split1, n_per_split1, world_size, smgr->ncclcomm, smgr->stream(comm1_stream), inplace);
        cudaEventRecord(input_ready[g], smgr->stream(comm1_stream));


        // 2. compute(comp_stream)
        cudaStreamWaitEvent(smgr->stream(comp_stream), input_ready[g], 0);
        // if (g>=dl_middle){ cudaStreamWaitEvent(smgr->stream(comp_stream), middle_copy_ready[g-dl_middle], 0); }
        in_feat = d_model; out_feat = d_hidden;
        for (int j = 0; j < compute_ws; ++j) {
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                middle_buf + n_per_node_split2 * j,
                dispatched_input_buf + n_per_node_split1 * j,
                weight1,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                0, core_op
            );
        }

        cudaEventRecord(input_to_mid_ready[g], smgr->stream(comp_stream));
        relu_kernel<<<CEIL(n_per_split2, NTH), NTH, 0, smgr->stream(comp_stream)>>>(middle_buf, n_per_split2);
        // cudaEventRecord(mid_ready[g], smgr->stream(comp_stream));

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());

        if (!inplace && g>=dl_dispatched_output){cudaStreamWaitEvent(smgr->stream(comp_stream), output_send_ready[g-dl_dispatched_output], 0);}
        in_feat = d_hidden; out_feat = d_model;
        for (int j = 0; j < compute_ws; ++j) {
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                dispatched_output_buf + n_per_node_split1 * j,
                middle_buf + n_per_node_split2 * j,
                weight2,
                batch_per_expert, in_feat,  out_feat, 
                num_local_expert,
                0, core_op
            );
        }

        cudaEventRecord(output_ready[g], smgr->stream(comp_stream));

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());

        // 0. copy dispathed_input from Device to Host
        //    copy middle from Device to Host
        // D2H
        if (!recompute_di){
            cudaStreamWaitEvent(smgr->stream(d2h_stream), input_ready[g], 0);
            checkCudaErrors(cudaMemcpyAsync(dispatched_input + n_per_split1 * g, dispatched_input_buf, n_per_split1 * sizeof(scalar_t), cudaMemcpyDeviceToHost, smgr->stream(d2h_stream)));
            cudaEventRecord(input_copy_ready[g], smgr->stream(d2h_stream));
        }
        if (!recompute_mi){
            // cudaStreamWaitEvent(smgr->stream(d2h_stream), mid_ready[g], 0);
            cudaStreamWaitEvent(smgr->stream(d2h_stream), input_to_mid_ready[g], 0);
            checkCudaErrors(cudaMemcpyAsync(middle + n_per_split2 * g, middle_buf, n_per_split2 * sizeof(scalar_t), cudaMemcpyDeviceToHost, smgr->stream(d2h_stream)));
            cudaEventRecord(middle_copy_ready[g], smgr->stream(d2h_stream));
        }

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());

        // 5. comm2

        cudaStreamWaitEvent(smgr->stream(comm2_stream), output_ready[g], 0);
        alltoall<scalar_t>(outputs + g * n_per_split1, dispatched_output_buf, n_per_split1, world_size, smgr->ncclcomm2, smgr->stream(comm2_stream), inplace);
        cudaEventRecord(output_send_ready[g], smgr->stream(comm2_stream));
    }

    smgr->sync(nstreams);
    checkCudaErrors(cudaGetLastError());

    for(int g=0; g<num_split; g++){
        cudaEventDestroy(input_ready[g]);
        cudaEventDestroy(output_ready[g]);
        cudaEventDestroy(input_to_mid_ready[g]);
        // cudaEventDestroy(mid_ready[g]);

        cudaEventDestroy(input_copy_ready[g]);
        cudaEventDestroy(middle_copy_ready[g]);
        cudaEventDestroy(output_send_ready[g]);

    }

    delete [] input_ready;
    delete [] output_ready;
    delete [] input_to_mid_ready;
    delete [] mid_ready;
    delete [] input_copy_ready;
    delete [] middle_copy_ready;
    delete [] output_send_ready;

    return;
}

template<typename scalar_t>
void pmoe_cuda_micro_backward_sharded_impl(
        scalar_t* grad_outs,
        scalar_t* inputs,
        std::vector<torch::Tensor> experts_params,

        float* dispatched_input,
        float* middle,
        // const scalar_t* dispatched_output,

        // scalar_t* grad_dispatched_out,
        // scalar_t* grad_middle,
        // scalar_t* grad_dispatched_in,
        scalar_t* grad_in,

        int d_model, int d_hidden, 
        int num_local_expert, int rank, int world_size, 
        int token_per_split,  int num_split,
        CudaStreamManager* smgr,
        scalar_t* grad_dispatched_input_buf1,
        scalar_t* grad_dispatched_input_buf2,
        scalar_t* grad_middle_buf,
        scalar_t* grad_dispatched_output_buf1,
        scalar_t* grad_dispatched_output_buf2,

        scalar_t* middle_buf1,
        scalar_t* middle_buf2,
        scalar_t* dispatched_input_buf1,
        scalar_t* dispatched_input_buf2,
        bool recompute_di,
        bool recompute_mi,
        int inplace, int core_op) {
    

    int n_per_split1 = token_per_split * d_model, n_per_split2 = token_per_split * d_hidden;
    int batch_per_expert = token_per_split / world_size / num_local_expert;
    int n_per_node_split1 = n_per_split1 / world_size;
    int n_per_node_split2 = n_per_split2 / world_size;

    const scalar_t alpha = 1, beta = 0;
    const scalar_t beta2 = 1;
    
    // int rank_in_group = rank % n_gran;
    // int group_rank = rank / n_gran;

    ncclComm_t ncclcomm2 = smgr->ncclcomm2; //? smgr->ncclcomm2 : smgr->ncclcomm;

    
    scalar_t *grad_dispatched_input_buf;
    scalar_t *grad_dispatched_output_buf;
    scalar_t *middle_buf;
    scalar_t *dispatched_input_buf;
    // dl: depend length
    int dl_grad_dispatched_input = grad_dispatched_input_buf2? 2:1; //(grad_dispatched_input_buf2 != 0);
    int dl_grad_dispatched_output = grad_dispatched_output_buf2? 2:1; //(grad_dispatched_output_buf2 != 0);
    int dl_middle = middle_buf2?2:1; //(middle_buf2 != 0);
    int dl_dispatched_input = dispatched_input_buf2?2:1; //(dispatched_input_buf2 != 0);
    if(!grad_dispatched_input_buf2)
        grad_dispatched_input_buf2 = grad_dispatched_input_buf1;
    if(!grad_dispatched_output_buf2)
        grad_dispatched_output_buf2 = grad_dispatched_output_buf1;
    if(!middle_buf2)
        middle_buf2 = middle_buf1;
    if(!dispatched_input_buf2)
        dispatched_input_buf2 = dispatched_input_buf1;

    cudaEvent_t *input_ready = new cudaEvent_t[num_split];
    cudaEvent_t *output_ready = new cudaEvent_t[num_split];
    cudaEvent_t *middle_buf_ready = new cudaEvent_t[num_split];
    cudaEvent_t *middle_buf_used_ready = new cudaEvent_t[num_split];
    cudaEvent_t *input_buf_ready = new cudaEvent_t[num_split];
    cudaEvent_t *input_buf_used_ready = new cudaEvent_t[num_split];
    // cudaEvent_t *grad_output_buf_used_ready = new cudaEvent_t[num_split]; // 用 middle_buf_used_ready 替代
    cudaEvent_t *grad_input_buf_send_ready = new cudaEvent_t[num_split];

    for (int i = 0; i < num_split; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
        cudaEventCreate(middle_buf_ready + i);
        cudaEventCreate(middle_buf_used_ready + i); // grad_output_buf_used_ready
        cudaEventCreate(input_buf_ready + i);
        cudaEventCreate(input_buf_used_ready + i);
        // cudaEventCreate(grad_output_buf_used_ready + i);
        cudaEventCreate(grad_input_buf_send_ready + i);
    }

    scalar_t *weight1, *weight2;
    scalar_t *grad_weight1, *grad_weight2;
    weight1 = experts_params[0].data_ptr<scalar_t>();
    weight2 = experts_params[1].data_ptr<scalar_t>();
    grad_weight1 = experts_params[0].mutable_grad().data_ptr<scalar_t>();
    grad_weight2 = experts_params[1].mutable_grad().data_ptr<scalar_t>();

    // printf("backward\n");

    int in_feat, out_feat;
    int h2d_stream=0, comm1_stream = 1, comp_stream=2, comm2_stream=3, nstreams=4;
    int compute_ws = num_local_expert == 1 ? 1 : world_size;
    batch_per_expert = num_local_expert == 1 ? batch_per_expert * world_size: batch_per_expert;


    for (int g=0; g < num_split; g++) {

        if(g%2==0){
            grad_dispatched_input_buf = grad_dispatched_input_buf1;
            grad_dispatched_output_buf = grad_dispatched_output_buf1;
            middle_buf = middle_buf1;
            dispatched_input_buf = dispatched_input_buf1;
        }else{
            grad_dispatched_input_buf = grad_dispatched_input_buf2;
            grad_dispatched_output_buf = grad_dispatched_output_buf2;
            middle_buf = middle_buf2;
            dispatched_input_buf = dispatched_input_buf2;
        }
        if(inplace){
            grad_dispatched_output_buf = grad_outs + g * n_per_split1;
            grad_dispatched_input_buf = grad_in + n_per_split1 * g;
            dispatched_input_buf = inputs + g * n_per_split1;
        }

        // 1. comm1

        if(!inplace && g>=dl_grad_dispatched_output){ cudaStreamWaitEvent(smgr->stream(comm1_stream), middle_buf_used_ready[g-dl_grad_dispatched_output], 0);}  // 近似相当于 grad_output_buf_used_ready
	    alltoall<scalar_t>(grad_dispatched_output_buf, grad_outs + g * n_per_split1, n_per_split1, world_size, smgr->ncclcomm, smgr->stream(comm1_stream), inplace);
        cudaEventRecord(input_ready[g], smgr->stream(comm1_stream));

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());
            // smgr -> group -> barrier();


        // 0. host -> device
        if (!recompute_mi){
            if (g>=dl_middle){ cudaStreamWaitEvent(smgr->stream(h2d_stream), middle_buf_used_ready[g-dl_middle], 0);  }
            checkCudaErrors(cudaMemcpyAsync(middle_buf, middle + n_per_split2 * g, n_per_split2 * sizeof(scalar_t), cudaMemcpyHostToDevice, smgr->stream(h2d_stream)));
            cudaEventRecord(middle_buf_ready[g], smgr->stream(h2d_stream));
        }

        if (recompute_di){
            if(!inplace){
                if (g>=dl_dispatched_input){ cudaStreamWaitEvent(smgr->stream(h2d_stream), input_buf_used_ready[g-dl_dispatched_input], 0);   }     
                alltoall<scalar_t>(dispatched_input_buf, inputs + g * n_per_split1, n_per_split1, world_size, smgr->ncclcomm, smgr->stream(h2d_stream), false);
            }
           cudaEventRecord(input_buf_ready[g], smgr->stream(h2d_stream));
        }else{
            if (!inplace && g>=dl_dispatched_input){ cudaStreamWaitEvent(smgr->stream(h2d_stream), input_buf_used_ready[g-dl_dispatched_input], 0);   }     
            checkCudaErrors(cudaMemcpyAsync(dispatched_input_buf, dispatched_input + g * n_per_split1, n_per_split1 * sizeof(scalar_t), cudaMemcpyHostToDevice, smgr->stream(h2d_stream)));
            cudaEventRecord(input_buf_ready[g], smgr->stream(h2d_stream));
        }
        
        if (recompute_mi){
            cudaStreamWaitEvent(smgr->stream(h2d_stream), input_buf_ready[g], 0); 
            if (g>=dl_middle){ cudaStreamWaitEvent(smgr->stream(h2d_stream), middle_buf_used_ready[g-dl_middle], 0);  }
            in_feat = d_model; out_feat = d_hidden;
            for (int j = 0; j < compute_ws; ++j) {
                stridedGemm<scalar_t>(
                    smgr->handle(h2d_stream),
                    middle_buf + n_per_node_split2 * j,
                    dispatched_input_buf + n_per_node_split1 * j,
                    weight1,
                    batch_per_expert, in_feat,  out_feat,  // m, k, n
                    num_local_expert,
                    0, core_op
                );
            }
            relu_kernel<<<CEIL(n_per_split2, NTH), NTH, 0, smgr->stream(h2d_stream)>>>(middle_buf, n_per_split2);
            cudaEventRecord(middle_buf_ready[g], smgr->stream(h2d_stream));
        }

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());

        // 2. computation
    // }
    // for (int g=0; g < num_split; g++) {        
        cudaStreamWaitEvent(smgr->stream(comp_stream), input_ready[g], 0);
        in_feat = d_hidden; out_feat = d_model;
        for (int j = 0; j < compute_ws; ++j) {
            // grad_middle_buf
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_dispatched_output_buf + n_per_node_split1 * j,
                grad_middle_buf + n_per_node_split2 * j,
                weight2,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                1, core_op
            );
        }

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());

        /*middle used*/
        cudaStreamWaitEvent(smgr->stream(comp_stream), middle_buf_ready[g], 0);
        relu_backward_kernel<<<CEIL(n_per_split2, NTH), NTH, 0, smgr->stream(comp_stream)>>>(
                middle_buf, grad_middle_buf, n_per_split2);

        // grad_w2
        //(row) grad_w2(e, m, h) <- grad_out(g, e, [b, m](T)) middle(g, e, b, h)
        //(col) grad_w2(h, m, e) <- middle(h, b, e, g) grad_out([m, b](T), e, g)
        in_feat = d_hidden; out_feat = d_model;
        for(int j=0; j<compute_ws; j++){
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_dispatched_output_buf + n_per_node_split1 * j,
                middle_buf + n_per_node_split2 * j,
                grad_weight2,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                2, core_op
            );
        }

        cudaEventRecord(middle_buf_used_ready[g], smgr->stream(comp_stream));

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());

        /*dispatched input used*/
        // grad_w1
        cudaStreamWaitEvent(smgr->stream(comp_stream), input_buf_ready[g], 0);
        in_feat = d_model; out_feat = d_hidden;
        for(int j=0; j<compute_ws; j++){
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_middle_buf + n_per_node_split2 * j,
                dispatched_input_buf + n_per_node_split1 * j,
                grad_weight1,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                2, core_op
            );
        }

        cudaEventRecord(input_buf_used_ready[g], smgr->stream(comp_stream));

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());

        if(!inplace && g>=dl_grad_dispatched_input) { cudaStreamWaitEvent(smgr->stream(comp_stream), grad_input_buf_send_ready[g-dl_grad_dispatched_input], 0);  }
        // similar to grad_middle
        in_feat = d_model; out_feat = d_hidden;
        for (int j = 0; j < compute_ws; ++j) {
            stridedGemm<scalar_t>(
                smgr->handle(comp_stream),
                grad_middle_buf + n_per_node_split2 * j,
                grad_dispatched_input_buf + n_per_node_split1 * j,
                weight1,
                batch_per_expert, in_feat,  out_feat,  // m, k, n
                num_local_expert,
                1, core_op
            );

        }
        cudaEventRecord(output_ready[g], smgr->stream(comp_stream));

// smgr->sync(nstreams);
// checkCudaErrors(cudaGetLastError());

        // comm2
        cudaStreamWaitEvent(smgr->stream(comm2_stream), output_ready[g], 0);
        alltoall<scalar_t>(grad_in + n_per_split1 * g, grad_dispatched_input_buf, n_per_split1, world_size, smgr->ncclcomm2, smgr->stream(comm2_stream), inplace);
        cudaEventRecord(grad_input_buf_send_ready[g], smgr->stream(comm2_stream));
        
    }

    smgr->sync(nstreams);
    checkCudaErrors(cudaGetLastError());
    
    for (int i = 0; i < num_split; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
        cudaEventDestroy(middle_buf_ready[i]);
        cudaEventDestroy(middle_buf_used_ready[i]);
        cudaEventDestroy(input_buf_ready[i]);
        cudaEventDestroy(input_buf_used_ready[i]);
        cudaEventDestroy(grad_input_buf_send_ready[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
    delete [] middle_buf_ready;
    delete [] middle_buf_used_ready;
    delete [] input_buf_ready;
    delete [] input_buf_used_ready;
    delete [] grad_input_buf_send_ready;
}


#endif  // SHARDED_FUSED_COMPUTE_H
