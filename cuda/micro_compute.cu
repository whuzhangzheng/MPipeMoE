
#include <cstdlib>
#include <vector>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvToolsExt.h>
#include <string.h>
#include "micro_compute.cuh"
#include "fused_compute.h"
#include <assert.h> 

// input aligned
std::vector<torch::Tensor> _micro_forward_pipe(
        torch::Tensor &inputs, // (num_split, world_size, nle, -1, d_model)
        std::vector<torch::Tensor> &experts_params,
        std::vector<int> recv_counts, std::vector<int> send_counts, 
        int d_model, int d_hidden, 
        int num_local_expert,
        // int num_token,
        int num_split, int inplace, int core_op
        ){
    // "the number tokens recived by different expert is not equal "

    auto smgr = getCudaStreamManager(inputs.device().index());
    int rank, world_size;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    NCCL_SAFE_CALL(ncclCommCount(smgr->ncclcomm, &world_size));
    // int num_total_experts = num_local_expert * world_size;

    // int max_recv_count, max_send_count;
    // max_recv_count = max(recv_counts);
    // max_send_count = max(send_counts);
    std::vector<int> recv_eles_per_split = std::vector<int>(recv_counts.size());
    std::vector<int> send_eles_per_split = std::vector<int>(send_counts.size());
    divide_mul(recv_counts, recv_eles_per_split, num_split, d_model);
    divide_mul(send_counts, send_eles_per_split, num_split, d_model);
    // assert(max_recv_count%num_split==0 && max_send_count%num_split==0);

    // torch::empty_like()
    auto output = torch::empty_like(inputs); //inputs.new_empty({num_token, d_model});
    int num_comp_tokens = sum(recv_counts);
    auto dispatched_input = inputs.new_empty({num_comp_tokens, d_model});
    auto middle = inputs.new_empty({num_comp_tokens, d_hidden});
    auto dispatched_output = inputs.new_empty({num_comp_tokens, d_model});

    // int num_split = -1;
    // if (num_split == -1) {
    //     char* p = getenv("num_split");
    //     if (p) {
    //         num_split = atoi(p);
    //     } else {
    //         num_split = 1;
    //     }
    // }
    // AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), 
            // "pmoe_cuda_micro_forward", ([&] {
        pipe_moe_cuda_micro_forward_impl<float>(
            inputs.data_ptr<float>(),
            experts_params,
            recv_counts, send_counts,
            recv_eles_per_split, send_eles_per_split,
            dispatched_input.data_ptr<float>(),
            middle.data_ptr<float>(),
            dispatched_output.data_ptr<float>(),
            output.data_ptr<float>(),
            d_model, d_hidden, num_local_expert, rank, world_size,
            num_split, smgr, core_op);
    // }));
    return {output, dispatched_input, middle};

}


std::vector<torch::Tensor> _micro_backward_pipe(
        torch::Tensor &grad_outs,
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        std::vector<int> recv_counts, std::vector<int> send_counts, 
        torch::Tensor dispatched_input,
        torch::Tensor middle,
        int d_model, int d_hidden, 
        int num_local_expert,
        // int num_token,
        int num_split, int inplace, int core_op
        ){
        
    auto smgr = getCudaStreamManager(inputs.device().index());
    int rank, world_size;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    NCCL_SAFE_CALL(ncclCommCount(smgr->ncclcomm, &world_size));
    // printf("rank:%d \t world_size:%d\n", rank, world_size);

    std::vector<int> recv_eles_per_split = std::vector<int>(recv_counts.size());
    std::vector<int> send_eles_per_split = std::vector<int>(send_counts.size());
    divide_mul(recv_counts, recv_eles_per_split, num_split, d_model);
    divide_mul(send_counts, send_eles_per_split, num_split, d_model);
    // int batch_per_rank = batch_size / world_size;

    int num_comp_tokens = sum(recv_counts);
    auto grad_dispatched_output = inputs.new_empty({num_comp_tokens, d_model});
    auto grad_middle = inputs.new_empty({num_comp_tokens, d_hidden});
    auto grad_dispatched_input = inputs.new_empty({num_comp_tokens, d_model});
    auto grad_in = torch::empty_like(inputs); // inputs.new_empty({num_token, d_model});
    
    for(auto p: experts_params){
        CHECK_INPUT(p);
        if (p.grad().defined()){
            CHECK_INPUT(p.grad());
            continue;
        }
        p.mutable_grad() = inputs.new_zeros(p.sizes());
    }


    // int num_split = -1;
    // if (num_split == -1) {
    //     char* p = getenv("num_split");
    //     if (p) {
    //         num_split = atoi(p);
    //     } else {
    //         num_split = 1;
    //     }
    // }

    // char *p2 = getenv("DEBUG");
    // if (p2 && strcmp(p2, "True") == 0){
    //     grad_dispatched_output_hook = grad_dispatched_output;
    //     grad_middle_hook = grad_middle;
    //     grad_dispatched_input_hook = grad_dispatched_input;
    // }

    // AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), 
        // "pmoe_cuda_micro_backward", ([&] {
            pipe_moe_cuda_micro_backward_impl<float>(
                grad_outs.data_ptr<float>(),
                inputs.data_ptr<float>(),
                experts_params,
                recv_counts, send_counts,
                recv_eles_per_split, send_eles_per_split, 

                dispatched_input.data_ptr<float>(),
                middle.data_ptr<float>(),
                // dispatched_output.data_ptr<scalar_t>(),
        
                grad_dispatched_output.data_ptr<float>(),
                grad_middle.data_ptr<float>(),
                grad_dispatched_input.data_ptr<float>(),
                grad_in.data_ptr<float>(),

                d_model, d_hidden, num_local_expert, rank, world_size,
                num_split, smgr, core_op);
        // }));
    return {grad_in};

}