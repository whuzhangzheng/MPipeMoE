
#include <cstdlib>
#include <vector>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvToolsExt.h>
#include <string.h>
#include "micro_seq.cuh"
#include "fused_compute.h"

void m_alltoall(torch::Tensor out, torch::Tensor inp, int num_split, std::vector<int> recv_counts, std::vector<int> send_counts){
    CudaStreamManager* smgr = getCudaStreamManager(inp.device().index());
    // _set_nccl(p, inp);

    int world_size, rank;
    NCCL_SAFE_CALL(ncclCommCount((smgr->ncclcomm), &world_size));    
    NCCL_SAFE_CALL(ncclCommUserRank((smgr->ncclcomm), &rank));   
    
    int d_model = inp.size(1);

    std::vector<int> recv_eles_per_split = std::vector<int>(recv_counts.size());
    std::vector<int> send_eles_per_split = std::vector<int>(send_counts.size());
    divide_mul(recv_counts, recv_eles_per_split, num_split, d_model);
    divide_mul(send_counts, send_eles_per_split, num_split, d_model);
    int send_offset=0, recv_offset=0;
    int send_offset_per_split = sum(send_eles_per_split);
    int recv_offset_per_split = sum(recv_eles_per_split);
    
    // int count_per_sr = inp.size(0) / world_size / split * feat;
    for(int i=0; i<num_split; i++){
        AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(),
                "alltoall", ([&] {
            alltoall<scalar_t>(  // ⚌~Y⚌~G~L⚌~Z~Dscalar_t ⚌~T该就⚌~X⚌inp.scalar_type()
                out.data_ptr<scalar_t>() + recv_offset,
                inp.data_ptr<scalar_t>() + send_offset,
                recv_eles_per_split, send_eles_per_split,  world_size, smgr->ncclcomm, smgr->stream(0), false);
        
        }));
        recv_offset += recv_offset_per_split; send_offset += send_offset_per_split;
    }

}

std::vector<torch::Tensor> _experts_ffn(
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        std::vector<int> recv_counts,
        int d_hidden, 
        int num_local_expert,
        int num_split, int inplace, int core_op
        ){
    // "the number tokens recived by different expert is not equal "
    assert(num_local_expert == 1);
    int d_model = inputs.size(1);
    int num_token = inputs.size(0);
    auto smgr = getCudaStreamManager(inputs.device().index());
    int rank, world_size;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    NCCL_SAFE_CALL(ncclCommCount(smgr->ncclcomm, &world_size));

    auto output = inputs.new_empty({num_token, d_model});
    auto middle = inputs.new_empty({num_token, d_hidden});

    std::vector<int> recv_eles_per_split = std::vector<int>(recv_counts.size());   
    divide_mul(recv_counts, recv_eles_per_split, num_split, d_model);

    // AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), 
            // "pmoe_cuda_micro_forward", ([&] {
        experts_ffn_forward_impl<float>(
            inputs.data_ptr<float>(),
            experts_params,
            middle.data_ptr<float>(),
            output.data_ptr<float>(),
            recv_eles_per_split,
            d_model, d_hidden, num_local_expert, rank, world_size,
            num_split, smgr, core_op);
    // }));
    return {output, middle};

}


std::vector<torch::Tensor> _experts_ffn_backward(
        torch::Tensor &grad_outs,  //
        torch::Tensor &inputs,  // 
        std::vector<torch::Tensor> &experts_params,
        // torch::Tensor dispatched_input,
        torch::Tensor middle,
        std::vector<int> recv_counts,
        int d_hidden, 
        int num_local_expert,
        int num_split, int inplace, int core_op
        ){
    int num_token = grad_outs.size(0);
    int d_model = inputs.size(1);

    auto smgr = getCudaStreamManager(inputs.device().index());
    int rank, world_size;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    NCCL_SAFE_CALL(ncclCommCount(smgr->ncclcomm, &world_size));
    // printf("rank:%d \t world_size:%d\n", rank, world_size);
    // int batch_per_rank = batch_size / world_size;

    auto grad_middle = inputs.new_empty({num_token, d_hidden});
    auto grad_in = inputs.new_empty({num_token, d_model});
    
    for(auto p: experts_params){
        CHECK_INPUT(p);
        if (p.grad().defined()){
            CHECK_INPUT(p.grad());
            continue;
        }
        p.mutable_grad() = inputs.new_zeros(p.sizes());
    }
    std::vector<int> recv_eles_per_split = std::vector<int>(recv_counts.size());   
    divide_mul(recv_counts, recv_eles_per_split, num_split, d_model);

    // AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), 
        // "pmoe_cuda_micro_backward", ([&] {
            experts_ffn_backward_impl<float>(
                grad_outs.data_ptr<float>(), //  grad_dispatched_output
                inputs.data_ptr<float>(),    // dispatched_input
                experts_params,
                recv_eles_per_split,

                middle.data_ptr<float>(),
        
                grad_middle.data_ptr<float>(),
                grad_in.data_ptr<float>(), // grad_dispatched_input

                d_model, d_hidden, num_local_expert, rank, world_size,
                num_split, smgr, core_op);
        // }));
    return {grad_in};

}
// deprecated
std::vector<torch::Tensor> _experts_ffn_deprecated(
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        int d_model, int d_hidden, 
        int num_local_expert,
        int num_token,
        int num_split, int inplace, int core_op
        ){
    // "the number tokens recived by different expert is not equal "

    auto smgr = getCudaStreamManager(inputs.device().index());
    int rank, world_size;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    NCCL_SAFE_CALL(ncclCommCount(smgr->ncclcomm, &world_size));

    auto output = inputs.new_empty({num_token, d_model});
    auto middle = inputs.new_empty({num_token, d_hidden});

    int token_per_split = num_token / num_split;

    // AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), 
            // "pmoe_cuda_micro_forward", ([&] {
        experts_ffn_forward_deprecated_impl<float>(
            inputs.data_ptr<float>(),
            experts_params,
            middle.data_ptr<float>(),
            output.data_ptr<float>(),
            d_model, d_hidden, num_local_expert, rank, world_size,
            token_per_split, num_split, smgr, core_op);
    // }));
    return {output, middle};

}


std::vector<torch::Tensor> _experts_ffn_backward_deprecated(
        torch::Tensor &grad_outs,  //
        torch::Tensor &inputs,  // 
        std::vector<torch::Tensor> &experts_params,
        // torch::Tensor dispatched_input,
        torch::Tensor middle,
        int d_model, int d_hidden, 
        int num_local_expert,
        int num_token,
        int num_split, int inplace, int core_op
        ){

    auto smgr = getCudaStreamManager(inputs.device().index());
    int rank, world_size;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    NCCL_SAFE_CALL(ncclCommCount(smgr->ncclcomm, &world_size));
    // printf("rank:%d \t world_size:%d\n", rank, world_size);
    // int batch_per_rank = batch_size / world_size;

    auto grad_middle = inputs.new_empty({num_token, d_hidden});
    auto grad_in = inputs.new_empty({num_token, d_model});
    
    for(auto p: experts_params){
        CHECK_INPUT(p);
        if (p.grad().defined()){
            CHECK_INPUT(p.grad());
            continue;
        }
        p.mutable_grad() = inputs.new_zeros(p.sizes());
    }


    int token_per_split = num_token / num_split;

    // AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), 
        // "pmoe_cuda_micro_backward", ([&] {
            experts_ffn_backward_deprecated_impl<float>(
                grad_outs.data_ptr<float>(), //  grad_dispatched_output
                inputs.data_ptr<float>(),    // dispatched_input
                experts_params,

                middle.data_ptr<float>(),
        
                grad_middle.data_ptr<float>(),
                grad_in.data_ptr<float>(), // grad_dispatched_input

                d_model, d_hidden, num_local_expert, rank, world_size,
                token_per_split, num_split, smgr, core_op);
        // }));
    return {grad_in};

}
void alltoall_deprecated(torch::Tensor out, torch::Tensor inp, int split, std::vector<int> recv_counts, std::vector<int> send_counts, int max_send_count, int max_recv_count, int flag){
    CudaStreamManager* smgr = getCudaStreamManager(inp.device().index());
    // _set_nccl(p, inp);
    int world_size, rank;
    NCCL_SAFE_CALL(ncclCommCount((smgr->ncclcomm), &world_size));    
    NCCL_SAFE_CALL(ncclCommUserRank((smgr->ncclcomm), &rank));   
    
    int feat = inp.size(1);
    // int count_per_sr = inp.size(0) / world_size / split * feat;
    AT_DISPATCH_FLOATING_TYPES(inp.scalar_type(),
            "alltoall", ([&] {
        alltoall_deprecated_impl<scalar_t>(  // ⚌~Y⚌~G~L⚌~Z~Dscalar_t ⚌~T该就⚌~X⚌inp.scalar_type()
            out.data_ptr<scalar_t>(),
            inp.data_ptr<scalar_t>(),
            split, 
            recv_counts, send_counts, flag,
            max_send_count, max_recv_count,
            feat, 
            world_size, rank, smgr);
      
    }));
}
