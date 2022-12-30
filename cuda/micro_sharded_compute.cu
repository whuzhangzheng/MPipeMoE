
#include <cstdlib>
#include <vector>
#include <torch/extension.h>
#include <assert.h>
// #include <c10/cuda/CUDAGuard.h>
#include <nvToolsExt.h>
#include "micro_sharded_compute.cuh"


torch::Tensor _micro_forward_sharded(
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        int d_model, int d_hidden, 
        int num_local_expert,
        int num_token, 
        std::string name,
        int num_split, int inplace, int core_op
        ){
    // int num_split = -1;
    // if (num_split == -1) {
    //     char* p = getenv("num_split");
    //     if (p) {
    //         num_split = atoi(p);
    //     } else {
    //         num_split = 1;
    //     }
    // }
    int token_per_split = num_token / num_split;
    
    int SHARD_IMPL = 0;
    char* p2 = getenv("SHARD_IMPL");
    if (p2) {
        SHARD_IMPL = atoi(p2);
    }
    
    auto smgr = getCudaStreamManager(inputs.device().index());
    int rank, world_size;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    NCCL_SAFE_CALL(ncclCommCount(smgr->ncclcomm, &world_size));

    // const auto d_hidden = experts_params[0].size(0);
    // const auto d_model = experts_params[0].size(1);

    // bool move_di = SHARD_IMPL/2==0?true:false;
    bool recompute_di = false, recompute_mi = false;
    if (SHARD_IMPL/2==1){
        recompute_di = true;
    }else if(SHARD_IMPL/2==2){
        recompute_mi = true;
    }else if(SHARD_IMPL/2==3){
        recompute_di = recompute_mi = true;
    }
    smgr->preAllocate(num_token, d_hidden, d_model, name, !recompute_di, !recompute_mi);
    float *dispatched_input_ptr = smgr->get_dispatched_input_ptr(name);
    float *middle_ptr=smgr->get_middle_ptr(name);

    // checkCudaErrors(cudaMallocHost((void**)&dispatched_input_ptr, inputs.size(0) * d_model * sizeof(int))); 
    // checkCudaErrors(cudaMallocHost((void**)&middle_ptr, inputs.size(0) * d_hidden * sizeof(int)));

    // auto global_middle_buf = inputs.new_empty({global_batch_size, d_hidden});
    // auto global_output_buf = inputs.new_empty({global_batch_size, d_model});
    // auto output_buf = inputs.new_empty({inputs.size(0), d_model});

    // auto method_ptr;
    // void (*method_ptr)( const scalar_t* , scalar_t* , scalar_t* , scalar_t* , scalar_t* , scalar_t* , scalar_t* , int , int , int , int , int , int , int , CudaStreamManager* , scalar_t* , scalar_t* , scalar_t* , scalar_t* , scalar_t* );


    float *dispatched_input_buf_ptr=0, *middle_buf_ptr=0, *dispatched_output_buf_ptr=0;
    float *dispatched_input_buf2_ptr=0, *middle_buf2_ptr=0, *dispatched_output_buf2_ptr=0;

    // sharded
    // int inplace = 1;
    // p2 = getenv("zz_inplace");
    // if (p2) {
    //     inplace = atoi(p2);
    // }
    // std::cout << ">inplace=" << inplace << std::endl;
    if (inplace != 1){
        auto dispatched_input_buf = inputs.new_empty({token_per_split, d_model});
        auto dispatched_output_buf = inputs.new_empty({token_per_split, d_model});  
        dispatched_input_buf_ptr = dispatched_input_buf.data_ptr<float>();
        dispatched_output_buf_ptr = dispatched_output_buf.data_ptr<float>();
        // std::cout << "inplace=0" << std::endl;
    }
    auto middle_buf = inputs.new_empty({token_per_split, d_hidden});
    middle_buf_ptr = middle_buf.data_ptr<float>();

    // not sharded
    auto outputs = inputs.new_empty({inputs.size(0), d_model});

    if (SHARD_IMPL % 2 == 0 && inplace !=1 ){
        auto dispatched_input_buf2 = inputs.new_empty({token_per_split, d_model});
        // auto middle_buf2 = inputs.new_empty({token_per_split, d_hidden});
        auto dispatched_output_buf2 = inputs.new_empty({token_per_split, d_model});   
        dispatched_input_buf2_ptr = dispatched_input_buf2.data_ptr<float>();
        // middle_buf2_ptr = middle_buf2.data_ptr<float>();
        dispatched_output_buf2_ptr = dispatched_output_buf2.data_ptr<float>();
    }

        // printf("forward: num_split=%d, SHARD_IMPL=%d, token_per_split:%d, ws:%d\n", num_split, SHARD_IMPL, token_per_split, world_size);
        pmoe_cuda_micro_forward_sharded_impl<float>(
            inputs.data_ptr<float>(),
            outputs.data_ptr<float>(),
            experts_params,
            dispatched_input_ptr, //dispatched_input.data_ptr<scalar_t>(),
            middle_ptr, //middle.data_ptr<scalar_t>(),
            d_model, d_hidden, num_local_expert, rank, world_size,
            token_per_split, num_split, 
            smgr,
            dispatched_input_buf_ptr,
            dispatched_input_buf2_ptr,
            middle_buf_ptr,
            middle_buf2_ptr,
            dispatched_output_buf_ptr,
            dispatched_output_buf2_ptr,
            recompute_di, recompute_mi, inplace, core_op);

    return outputs;
}


std::vector<torch::Tensor> _micro_backward_sharded(
        torch::Tensor &grad_outs,
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        int d_model, int d_hidden, 
        int num_local_expert,
        int num_token, 
        std::string name,
        int num_split, int inplace, int core_op
        ){


    // long num_split = -1;
    // if (num_split == -1) {
    //     char* p = getenv("num_split");
    //     if (p) {
    //         num_split = atoi(p);
    //     } else {
    //         num_split = 1;
    //     }
    // }
    int token_per_split = num_token / num_split;
    int SHARD_IMPL = 0;
    char* p2 = getenv("SHARD_IMPL");
    if (p2) {
        SHARD_IMPL = atoi(p2);
    }
    bool recompute_di = false, recompute_mi = false;
    if (SHARD_IMPL/2==1){
        recompute_di = true;
    }else if(SHARD_IMPL/2==2){
        recompute_mi = true;
    }else if(SHARD_IMPL/2==3){
        recompute_di = recompute_mi = true;
    }

    int rank, world_size;
    auto smgr = getCudaStreamManager(inputs.device().index());
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    NCCL_SAFE_CALL(ncclCommCount(smgr->ncclcomm, &world_size));

    // assert(num_split == 1);

    float *dispatched_input_ptr = smgr->get_dispatched_input_ptr(name);
    float *middle_ptr=smgr->get_middle_ptr(name);
    

    // int inplace = 1;
    // p2 = getenv("zz_inplace");
    // if (p2) {
    //     inplace = atoi(p2);
    // }
    

    float *grad_dispatched_input_buf1_ptr=0, *grad_dispatched_output_buf1_ptr=0, *middle_buf1_ptr=0, *dispatched_input_buf1_ptr=0;
    if(!inplace){
        auto grad_dispatched_output_buf1 = inputs.new_empty({token_per_split, d_model});
        auto grad_dispatched_input_buf1 = inputs.new_empty({token_per_split, d_model});
        auto dispatched_input_buf1 = inputs.new_empty({token_per_split, d_model});
        grad_dispatched_output_buf1_ptr = grad_dispatched_output_buf1.data_ptr<float>();
        grad_dispatched_input_buf1_ptr = grad_dispatched_input_buf1.data_ptr<float>();
        dispatched_input_buf1_ptr = dispatched_input_buf1.data_ptr<float>();
        // std::cout << "inplace=0" << std::endl;
    }
    
    auto middle_buf1 = inputs.new_empty({token_per_split, d_hidden});
    middle_buf1_ptr = middle_buf1.data_ptr<float>();

    // auto grad_dispatched_out = inputs.new_empty({batch_per_rank, d_model});
    // auto grad_middle = inputs.new_empty({batch_size, d_hidden});
    // auto grad_dispatched_in = inputs.new_empty({batch_size, d_model});

    auto grad_middle_buf = inputs.new_empty({token_per_split, d_hidden});
    auto grad_in = inputs.new_empty({num_token, d_model});


    for(auto p: experts_params){
        CHECK_INPUT(p);
        if (p.grad().defined()){
            CHECK_INPUT(p.grad());
            continue;
        }
        p.mutable_grad() = inputs.new_zeros(p.sizes()); // 这里不能用 new empty
    }

    float *grad_dispatched_input_buf2_ptr=0, *grad_dispatched_output_buf2_ptr=0, *middle_buf2_ptr=0, *dispatched_input_buf2_ptr=0;
    if (SHARD_IMPL %2 == 0){
        if (!inplace){
            auto grad_dispatched_output_buf2 = inputs.new_empty({token_per_split, d_model});
            auto grad_dispatched_input_buf2 = inputs.new_empty({token_per_split, d_model});
            auto dispatched_input_buf2 = inputs.new_empty({token_per_split, d_model});
            grad_dispatched_input_buf2_ptr = grad_dispatched_input_buf2.data_ptr<float>();
            grad_dispatched_output_buf2_ptr = grad_dispatched_output_buf2.data_ptr<float>();
            dispatched_input_buf2_ptr = dispatched_input_buf2.data_ptr<float>();
        }

        auto middle_buf2 = inputs.new_empty({token_per_split, d_hidden});
        middle_buf2_ptr = middle_buf2.data_ptr<float>();
    }
    // printf("backward: num_split=%d, SHARD_IMPL=%d, token_per_split:%d, ws:%d\n", num_split, SHARD_IMPL, token_per_split, world_size);

    pmoe_cuda_micro_backward_sharded_impl<float>(
        grad_outs.data_ptr<float>(),
        inputs.data_ptr<float>(),
        experts_params,
        dispatched_input_ptr, //dispatched_input.data_ptr<float>(),
        middle_ptr, // middle.data_ptr<float>(),
        grad_in.data_ptr<float>(),
        d_model, d_hidden, 
        num_local_expert, rank, world_size, 
        token_per_split, num_split, 
        smgr,
        grad_dispatched_input_buf1_ptr,
        grad_dispatched_input_buf2_ptr,
        grad_middle_buf.data_ptr<float>(),
        grad_dispatched_output_buf1_ptr,
        grad_dispatched_output_buf2_ptr,

        middle_buf1_ptr,
        middle_buf2_ptr,
        dispatched_input_buf1_ptr,
        dispatched_input_buf2_ptr,
        recompute_di, recompute_mi,
        inplace, core_op
    );

    // smgr->deallocate();
    // AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), 
    //     "pmoe_cuda_fused_backward_sharded", ([&] {
    //         pmoe_cuda_fused_backward_sharded_impl(
    //             grad_outs.data_ptr<scalar_t>(),
    //             inputs.data_ptr<scalar_t>(),
    //             experts_params,
    //             dispatched_input_ptr, //dispatched_input.data_ptr<scalar_t>(),
    //             middle_ptr, // middle.data_ptr<scalar_t>(),
    //             grad_in.data_ptr<scalar_t>(),
    //             d_model, d_hidden, 
    //             num_expert, rank, world_size, batch_per_rank,
    //             num_split, smgr,
    //             grad_dispatched_input_buf1.data_ptr<scalar_t>(),
    //             grad_dispatched_input_buf2.data_ptr<scalar_t>(),
    //             grad_middle_buf.data_ptr<scalar_t>(),
    //             grad_dispatched_output_buf1.data_ptr<scalar_t>(),
    //             grad_dispatched_output_buf2.data_ptr<scalar_t>(),

    //             middle_buf1.data_ptr<scalar_t>(),
    //             middle_buf2.data_ptr<scalar_t>(),
    //             dispatched_input_buf1.data_ptr<scalar_t>(),
    //             dispatched_input_buf2.data_ptr<scalar_t>()
    //         );
    //     }));

    return {grad_in,};

}

