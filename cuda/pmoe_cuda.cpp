#include <iostream>
#include <vector>
#include <torch/extension.h>
#include <string>
#include <c10d/ProcessGroupNCCL.hpp>


std::vector<torch::Tensor> _micro_forward_pipe(
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        int d_model, int d_hidden, 
        int num_local_expert,
        int num_token,
        int num_split, int inplace, int core_op
        );
std::vector<torch::Tensor> _micro_backward_pipe(
        torch::Tensor &grad_outs,
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        torch::Tensor dispatched_input,
        torch::Tensor middle,
        int d_model, int d_hidden, 
        int num_local_expert,
        int num_token,
        int num_split, int inplace, int core_op
        );

std::vector<torch::Tensor> _micro_backward_sharded(
        torch::Tensor &grad_outs,
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        int d_model, int d_hidden, 
        int num_local_expert,
        int num_token, 
        std::string name,
        int num_split, int inplace, int core_op);

torch::Tensor _micro_forward_sharded(
        torch::Tensor &inputs,
        std::vector<torch::Tensor> &experts_params,
        int d_model, int d_hidden, 
        int num_local_expert,
        int num_token, 
        std::string name,
        int num_split, int inplace, int core_op);


void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t, int batch, int d_model, int d_hidden);
void _ensure_nccl2(c10d::ProcessGroupNCCL& p, 
    c10d::ProcessGroupNCCL& p2, 
    torch::Tensor t, int batch, int d_model, int d_hidden, 
    std::vector<std::vector<int>> topos);


static void init_nccl(
    const torch::Tensor &nccl_unique_id_tensor,
    int world_size,
    int world_rank,
    int max_num_split);




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {


    m.def("micro_forward_pipe", &_micro_forward_pipe, "_micro_forward_pipe");
    m.def("micro_backward_pipe", &_micro_backward_pipe, "_micro_backward_pipe");

    m.def("micro_backward_sharded", &_micro_backward_sharded, "_micro_backward_sharded");
    m.def("micro_forward_sharded", &_micro_forward_sharded, "_micro_forward_sharded");


    m.def("ensure_nccl", &_ensure_nccl, "FastMoE ensure torch nccl comm");
    m.def("ensure_nccl2", &_ensure_nccl2, "FastMoE ensure torch nccl comm");

}
