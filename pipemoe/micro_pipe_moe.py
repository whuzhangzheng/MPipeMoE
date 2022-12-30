

from collections import defaultdict
from deepspeed.utils.timer import ThroughputTimer, SynchronizedWallClockTimer
from typing import Callable, Dict, TYPE_CHECKING, Any, Optional, Tuple, Union, cast
from torch import nn
import time, os
from time import perf_counter
import numpy as np
import torch
import torch.autograd as autograd
from torch import Tensor
import torch.distributed as dist
from torch.nn import Module, ModuleList
import nvtx
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch import nn
import torch.distributed as dist
import pipemoe_cuda 
from .functions import initial
from .gates import Gate




if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

def inverse_indices(indices):
    v, inv_indices= indices.sort()
    return inv_indices


# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any,
                group: dist.ProcessGroup,
                input: Tensor,
                output_split_sizes=None,
                input_split_sizes=None,
                async_op=False) -> Tensor:  # type: ignore
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        input = input.contiguous()
        if output_split_sizes:
            output = input.new_empty(output_split_sizes.sum(), input.size(1))
            dist.all_to_all_single(output, input, output_split_sizes.tolist(), input_split_sizes.tolist(), group=group, async_op=async_op)
        else:
            # print(input.shape)
            output = torch.empty_like(input)
            dist.all_to_all_single(output, input, group=group, async_op=async_op)
        return output
    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return None, _AllToAll.apply(ctx.group, *grad_output), None, None, None


class StreamManager():
    def __init__(self, num_split=2) -> None:
        self.num_split = num_split
        self.reset()    
    def reset(self):
        self.comp_stream = new_stream()
        self.comm1_stream = new_stream()
        self.comm2_stream = new_stream()
        self.comm1_end_events = [torch.cuda.Event() for i in range(self.num_split)]
        self.comp_end_event = [torch.cuda.Event() for i in range(self.num_split)]
        pass
    def sync(self):
        self.comp_stream.synchronize()
        self.comm1_stream.synchronize()
        self.comm2_stream.synchronize()

def split_generator(send_counts: np.ndarray, recv_counts: np.ndarray, num_split):
    send_counts, recv_counts = send_counts.copy(), recv_counts.copy()
    max_mun_token = max(send_counts.max(), recv_counts.max())
    num_token_per_split = (max_mun_token + num_split -1) // num_split
    def minimum_and_reduce(array, ceil_value):
        output = np.minimum(array, ceil_value)
        array[:] = array - output
        return output
    for i in range(num_split):
        send_counts_split = minimum_and_reduce(send_counts, num_token_per_split)
        recv_counts_split = minimum_and_reduce(recv_counts, num_token_per_split)
        yield send_counts_split, recv_counts_split

def pad_indices(sort_indices, send_counts, recv_counts, num_split):
    if isinstance(send_counts, list):
        max_num_token_one_expert = (max(send_counts+recv_counts) + num_split - 1) // num_split * num_split
    else:
        max_num_token_one_expert = (max(send_counts.max().item(), recv_counts.max().item()) + num_split - 1) // num_split * num_split
    padded_send_indices = torch.empty(max_num_token_one_expert * len(send_counts), device=sort_indices.device, dtype=torch.long)
    padded_indices_mask = torch.zeros(max_num_token_one_expert * len(send_counts), device=sort_indices.device, dtype=torch.bool)
    # padded_inversed_send_indices= torch.zeros_like(padded_send_indices)
    cumsum_send_counts = np.cumsum([0] + send_counts).tolist()
    for i,length in enumerate(send_counts):
        start_idx = i*max_num_token_one_expert
        padded_send_indices[start_idx: start_idx+length] = sort_indices[cumsum_send_counts[i]: cumsum_send_counts[i+1]]
        # padded_inversed_send_indices[start_idx: start_idx+length] = inversed_sort_indices[cumsum_send_counts[i]: cumsum_send_counts[i+1]]
        padded_indices_mask[start_idx: start_idx+length] = True

    return padded_send_indices, padded_indices_mask


import numpy as np
import copy
def input_generator(send_counts: np.ndarray, recv_counts: np.ndarray, num_split, sort_indices: torch.Tensor=None, world_size=None, padding=True):
    """
    inp in not padded
    """
    if sort_indices == None:
        sort_indices = torch.arange(sum(send_indices))
    max_num_token = max(send_counts.max(), recv_counts.max())
    max_per_split_per_node = (max_num_token + num_split -1) // num_split
    spliter = split_generator(send_counts, recv_counts, num_split)
    send_counts_offset = np.zeros(len(send_counts)+1, dtype=int)
    np.cumsum(send_counts, out=send_counts_offset[1:])
    output_token_offset = 0
    def pad(part_sort_indices):
        indices_mask = torch.ones(max_per_split_per_node, dtype=torch.bool, device=sort_indices.device)
        if len(part_sort_indices) < max_per_split_per_node:
            indices_mask[len(part_sort_indices):] = False
            return torch.cat([part_sort_indices, sort_indices[:(max_per_split_per_node-len(part_sort_indices))]], dim=0), indices_mask
        return part_sort_indices, indices_mask
    for send_counts_split, recv_counts_split in spliter:
        if padding:
            ret = [pad(sort_indices[offset: offset+num]) for offset, num in zip(send_counts_offset, send_counts_split)]
            send_indices = [x[0] for x in ret]
            indices_mask = [x[1] for x in ret]
            if not world_size:
                world_size = dist.get_world_size()
            output_token_offset += max_per_split_per_node * world_size
        else:
            indices_mask = torch.ones(max_per_split_per_node, dtype=torch.bool, device=sort_indices.device)
            send_indices = [sort_indices[offset: offset+num] for offset, num in zip(send_counts_offset, send_counts_split)]
            output_token_offset += sum(recv_counts_split)
        # send_indices = [y for x in send_indices for y in x]
        send_indices = torch.cat(send_indices, dim=0)
        # splited_inputs = inp[send_indices]
        yield send_indices, indices_mask, output_token_offset, send_counts_split, recv_counts_split
        send_counts_offset += max_per_split_per_node


class MicroBatchPipeMOELayer(Base):    
    route_data = {}
    args = None
    epoch = 0
    def __init__(self,
                 d_model, d_hidden,
                 num_experts,
                 experts: Module,
                 gate: Module,
                 num_local_experts: int,
                 capacity_factor: float = 1,
                 min_capacity: int = -1,
                 group: Optional[Any] = None, 
                 name="moe-layer", 
                 args=None, 
                 debug=False, 
                 normalize_weights=False,
                 k = 2,
                 fix_capacity=True,
                 num_split=2) -> None:
        super().__init__()
        self.gate = gate
        self.top_k = k
        self.fix_capacity = fix_capacity
        # self.gate = gate
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.experts = experts
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.num_local_experts = num_local_experts
        self.capacity_factor = capacity_factor
        # if self.capacity_factor != 1:
        #     raise NotImplementedError("capacity_factor can only be 1")
        self.min_capacity = min_capacity
        self.time_falltoall = 0.0
        self.time_salltoall = 0.0
        self.time_moe = 0.0
        self.timers = SynchronizedWallClockTimer()
        self.wall_clock_breakdown = False
        self.name = name
        self.pipe_type = args.pipe_type
        # assert self.pipe_type in ['seq', "pipe", 'original', 'seq2'] or self.pipe_type.startswith("sharded")
        # self.num_experts = self.world_size * self.num_local_experts
        self.args = args

        self.ep_ws = self.args.ep_world_size
        if MicroBatchPipeMOELayer.args == None:
            MicroBatchPipeMOELayer.args = args
        self.debug = debug
        self.expert_capacity = None
        self.normalize_weights = normalize_weights
        self.times_statistic = {}
        self.inplace = int(os.environ.get("zz_inplace", "0"))
        self.core_op = int(os.environ.get("zz_core_op", "0"))
        # self.mem_manager = MemoryManager()
        self.group1 = dist.distributed_c10d._get_default_group()
        self.group2 = dist.new_group(list(range(dist.get_world_size())), backend='nccl')
        self.num_split = num_split
        self.stream_manater = StreamManager(num_split)
    def __str__(self) -> str:
        return f"MicroBatchPipeMOELayer(pipe_type={self.pipe_type}, num_split={self.num_split} inplace={self.inplace}, core_op={self.core_op})"
    def timer(self, name, attr_name="", start=True):
        if self.wall_clock_breakdown:
            if start:
                self.timers(name).start()
                # nvtx.push_range(name)
            else:
                self.timers(name).stop()
                # nvtx.pop_range()
                self.__setattr__(attr_name, self.timers(name).elapsed(reset=False) * 1000)
                self.times_statistic[name] = self.timers(name).elapsed(reset=False) * 1000

    def get_experts_num(self, expert_ids):
        experts, counts = expert_ids.unique(return_counts=True)
        if not self.fix_capacity:
            if (experts.size(0) != self.num_experts):
                counts = counts.new_zeros(self.num_experts).scatter(0, experts, counts)
            counts = counts.reshape(-1, self.num_local_experts)
            recv_counts = torch.empty_like(counts)
            dist.all_to_all_single(recv_counts, counts, group=self.group)
        else:
            recv_counts = counts
        return counts.reshape(-1), recv_counts.reshape(-1)
        
    def forward(self, *input: Tensor, **kwargs: Any) -> Tensor:
        inp = input[0]
        input_shape = inp.shape

        # self.timer("moe", start=True)


        inp = inp.reshape(-1, inp.shape[-1])
        d_model = inp.shape[-1]
        num_tokens = inp.shape[0]

        # self.timer("gate", start=True)
        self.l_aux, combile_weights, sort_indices, expert_ids = self.gate(inp, normalize_weights=self.normalize_weights)
        # print(combile_weights.shape, sort_indices.shape, expert_ids.shape)
        send_counts, recv_counts = self.get_experts_num(expert_ids)
        # send_counts = send_counts.cpu().numpy()
        # recv_counts = recv_counts.cpu().numpy()

        # self.timer("gate",attr_name="time_gate", start=False)

        # inputs = _LOCAL_SCATTER.apply(inputs, sort_indices)

        # if dist.get_rank() == 0:
        #     print(dist.get_rank(), "send_counts:", send_counts.tolist())
        #     print(dist.get_rank(), "recv counts:", recv_counts.tolist())
        # print(sort_indices.shape)

        self.timer("main_forward", start=True)
        if self.pipe_type == 'original':
            inp = inp[sort_indices]
            # print('original')
            # recv_counts = recv_counts.reshape(-1, self.num_local_experts)
            # send_counts = send_counts.reshape(-1, self.num_local_experts)
            dispatched_input = _AllToAll.apply(self.group, inp)
            expert_output = self.experts(dispatched_input)
            expert_output = expert_output.reshape(-1, d_model)
            combined_out = _AllToAll.apply(self.group, expert_output)
            combined_out = combined_out[self.gate.get_inversed_indices()]        
            # combined_out = combined_out[inverse_indices(sort_indices)]
        elif self.pipe_type.endswith("pipe"):
            # print("pipe")
            sort_indices = sort_indices.reshape(self.world_size, self.num_local_experts, self.num_split, -1).permute(2, 0, 1, 3).reshape(-1)
            inp = inp[sort_indices]
            combined_out = _DIST_MICRO_PIPE_FUNC.apply(inp, self.experts, self.num_local_experts, self.d_hidden, self.num_split, self.inplace, self.core_op)
            combined_out = combined_out[self.gate.get_inversed_indices()]        
            # combined_out = combined_out[inverse_indices(sort_indices)]       
        elif self.pipe_type.endswith("py"):
            print("micro-py")
            combined_out = _DIST_MICRO_PIPE_PY(None, inp, sort_indices, send_counts, recv_counts, self.experts, self.num_local_experts, self.world_size, self.stream_manater, self.group1, self.group2)
        else:
            if not self.fix_capacity:
                send_counts, recv_counts = send_counts.tolist(), recv_counts.tolist()
                padded_send_indices, padded_indices_mask = pad_indices(sort_indices, send_counts, recv_counts, self.num_split)
                padded_send_indices = padded_send_indices.reshape(self.world_size, self.num_local_experts, self.num_split, -1).permute(2, 0, 1, 3).reshape(-1)
                inp = inp[padded_send_indices]
            else:
                sort_indices = sort_indices.reshape(self.world_size, self.num_local_experts, self.num_split, -1).permute(2, 0, 1, 3).reshape(-1)
                inp = inp[sort_indices]
            combined_out = _DIST_MICRO_SHARDED_FUNC.apply(inp, self.experts, self.num_local_experts, self.d_hidden, self.name, self.num_split, self.inplace, self.core_op)
            if not self.fix_capacity:
                combined_out = combined_out[inverse_indices(padded_send_indices[padded_indices_mask])]
            else:
                # combined_out = combined_out[inverse_indices(sort_indices)]
                combined_out = combined_out[self.gate.get_inversed_indices()]

        self.timer("main_forward", attr_name="time_main_forward", start=False)
        

        # combined_out[sort_indices.long()] = combined_out.clone()
        # combined_out = _LOCAL_GATHER.apply(combined_out, sort_indices)
        # combined_out = _local_gather(combined_out, sort_indices, combined_out.shape[0], maybe_overlap=False)

        d_model_out = combined_out.shape[-1]

        # combined_out = combile_weights.unsqueeze(1) * combined_out
        combined_out.mul_(combile_weights.reshape(-1, 1)) 
        combined_out = combined_out.reshape(num_tokens, self.top_k, -1).sum(axis=1)

        combined_out = combined_out.reshape(*input_shape[:-1], d_model_out)
        # self.timer("moe", "time_moe", start=False)
        
        return combined_out


class _DIST_MICRO_PIPE_FUNC(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inp, experts, num_local_experts, d_hidden, num_split, inplace, core_op) -> Tensor:  # type: ignore
        d_model = inp.size(-1)
        num_token = inp.size(0)
        initial(inp, d_hidden)

        # combined_out = pipemoe_cuda.fused_forward2(inputs, tuple(experts.parameters()), num_local_experts, mem_manager.dispatched_input, mem_manager.middle, mem_manager.dispatched_output)
        combined_out, dispatched_input, middle_output  = pipemoe_cuda.micro_forward_pipe(inp, tuple(experts.parameters()), d_model, d_hidden, num_local_experts, num_token, num_split, inplace, core_op)

        ctx.saved_for_backward = tuple(experts.parameters()), inp, d_model, d_hidden, num_local_experts, num_token, dispatched_input, middle_output, num_split, inplace, core_op

        return combined_out
    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor):
        expert_params, inp, d_model, d_hidden, num_local_experts, num_token, dispatched_input, middle_output, num_split, inplace, core_op = ctx.saved_for_backward
        (grad_in,)  = pipemoe_cuda.micro_backward_pipe(grad_output[0], inp, expert_params, dispatched_input, middle_output, d_model, d_hidden, num_local_experts, num_token, num_split, inplace, core_op)
        return grad_in, None, None, None, None, None, None

class _DIST_MICRO_SHARDED_FUNC(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inp, experts, num_local_experts, d_hidden, name, num_split, inplace, core_op) -> Tensor:  # type: ignore
        d_model = inp.size(-1)
        num_token = inp.size(0)
        initial(inp, d_hidden)

        # combined_out = pipemoe_cuda.fused_forward2(inputs, tuple(experts.parameters()), num_local_experts, mem_manager.dispatched_input, mem_manager.middle, mem_manager.dispatched_output)
        combined_out  = pipemoe_cuda.micro_forward_sharded(inp, tuple(experts.parameters()), d_model, d_hidden, num_local_experts, num_token, name, num_split, inplace, core_op)

        ctx.saved_for_backward = tuple(experts.parameters()), inp, d_model, d_hidden, num_local_experts, num_token, name, num_split, inplace, core_op

        return combined_out
    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor):
        expert_params, inp, d_model, d_hidden, num_local_experts, num_token, name, num_split, inplace, core_op = ctx.saved_for_backward
        (grad_in,)  = pipemoe_cuda.micro_backward_sharded(grad_output[0], inp, expert_params, d_model, d_hidden, num_local_experts, num_token, name, num_split, inplace, core_op)
        return grad_in, None, None, None, None, None, None, None
                

def new_stream(device=torch.device('cuda', 0)):
    return torch.cuda.Stream(device)


    

