from enum import EnumMeta
import torch
from torch import nn, topk
import torch.nn.functional as F
from typing import Optional
# SynchronizedWallClockTimer
from deepspeed.utils.timer import ThroughputTimer, SynchronizedWallClockTimer
import os

class Gate(nn.Module):
    wg: torch.nn.Linear
    def __init__(self,
                 model_dim: int,
                 num_experts: int,
                #  args,
                 k: int = 2,
                 capacity_factor: float = 1.0,
                #  min_capacity: int = 4,
                 fix_capacity=True) -> None:
        super(Gate, self).__init__()
        # torch.manual_seed(1)
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False).float()
        self.k = k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        # self.min_capacity = min_capacity
        self.wall_clock_breakdown = False
        
        self.num_split = int(os.environ.get("num_split", "2"))
        self.timers = SynchronizedWallClockTimer()
        self.gate_time = 0.0
        self.wall_clock_breakdown = False
        self.fix_capacity = fix_capacity
        self.capacity = None
        self.wg_init()
        self.weight_versoin = 1
    
    def wg_init(self):
        # self.wg.weight.data.zero_()
        # self.wg.weight.data.fill_diagonal_(1)

        # gain=nn.init.calculate_gain('relu')
        nn.init.kaiming_normal_(self.wg.weight)

    def set_capacity(self, num_tokens):
        # print(self.num_experts)
        self.capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        # if self.min_capacity >0 :
        #     self.capacity = max(self.min_capacity, self.capacity)

    def forward(self, input: torch.Tensor, normalize_weights=False):
        self.num_tokens = input.shape[0]
        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()

        logits = self.wg(input)
        if not self.capacity:
            self.set_capacity(logits.size(0))

        l_aux, indices, gates = loss_compute(logits)
    
        # print("l_aux", self.l_aux)
        # if self.weight_version == 1:
            # combile_weights, indices = compute_weight_and_sort_indices1(gates, self.capacity)
        # elif self.weight_version == 2:
            # combile_weights, indices = compute_weight_and_sort_indices2(gates, self.capacity)
        # elif self.weight_version == 3:
            # combile_weights, indices = compute_weight_and_sort_indices3(gates, self.capacity)
        if not self.fix_capacity:
            combile_weights, indices, expert_ids = compute_weight_and_sort_indices(gates,  self.k, normalize=normalize_weights)
        else:
            # combile_weights, indices, expert_ids = compute_weight_and_sort_indices_fix_capacity1(gates,  self.k, self.capacity_factor, normalize=normalize_weights)
            combile_weights, indices, expert_ids, mask = compute_weight_and_sort_indices_fix_capacity_simple(gates,  self.k, self.capacity_factor, normalize=normalize_weights, num_split=self.num_split)
            self.mask = mask
        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False) * 1000
        self.indices = indices
        # print(expert_ids.unique(return_counts=True)[1])
        return l_aux, combile_weights, indices, expert_ids
    def get_inversed_indices(self):
        inversed_indices = inverse_indices(self.indices)
        if self.fix_capacity:
            inversed_indices_full = torch.zeros(self.num_tokens, device=inversed_indices.device, dtype=inversed_indices.dtype)
            inversed_indices = inversed_indices[self.mask]
            inversed_indices_full[self.indices[self.mask]] = inversed_indices
            # print(inversed_indices_full.shape)
            return inversed_indices_full

        return inversed_indices

    def update_indices(self, indices):
        self.indices = indices

def loss_compute(logits, used_token: torch.Tensor = None, noisy_gate_policy: Optional[str] = None):
    """
    top1
    """
    gates = F.softmax(logits, dim=1)
    # gates has shape of SE
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    indices1_s = torch.argmax( gates, dim=1)
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)
    # mask only used tokens
    if used_token is not None:
        mask1 = torch.einsum("s,se->se", used_token, mask1)


    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts

    return l_aux, indices1_s, gates


def compute_weight_and_sort_indices_softmove(gates, k, normalize=False):
    # todo
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    capacity = k * num_tokens // num_experts

    values = gates
    template = torch.arange(values.numel()) 
    final_topk_indices = (torch.ones_like(values, dtype=torch.long) * torch.arange(num_experts, device=gates.device).unsqueeze(0)).reshape(-1)
    index_base = torch.ones_like(values, dtype=torch.long) * torch.arange(num_tokens, device=gates.device).unsqueeze(1) * num_experts
    pad_val = 1e-2
    while True:
        values, topk_indices = torch.topk(values, k=num_experts, dim=1)        
        final_topk_indices = final_topk_indices[(topk_indices+index_base).reshape(-1)]
        topk_indices = final_topk_indices.reshape(num_tokens, num_experts).transpose(0,1).reshape(-1)
        values = values.transpose(0,1).reshape(-1)
        # expert_ids, order_indices = topk_indices[:, :k].transpose(0, 1).reshape(-1).sort()
        ok = True
        experts, counts = topk_indices[:capacity*num_experts].unique(return_counts=True)
        # print(counts)
        # 注意有可能一次超出的expert 太多，导致
        for i, expert in enumerate(experts):
            if counts[i] > capacity:
                ok = False
                values[template[topk_indices == expert][capacity:]]= pad_val
                pad_val = pad_val/10  # 每次填充不一样的'小'数
        values = values.reshape(num_experts, num_tokens).transpose(0,1)
        if ok:
            break
    topk_indices = final_topk_indices.reshape(num_tokens, num_experts)[:, :k]
    weights = values[:, :k]


    expert_ids, order_indices = topk_indices.reshape(-1).sort() # tokens_indices: for restoring original orider
    index_base2 = torch.ones_like(topk_indices, dtype=torch.long) * torch.arange(num_tokens, device=gates.device).unsqueeze(1) 
    indices = index_base2.reshape(-1)[order_indices] 
    # weights = torch.zeros(num_experts * capacity, device=gates.device, dtype=values.dtype, requires_grad=False)
    # weights[indices.reshape(-1)] = values.reshape(-1)    
    if normalize:
        weights = F.softmax(weights, dim=-1)
    return weights, indices,  expert_ids

# def compute_weight_and_sort_indices_capacity(gates, k=1, normalize=False):
#     num_tokens = int(gates.shape[0])
#     num_experts = int(gates.shape[1])
#     gates = gates.detach()
#     values, topk_indices = torch.topk(gates, k=k, dim=1) # indices： expert idx in #(num_tokens, topk)
#     weights = values
#     expert_ids, order_indices = topk_indices.reshape(-1).sort() # tokens_indices: for restoring original orider
#     indices = (torch.ones_like(values, dtype=torch.long) * torch.arange(num_tokens, device=gates.device).unsqueeze(1)).reshape(-1)[order_indices] 
#     # weights = torch.zeros(num_experts * capacity, device=gates.device, dtype=values.dtype, requires_grad=False)
#     # weights[indices.reshape(-1)] = values.reshape(-1)    
#     if normalize:
#         weights = F.softmax(weights, dim=-1)
#     return weights, indices,  expert_ids

def inverse_indices(indices):
    v, inv_indices= indices.sort()
    return inv_indices

def compute_weight_and_sort_indices_fix_capacity1(gates, k, capacity_factor=1, normalize=False):
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    capacity = int(k * num_tokens // num_experts * capacity_factor)
    # print(capacity)
    assert k==1
    values, indices = torch.topk(gates, k=capacity, dim=0)  # [0.7032, 0.6803, 0.5373,  ..., 0.5320, 0.5373, 0.4983], 
    weights = torch.zeros(num_tokens * k, device=gates.device, dtype=values.dtype, requires_grad=False)
    # weights[indices.reshape(-1)] = values.reshape(-1)    
    weights[indices.reshape(-1)] = values.reshape(-1)    
    expert_ids = torch.ones(num_experts, capacity, device=gates.device) * torch.arange(num_experts, device=gates.device).unsqueeze(-1)
    return weights.reshape(num_tokens, k), indices.transpose(0,1).reshape(-1), expert_ids.reshape(-1)

def compute_weight_and_sort_indices_fix_capacity_simple(gates, k, capacity_factor=1, normalize=False, num_split=1):
    """
    realloc: 让分布尽可能均匀, 把超出expert容量的token 移到另外的expert
    """
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    capacity = int(k * num_tokens // num_experts * capacity_factor) // num_split * num_split
    values, topk_indices = torch.topk(gates, k=k, dim=1)  # [0.7032, 0.6803, 0.5373,  ..., 0.5320, 0.5373, 0.4983], 
    weights = values
    experts, idx = topk_indices.reshape(-1).sort()
    idx = (torch.ones_like(topk_indices) * torch.arange(num_tokens, device=gates.device).unsqueeze(1)).reshape(-1)[idx]
    indices = torch.zeros(capacity, num_experts, device=gates.device, dtype=topk_indices.dtype) + -1
    for i in range(num_experts):
        num_expert_token = (experts == i).sum().item()
        num_move = min(num_expert_token, capacity)
        indices[:num_move, i] = idx[:num_move] 
        idx = idx[num_expert_token:]
    indices = indices[:capacity, :].transpose(0, 1).reshape(-1)
    mask = indices >= 0 
    # indices[~mask] = 0
    # weights = torch.zeros(num_experts * capacity, device=gates.device, dtype=values.dtype, requires_grad=False)
    
    expert_ids = torch.ones(num_experts, capacity, device=gates.device) * torch.arange(num_experts, device=gates.device).unsqueeze(-1)
    # print("ca", capacity)
    return weights, indices, expert_ids.reshape(-1), mask


def compute_weight_and_sort_indices_fix_capacity(gates, k, capacity_factor=1, normalize=False, realloc=True):
    """
    realloc: 让分布尽可能均匀, 把超出expert容量的token 移到另外的expert
    """
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    capacity = int(k * num_tokens // num_experts * capacity_factor)
    values, topk_indices = torch.topk(gates, k=k, dim=1)  # [0.7032, 0.6803, 0.5373,  ..., 0.5320, 0.5373, 0.4983], 
    weights = values
    experts, idx = topk_indices.reshape(-1).sort()
    idx = (torch.ones_like(topk_indices) * torch.arange(num_tokens, device=gates.device).unsqueeze(1)).reshape(-1)[idx]
    tmp_factor = num_experts if realloc else 1
    indices = torch.zeros(capacity*tmp_factor, num_experts, device=gates.device, dtype=topk_indices.dtype)
    for i in range(num_experts):
        num_expert_token = (experts == i).sum().item()
        num_move = min(num_expert_token, capacity * tmp_factor)
        indices[:num_move, i] = idx[:num_move] + 1
        idx = idx[num_expert_token:]
    if realloc:
        abundent = indices[capacity:, :][indices[capacity:, :] != 0]
        indices[:capacity, :][indices[:capacity, :]==0] = abundent  
    indices = indices - 1
    indices = indices[:capacity, :].transpose(0, 1).reshape(-1)
    # weights = torch.zeros(num_experts * capacity, device=gates.device, dtype=values.dtype, requires_grad=False)
    
    expert_ids = torch.ones(num_experts, capacity, device=gates.device) * torch.arange(num_experts, device=gates.device).unsqueeze(-1)
    # print("ca", capacity)
    return weights, indices, expert_ids.reshape(-1)


def compute_weight_and_sort_indices(gates, k=1, normalize=False):
    num_tokens = int(gates.shape[0])
    num_experts = int(gates.shape[1])
    gates = gates.detach()
    values, topk_indices = torch.topk(gates, k=k, dim=1) # indices： expert idx in #(num_tokens, topk)
    weights = values
    expert_ids, order_indices = topk_indices.reshape(-1).sort() # tokens_indices: for restoring original orider
    indices = (torch.ones_like(values, dtype=torch.long) * torch.arange(num_tokens, device=gates.device).unsqueeze(1)).reshape(-1)[order_indices] 
    # weights = torch.zeros(num_experts * capacity, device=gates.device, dtype=values.dtype, requires_grad=False)
    # weights[indices.reshape(-1)] = values.reshape(-1)    
    if normalize:
        weights = F.softmax(weights, dim=-1)
    return weights, indices,  expert_ids



if __name__ == '__main__':
    gates = torch.rand(16, 4)
    k = 2
    weights, indices,  expert_ids = compute_weight_and_sort_indices(gates, k)
    weights2, indices2,  expert_ids2 = compute_weight_and_sort_indices_softmove(gates, k)
    weights3, indices3,  expert_ids3 = compute_weight_and_sort_indices_fix_capacity(gates, k, capacity_factor=1)


    k = 1
    weights, indices,  expert_ids = compute_weight_and_sort_indices(gates, k)
    weights2, indices2,  expert_ids2 = compute_weight_and_sort_indices_softmove(gates, k)
    weights3, indices3,  expert_ids3 = compute_weight_and_sort_indices_fix_capacity(gates, k)
    weights4, indices4,  expert_ids4 = compute_weight_and_sort_indices_fix_capacity1(gates, k, capacity_factor=2)
    print(weights3.shape, indices3.shape, expert_ids3.shape)
    print(weights3.shape, indices4.shape, expert_ids4.shape)
    pass

# from fmoe.gates.base_gate import BaseGate

class BaseGate(nn.Module):
    def __init__(self, num_expert, world_size):
        super().__init__()
        self.world_size = world_size
        self.num_expert = num_expert
        self.tot_expert = world_size * num_expert
        self.loss = None

    def forward(self, x):
        raise NotImplementedError('Base gate cannot be directly used for fwd')

    def set_loss(self, loss):
        self.loss = loss

    def get_loss(self, clear=True):
        loss = self.loss
        if clear:
            self.loss = None
        return loss


class BalanceGate(BaseGate):
    def __init__(self, d_model, num_local_expert, world_size, top_k=2):
        super().__init__(num_local_expert, world_size)
        self.top_k = top_k
        self.gate = Gate(d_model, num_local_expert * world_size, top_k, fix_capacity=True)
    def forward(self, x):
        l_aux, combile_weights, indices, expert_ids = self.gate(x)
        # gate_top_k_idx = torch.zeros((x.shape[0], self.top_k), device=x.device)
        # _, reverse_indices = indices.sort()
        reverse_indices = self.gate.get_inversed_indices()
        gate_top_k_idx = expert_ids[reverse_indices].reshape(x.shape[0], self.top_k)
        return gate_top_k_idx.long(), combile_weights

class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.wg_init()
        
        self.timers = SynchronizedWallClockTimer()
        self.gate_time = 0.0
        self.wall_clock_breakdown = False

    def forward(self, inp, return_all_scores=False):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        if self.wall_clock_breakdown:
            self.timers('TopKGate').start()
        gate = self.gate(inp)
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=self.top_k, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_val = gate_top_k_val.view(-1, self.top_k)

        # (BxL) x 1 x top_k
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        if return_all_scores:
            return gate_top_k_idx, gate_top_k_val, gate

        if self.wall_clock_breakdown:
            self.timers('TopKGate').stop()
            self.gate_time = self.timers('TopKGate').elapsed(reset=False) * 1000

        return gate_top_k_idx, gate_top_k_val

    def wg_init(self):
        self.gate.weight.data.zero_()
        self.gate.weight.data.fill_diagonal_(1)

