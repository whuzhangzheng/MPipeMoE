
import torch.nn.init as init
import torch
import torch.distributed as dist
from torch import nn

from .utils import log_dist
from .micro_pipe_moe import MicroBatchPipeMOELayer

from .gates import Gate

from .experts import ConExperts
import typing


class MicroBatchPipeMoE(torch.nn.Module):
    def __init__(self,
                 d_model,
                 d_hidden,
                 moe_group=None,
                 num_experts=1,
                 k=1,
                 capacity_factor=1.,
                 min_capacity=4,
                 name="moe-layer",
                 args=None,
                 fix_capacity=True,
                 num_split=2):
        """Initialize an MoE layer.

        Arguments:
            hidden_size (int): the hidden dimension of the model, importantly this is also the input and output dimension.

            expert (torch.nn.Module): the torch module that defines the expert (e.g., MLP, torch.linear).

            num_experts (int, optional): default=1, the total number of experts per layer.

            k (int, optional): default=1, top-k gating value, only supports k=1 or k=2.

            capacity_factor (float, optional): default=1.0, the capacity of the expert at training time.

            eval_capacity_factor (float, optional): default=1.0, the capacity of the expert at eval time.

            min_capacity (int, optional): default=4, the minimum capacity per expert regardless of the capacity_factor.

            noisy_gate_policy (str, optional): default=None, noisy gate policy, valid options are 'Jitter', 'RSample' or 'None'.
        """

        super(MicroBatchPipeMoE, self).__init__()


        num_local_experts = num_experts // dist.get_world_size(moe_group)

        log_dist(f'num_experts: {num_experts} | num_local_experts: {num_local_experts}')

        self.num_experts = num_experts
        # if args.experts_type == "seq":
        #     experts = Experts(nn.Sequential(nn.Linear(d_model, d_hidden, bias=False), nn.GELU(), nn.Linear(d_hidden, d_model, bias=False)), num_local_experts)
        # elif args.experts_type == "con":
        #     experts = ConExperts(num_local_experts, d_model=d_model, d_hidden=d_hidden)
        # experts = SeqExperts(num_local_experts, d_model=d_model, d_hidden=d_hidden)
        
        experts = ConExperts(num_local_experts, d_model=d_model, d_hidden=d_hidden)
        gate  = Gate(d_model, num_experts, k, capacity_factor, min_capacity, fix_capacity=fix_capacity)

        self.deepspeed_moe = MicroBatchPipeMOELayer(d_model, d_hidden,
                                      num_experts,
                                      experts,
                                      gate,
                                      num_local_experts,
                                      capacity_factor=capacity_factor,
                                      min_capacity=min_capacity,
                                      group=moe_group,
                                      name=name,
                                      args=args,
                                      debug=False,
                                      normalize_weights=False,
                                      k=k,
                                      fix_capacity=fix_capacity,
                                      num_split=num_split)

    def forward(self, hidden_states, used_token=None):
        """ MoE forward

        Arguments:
            hidden_states (Tensor): input to the layer
            used_token (Tensor, optional): default: None, mask only used tokens

        Returns:
            A tuple including output, gate loss, and expert count.

            * output (Tensor): output of the model

            * l_aux (Tensor): gate loss value

            * exp_counts (int): expert count
        """
        output = self.deepspeed_moe(hidden_states, used_token)
        #return output, self.deepspeed_moe.l_aux, self.deepspeed_moe.exp_counts
        return output

