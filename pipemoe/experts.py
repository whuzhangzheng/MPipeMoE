
import torch
import copy
import math
from torch import nn

class SeqExperts(torch.nn.Module):
    def __init__(self, num_local_experts, d_model=1024, d_hidden=4096):
        super(SeqExperts, self).__init__()
        expert = nn.Sequential(nn.Linear(d_model, d_hidden, bias=False), nn.ReLU(inplace=True), nn.Linear(d_hidden, d_model, bias=False))
        self.experts = torch.nn.ModuleList(
            [copy.deepcopy(expert) for i in range(num_local_experts)])

        self.num_local_experts = num_local_experts

        # TODO: revisit allreduce for moe.gate...
        for expert in self.experts:
            for name, param in expert.named_parameters():
                param.allreduce = False


    def forward(self, inputs, splits, return_middle=False):
        chunks = inputs.split(splits.tolist(), dim=0)
        expert_outputs = []
        for i, chunk in enumerate(chunks):
        # for chunk, expert in zip(chunks, self.deepspeed_experts):
            out = self.experts[i%self.num_local_experts](chunk)
            expert_outputs += [out]

        expert_output = torch.cat(expert_outputs, dim=0)   # allocate new memory
        return expert_output

class ConExperts(torch.nn.Module):
    def __init__(self, expert_num, world_size=1, d_model=1024, d_hidden=4096):
        super(ConExperts, self).__init__()
        self.d_hidden = d_hidden
        self.d_model = d_model
        self.world_size = world_size
        self.expert_num = expert_num
        self.weight1 = nn.Parameter(torch.Tensor(expert_num, d_hidden, d_model))
        self.act = nn.ReLU(inplace=True)
        self.weight2 = nn.Parameter(torch.Tensor(expert_num, d_model, d_hidden))

        # self.weight1.data = torch.ones((expert_num, d_model, d_hidden))*torch.arange(1, expert_num+1).unsqueeze_(1).unsqueeze_(1)
        # self.weight2.data = torch.ones((expert_num, d_hidden, d_model))*torch.arange(1, expert_num+1).unsqueeze_(1).unsqueeze_(1)
        self.reset_parameters()
        for name, param in self.named_parameters():
            param.allreduce = False

    def forward(self, x):
        shape = x.shape
        x = x.reshape(self.world_size, self.expert_num, -1, self.d_model)
        ffn1 = torch.einsum("ge...m,ehm -> ge...h", x, self.weight1)
        ffn1 = self.act(ffn1)
        y = torch.einsum("ge...h,emh -> ge...m", ffn1, self.weight2)
        y = y.reshape(x.shape)
        return y
        # return y, ffn1, ffn1
    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88
        torch.nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
