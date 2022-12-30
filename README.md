## MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism
===
This repository is the open-source codebase of the IPDPS'23 paper, `MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism`
 It is a prototype to verify the ideas in the paper. 

### install
```
cd MPipeMoE
python setup.py install
```

### usage

```py
from pipemoe import layer

MOE_FFN: nn.Module = layer.PipeMoE(
        d_model=d_model,
        d_hidden=d_hidden,
        moe_group=dist.distributed_c10d._get_default_group(),
        num_experts=num_experts,
        k=top_k,
        name=f"layer1",
        args=args)
```

