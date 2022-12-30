import sys
import argparse
import torch.distributed as dist

def log_dist(*args, **kwargs):
    if 'ranks' not in kwargs:
        kwargs['ranks'] = [0]
    ranks = kwargs['ranks']
    if dist.get_rank() in ranks:
        print(f"[{dist.get_rank}]", *args)
