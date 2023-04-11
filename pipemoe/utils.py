import sys
import argparse
import torch.distributed as dist

def log_dist(*args, **kwargs):
    if 'ranks' not in kwargs:
        kwargs['ranks'] = [0]
    ranks = kwargs['ranks']
    if dist.get_rank() in ranks:
        print(f"[{dist.get_rank()}]", *args)


def construct_min_args():
    parser = argparse.ArgumentParser()
    parser = add_moe_arguments(parser)
    args = parser.parse_args()
    return args

def add_moe_arguments(parser):
    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        default=1,
                        type=int,
                        help='(moe) number of total experts')

    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument('--capacity-factor',
                        default=1.0,
                        type=float,
                        help='the capacity of the expert at training time.')
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    parser.add_argument(
        '--aux-loss',
        default=0,
        type=float,
        help=
        '(moe) auxiliary loss to for load balance. Valid values are None, enable'
    )
    parser.add_argument(
        '--save-route',
        default=False,
        type=bool,
        help=
        '(moe) if save route data'
    )

    parser.add_argument(
        '--moe-layer-delta',
        default=2,
        type=int,
        help=
        '(moe) one of every `moe-layer-delta` layer replaced with moelayer'
    )
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )

    parser.add_argument(
        '--pipe',
        default=False,
        action='store_true',
        help=
        '(moe) whether use PipeMoeLayer'
    )
    parser.add_argument(
        '--experts_type',
        default='con',
        type=str,
        help=
        '(moe) use fused experts(con) or sequential experts(seq)'
    )
    parser.add_argument(
        '--loss_version',
        default=1,  # 1
        type=int,
        help=
        '(moe) how to compute auxiliary loss'
    )
    parser.add_argument(
        '--weight_version', 
        default=1,      # 1,2,3
        type=int,
        help=
        '(moe) how to compute combined weights'
    )

    parser.add_argument(
        '--pipe_version',
        default=2, # 0 2
        type=int,
        help=
        '(moe) how to compute combined weights'
    )

    return parser