import torch.distributed as dist
import torch
import pipemoe_cuda
import os
from pipemoe.utils import log_dist

initialized = False

def generate_rank_topo(n_nodes, n_gran=None):
    assert dist.is_initialized()
    ws = dist.get_world_size()
    n_gran = int(os.environ.get("PMOE_FUSE_GRAN", "1"))
    topo_type = os.environ.get("topo_type", "exchange")
    print("zzzz topo_type:", topo_type, "\t n_gran: ", n_gran)
    topos = []
    if topo_type == "exchange":
        topos = generate_exchange_topo(n_nodes)
    elif topo_type == "star":
        topos = generate_star_topo(n_nodes)
    elif topo_type == "ring":
        topos = generate_ring_topo(n_nodes, n_gran)
    else:
        topos = generate_star_topo(n_nodes)
        pass
        # raise NotImplementedError()

    for i in range(ws):
        if not dist.is_initialized() or dist.get_rank() == 0:
            if len(topos) == ws:
                log_dist(i, "send, recv:", topos[i])
            else:
                log_dist(i, "send:", topos[i], ";\trecv:", topos[i+ws])
    return topos

def generate_ring_topo(n_nodes, n_gran):
    n_groups = dist.get_world_size() // n_gran
    send_topos = []
    recv_topos = []
    for rank in range(n_nodes):
        group_rank = rank // n_gran
        rank_in_group = rank % n_gran
        send_topo = []
        recv_topo = []
        for i in range(n_nodes):
            g = i // n_gran
            group_send = (group_rank + g) % n_groups * n_gran
            group_recv = (group_rank - g + n_groups) % n_groups * n_gran
            j = i % n_gran
            rank_send = group_send + (rank_in_group + j) % n_gran
            rank_recv = group_recv + (rank_in_group -j + n_gran) % n_gran
            send_topo.append(rank_send)
            recv_topo.append(rank_recv)
        send_topos.append(send_topo)
        recv_topos.append(recv_topo)
    # check
    for i, topo in enumerate(send_topos):
        for t, j in enumerate(topo):
            assert recv_topos[j][t] == i

    send_topos.extend(recv_topos)
    return send_topos

def generate_star_topo(n_nodes):
    topos = []
    for i in range(n_nodes):
        topos.append(list(range(n_nodes)))
    return topos

def generate_exchange_topo(n_nodes):
    topos = []
    topos.append(torch.arange(n_nodes))
    group_size = 2
    while group_size <= n_nodes:
        for i, rank in enumerate(range(group_size//2, group_size)):
            topo = topos[i].clone().reshape(-1, 2, group_size//2)[:, [1, 0], :].reshape(-1)
            topos.append(topo)
        group_size *= 2

    # check
    for i, topo in enumerate(topos):
        topos[i] = topo.tolist()
        for t, j in enumerate(topo):
            assert topos[j][t] == i
    return topos

def initial(inputs, d_hidden, group=None, two_comm=True):
    global initialized
    if initialized:
        return
    if two_comm:
        group =  dist.distributed_c10d._get_default_group()
        group2 = dist.new_group(list(range(dist.get_world_size())), backend='nccl')
        pipemoe_cuda.ensure_nccl2(group, group2, inputs, inputs.size(0), inputs.size(1), d_hidden, generate_rank_topo(dist.get_world_size()))
    else:
        pipemoe_cuda.ensure_nccl(dist.distributed_c10d._get_default_group(), inputs, inputs.size(0), inputs.size(1), d_hidden)
    initialized = True