import torch
import itertools
import torch.distributed as dist

def formular(N, K, n , M, B, S, double_buf=True):
    d = 2 if double_buf else 1
    return 12 * (1-d /N) / (14 + (K + 2)*n * M * 4/ (B *S))

def formular2(N, K, n , M, B, S, double_buf=True):
    d = 2 if double_buf else 1
    return 6 * (1-d /N) / (8 + (K + 2)*n * M * 4/ (B *S))

def generator(NS, KS, ns, MS, BS, SS):
    for it in itertools.product(NS, KS, ns, MS, BS, SS):
        yield it

def estimate_reserved_memory(strategy, N, K, nle , M, B, S=128):
    model_State = nle * (2 + K) * M * 4 * M * 2 # 2 个FFN
    # print("model_State", model_State * 4// 1024// 1024)
    # print("x:", M*B*S*4//1024//1024)
    # print("expert:", M*M*4*2*4//1024//1024)
    fix_acts =  B * S * M * 2 * 2 # input + output; act+grad
    # fix_acts += B * S * M   # indexed input
    if strategy == "pipe" :
        acts = (4 + 2) * B * S * M 
        bw_buf = (4 + 2) * B * S * M   
    elif strategy.startswith("sharded"):
        impl = int(strategy[-1])
        if impl % 2 == 0:
            acts = (6/N*2) * B * S * M 
            bw_buf = (6/N*2) * B * S * M   
        elif impl % 2 == 1:
            acts = (6/N) * B * S * M 
            bw_buf = (6/N) * B * S * M   
    else:
    # if strategy == "original" or strategy == "seq":
        acts = (4 + 2) * B * S * M 
        bw_buf = (4 + 1) * B * S * M   
    
    return (model_State + acts + bw_buf + fix_acts) * 4 // (1024 * 1024)

class MemoryTracer:
    allocated_before = 0
    reserved_before = 0
    @staticmethod
    def unit(x):
        if abs(x) >= 1024 * 1024 *1024:
            return x//(1024*1024*1024), "GB", x%(1024*1024*1024)
        elif abs(x) >= 1024 * 1024 :
            return x//(1024*1024), "MB", x%(1024*1024)
        elif abs(x) >= 1024:
            return x//1024, "KB", x%1024
        else:
            return x, "B", 0
    @staticmethod
    def reset():
        MemoryTracer.allocated_now = torch.cuda.memory_allocated()
        MemoryTracer.reserved_now = torch.cuda.memory_reserved()
    @staticmethod
    def parse(x):
        des = ""
        while x != 0:
            v, u, x = MemoryTracer.unit(x)
            des = f"{des}{v}{u},"
        if des == "":
            des = "0B"
        return des

    @staticmethod
    def allocated(tag="", total=False, log=True):
        allocated_now = torch.cuda.memory_allocated()
        if log:
            if total:
                MemoryTracer.log(f"({tag}) allocated: {MemoryTracer.parse(allocated_now)}")
            # v, u = MemoryTracer.unit(allocated_now-MemoryTracer.allocated_before)
            MemoryTracer.log(f"({tag}) step allocated: {MemoryTracer.parse(allocated_now-MemoryTracer.allocated_before)}")
        MemoryTracer.allocated_before = allocated_now
        return allocated_now // (1024*1024)
    @staticmethod
    def reserved(tag="", total=False):
        reserved_now = torch.cuda.memory_reserved()
        if total:
            MemoryTracer.log(f"({tag}) reserved: {MemoryTracer.parse(reserved_now)} MB")
        # v, u = MemoryTracer.unit(reserved_now-MemoryTracer.reserved_before)
        MemoryTracer.log(f"({tag}) step reserved: {MemoryTracer.parse(reserved_now-MemoryTracer.reserved_before)}")
        MemoryTracer.reserved_before = reserved_now
        return reserved_now // (1024*1024)
    @staticmethod
    def peak(tag="", reset=False):
        peak_mem = torch.cuda.max_memory_allocated()
        MemoryTracer.log(f"({tag}) max memory allocated: {MemoryTracer.parse(peak_mem)}")
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return peak_mem  // (1024*1024)

    @staticmethod
    def log(*args):
        if not dist.is_initialized() or dist.get_rank() * 0 == 0:
            print(*args)


if __name__ == '__main__':
    heads = ['N', 'K', 'M', 'B\\times S', "rate1", "rate2"]
    print(" & ".join(heads))
    NS = [4, 16] #[2, 4, 8, 16]
    KS = [1, 3] #[0, 1, 3]
    ns = [1]
    MS = [512, 1024]
    BS = [16, 64] #[16, 32, 64, 128, 256]
    SS = [128]
    for N, K, n , H, B, S in itertools.product(NS, KS, ns, MS, BS, SS):
        rate1 = formular(N, K, n , H, B, S, False)
        rate2 = formular(N, K, n , H, B, S, True)
        rate3 = formular2(N, K, n , H, B, S, True)
        line = [N, K , H, f"{B}\\times {S}", f"{rate1:.2f}", f"{rate2:.2f}", f"{rate3:.2f} \\\\"]
        line = [str(x) for x in line]
        print(' & '.join(line))