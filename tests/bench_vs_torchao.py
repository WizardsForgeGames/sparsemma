"""SparseMMA fused sparse INT8 layer vs compiled dense INT8 alternatives on Windows.

Run from the sparsemma repo root. Per-iteration CUDA-event timing, median of 200 after 30 warmup,
3 interleaved rounds (median of round medians reported, plus min/max across rounds).
"""
import copy, os, statistics, sys
sys.path.insert(0, os.getcwd())
import torch, torch.nn as nn
import torchao
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
from python.quantize_utils import Int8Linear

torch.manual_seed(0)


def t_ms(fn, x, warmup=30, reps=200):
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(x); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def relerr(y, ref):
    return ((y.float() - ref.float()).norm() / ref.float().norm()).item()


print(f"torch {torch.__version__} | torchao {torchao.__version__} | {torch.cuda.get_device_name(0)}")
w = torch.randn(4096, 4096, device="cuda", dtype=torch.half)
for _ in range(300):
    w = w @ w; w = w / w.abs().max()
torch.cuda.synchronize(); del w

for K, N in [(1536, 6144), (6144, 1536), (1280, 5120), (5120, 1280)]:
    base = nn.Linear(K, N).cuda().half()
    sp = Int8Linear.from_linear_sparse(base); sp.mode = "sparse"
    ao = copy.deepcopy(base); quantize_(ao, int8_dynamic_activation_int8_weight())
    ao_c = torch.compile(ao, mode="max-autotune-no-cudagraphs")
    fp16_c = torch.compile(copy.deepcopy(base), mode="max-autotune-no-cudagraphs")
    paths = {"fp16": base, "fp16_compiled": fp16_c, "torchao_int8_compiled": ao_c, "sparsemma": sp}
    for M in (4096, 8192):
        x = torch.randn(M, K, device="cuda", dtype=torch.half)
        ref = base(x)
        errs = {k: relerr(f(x), ref) for k, f in paths.items()}
        rounds = {k: [] for k in paths}
        for _ in range(3):
            for k, f in paths.items():
                rounds[k].append(t_ms(f, x))
        cells = "  ".join(f"{k}={statistics.median(v):.3f}ms[{min(v):.3f}-{max(v):.3f}] err={errs[k]:.3f}"
                          for k, v in rounds.items())
        print(f"{K}x{N} M={M}: {cells}")
