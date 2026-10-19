"""
Kernel-level benchmark: sparse GEMM alone vs the full layer vs cuBLAS baselines.

Columns (all CUDA-event medians of GPU time, see "Timing" below):
  quant      sparse_quantize_act: per-token activation quantize kernel alone (fp16 -> int8 + scales)
  sp GEMM    sparse_int8_gemm on pre-quantized int8 inputs: 2:4 sparse mma.sp GEMM + dequant + bias
             epilogue, fp16 out. Apples to apples with _int_mm (which outputs int32, no epilogue).
  sp layer   Int8Linear sparse forward: quant + GEMM (what a model sees)
  _int_mm    torch._int_mm (cuBLASLt dense INT8 GEMM), pre-quantized inputs, int32 output
  fp16       nn.Linear FP16 (cuBLAS)

Timing: each call is captured n times into a CUDA graph (n chosen so one replay is >= 0.2 ms of
GPU work); 100 warmup replays, then 300 replays each bracketed by CUDA events and enqueued
back-to-back; per-call time = median(replay) / n. Host launch overhead is excluded, so small-M
numbers are GPU time, not Python/dispatch time. Inputs stay L2-resident across replays.

Usage (from the repo root):
    python tests/bench_kernels.py              # headline shapes
    python tests/bench_kernels.py --rounds 3   # repeat and report median [min-max] across rounds
"""
import argparse
import os
import statistics
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.quantize_utils import Int8Linear, _load_sp  # noqa: E402


def time_graph(fn, warmup=100, reps=300, min_sample_ms=0.2):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        fn()
    e.record()
    torch.cuda.synchronize()
    est = s.elapsed_time(e) / 20
    n_inner = max(1, int(min_sample_ms / max(est, 1e-3) + 0.999))
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(st):
        fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=st):
            for _ in range(n_inner):
                fn()
    torch.cuda.current_stream().wait_stream(st)
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(reps)]
    for s, e in ev:
        s.record(); g.replay(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in ev) / n_inner


def tops(M, K, N, ms):
    return 2 * M * K * N / (ms * 1e-3) / 1e12


SHAPES = [(1536, 6144), (6144, 1536), (1280, 5120), (5120, 1280)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--ms", type=int, nargs="*", default=[64, 1024, 4096, 8192])
    args = ap.parse_args()

    sp_mod = _load_sp()
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
    w = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    for _ in range(300):
        w = w @ w; w = w / w.abs().max()
    torch.cuda.synchronize(); del w

    cols = ["quant", "sp GEMM", "sp layer", "_int_mm", "fp16"]
    hdr = f"{'K x N':>11} {'M':>5} | " + " | ".join(f"{c:>16}" for c in cols)
    print(hdr); print("-" * len(hdr))
    torch.manual_seed(0)
    for K, N in SHAPES:
        lin = nn.Linear(K, N).cuda().half()
        sp = Int8Linear.from_linear_sparse(lin); sp.mode = "sparse"
        wq = (lin.weight.float() / (lin.weight.float().abs().amax(1, keepdim=True) / 127)
              ).round().clamp(-128, 127).to(torch.int8)
        wq_t = wq.t()
        for M in args.ms:
            x = torch.randn(M, K, dtype=torch.float16, device="cuda")
            xq, xs = sp_mod.sparse_quantize_act(x)
            fns = {
                "quant": lambda: sp_mod.sparse_quantize_act(x),
                "sp GEMM": lambda: sp_mod.sparse_int8_gemm(
                    xq, xs, sp.weight_sparse_comp, sp.weight_sparse_meta, sp.weight_scale, sp.bias),
                "sp layer": lambda: sp(x),
                "_int_mm": (lambda: torch._int_mm(xq, wq_t)) if M > 16 else None,
                "fp16": lambda: lin(x),
            }
            res = {c: [] for c in cols}
            for _ in range(args.rounds):
                for c in cols:
                    if fns[c] is not None:
                        res[c].append(time_graph(fns[c]))
            cells = []
            for c in cols:
                v = res[c]
                if not v:
                    cells.append(f"{'n/a':>16}")
                    continue
                med = statistics.median(v)
                t = "" if c == "quant" else f" {tops(M, K, N, med):4.0f}T"
                cells.append(f"{med:7.4f}ms{t:>6}" if c != "quant" else f"{med:7.4f}ms      ")
            print(f"{K:>5}x{N:<5} {M:>5} | " + " | ".join(f"{c:>16}" for c in cells))
            if args.rounds > 1:
                spread = "  ".join(f"{c}:[{min(res[c]):.4f}-{max(res[c]):.4f}]" for c in cols if res[c])
                print(f"{'':>19}   {spread}")
        del lin, sp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
