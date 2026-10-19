"""
Benchmark: SparseMMA sparse INT8 layer vs FP16 cuBLAS vs dense INT8 baselines.

Columns:
  FP16 cuBLAS      nn.Linear in FP16 (cuBLAS).
  SparseMMA layer  Int8Linear, mode="sparse": activation quantize kernel + 2:4 sparse
                   INT8 mma.sp GEMM with dequant + bias epilogue. FP16 in, FP16 out.
  own dense TC     Int8Linear, mode="tc": this repo's dense INT8 wmma kernel.
  _int_mm GEMM     torch._int_mm (cuBLASLt dense INT8 GEMM) on pre-quantized int8
                   inputs, int32 output. GEMM only: no quantize, no dequant.
  INT8 e2e torch   Unfused eager PyTorch: quantize -> torch._int_mm -> dequant + bias.

Timing: CUDA events, median of 100 iterations after 20 warmup iterations.
TOPS are dense-equivalent (2*M*K*N) in every column, so columns are directly comparable
even though the sparse kernel only executes half the multiply-adds.

Usage (from the repo root):
    python tests/bench_vs_dense_int8.py

Results from an RTX 4090 run are in results/.
"""
import os
import statistics
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.quantize_utils import Int8Linear, sp_available, tc_available


def t_ms(fn, warmup=20, reps=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def tops(M, K, N, ms):
    return 2 * M * K * N / (ms * 1e-3) / 1e12


def main():
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | {torch.cuda.get_device_name(0)}")
    print(f"sparse available: {sp_available()}  dense-TC available: {tc_available()}")
    # Spin the GPU up to steady-state clocks before timing anything.
    w = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    for _ in range(300):
        w = w @ w; w = w / w.abs().max()
    torch.cuda.synchronize(); del w

    shapes = [(1536, 6144), (6144, 1536), (1280, 5120), (5120, 1280)]
    Ms = [1024, 4096, 8192]
    hdr = (f"{'K x N':>11} {'M':>5} | {'FP16 cuBLAS':>11} | {'SparseMMA layer':>15} | {'own dense TC':>12} | "
           f"{'_int_mm GEMM':>12} | {'INT8 e2e torch':>14}")
    print(hdr); print("-" * len(hdr))
    for K, N in shapes:
        lin = nn.Linear(K, N).cuda().half()
        sp = Int8Linear.from_linear_sparse(lin); sp.mode = "sparse"
        tc = Int8Linear.from_linear(lin); tc.mode = "tc"
        wq = (lin.weight.float() / (lin.weight.float().abs().amax(1, keepdim=True) / 127)).round().clamp(-128, 127).to(torch.int8)
        wq_t = wq.t()  # K x N, column-major view
        wscale = (lin.weight.float().abs().amax(1) / 127).half()
        for M in Ms:
            x = torch.randn(M, K, dtype=torch.float16, device="cuda")
            xq = (x.float() / (x.float().abs().amax(1, keepdim=True) / 127)).round().clamp(-128, 127).to(torch.int8)

            def e2e():
                xs = x.float().abs().amax(1, keepdim=True) / 127
                q = (x.float() / xs).round().clamp(-128, 127).to(torch.int8)
                acc = torch._int_mm(q, wq_t)
                return (acc.float() * xs * wscale.float()).half() + lin.bias

            r = {
                "fp16": t_ms(lambda: lin(x)),
                "sp": t_ms(lambda: sp(x)),
                "tc": t_ms(lambda: tc(x)),
                "intmm": t_ms(lambda: torch._int_mm(xq, wq_t)),
                "e2e": t_ms(e2e),
            }
            cells = [f"{r[k]:6.3f}ms {tops(M, K, N, r[k]):4.0f}T" for k in ("fp16", "sp", "tc", "intmm", "e2e")]
            print(f"{K:>5}x{N:<5} {M:>5} | {cells[0]:>11} | {cells[1]:>15} | {cells[2]:>12} | {cells[3]:>12} | {cells[4]:>14}")
        del lin, sp, tc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
