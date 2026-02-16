"""
Minimal script for Nsight Compute profiling of the sparse INT8 kernel.
Runs a single shape so ncu can capture kernel metrics.

Usage:
    ncu --set full -o sparse_profile python tests/profile_sparse.py
    ncu --metrics ... python tests/profile_sparse.py
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.quantize_utils import Int8Linear

# Build the sparse kernel
layer_fp16 = torch.nn.Linear(1536, 6144, bias=True).cuda().half()
layer_sparse = Int8Linear.from_linear_sparse(layer_fp16)
del layer_fp16

# Shape: M=4096, K=1536, N=6144 (our peak-performance shape)
x = torch.randn(4096, 1536, dtype=torch.float16, device="cuda")

# Warmup (outside profiled region)
for _ in range(5):
    _ = layer_sparse(x)
torch.cuda.synchronize()

# Profiled iteration
torch.cuda.nvtx.range_push("sparse_gemm")
y = layer_sparse(x)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print(f"Output shape: {y.shape}, dtype: {y.dtype}")
