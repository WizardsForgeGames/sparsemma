# sparsemma

2:4-sparse INT8 Tensor Core GEMM for PyTorch on Windows, where PyTorch's own 2:4-sparse INT8 path doesn't run.

## Why this exists

The goal was to get 2:4-sparse INT8 GEMM on Sparse Tensor Cores working on Windows at all, and then to optimize it toward cuBLAS. It is not meant as a cuBLAS replacement.

In a stock PyTorch install on Windows, the built-in 2:4 sparse path does not work for int8. On Windows 11 with PyTorch 2.8.0+cu126, `torch.sparse.to_sparse_semi_structured` on an int8 tensor fails with both of its backends:

- CUTLASS backend: `sparse_semi_structured_mad_op : CUTLASS not supported`
- cuSPARSELt backend: `cuSPARSELt not supported on your machine.`

The cuSPARSELt library itself supports Windows, but PyTorch's Windows wheels ship neither backend for this path. Dense INT8 does work on Windows through `torch._int_mm` (cuBLASLt), so the gap is specifically 2:4-sparse INT8.

sparsemma fills that gap with an `Int8Linear` drop-in replacement for `nn.Linear` that runs on NVIDIA's Sparse Tensor Cores. Weights are quantized to INT8 and 2:4-pruned, and the layer takes FP16 in and returns FP16 out. The kernels are hand-written PTX `mma.sp` code with no dependency on cuSPARSELt or CUTLASS. They JIT-compile on first use through `torch.utils.cpp_extension`, and on Windows the build script finds MSVC by itself.

## Status

This is a personal tool, built for my own projects and shared in case anyone else finds it useful. I update it when I come back to it. There are no support guarantees. The successor work, more general and actively developed, is in [Forge](https://github.com/WizardsForgeGames/Forge).

## Key Numbers

Measured on one RTX 4090 (see [Performance](#performance)):

- **2.2–2.9x faster than cuBLAS FP16** `nn.Linear` at M >= 4096 on the four FFN shapes tested (2.4–2.8x at M = 1024)
- The sparse GEMM alone is **1.02–1.77x faster than cuBLASLt dense INT8** (`torch._int_mm`) at M >= 1024, up to 583 dense-equivalent TOPS
- **69% less weight memory** than FP16 (INT8 + 2:4 compression, including sparsity metadata)
- All speed comparisons run a **2:4-pruned** layer against a **dense** one. They are speed at different numerics: pruning changes the layer's output, and real models need validation or pruning-aware fine-tuning (see the accuracy caveat below).
- **No dependencies** beyond PyTorch, the CUDA toolkit, and a host C++ compiler (MSVC on Windows, auto-detected)

## Where it stands

- The fused sparse layer (FP16 in, FP16 out) is **2.2–2.9x faster than cuBLAS FP16** at M >= 4096 and **2.4–2.8x at M = 1024**, and stores weights in **69% less memory**. At M = 64 it ranges from parity to 1.6x.
- The **sparse GEMM alone beats cuBLASLt's dense INT8 GEMM** (`torch._int_mm`, pre-quantized inputs, int32 out) on all 12 shape/batch pairs at M >= 1024: 1.3–1.8x, except 1.02–1.09x at the two K-heavy shapes at M = 8192. The full layer, including its activation quantize and dequant epilogue, still beats `_int_mm` on 10 of those 12; at 6144x1536 and 5120x1280, M = 8192, quantizing an 84–100 MB activation that no longer fits the 72 MB L2 costs more than the sparse math saves. At M = 64, `_int_mm` is faster on the two up-projection shapes.
- It is **7.7–9.2x faster than an unfused eager-PyTorch INT8 path** (quantize, `torch._int_mm`, dequantize) at M >= 4096, and 4.8–5.3x faster at M = 1024.
- Against **torchao's compiled dense INT8** (`int8_dynamic_activation_int8_weight` + `torch.compile`, triton-windows 3.4, torchao 0.7.0), the sparse layer is faster on all 6 valid shape/batch pairs, 1.4–3.1x, e.g. 0.181 vs 0.516 ms at 1536x6144, M=4096 and 0.459 vs 0.646 ms at 6144x1536, M=8192. On random weights the pruned layer's relative error vs the dense layer is 0.45, versus 0.01 for torchao's dense INT8, so this is speed at different numerics.
- The dense INT8 `tc` backend runs at 0.81–1.13x FP16 speed, so it is not a speed win over cuBLAS FP16.
- The sparse GEMM peaks at 583 dense-equivalent TOPS, 44% of the RTX 4090's 1321 TOPS sparse INT8 peak. Nsight Compute puts the main loop at ~80% of L2 throughput, so L2→SM bandwidth, not Tensor Core math, is the current limit.

## Performance

**Hardware/software:** RTX 4090 (sm_89), PyTorch 2.8.0+cu126, CUDA 12.6, Windows 11.
**Method:** [`tests/bench_kernels.py --rounds 3`](tests/bench_kernels.py). Each call is captured into a CUDA graph (enough copies that one replay is >= 0.2 ms), timed over 300 replays after 100 warmup with CUDA events, and the median of 3 rounds is reported. This is GPU time; host launch overhead is excluded.
**TOPS** are *effective (dense-equivalent) TOPS*: 2·M·K·N divided by time, for every column. The sparse kernel does half the multiply-adds, so its effective TOPS can exceed what it physically executes. The columns are directly comparable.

Effective TOPS (latency in ms):

| K x N | M | FP16 cuBLAS | **SparseMMA layer** | SparseMMA GEMM only¹ | `_int_mm` GEMM only² |
|---|---|---|---|---|---|
| 1536 x 6144 | 1024 | 166 (0.117) | **449 (0.043)** | 474 (0.041) | 293 (0.066) |
| 1536 x 6144 | 4096 | 169 (0.458) | **493 (0.157)** | 543 (0.142) | 396 (0.195) |
| 1536 x 6144 | 8192 | 166 (0.930) | **486 (0.319)** | 537 (0.288) | 407 (0.380) |
| 6144 x 1536 | 1024 | 156 (0.124) | **437 (0.044)** | 499 (0.039) | 282 (0.069) |
| 6144 x 1536 | 4096 | 158 (0.488) | **368 (0.210)** | 535 (0.144) | 304 (0.254) |
| 6144 x 1536 | 8192 | 161 (0.963) | **387 (0.399)** | 583 (0.265) | 572 (0.270) |
| 1280 x 5120 | 1024 | 152 (0.088) | **405 (0.033)** | 433 (0.031) | 279 (0.048) |
| 1280 x 5120 | 4096 | 168 (0.321) | **474 (0.113)** | 502 (0.107) | 371 (0.145) |
| 1280 x 5120 | 8192 | 165 (0.650) | **441 (0.244)** | 504 (0.213) | 381 (0.282) |
| 5120 x 1280 | 1024 | 149 (0.090) | **362 (0.037)** | 410 (0.033) | 238 (0.056) |
| 5120 x 1280 | 4096 | 160 (0.336) | **423 (0.127)** | 514 (0.104) | 306 (0.176) |
| 5120 x 1280 | 8192 | 159 (0.673) | **350 (0.307)** | 542 (0.198) | 499 (0.215) |

- **SparseMMA layer:** `Int8Linear` in sparse mode. Activation quantize kernel, then the 2:4 sparse INT8 `mma.sp` GEMM with dequant and bias in its epilogue. FP16 in, FP16 out.
- ¹ `sparse_int8_gemm`: the sparse GEMM with its dequant + bias epilogue on pre-quantized int8 activations, FP16 out (the layer minus the activation quantize).
- ² `torch._int_mm`: cuBLASLt dense INT8 GEMM only. Pre-quantized int8 inputs, int32 output, no quantize/dequant.

Full runs are in [`results/rtx4090_2026-10-19_optimized.txt`](results/rtx4090_2026-10-19_optimized.txt): this table with the quantize-kernel times and M = 64, the per-call-event comparison against the eager INT8 path and the dense `tc` backend ([`tests/bench_vs_dense_int8.py`](tests/bench_vs_dense_int8.py)), and the torchao comparison ([`tests/bench_vs_torchao.py`](tests/bench_vs_torchao.py)). The same day's numbers before the kernel optimizations are in [`results/rtx4090_2026-10-19.txt`](results/rtx4090_2026-10-19.txt). Compared with those, the layer is 1.6–3.1x faster and the GEMM 1.7–3.2x faster at M >= 1024.

**Weight memory:** the sparse layout stores 0.5 bytes/weight of INT8 values plus 0.125 bytes/weight of 2:4 metadata (plus one FP16 scale per output channel). That is 69% less than FP16's 2 bytes/weight. The theoretical 75% figure (INT8 × 50% sparsity) applies only before metadata and scales are counted.

## Requirements

- Python 3.8+
- PyTorch with CUDA support (tested: 2.8.0+cu126; `setup.py` declares `torch>=2.0`, but older versions are untested)
- CUDA Toolkit 11.8+ (needed to compile for `sm_89`; tested: 12.6)
- NVIDIA GPU with Sparse Tensor Cores (Ampere or newer). Only the RTX 4090 has been tested; see [Target Hardware](#target-hardware)
- **Windows**: Visual Studio 2019/2022 with C++ build tools. Auto-detected from the default install locations, or set `MSVC_PATH`
- **Linux**: GCC compatible with your CUDA version (untested)

## Installation

```bash
git clone https://github.com/WizardsForgeGames/sparsemma
cd sparsemma
pip install -e . --no-build-isolation   # setup.py imports torch at build time
```

You can also skip installation and run from the repo root. The Python API JIT-compiles the kernels on first use and caches them afterwards:

```python
from csrc.build import load_sparse_tc
sp = load_sparse_tc()  # compiles on first call, cached after
```

## Quick Start

### Quantize a whole model

```python
from python.quantize_utils import quantize_model_sparse

model = load_your_model()  # any PyTorch model with nn.Linear layers (FP16, on CUDA)
quantize_model_sparse(model)

# Linear layers with in_features >= 640 (and both dims >= 128) are now:
#   - INT8 quantized (per output channel)
#   - 2:4 pruned (50% structured sparsity, magnitude-based)
#   - Running on Sparse Tensor Cores
output = model(input)
```

**Accuracy caveat:** `quantize_model_sparse` applies 2:4 magnitude pruning with no fine-tuning. That throws away half of every layer's weights and can cost real accuracy, depending on the model. Validate the quantized model on your own task before relying on it. Sparse layers need `in_features % 64 == 0` and `out_features % 16 == 0`. `quantize_model_sparse` leaves other layers as FP16 `nn.Linear`.

### Quantize a single layer

```python
import torch
from python.quantize_utils import Int8Linear

layer_fp16 = torch.nn.Linear(1280, 5120, bias=True).cuda().half()
layer_sparse = Int8Linear.from_linear_sparse(layer_fp16)

x = torch.randn(1024, 1280, dtype=torch.float16, device="cuda")
y = layer_sparse(x)  # (1024, 5120) fp16 output
```

### Use the kernel directly

```python
from csrc.build import load_sparse_tc

sp = load_sparse_tc()

# Prune + compress weights (offline, once)
compressed, metadata = sp.sparse_prune_and_pack(weight_int8)

# Run sparse GEMM (per forward pass)
output = sp.sparse_int8_linear(x, compressed, metadata, weight_scale, bias)

# Or in two steps: per-token activation quantize, then the GEMM on int8 inputs
x_int8, act_scale = sp.sparse_quantize_act(x)
output = sp.sparse_int8_gemm(x_int8, act_scale, compressed, metadata, weight_scale, bias)
```

## How It Works

### 2:4 Structured Sparsity

For every group of 4 values along the K dimension, the two smallest-magnitude values are pruned to zero. The result is a **2:4 pattern**, which NVIDIA's Sparse Tensor Cores execute natively at up to 2x the dense peak math rate:

```
Original:   [10, 1, 20, 2]  →  keep [10, 20], prune [1, 2]
Compressed: [10, 20]  +  metadata: indices (0, 2)
```

The stored values shrink to half the dense size. Combined with INT8 quantization, the values take 25% of FP16's memory. Once the 2:4 metadata is added, the total is 31% of FP16, a 69% saving.

### The Kernel

The core kernel (`int8_sparse_tc.cu`) uses:

- **PTX `mma.sp.sync.aligned.m16n8k64` instructions** (one per 64-deep K step per 16x8 output tile) for direct Sparse Tensor Core access, with no cuSPARSELt dependency
- **`ldmatrix.sync.aligned.m8n8.x4` fragment loads**, each A and B fragment loaded once per K step and reused across the warp's sub-tiles
- **3-stage `cp.async` pipeline** for weights, activations and the 2:4 metadata. Per-thread copy addresses are computed once per tile, and out-of-range rows are clamped instead of branched around. For the large tiles, metadata is fetched four K steps at a time so each request fills a 32-byte L2 sector
- **XOR shared-memory swizzles**, chunk-level permutations that keep fragment, metadata and epilogue accesses free of bank conflicts
- **Register epilogue**: dequant and bias are applied in registers, and each 8x8 block is transposed with `movmatrix`. Results are then staged through shared memory so each lane writes 16-byte vectors
- **Persistent kernel**: one CTA per SM (the launch reserves just over half an SM's shared memory to guarantee it) loops over output tiles in M-fastest order, so consecutive tiles of a CTA reuse the same weight tile
- **Cost-model dispatch**: the tile shape (256x128, 128x128 or 128x64) and the split-K factor are picked by a small wave-quantization-aware cost model whose coefficients were fitted on an RTX 4090 sweep of 240 shapes; the wave count uses the device's SM count
- **Split-K decomposition** spreads the K dimension across CTAs when there are too few output tiles to fill the SMs. Partial sums are combined with FP32 `atomicAdd`, so split-K results are not bit-reproducible run to run (differences are at FP32 rounding level)
- **Single-pass activation quantize**: one read of each FP16 row (16-byte vectors held in registers), then per-token scale and 8-byte INT8 stores

### Three Compute Backends

| Backend | Method | When Used |
|---|---|---|
| `sparse` | PTX `mma.sp` (2:4 Sparse Tensor Cores) | `quantize_model_sparse()`. 69% less weight memory than FP16 |
| `tc` | `wmma` intrinsics (Dense INT8 Tensor Cores) | `quantize_model_tensorcore()`, no pruning. One INT8 copy plus per-channel scales: 50% less weight memory than FP16 |
| `dequant` | INT8→FP16 on-the-fly + cuBLAS FP16 matmul | Fallback for `tc` layers when M < 768 or when the shape isn't `K % 64 == 0` / `N % 16 == 0` |

## Project Structure

```
sparsemma/
  csrc/
    int8_sparse_tc.cu      # Sparse INT8 GEMM: hand-rolled PTX mma.sp kernel
    int8_gemm_tc.cu        # Dense INT8 GEMM: wmma-based Tensor Core kernel
    int8_kernels.cu        # Fused quantize/dequantize helper kernels
    build.py               # JIT compilation (auto-detects MSVC on Windows)
  python/
    quantize_utils.py      # Int8Linear module + quantize_model_* APIs
  tests/
    test_sparse_tc.py      # Correctness tests (pytest)
    bench_kernels.py       # Quantize / GEMM / layer vs _int_mm and FP16, CUDA-graph timing (table above)
    bench_vs_dense_int8.py # FP16 vs sparse vs dense INT8 baselines, incl. eager INT8 and the tc backend
    bench_vs_torchao.py    # Sparse layer vs torchao compiled dense INT8
    benchmark.py           # FP16 vs Sparse INT8 vs Dense INT8, with VRAM
    profile_sparse.py      # Nsight Compute profiling script
  results/                 # Recorded benchmark runs
  setup.py
```

**If a kernel fails to build**, `Int8Linear` falls back to a slower path and prints a boxed warning naming the kernel and the error (once per process). Set `INT8_KERNELS_STRICT=1` to make that an error instead. On Windows with a non-UTF-8 code page, older PyTorch versions can fail to read kernel sources; the sources are ASCII-only for that reason, and `PYTHONUTF8=1` also avoids it.

## Running Tests

```bash
# Correctness
python -m pytest tests/test_sparse_tc.py -v

# Sparse vs dense micro-benchmark (from the test file)
python tests/test_sparse_tc.py --bench

# Kernel-level comparison against FP16 and _int_mm (the table above)
python tests/bench_kernels.py --rounds 3

# Comparison against FP16, dense INT8 (eager and the tc backend) and torchao
python tests/bench_vs_dense_int8.py
python tests/bench_vs_torchao.py

# Benchmark with VRAM measurements
python tests/benchmark.py
python tests/benchmark.py --no-sparse   # dense INT8 + FP16 only

# Profile with Nsight Compute
ncu --set full -o sparse_profile python tests/profile_sparse.py
```

## Target Hardware

Sparse Tensor Cores require Ampere or newer. The build targets `sm_80`, `sm_86` and `sm_89`.

| GPU | Architecture | Compute Capability | Status |
|---|---|---|---|
| RTX 4090 | Ada Lovelace | sm_89 | **Tested** (Windows 11) |
| RTX 4080 / 4070 | Ada Lovelace | sm_89 | Untested |
| RTX 3090 / 3080 / 3070 | Ampere | sm_86 | Untested |
| A100 / A6000 | Ampere | sm_80 / sm_86 | Untested |
| RTX 2080 / V100 | Turing / Volta | sm_75 / sm_70 | Not supported (no Sparse Tensor Cores) |

Linux builds are untested.

## License

MIT License. See [LICENSE](LICENSE).
