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

- **1.3–1.5x faster than cuBLAS FP16** `nn.Linear` at M >= 4096 on the four FFN shapes tested
- **69% less weight memory** than FP16 (INT8 + 2:4 compression, including sparsity metadata)
- All speed comparisons run a **2:4-pruned** layer against a **dense** one. They are speed at different numerics: pruning changes the layer's output, and real models need validation or pruning-aware fine-tuning (see the accuracy caveat below).
- **No dependencies** beyond PyTorch, the CUDA toolkit, and a host C++ compiler (MSVC on Windows, auto-detected)

## Where it stands

- The fused sparse layer is **1.3–1.5x faster than cuBLAS FP16** at M >= 4096 and stores weights in **69% less memory**.
- It is **4.2–5.8x faster than an unfused eager-PyTorch INT8 path** (quantize, `torch._int_mm`, dequantize) at M >= 4096, and 1.6–2.5x faster at M = 1024.
- **cuBLASLt's dense INT8 GEMM alone (`torch._int_mm`) is faster** than the fused sparse layer: 1.4–2.6x at M >= 4096. That comparison flatters `_int_mm`, because it is given pre-quantized inputs and returns int32 with no quantize or dequantize step. Even so, sparsemma's value today is **memory savings plus a fused FP16-in/FP16-out drop-in layer**, not peak INT8 throughput.
- At M = 1024 the sparse layer beats FP16 only on the up-projection shapes (1536x6144, 1280x5120: 1.1–1.2x). On the down-projection shapes (6144x1536, 5120x1280) it is slower than FP16.
- Against **torchao's compiled dense INT8** (`int8_dynamic_activation_int8_weight` + `torch.compile`, triton-windows 3.4, torchao 0.7.0), the sparse layer was faster on 5 of 6 valid shape/batch pairs, e.g. 0.308 vs 0.428 ms at 1536x6144, M=4096; torchao was faster at 6144x1536, M=8192 (0.573 vs 0.674 ms). On random weights the pruned layer's relative error vs the dense layer is 0.45, versus 0.01 for torchao's dense INT8, so this is speed at different numerics.
- The dense INT8 `tc` backend runs at 0.67–1.12x FP16 speed across these runs, so it is not a speed win over cuBLAS FP16.

## Performance

**Hardware/software:** RTX 4090 (sm_89), PyTorch 2.8.0+cu126, CUDA 12.6, Windows 11.
**Method:** CUDA events, median of 100 iterations after 20 warmup iterations ([`tests/bench_vs_dense_int8.py`](tests/bench_vs_dense_int8.py)).
**TOPS** are *effective (dense-equivalent) TOPS*: 2·M·K·N divided by time, for every column. The sparse kernel does half the multiply-adds, so its effective TOPS can exceed what it physically executes. The columns are directly comparable.

Effective TOPS (latency in ms):

| K x N | M | FP16 cuBLAS | **SparseMMA layer** | own dense TC | `_int_mm` GEMM only¹ | INT8 e2e eager² |
|---|---|---|---|---|---|---|
| 1536 x 6144 | 4096 | 166 (0.466) | **248 (0.312)** | 177 (0.437) | 387 (0.200) | 59 (1.300) |
| 1536 x 6144 | 8192 | 166 (0.932) | **254 (0.609)** | 173 (0.895) | 400 (0.387) | 54 (2.844) |
| 6144 x 1536 | 4096 | 162 (0.478) | **225 (0.343)** | 149 (0.518) | 315 (0.246) | 42 (1.833) |
| 6144 x 1536 | 8192 | 165 (0.938) | **223 (0.692)** | 150 (1.033) | 583 (0.265) | 42 (3.646) |
| 1280 x 5120 | 4096 | 163 (0.330) | **238 (0.225)** | 183 (0.293) | 357 (0.150) | 52 (1.037) |
| 1280 x 5120 | 8192 | 166 (0.648) | **229 (0.469)** | 165 (0.649) | 373 (0.288) | 47 (2.296) |
| 5120 x 1280 | 4096 | 159 (0.338) | **208 (0.258)** | 152 (0.353) | 307 (0.175) | 36 (1.479) |
| 5120 x 1280 | 8192 | 158 (0.680) | **207 (0.519)** | 140 (0.769) | 502 (0.214) | 36 (2.999) |

- **SparseMMA layer:** `Int8Linear` in sparse mode. Activation quantize kernel, then the 2:4 sparse INT8 `mma.sp` GEMM with dequant and bias in its epilogue. FP16 in, FP16 out.
- ¹ `torch._int_mm`: cuBLASLt dense INT8 GEMM only. Pre-quantized int8 inputs, int32 output, no quantize/dequant.
- ² Eager PyTorch: quantize, then `torch._int_mm`, then dequant + bias, as separate ops.

The full run, including the M = 1024 rows, is in [`results/rtx4090_2026-10-19.txt`](results/rtx4090_2026-10-19.txt).

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

- **PTX `mma.sp.sync.aligned.m16n8k32` instructions** for direct Sparse Tensor Core access, with no cuSPARSELt dependency
- **`ldmatrix.sync.aligned.m8n8.x2` fragment loads**: hardware shared-memory-to-register transfers
- **`cp.async` pipeline**, which overlaps global-to-shared copies with Tensor Core compute across 3–4 buffer stages
- **XOR shared-memory swizzle**, a chunk-level permutation that avoids smem bank conflicts
- **Persistent kernel**: 128 CTAs stay resident and loop over output tiles in M-fastest order. (A grouped "L2 swizzle" ordering is in the code but, with the current constants, reduces to plain M-fastest order; it is not a working L2 optimization yet.)
- **Split-K decomposition** spreads the K dimension across CTAs when there are too few output tiles to fill the SMs

Launch heuristics, such as the CTA count, are tuned for the RTX 4090 (128 SMs).

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
    bench_vs_dense_int8.py # FP16 vs sparse vs dense INT8 baselines (table above)
    benchmark.py           # FP16 vs Sparse INT8 vs Dense INT8, with VRAM
    profile_sparse.py      # Nsight Compute profiling script
  results/                 # Recorded benchmark runs
  setup.py
```

## Running Tests

```bash
# Correctness
python -m pytest tests/test_sparse_tc.py -v

# Sparse vs dense micro-benchmark (from the test file)
python tests/test_sparse_tc.py --bench

# Comparison against FP16 and dense INT8 baselines (the table above)
python tests/bench_vs_dense_int8.py

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
