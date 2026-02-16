# sparsemma

INT8 Sparse Tensor Core GEMM kernels for PyTorch — **built for Windows**.

Most INT8 / sparse inference libraries assume Linux. cuSPARSELt doesn't ship Windows builds. CUTLASS needs CMake gymnastics. TensorRT is a whole ecosystem. If you're on Windows with an NVIDIA GPU and just want fast INT8 inference in PyTorch, your options are... limited.

**sparsemma fixes that.** One pip install, auto-detects your MSVC compiler, JIT-compiles the kernels, and drops into any PyTorch model. No build system to fight. No Linux-only dependencies. Just `quantize_model_sparse(model)` and go.

Under the hood, it's hand-rolled PTX `mma.sp` instructions driving Sparse Tensor Cores directly — the same hardware path that cuSPARSELt uses on Linux, but implemented from scratch with zero external dependencies.

## Key Numbers

- **263 TOPS** on RTX 4090 — 1.5x faster than FP16 at large batch sizes
- **75% weight VRAM savings** — INT8 quantization + 2:4 sparsity compression
- **One line** to quantize a model — `quantize_model_sparse(model)`
- **Zero dependencies** beyond PyTorch + CUDA toolkit
- **Windows + Linux** — auto-detects MSVC, works out of the box on both

## Performance

RTX 4090, M=4096 (large batch), median of 50 iterations:

| Layer Shape | FP16 | Sparse INT8 | Speedup | VRAM Saved |
|---|---|---|---|---|
| 1536 x 6144 (DINOv2 FFN) | 174 TOPS | **263 TOPS** | **1.51x** | 69% |
| 1280 x 5120 (UNet FFN) | 157 TOPS | **222 TOPS** | **1.42x** | 69% |
| 6144 x 1536 | 168 TOPS | **232 TOPS** | **1.38x** | 69% |
| 5120 x 1280 | 156 TOPS | **203 TOPS** | **1.30x** | 69% |
| 1280 x 1280 | 139 TOPS | **164 TOPS** | **1.18x** | 69% |
| 1024 x 1024 | 158 TOPS | **179 TOPS** | **1.13x** | 69% |

Sparse INT8 is fastest on large layers with large batch sizes (M >= 1024, K >= 1024). For small layers or small batches, FP16 can be faster due to kernel launch overhead.

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- CUDA Toolkit 11.8+ (for `sm_80`+)
- NVIDIA GPU with Sparse Tensor Cores: Ampere (RTX 3090, A100) or Ada Lovelace (RTX 4090)
- **Windows**: Visual Studio 2019/2022 with C++ build tools (auto-detected)
- **Linux**: GCC compatible with your CUDA version

## Installation

```bash
pip install -e .
```

Or use the JIT compiler directly (no install needed):

```python
from csrc.build import load_sparse_tc
sp = load_sparse_tc()  # compiles on first call, cached after
```

## Quick Start

### Quantize a whole model (one line)

```python
from python.quantize_utils import quantize_model_sparse

model = load_your_model()  # any PyTorch model with nn.Linear layers
quantize_model_sparse(model)

# That's it. All eligible Linear layers are now:
#   - INT8 quantized (per-channel)
#   - 2:4 pruned (50% structured sparsity)
#   - Running on Sparse Tensor Cores
output = model(input)
```

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

For every group of 4 values along the K dimension, the two smallest-magnitude values are pruned to zero. This creates a **2:4 pattern** that NVIDIA's Sparse Tensor Cores execute natively at 2x throughput:

```
Original:   [10, 1, 20, 2]  →  keep [10, 20], prune [1, 2]
Compressed: [10, 20]  +  metadata: indices (0, 2)
```

The weight matrix shrinks to half its original size. Combined with INT8 quantization (another 2x), total weight memory is **25% of FP16**.

### The Kernel

The core kernel (`int8_sparse_tc.cu`) uses:

- **PTX `mma.sp.sync.aligned.m16n8k32` instructions** — direct Sparse Tensor Core access without cuSPARSELt overhead
- **`ldmatrix.sync.aligned.m8n8.x2` fragment loads** — hardware-optimized shared memory to register transfer
- **`cp.async` pipeline** — overlaps global-to-shared memory copies with Tensor Core compute across 3-4 buffer stages
- **XOR bank-conflict-free shared memory swizzle** — chunk-level permutation eliminates smem bank conflicts
- **Persistent kernel with L2 CTA swizzle** — tiles stay resident on SMs, weight data stays hot in L2 cache
- **Split-K decomposition** — distributes large K dimensions across CTAs for better SM utilization

### Three Compute Backends

| Backend | Method | When Used |
|---|---|---|
| `sparse` | PTX `mma.sp` (2:4 Sparse Tensor Cores) | `quantize_model_sparse()` — best throughput + VRAM |
| `tc` | `wmma` intrinsics (Dense INT8 Tensor Cores) | `quantize_model_tensorcore()` — no pruning |
| `dequant` | INT8→FP16 on-the-fly + cuBLAS FP16 matmul | Automatic fallback for small layers |

## Project Structure

```
sparsemma/
  csrc/
    int8_sparse_tc.cu    # Sparse INT8 GEMM — hand-rolled PTX mma.sp kernel
    int8_gemm_tc.cu      # Dense INT8 GEMM — wmma-based Tensor Core kernel
    int8_kernels.cu      # Fused quantize/dequantize helper kernels
    build.py             # JIT compilation (auto-detects MSVC on Windows)
  python/
    quantize_utils.py    # Int8Linear module + quantize_model_* APIs
  tests/
    test_sparse_tc.py    # Correctness tests (10 steps)
    benchmark.py         # FP16 vs Sparse INT8 vs Dense INT8 benchmarks
    profile_sparse.py    # Nsight Compute profiling script
  setup.py
```

## Running Tests

```bash
# Correctness (all 10 tests)
python tests/test_sparse_tc.py

# Benchmark
python tests/benchmark.py

# Benchmark without sparse (dense INT8 + FP16 only)
python tests/benchmark.py --no-sparse

# Profile with Nsight Compute
ncu --set full -o sparse_profile python tests/profile_sparse.py
```

## Target Hardware

Sparse Tensor Cores require Ampere or newer:

| GPU | Architecture | Compute Capability | Status |
|---|---|---|---|
| RTX 4090 / 4080 / 4070 | Ada Lovelace | sm_89 | Tested |
| RTX 3090 / 3080 / 3070 | Ampere | sm_86 | Supported |
| A100 / A6000 | Ampere | sm_80 | Supported |
| RTX 2080 / V100 | Turing / Volta | sm_75 / sm_70 | Not supported (no Sparse TC) |

## License

MIT License. See [LICENSE](LICENSE).
