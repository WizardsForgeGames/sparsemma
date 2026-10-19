"""
Int8 quantization utilities — standalone edition.

Three compute backends (auto-selected per layer):
  1. "sparse" — 2:4 Sparse INT8 Tensor Core GEMM via PTX mma.sp.
     Prunes 50% of weights (2:4 pattern), compresses, and uses Sparse
     Tensor Cores for 2x throughput. Requires K % 64 == 0, N % 16 == 0.
     ~69% weight VRAM savings vs FP16: 0.5 B/weight of int8 values + 0.125 B/weight
     of 2:4 metadata (theoretical 75% before metadata and scales).

  2. "tc" — Custom INT8 Tensor Core GEMM via wmma intrinsics. Fuses
     GEMM + dequant + bias in one kernel. Requires K % 64 == 0, N % 16 == 0.

  3. "dequant" — Dequantize int8 weights to fp16 on-the-fly, then fp16
     matmul. Fallback for "tc" layers (M < 768 or unaligned shapes).

  Note: dense layers ("tc"/"dequant") keep both an (N, K) and a (K, N) int8 copy
  of the weight, so their weight memory is about the same as FP16.

Usage:
    from python.quantize_utils import quantize_model_sparse, quantize_model_tensorcore

    # Best: sparse INT8 (2:4 pruning + Sparse Tensor Cores)
    quantize_model_sparse(model)

    # Dense INT8: custom TC GEMM (auto-fallback to dequant)
    quantize_model_tensorcore(model)
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
#  Custom TC GEMM module (lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════════

_TC_MODULE = None


def _load_tc():
    """Lazy-load the custom Tensor Core INT8 GEMM extension."""
    global _TC_MODULE
    if _TC_MODULE is not None:
        return _TC_MODULE
    try:
        from csrc.build import load_tc_gemm
        _TC_MODULE = load_tc_gemm()
        return _TC_MODULE
    except Exception as e:
        print(f"  Warning: TC GEMM compilation failed ({e}), using dequant fallback")
        return None


def tc_available() -> bool:
    """Check if the custom TC GEMM kernel is available."""
    return _load_tc() is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  Sparse TC GEMM module (lazy-loaded)
# ═══════════════════════════════════════════════════════════════════════════════

_SP_MODULE = None


def _load_sp():
    """Lazy-load the sparse INT8 Tensor Core GEMM extension (PTX mma.sp)."""
    global _SP_MODULE
    if _SP_MODULE is not None:
        return _SP_MODULE
    try:
        from csrc.build import load_sparse_tc
        _SP_MODULE = load_sparse_tc()
        return _SP_MODULE
    except Exception as e:
        print(f"  Warning: Sparse TC compilation failed ({e}), using dense fallback")
        return None


def sp_available() -> bool:
    """Check if the sparse INT8 TC GEMM kernel is available."""
    return _load_sp() is not None


# ═══════════════════════════════════════════════════════════════════════════════
#  Int8Linear — stores weights as int8, computes via Tensor Cores or fallback
# ═══════════════════════════════════════════════════════════════════════════════

class Int8Linear(nn.Module):
    """Drop-in nn.Linear replacement with int8 weight storage.

    Three compute paths (auto-selected):
      1. sparse: 2:4 Sparse INT8 TC GEMM via PTX mma.sp (fused quant+GEMM+dequant+bias).
         Requires K % 64 == 0, N % 16 == 0. Uses from_linear_sparse().
      2. TC: Dense INT8 Tensor Core GEMM (fused quant+GEMM+dequant+bias).
         Requires K % 64 == 0, N % 16 == 0.
      3. dequant: Dequantize int8->fp16 on-the-fly + fp16 matmul. Fallback.

    Weight quantization (offline, per output-channel):
        scale_j = max(|W[j,:]|) / 127
        W_int8[j,:] = round(W[j,:] / scale_j).clamp(-128, 127)
    """

    __constants__ = ["in_features", "out_features", "_tc_ok", "_sparse_ok"]

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self._tc_ok = (in_features % 64 == 0) and (out_features % 16 == 0)
        self._sparse_ok = (in_features % 64 == 0) and (out_features % 16 == 0)

        self.register_buffer(
            "weight_int8_nk",
            torch.zeros(out_features, in_features, dtype=torch.int8),
        )
        self.register_buffer(
            "weight_int8_t",
            torch.zeros(in_features, out_features, dtype=torch.int8),
        )
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float16),
        )
        self.register_buffer("weight_sparse_comp", None)
        self.register_buffer("weight_sparse_meta", None)

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float16),
            )
        else:
            self.bias = None

    @staticmethod
    def from_linear(linear: nn.Linear) -> "Int8Linear":
        """Convert an nn.Linear to Int8Linear with per-channel weight quantization."""
        has_bias = linear.bias is not None
        layer = Int8Linear(linear.in_features, linear.out_features, bias=has_bias)

        weight_f32 = linear.weight.data.float()
        scale = weight_f32.abs().amax(dim=1) / 127.0
        scale = scale.clamp(min=1e-8)

        weight_int8 = (weight_f32 / scale[:, None]).round().clamp(-128, 127).to(torch.int8)

        layer.weight_int8_nk = weight_int8.contiguous()
        layer.weight_int8_t = weight_int8.t().contiguous()
        layer.weight_scale = scale.to(torch.float16)

        if has_bias:
            layer.bias = linear.bias.data.to(torch.float16)

        return layer.to(linear.weight.device)

    @staticmethod
    def from_linear_sparse(linear: nn.Linear) -> "Int8Linear":
        """Convert nn.Linear to sparse Int8Linear (2:4 pruning + int8 quantization)."""
        sp_mod = _load_sp()
        if sp_mod is None:
            raise RuntimeError("Sparse TC module not available.")

        has_bias = linear.bias is not None
        layer = Int8Linear(linear.in_features, linear.out_features, bias=has_bias)

        weight_f32 = linear.weight.data.float()
        scale = weight_f32.abs().amax(dim=1).clamp(min=1e-8) / 127.0
        weight_int8 = (weight_f32 / scale[:, None]).round().clamp(-128, 127).to(torch.int8)

        device = linear.weight.device
        results = sp_mod.sparse_prune_and_pack(weight_int8.to(device))
        layer.weight_sparse_comp = results[0]
        layer.weight_sparse_meta = results[1]
        layer.weight_scale = scale.to(torch.float16)

        layer.register_buffer("weight_int8_nk", torch.empty(0, dtype=torch.int8))
        layer.register_buffer("weight_int8_t", torch.empty(0, dtype=torch.int8))

        if has_bias:
            layer.bias = linear.bias.data.to(torch.float16)

        layer.mode = "sparse"
        return layer.to(device)

    _tc_ref = None
    _sp_ref = None
    _empty_bias = None

    def _forward_sparse(self, x: torch.Tensor) -> torch.Tensor:
        if self._sp_ref is None:
            self._sp_ref = _load_sp()
        if self.bias is not None:
            bias = self.bias
        else:
            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, dtype=torch.float16, device=x.device)
            bias = self._empty_bias
        return self._sp_ref.sparse_int8_linear(
            x, self.weight_sparse_comp, self.weight_sparse_meta,
            self.weight_scale, bias
        )

    def _forward_tc(self, x: torch.Tensor) -> torch.Tensor:
        if self._tc_ref is None:
            self._tc_ref = _load_tc()
        if self.bias is not None:
            bias = self.bias
        else:
            if self._empty_bias is None or self._empty_bias.device != x.device:
                self._empty_bias = torch.empty(0, dtype=torch.float16, device=x.device)
            bias = self._empty_bias
        return self._tc_ref.int8_linear_tc(x, self.weight_int8_nk, self.weight_scale, bias)

    def _forward_dequant(self, x: torch.Tensor) -> torch.Tensor:
        w_fp16 = self.weight_int8_t.to(torch.float16) * self.weight_scale[None, :]
        return torch.mm(x.to(torch.float16), w_fp16)

    mode: str = "tc"
    _TC_MIN_M: int = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        M = x.shape[0]

        if self.mode == "sparse" and self._sparse_ok and sp_available():
            y = self._forward_sparse(x)
        elif self.mode == "tc" and self._tc_ok and tc_available() and M >= self._TC_MIN_M:
            y = self._forward_tc(x)
        else:
            y = self._forward_dequant(x)
            if self.bias is not None:
                y = y + self.bias
            if len(orig_shape) > 2:
                y = y.reshape(*orig_shape[:-1], self.out_features)
            return y

        if len(orig_shape) > 2:
            y = y.reshape(*orig_shape[:-1], self.out_features)
        return y

    def extra_repr(self) -> str:
        if self.mode == "sparse" and self._sparse_ok and sp_available():
            mode_str = "sparse_tc (2:4 Sparse INT8 Tensor Core)"
        elif self.mode == "tc" and self._tc_ok and tc_available():
            mode_str = "tc_gemm (custom INT8 Tensor Core)"
        else:
            mode_str = "dequant (int8->fp16 + fp16 matmul)"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={mode_str}"
        )


def quantize_model_tensorcore(model, min_features=128, verbose=True):
    """Replace nn.Linear layers with Int8Linear (Tensor Core int8 or dequant fallback)."""
    stats = {"total_params": 0, "quantized_params": 0, "skipped_params": 0,
             "num_replaced": 0, "tc_eligible": 0}
    _replace_linear_tensorcore(model, min_features, stats)

    if verbose:
        tc_ok = tc_available()
        mode = "TC GEMM (custom INT8 Tensor Core)" if tc_ok else "dequant fallback"
        print(f"  Quantized {stats['num_replaced']} Linear layers -> Int8Linear [{mode}]")
        print(f"  TC-eligible: {stats['tc_eligible']} layers")
        print(f"  Linear params: {stats['total_params'] / 1e6:.1f}M total, "
              f"{stats['quantized_params'] / 1e6:.1f}M quantized, "
              f"{stats['skipped_params'] / 1e6:.1f}M skipped")
        print("  Weight VRAM: ~same as FP16 (int8 weights stored in both (N,K) and (K,N) layouts)")

    return stats


def _replace_linear_tensorcore(module, min_features, stats):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            num_params = child.in_features * child.out_features
            stats["total_params"] += num_params

            if child.in_features >= min_features and child.out_features >= min_features:
                int8_layer = Int8Linear.from_linear(child)
                setattr(module, name, int8_layer)
                stats["quantized_params"] += num_params
                stats["num_replaced"] += 1
                if int8_layer._tc_ok:
                    stats["tc_eligible"] += 1
            else:
                stats["skipped_params"] += num_params
        else:
            _replace_linear_tensorcore(child, min_features, stats)


def quantize_model_sparse(model, min_features=128, sparse_min_k=640, verbose=True):
    """Replace nn.Linear layers with sparse Int8Linear (2:4 pruning + Sparse TC GEMM)."""
    if not sp_available():
        raise RuntimeError(
            "Sparse TC module not available. Compile with: python csrc/build.py sparse_tc"
        )

    stats = {"total_params": 0, "quantized_params": 0, "skipped_params": 0,
             "num_replaced": 0, "sparse_eligible": 0, "kept_fp16": 0}
    _replace_linear_sparse(model, min_features, sparse_min_k, stats)

    if verbose:
        fp16_mb = stats["quantized_params"] * 2 / 1e6
        # 0.5 B/weight int8 values (K/2 per row) + 0.125 B/weight metadata (K/32 int32 per row)
        sparse_mb = stats["quantized_params"] * 0.625 / 1e6
        saved_mb = fp16_mb - sparse_mb
        print(f"  Quantized {stats['num_replaced']} Linear layers -> sparse Int8Linear")
        print(f"  Sparse-eligible: {stats['sparse_eligible']} layers")
        print(f"  Kept as FP16: {stats['kept_fp16']} layers "
              f"(K < {sparse_min_k}, K % 64 != 0, or N % 16 != 0)")
        print(f"  Estimated VRAM saved: ~{saved_mb:.0f} MB (vs FP16)")

    return stats


def _replace_linear_sparse(module, min_features, sparse_min_k, stats):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            num_params = child.in_features * child.out_features
            stats["total_params"] += num_params

            if child.in_features < min_features or child.out_features < min_features:
                stats["skipped_params"] += num_params
            elif (child.in_features < sparse_min_k
                  or child.in_features % 64 != 0
                  or child.out_features % 16 != 0):
                # The sparse kernel needs K % 64 == 0 and N % 16 == 0. A sparse Int8Linear
                # with any other shape would fall through to the dequant path, which has
                # no dense weights in sparse mode, so keep these layers as FP16.
                stats["kept_fp16"] += 1
            else:
                int8_layer = Int8Linear.from_linear_sparse(child)
                setattr(module, name, int8_layer)
                stats["quantized_params"] += num_params
                stats["num_replaced"] += 1
                if int8_layer._sparse_ok:
                    stats["sparse_eligible"] += 1
        else:
            _replace_linear_sparse(child, min_features, sparse_min_k, stats)
