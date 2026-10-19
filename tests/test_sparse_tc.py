"""
Correctness tests for the hand-rolled sparse INT8 Tensor Core GEMM (PTX mma.sp).

Covers: compilation, 2:4 prune/compress, single-tile mma.sp exactness (uniform,
varied and random data), the tiled GEMM against a reference matmul, UNet-scale
shapes, and the Int8Linear / quantize_model_sparse integration.

Usage:
    python -m pytest tests/test_sparse_tc.py -v
    python tests/test_sparse_tc.py           # same, via pytest.main
    python tests/test_sparse_tc.py --bench   # sparse vs dense TC micro-benchmark
"""

import os
import sys

import pytest
import torch

# Add project root to path (parent of tests/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(scope="module")
def mod():
    """Compile (or load from cache) the sparse TC extension once per test module."""
    from csrc.build import load_sparse_tc
    return load_sparse_tc()


def pack_metadata_for_hardware(meta_raw, N, K):
    """
    Reorder raw sequential metadata into per-lane hardware format for mma.sp.

    For m16n8k32 INT8 sparse with spsel=0:
      - 32 threads per warp, grouped in quads (4 threads per group)
      - groupID = lane >> 2  (0..7), handles rows groupID and groupID+8
      - tid_in_group = lane & 3  (0..3)
      - spsel=0 reads metadata from tid_in_group 0 and 1:
        * tid_in_group=0: K-groups 0,1,2,3  for rows (groupID, groupID+8)
        * tid_in_group=1: K-groups 4,5,6,7  for rows (groupID, groupID+8)
      - E bits[0..15]: 4 nibbles for row groupID
      - E bits[16..31]: 4 nibbles for row groupID+8

    Args:
        meta_raw: (N * K/4,) uint8 tensor, sequential nibbles
        N: number of rows (must be 16 for test)
        K: original dense K (must be 32 for test)

    Returns:
        (32,) int32 tensor, per-lane metadata
    """
    import numpy as np
    groups_per_row = K // 4  # 8 for K=32
    meta_np = meta_raw.cpu().numpy().reshape(N, groups_per_row)

    hw_meta = np.zeros(32, dtype=np.uint32)

    for groupID in range(8):
        row_a = groupID       # rows 0..7
        row_b = groupID + 8   # rows 8..15

        # tid_in_group=0: all 8 group nibbles for row groupID
        e0 = np.uint32(0)
        for g in range(groups_per_row):
            nib = int(meta_np[row_a, g]) & 0xF
            e0 |= np.uint32(nib << (g * 4))
        hw_meta[groupID * 4 + 0] = e0

        # tid_in_group=1: all 8 group nibbles for row groupID+8
        e1 = np.uint32(0)
        for g in range(groups_per_row):
            nib = int(meta_np[row_b, g]) & 0xF
            e1 |= np.uint32(nib << (g * 4))
        hw_meta[groupID * 4 + 1] = e1

    return torch.from_numpy(hw_meta.view(np.int32)).clone()


def reference_sparse_matmul(A_dense, B_nk):
    """
    Reference: C = A_sparse @ B^T where B is stored as (N_out, K).
    But for our test, B is (8, 32) and we want C = A(16,32) @ B^T(32,8) = (16, 8).

    Actually the MMA computes: C[m,n] = sum_k A[m,k] * B[k,n]
    where B is col-major (K=32, N=8).
    B stored as (8, 32) row-major = col-major (32, 8).
    So B_col[k, n] = B_rm[n, k] = B_rm[n][k].

    C[m, n] = sum_k A_dense[m, k] * B_rm[n, k]
            = (A_dense @ B_rm^T)[m, n]
    """
    # A_dense: (16, 32) int8, B_nk: (8, 32) int8
    A = A_dense.to(torch.int32)
    B = B_nk.to(torch.int32)
    return A @ B.T  # (16, 32) @ (32, 8) = (16, 8)


def test_compile(mod):
    """Step 1: The module compiles and exports the expected functions."""
    funcs = [x for x in dir(mod) if not x.startswith('_')]
    print(f"  Exported functions: {funcs}")
    for name in ("test_mma_sp", "sparse_prune_compress",
                 "sparse_prune_and_pack", "sparse_int8_linear"):
        assert name in funcs, f"{name} not found!"


def test_instruction_runs(mod):
    """Step 2: Verify mma.sp doesn't crash (all-ones test)."""
    print()
    print("=" * 60)
    print("Step 2: Testing mma.sp instruction execution (all-ones)...")
    print("=" * 60)

    # A_comp (16, 16) all ones - represents A_dense (16, 32) with [1,1,0,0] pattern
    A_comp = torch.ones(16, 16, dtype=torch.int8, device='cuda')

    # B (8, 32) all ones
    B_dense = torch.ones(8, 32, dtype=torch.int8, device='cuda')

    # Metadata: all groups have pattern keep[0,1] -> nibble = 0|1<<2 = 0x4
    # For the uniform case, all E values = 0x44444444
    meta = torch.full((32,), 0x44444444, dtype=torch.int32, device='cuda')

    C = mod.test_mma_sp(A_comp, B_dense, meta)
    print(f"  C shape: {C.shape}, dtype: {C.dtype}")
    print(f"  C:\n{C.cpu()}")

    # Reference: each output = 16 non-zeros per row × 1 × 1 = 16
    # (8 groups × 2 kept values × 1 × 1)
    expected_val = 16
    C_ref = torch.full((16, 8), expected_val, dtype=torch.int32)

    assert torch.equal(C.cpu(), C_ref), (
        f"expected all elements = {expected_val}, got sum {C.sum().item()}")


def test_correctness(mod):
    """Step 3: Test with varied values to verify metadata correctness."""
    print()
    print("=" * 60)
    print("Step 3: Correctness test (varied values)...")
    print("=" * 60)

    # A_dense (16, 32): pattern [val, val+1, 0, 0] per group
    # Keep positions 0,1 (largest magnitude if we use values > 0)
    A_dense = torch.zeros(16, 32, dtype=torch.int8)
    for row in range(16):
        for g in range(8):
            A_dense[row, g * 4 + 0] = (g + 1)        # keep
            A_dense[row, g * 4 + 1] = (g + 1) * 2    # keep (larger)
            A_dense[row, g * 4 + 2] = 0               # prune
            A_dense[row, g * 4 + 3] = 0               # prune

    # A_comp (16, 16): non-zeros in order
    A_comp = torch.zeros(16, 16, dtype=torch.int8)
    for row in range(16):
        for g in range(8):
            A_comp[row, g * 2 + 0] = A_dense[row, g * 4 + 0]
            A_comp[row, g * 2 + 1] = A_dense[row, g * 4 + 1]

    # B (8, 32): value = k+1 for all columns (tests that metadata selects right k)
    B_dense = torch.zeros(8, 32, dtype=torch.int8)
    for n in range(8):
        for k in range(32):
            B_dense[n, k] = (k % 15) + 1  # keep in int8 range

    # Metadata: keep[0,1] for all groups -> nibble 0x4
    meta = torch.full((32,), 0x44444444, dtype=torch.int32, device='cuda')

    C = mod.test_mma_sp(A_comp.cuda(), B_dense.cuda(), meta)
    C_ref = reference_sparse_matmul(A_dense, B_dense)

    print(f"  C (mma.sp):\n{C.cpu()}")
    print(f"  C_ref (dense matmul):\n{C_ref}")

    diff = (C.cpu() - C_ref).abs()
    assert torch.equal(C.cpu(), C_ref), f"mismatch: max diff = {diff.max().item()}"


def test_prune_compress(mod):
    """Test the prune + compress kernel."""
    print()
    print("=" * 60)
    print("Step 4: Testing prune_compress_kernel...")
    print("=" * 60)

    # Create weight with known pattern: [10, 20, 1, 2] -> keep [10, 20], prune [1, 2]
    W = torch.zeros(4, 8, dtype=torch.int8, device='cuda')
    for row in range(4):
        for g in range(2):  # 2 groups per row (K=8, 8/4=2)
            W[row, g * 4 + 0] = 10
            W[row, g * 4 + 1] = 20
            W[row, g * 4 + 2] = 1
            W[row, g * 4 + 3] = 2

    comp, meta_raw = mod.sparse_prune_compress(W)
    print(f"  Input weight (4, 8):\n{W.cpu()}")
    print(f"  Compressed (4, 4):\n{comp.cpu()}")
    print(f"  Raw metadata ({meta_raw.shape}):\n{meta_raw.cpu()}")

    # Check compressed values
    expected_comp = torch.zeros(4, 4, dtype=torch.int8)
    for row in range(4):
        for g in range(2):
            expected_comp[row, g * 2 + 0] = 10
            expected_comp[row, g * 2 + 1] = 20

    assert torch.equal(comp.cpu(), expected_comp), "compressed values mismatch"

    # Check metadata nibbles: keep positions 0,1 -> nibble = 0 | (1 << 2) = 0x4
    meta_vals = meta_raw.cpu()
    all_correct = all(meta_vals[i].item() == 0x4 for i in range(meta_vals.numel()))
    print(f"  Metadata nibbles: {[hex(x.item()) for x in meta_vals]}")
    assert all_correct, "metadata nibbles should all be 0x4"


def test_random_data(mod):
    """Step 5: Random data + random 2:4 patterns - exhaustive correctness check."""
    print()
    print("=" * 60)
    print("Step 5: Random data with varied 2:4 patterns...")
    print("=" * 60)

    torch.manual_seed(42)

    # Random dense weight (16, 32) with values in [-50, 50]
    A_dense = torch.randint(-50, 51, (16, 32), dtype=torch.int8)

    # Apply 2:4 pruning: for each group of 4, zero the 2 smallest
    A_pruned = A_dense.clone()
    A_comp = torch.zeros(16, 16, dtype=torch.int8)
    meta_nibbles = torch.zeros(16, 8, dtype=torch.uint8)  # (rows, groups)

    for row in range(16):
        for g in range(8):
            vals = A_dense[row, g*4 : g*4+4].abs()
            # Find 2 largest positions
            _, top2 = vals.topk(2)
            top2_sorted = top2.sort().values
            keep0, keep1 = top2_sorted[0].item(), top2_sorted[1].item()

            # Zero the pruned positions
            mask = torch.zeros(4, dtype=torch.bool)
            mask[keep0] = True
            mask[keep1] = True
            for i in range(4):
                if not mask[i]:
                    A_pruned[row, g*4+i] = 0

            # Store compressed
            A_comp[row, g*2+0] = A_dense[row, g*4+keep0]
            A_comp[row, g*2+1] = A_dense[row, g*4+keep1]

            # Store metadata nibble
            meta_nibbles[row, g] = keep0 | (keep1 << 2)

    # Random B (8, 32)
    B_dense = torch.randint(-30, 31, (8, 32), dtype=torch.int8)

    # Pack metadata for hardware
    # Layout (empirically verified on RTX 4090):
    #   Lane groupID*4+0 (tid=0): all 8 nibbles for row groupID
    #   Lane groupID*4+1 (tid=1): all 8 nibbles for row groupID+8
    #   Lane groupID*4+2 (tid=2): unused by spsel=0
    #   Lane groupID*4+3 (tid=3): unused by spsel=0
    #   Nibble i at bits[i*4..i*4+3] = K-group i
    import numpy as np
    hw_meta_np = np.zeros(32, dtype=np.uint32)
    for groupID in range(8):
        row_a = groupID
        row_b = groupID + 8

        # tid_in_group=0: all 8 groups for row groupID
        e0 = np.uint32(0)
        for g in range(8):
            nib = int(meta_nibbles[row_a, g]) & 0xF
            e0 |= np.uint32(nib << (g * 4))
        hw_meta_np[groupID * 4 + 0] = e0

        # tid_in_group=1: all 8 groups for row groupID+8
        e1 = np.uint32(0)
        for g in range(8):
            nib = int(meta_nibbles[row_b, g]) & 0xF
            e1 |= np.uint32(nib << (g * 4))
        hw_meta_np[groupID * 4 + 1] = e1

    hw_meta = torch.from_numpy(hw_meta_np.view(np.int32)).clone()

    # Run MMA
    C = mod.test_mma_sp(A_comp.cuda(), B_dense.cuda(), hw_meta.cuda())

    # Reference: C = A_pruned @ B^T (sparse matmul using the pruned dense matrix)
    C_ref = reference_sparse_matmul(A_pruned, B_dense)

    print(f"  C (mma.sp) [0:4, :]:\n{C.cpu()[0:4, :]}")
    print(f"  C_ref      [0:4, :]:\n{C_ref[0:4, :]}")

    diff = (C.cpu() - C_ref).abs()
    n_wrong = int((diff > 0).sum().item())
    assert torch.equal(C.cpu(), C_ref), (
        f"mismatch: max diff = {diff.max().item()}, wrong elements {n_wrong}/{16*8}")


def test_tiled_gemm_small(mod):
    """Step 6: Test tiled sparse GEMM at small sizes (verifies tiling logic)."""
    print()
    print("=" * 60)
    print("Step 6: Tiled sparse GEMM - small sizes (N=64, K=64, M=32)...")
    print("=" * 60)

    import numpy as np
    torch.manual_seed(123)
    N_out, K, M = 64, 64, 32

    # Random dense weight
    W_dense = torch.randint(-50, 51, (N_out, K), dtype=torch.int8, device='cuda')

    # Use CUDA prune for both kernel and reference (avoids tie-breaking diffs)
    results = mod.sparse_prune_compress(W_dense)
    w_comp_raw, meta_raw = results[0], results[1]
    results2 = mod.sparse_prune_and_pack(W_dense)
    w_comp, w_meta = results2[0], results2[1]

    # Reconstruct pruned dense matrix from CUDA prune output
    meta_raw_np = meta_raw.cpu().numpy()
    w_comp_np = w_comp_raw.cpu().numpy()
    W_pruned = torch.zeros(N_out, K, dtype=torch.int8)
    for row in range(N_out):
        for g in range(K // 4):
            nib = int(meta_raw_np[row * (K // 4) + g]) & 0xF
            k0, k1 = nib & 0x3, (nib >> 2) & 0x3
            W_pruned[row, g*4+k0] = w_comp_np[row, g*2+0]
            W_pruned[row, g*4+k1] = w_comp_np[row, g*2+1]

    w_scale = W_pruned.float().abs().amax(dim=1).clamp(min=1e-8) / 127.0

    # Random activation
    x = torch.randn(M, K, dtype=torch.float16, device='cuda')

    w_scale_half = w_scale.half().cuda()
    bias = torch.zeros(N_out, dtype=torch.float16, device='cuda')
    y = mod.sparse_int8_linear(x, w_comp, w_meta, w_scale_half, bias)

    # Reference
    x_float = x.float().cpu()
    act_scale = x_float.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    x_int8 = (x_float / act_scale[:, None]).round().clamp(-128, 127).to(torch.int8)

    C_ref_int32 = W_pruned.int() @ x_int8.int().T  # (N_out, M)
    y_ref = (C_ref_int32.float().T * act_scale[:, None].float() *
             w_scale[None, :].float()).half()

    y_cpu = y.cpu()
    diff = (y_cpu.float() - y_ref.float()).abs()
    denom = y_ref.float().abs().clamp(min=1.0)
    rel_err = diff / denom
    max_abs = diff.max().item()
    max_rel = rel_err.max().item()
    mean_rel = rel_err.mean().item()

    print(f"  Output shape: {y_cpu.shape}")
    print(f"  y[0, :8]:     {y_cpu[0, :8]}")
    print(f"  y_ref[0, :8]: {y_ref[0, :8]}")
    print(f"  Max abs diff:  {max_abs:.4f}")
    print(f"  Max rel error: {max_rel:.4f} ({max_rel*100:.2f}%)")
    print(f"  Mean rel error: {mean_rel:.6f} ({mean_rel*100:.4f}%)")

    assert max_rel < 0.05, f"max rel error {max_rel:.4f}"
    assert mean_rel < 0.02, f"mean rel error {mean_rel:.6f}"


def test_tiled_gemm_unet(mod):
    """Step 7: Test tiled sparse GEMM at UNet-scale sizes."""
    print()
    print("=" * 60)
    print("Step 7: Tiled sparse GEMM - UNet shapes...")
    print("=" * 60)

    shapes = [
        (1536, 1280, 1280),  # Typical UNet linear
        (1536, 1280, 5120),  # FFN up-projection
        (1536, 5120, 1280),  # FFN down-projection
        (1536,  640,  640),  # Smaller UNet block
        (1536,  320,  320),  # Smallest UNet block
    ]

    torch.manual_seed(42)

    for M, N_out, K in shapes:
        print(f"\n  Shape: M={M}, N_out={N_out}, K={K}")

        # Random dense weight
        W_dense = torch.randint(-50, 51, (N_out, K), dtype=torch.int8, device='cuda')

        # Prune+compress+pack
        w_comp, w_meta = mod.sparse_prune_and_pack(W_dense)

        # Weight scale (per-channel)
        w_scale = torch.rand(N_out, dtype=torch.float16, device='cuda') * 0.01 + 0.001

        # Random activation
        x = torch.randn(M, K, dtype=torch.float16, device='cuda') * 0.5

        # Empty bias
        bias = torch.empty(0, dtype=torch.float16, device='cuda')

        y = mod.sparse_int8_linear(x, w_comp, w_meta, w_scale, bias)
        print(f"    Output: {y.shape}, dtype={y.dtype}")
        print(f"    Range: [{y.min().item():.2f}, {y.max().item():.2f}]")
        assert y.shape == (M, N_out)
        assert y.dtype == torch.float16
        nan_count = y.isnan().sum().item()
        assert nan_count == 0, f"{nan_count} NaN values for M={M}, N={N_out}, K={K}"


def test_tiled_gemm_correctness(mod):
    """Step 8: Correctness of tiled sparse GEMM at 128x128x64.

    Uses CUDA prune+compress for both kernel and reference to avoid
    tie-breaking differences between Python and CUDA pruning.
    """
    print()
    print("=" * 60)
    print("Step 8: Tiled GEMM correctness (128x128x64)...")
    print("=" * 60)

    import numpy as np
    torch.manual_seed(99)
    N_out, K, M = 128, 128, 64

    # Random dense weight
    W_dense = torch.randint(-40, 41, (N_out, K), dtype=torch.int8, device='cuda')

    # Use CUDA prune+compress to get canonical compressed weight + raw metadata
    results = mod.sparse_prune_compress(W_dense)
    w_comp_cuda = results[0]     # (N_out, K/2) int8
    meta_raw_cuda = results[1]   # (N_out * K/4) uint8

    # Also get packed metadata for the kernel
    results2 = mod.sparse_prune_and_pack(W_dense)
    w_comp = results2[0]
    w_meta = results2[1]

    # Reconstruct pruned dense matrix from CUDA output for reference
    meta_raw_np = meta_raw_cuda.cpu().numpy()
    w_comp_np = w_comp_cuda.cpu().numpy()
    W_pruned = torch.zeros(N_out, K, dtype=torch.int8)
    for row in range(N_out):
        for g in range(K // 4):
            nib = int(meta_raw_np[row * (K // 4) + g]) & 0xF
            k0 = nib & 0x3
            k1 = (nib >> 2) & 0x3
            W_pruned[row, g * 4 + k0] = w_comp_np[row, g * 2 + 0]
            W_pruned[row, g * 4 + k1] = w_comp_np[row, g * 2 + 1]

    # Random int8 activation
    x_int8 = torch.randint(-30, 31, (M, K), dtype=torch.int8)

    # Reference matmul: W_pruned(N,K) @ x_int8^T(K,M) = C(N,M), then C^T = (M,N)
    C_ref = (W_pruned.int() @ x_int8.int().T).T  # (M, N_out)

    # Build fp16 activation that the kernel will quantize back to x_int8 exactly
    x_float = x_int8.float()
    act_scales_ref = x_float.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    x_fp16 = (x_float * act_scales_ref[:, None]).half()

    # Weight scale = 1.0, no bias
    w_scale = torch.ones(N_out, dtype=torch.float16, device='cuda')
    bias = torch.empty(0, dtype=torch.float16, device='cuda')

    y = mod.sparse_int8_linear(x_fp16.cuda(), w_comp, w_meta, w_scale, bias)

    # Expected: C_ref * act_scale, as fp16
    y_expected = (C_ref.float() * act_scales_ref[:, None]).half()

    y_cpu = y.cpu()
    diff = (y_cpu.float() - y_expected.float()).abs()
    max_diff = diff.max().item()
    denom = y_expected.float().abs().clamp(min=1.0)
    rel_err = diff / denom
    max_rel = rel_err.max().item()
    mean_rel = rel_err.mean().item()

    print(f"  Output shape: {y_cpu.shape}")
    print(f"  y[0, :8]:     {y_cpu[0, :8]}")
    print(f"  y_exp[0, :8]: {y_expected[0, :8]}")
    print(f"  Max abs diff:  {max_diff:.2f}")
    print(f"  Max rel error: {max_rel:.4f} ({max_rel*100:.2f}%)")
    print(f"  Mean rel error: {mean_rel:.6f} ({mean_rel*100:.4f}%)")

    # Tolerance: quantization noise scales with K (each off-by-1 int8 * weight ~40 * scale)
    # For K=128: max abs diff ~20 is expected. Mean relative ~2% is fine.
    assert max_diff < 30, f"max abs diff {max_diff:.2f}"
    assert mean_rel < 0.03, f"mean rel error {mean_rel:.6f}"


@pytest.mark.parametrize("K,N_out,M", [
    (1280, 320, 70000),   # narrow N, activation larger than L2: auto picks the fused path
    (1024, 336, 5000),    # ragged N (last weight tile partly past N_out), ragged M
    (640, 1296, 2049),
    (64, 16, 5),          # single K step, single tile
])
def test_fused_quantize_path_matches(mod, K, N_out, M):
    """The fused quantize + GEMM path and the N-fastest tile order give bit-identical output
    to quantize kernel + GEMM (int32 accumulation is exact; same per-element epilogue)."""
    torch.manual_seed(K + N_out + M)
    W = torch.randn(N_out, K, device="cuda") * 0.02
    w_scale = W.abs().amax(1).clamp(min=1e-8) / 127.0
    wq = (W / w_scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    wc, wm = mod.sparse_prune_and_pack(wq.contiguous())
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = (torch.randn(N_out, device="cuda") * 0.1).half()
    ws = w_scale.half()

    ref = mod.sparse_int8_linear_ex(x, wc, wm, ws, b, 0)       # quantize kernel + GEMM
    fused = mod.sparse_int8_linear_ex(x, wc, wm, ws, b, 1)     # fused path, forced
    auto = mod.sparse_int8_linear(x, wc, wm, ws, b)
    assert torch.equal(fused, ref)
    assert torch.equal(auto, ref)
    xq, xs = mod.sparse_quantize_act(x)
    for tile in (0, 1, 2):
        for order in (0, 1):
            y = mod.sparse_int8_gemm(xq, xs, wc, wm, ws, b, tile, 1, order)
            assert torch.equal(y, ref), (tile, order)
    if (K, N_out, M) == (1280, 320, 70000):
        assert mod.sparse_int8_linear_path(M, K, N_out) == 1
    assert mod.sparse_int8_linear_path(4096, 1280, 5120) == 0   # headline shapes stay two-kernel


def benchmark_sparse_vs_dense(mod):
    """Step 9: Benchmark sparse TC GEMM vs dense TC GEMM."""
    print()
    print("=" * 60)
    print("Step 9: Benchmark - Sparse vs Dense TC GEMM...")
    print("=" * 60)

    try:
        from csrc.build import load_tc_gemm
        tc_mod = load_tc_gemm()
    except Exception as e:
        print(f"  SKIP: Could not load dense TC module: {e}")
        return

    shapes = [
        (1536, 1280, 1280),
        (1536, 1280, 5120),
        (1536, 5120, 1280),
        (768,  1280, 1280),
        (384,  1280, 1280),
    ]

    warmup = 20
    iters = 100

    print(f"\n  {'M':>6} {'N_out':>6} {'K':>6} | {'Sparse':>10} {'Dense':>10} {'Speedup':>8}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} | {'-'*10} {'-'*10} {'-'*8}")

    for M, N_out, K in shapes:
        torch.manual_seed(0)

        # Prepare sparse inputs
        W_dense = torch.randint(-50, 51, (N_out, K), dtype=torch.int8, device='cuda')
        w_comp, w_meta = mod.sparse_prune_and_pack(W_dense)
        w_scale_sp = torch.rand(N_out, dtype=torch.float16, device='cuda') * 0.01 + 0.001
        bias_empty = torch.empty(0, dtype=torch.float16, device='cuda')
        x = torch.randn(M, K, dtype=torch.float16, device='cuda') * 0.5

        # Prepare dense inputs (reuse W_dense as weight)
        w_scale_dn = w_scale_sp.clone()
        bias_dn = torch.zeros(N_out, dtype=torch.float16, device='cuda')

        # Warmup sparse
        for _ in range(warmup):
            mod.sparse_int8_linear(x, w_comp, w_meta, w_scale_sp, bias_empty)
        torch.cuda.synchronize()

        # Time sparse
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            mod.sparse_int8_linear(x, w_comp, w_meta, w_scale_sp, bias_empty)
        end.record()
        torch.cuda.synchronize()
        sparse_ms = start.elapsed_time(end) / iters

        # Warmup dense
        for _ in range(warmup):
            tc_mod.int8_linear_tc(x, W_dense, w_scale_dn, bias_dn)
        torch.cuda.synchronize()

        # Time dense
        start.record()
        for _ in range(iters):
            tc_mod.int8_linear_tc(x, W_dense, w_scale_dn, bias_dn)
        end.record()
        torch.cuda.synchronize()
        dense_ms = start.elapsed_time(end) / iters

        speedup = dense_ms / sparse_ms if sparse_ms > 0 else 0
        print(f"  {M:>6} {N_out:>6} {K:>6} | {sparse_ms:>8.3f}ms {dense_ms:>8.3f}ms {speedup:>7.2f}x")


def test_int8linear_sparse_integration():
    """Step 10: Integration test - Int8Linear sparse mode via quantize_utils."""
    print()
    print("=" * 60)
    print("Step 10: Int8Linear sparse integration test...")
    print("=" * 60)

    from python.quantize_utils import Int8Linear, quantize_model_sparse, sp_available

    if not sp_available():
        pytest.skip("Sparse TC module not available")

    # Test 1: from_linear_sparse on a single layer
    print("\n  --- Test 1: from_linear_sparse ---")
    torch.manual_seed(42)
    linear = torch.nn.Linear(1280, 1280, bias=True).cuda().half()
    sparse_layer = Int8Linear.from_linear_sparse(linear)
    print(f"  Mode: {sparse_layer.mode}")
    print(f"  Sparse comp shape: {sparse_layer.weight_sparse_comp.shape}")
    print(f"  Sparse meta shape: {sparse_layer.weight_sparse_meta.shape}")
    print(f"  Dense weight freed: {sparse_layer.weight_int8_nk.numel() == 0}")
    print(f"  repr: {sparse_layer.extra_repr()}")

    assert sparse_layer.mode == "sparse"
    assert sparse_layer.weight_sparse_comp.shape == (1280, 640)  # K/2
    assert sparse_layer.weight_sparse_meta.shape == (1280, 40)   # K/32
    assert sparse_layer.weight_int8_nk.numel() == 0

    # Forward pass
    x = torch.randn(64, 1280, dtype=torch.float16, device='cuda')
    y = sparse_layer(x)
    print(f"  Forward: x {x.shape} -> y {y.shape}")
    assert y.shape == (64, 1280)
    assert not y.isnan().any(), "NaN in output!"
    print("  PASS: Single layer sparse forward works")

    # Test 2: 3D input (batched)
    print("\n  --- Test 2: 3D input ---")
    x3d = torch.randn(2, 32, 1280, dtype=torch.float16, device='cuda')
    y3d = sparse_layer(x3d)
    print(f"  Forward: x {x3d.shape} -> y {y3d.shape}")
    assert y3d.shape == (2, 32, 1280)
    assert not y3d.isnan().any()
    print("  PASS: 3D input works")

    # Test 3: No-bias layer
    print("\n  --- Test 3: No-bias layer ---")
    linear_nobias = torch.nn.Linear(640, 320, bias=False).cuda().half()
    sparse_nobias = Int8Linear.from_linear_sparse(linear_nobias)
    y_nb = sparse_nobias(torch.randn(16, 640, dtype=torch.float16, device='cuda'))
    assert y_nb.shape == (16, 320)
    assert not y_nb.isnan().any()
    print("  PASS: No-bias sparse layer works")

    # Test 4: quantize_model_sparse on a small model
    print("\n  --- Test 4: quantize_model_sparse ---")
    model = torch.nn.Sequential(
        torch.nn.Linear(1280, 5120, bias=True),
        torch.nn.ReLU(),
        torch.nn.Linear(5120, 1280, bias=True),
    ).cuda().half()

    stats = quantize_model_sparse(model, min_features=128, verbose=True)
    print(f"  Stats: {stats}")
    assert stats['num_replaced'] == 2
    assert stats['sparse_eligible'] == 2

    # Forward pass through quantized model
    x_model = torch.randn(32, 1280, dtype=torch.float16, device='cuda')
    y_model = model(x_model)
    print(f"  Model forward: x {x_model.shape} -> y {y_model.shape}")
    assert y_model.shape == (32, 1280)
    assert not y_model.isnan().any()
    print("  PASS: Full model sparse quantization + forward works")

    # Test 5: Verify mode shows correctly
    print("\n  --- Test 5: Layer inspection ---")
    for name, m in model.named_modules():
        if isinstance(m, Int8Linear):
            print(f"  {name}: {m.extra_repr()}")
            assert m.mode == "sparse"


def test_quantize_model_sparse_skips_unaligned_k():
    """quantize_model_sparse must leave layers with K % 64 != 0 as FP16 nn.Linear.

    K=672 passes the K >= 640 threshold and the prune kernel's K % 32 check, but the
    sparse GEMM needs K % 64 == 0. Converting it used to crash on forward.
    """
    from python.quantize_utils import Int8Linear, quantize_model_sparse, sp_available

    if not sp_available():
        pytest.skip("Sparse TC module not available")

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(672, 1280, bias=True),   # K % 64 == 32 -> must stay FP16
        torch.nn.ReLU(),
        torch.nn.Linear(1280, 1280, bias=True),  # aligned -> sparse
    ).cuda().half()

    stats = quantize_model_sparse(model, verbose=False)
    assert stats["num_replaced"] == 1
    assert stats["kept_fp16"] == 1
    assert type(model[0]) is torch.nn.Linear
    assert isinstance(model[2], Int8Linear) and model[2].mode == "sparse"

    y = model(torch.randn(64, 672, dtype=torch.float16, device="cuda"))
    assert y.shape == (64, 1280)
    assert not y.isnan().any()


if __name__ == "__main__":
    if "--bench" in sys.argv:
        from csrc.build import load_sparse_tc
        print(f"PyTorch: {torch.__version__}  GPU: {torch.cuda.get_device_name(0)}")
        benchmark_sparse_vs_dense(load_sparse_tc())
    else:
        sys.exit(pytest.main([__file__, "-v"]))
