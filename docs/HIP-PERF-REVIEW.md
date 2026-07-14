# colibrì HIP Backend — Performance Review & Opportunities

**Date:** 2026-07-14
**Scope:** `c/backend_hip.cpp` and the host GPU integration in `c/glm.c`, reviewed
for throughput improvements on the AMD Radeon 8060S (`gfx1151`, APU, unified
memory). This is a findings document for later reuse — not an implementation.

**Profile baseline** (`coli run --gpu 0 --vram 48 …`, real 744B GLM-5.2 model,
9 decode tokens, wall 81.9s):
```
PROFILE: expert-disk 51.7s service / 0.0s wait | expert-matmul 9.6s | attention 15.6s (kvb 1.8s) | lm_head 0.8s | other 55.9s
ATTENTION: projection/RoPE 4.8s | score-softmax-value 0.5s | output projection 8.5s
```
`accounted = t_ewait + t_emm + t_attn + t_head` = ~26s, so **"other" (55.9s) is
the real CPU/GPU bottleneck**, not disk (disk service overlaps; wait≈0).

---

## What "other" contains

`other = elapsed - (ewait + emm + attn + head)`. The buckets `t_emm` (routed
expert matmuls) and `t_attn` (MLA: q_a/q_b/kv_a/kv_b/o + RoPE + softmax + value)
are instrumented. **Uninstrumented** GPU work falls into "other":

- **Shared expert** (`glm.c:2074-2080`): 3 separate `matmul_qt` calls per layer
  (`sh_gate`, `sh_up`, `sh_down`) → 78 layers × 3 = **234 synchronous
  H2D→kernel→D2H round-trips per token**, plus SiLU on the CPU in between.
- **`dense_mlp`** (first 3 layers, `glm.c:2086`): same 3-call pattern.
- Layernorms, router, RoPE, embedding, residual adds, MTP head, sampling.

So the shared-expert / dense-MLP GPU round-trips and their host sync are the
actionable HIP part of "other".

---

## Confirmed hot spots (code-level)

1. **Shared expert unfused** — 3 GPU round-trips/layer where 1 fused call would
   do. `coli_cuda_expert_mlp()` already fuses gate+SiLU·up+down with one H2D/D2H
   but the shared-expert path calls `matmul_qt` three times.
2. **`dense_mlp` unfused** — same pattern (`glm.c:2086`).
3. **Attention on CPU by default** (15.6s). `COLI_CUDA_ATTN=1` moves it to GPU
   but is off; the kernel also has **low occupancy** (grid = `H`=64 blocks) and a
   **single-threaded softmax over T≤4096** (`attention_absorb_kernel`).
4. **`quant_matmul` int4 reads each packed byte twice** via `weight_at` (one
   nibble per global read) instead of unpacking both nibbles at once.
5. **No tensor cores** — W4A32 is scalar int4 unpack + FP32 FMA, memory-bound.
   `gfx1151` supports WMMA (int8/fp16); rocWMMA could mirror the CUDA WMMA path.

---

## Prioritized opportunities

### Tier 1 — high impact, low/medium effort

#### A. Fuse shared-expert + dense MLP through `coli_cuda_expert_mlp`
Wire `sh_gate/sh_up/sh_down` (and `gate_proj/up_proj/down_proj`) into one fused
GPU call when all three are `cuda_eligible`. Shapes already satisfy the
contract: shared expert is `sh_gate[2048,6144]`, `sh_up[2048,6144]`,
`sh_down[6144,2048]` → `down->I(2048)==gate->O(2048)`,
`down->O(6144)==gate->I(6144)`. ✓

**Win:** 3→1 round-trips per layer, SiLU moves to GPU, `x` uploaded once instead
of 3×. ~156 fewer syncs/token. Cleanest attack on "other". ~30 lines of host
glue in the shared-expert block and `dense_mlp`. Low risk: `coli_cuda_expert_mlp`
is already unit-tested.

**Code touch points:** `glm.c:2074-2080` (shared expert), `glm.c:2086`
(`dense_mlp`), behind `#ifdef COLI_CUDA` with `cuda_eligible` guards + fallback
to the existing 3× `matmul_qt`.

#### B. Enable `COLI_CUDA_ATTN` for HIP + raise attention kernel occupancy
Move ~15.6s of attention off the CPU. Then fix the kernel
(`backend_hip.cpp` `attention_absorb_kernel`):
- (a) **Parallel softmax** — replace the one-thread loop over T with a warp
  shuffle reduction (T≤4096 → a few warps).
- (b) **More blocks** — grid is only `H`=64; split the T (or K) dimension across
  blocks so 40 CUs are fed (currently ~1.6 blocks/CU → severe underfill).

**Win:** attention goes GPU-side and actually uses the GPU. Medium effort.

### Tier 2 — medium impact

#### C. `quant_matmul` int4 micro-opts
- Unpack **both nibbles per byte load** (use the `unpack_s4` idiom from
  `grouped_hidden_w4`), halving global weight reads for the dense/shared int4
  path.
- Cache the `x` vector in **shared memory** for S=1 decode (re-read today by
  every output-row block; 24 KB fits easily).
Helps every dense/shared matmul. Low-medium effort.

#### D. `grouped_hidden_w4_dual`: cache activation row in shared memory
Per `(s,c)` load the activation row into shared once; currently every O block
re-reads it from global. Low effort, minor win (L2 likely absorbs some today).

### Tier 3 — high effort, highest ceiling

#### E. rocWMMA tensor-core path
Add an int8 (W8A8) or fp16 WMMA GEMM for grouped + shared experts behind
`COLI_CUDA_TC_INT4` (or a new flag) — the HIP analogue of the CUDA
`grouped_s4_wmma` kernel (which is guarded out on HIP today). `gfx1151` supports
WMMA. Biggest win once the expert cache is warm and the workload is
matmul-bound rather than disk-bound. Requires rocWMMA (`/opt/rocm/include/rocwmma`)
or the `__builtin_amdgcn_wmma_*` intrinsics.

#### F. Keep dense activations on-device across the MLA chain
Today every dense matmul (`q_a`, `q_b`, `kv_a`, `kv_b`, `o`) does its own
H2D/D2H. Keep a persistent device activation buffer and chain matmuls without
D2H between them, only copying back when the host needs the result (layernorm,
router, residual). Largest sync-overhead win; requires reworking the host
forward to hold device buffers and push layernorms/router to the GPU.

### Micro
- **G.** Fusion (A) fixes the shared-expert `x` triple-upload for free.
- **H.** Wave-shuffle reductions instead of the 256-wide shared-memory tree in
  `quant_matmul` / `grouped_*`.
- **I.** Vectorized loads (`uchar4` / `float4`) in the dequant kernels.

---

## Recommended next steps

1. **Implement A** (fuse shared expert + dense MLP via `coli_cuda_expert_mlp`).
   Smallest, lowest-risk change that directly removes the largest chunk of
   "other".
2. **Implement B** (enable `COLI_CUDA_ATTN` + parallel softmax + more blocks).
   Removes the 15.6s attention bucket from the CPU.
3. Re-measure with `COLI_CUDA_PROFILE=1` and compare "other" / "attention" /
   wall time.
4. Only then consider **C/D** (micro-kernel opts) and **E/F** (tensor cores /
   on-device activation chain), which are higher-effort.

## Non-GPU note (context, not HIP scope)

The remaining wall time is dominated by **disk I/O for cold streaming experts**
(expert-disk 51.7s service, ~50% hit rate). The GPU can't help here. Raising the
expert hit rate — more `--ram`, more usage history → more pinning, `--topp 0.7`
— improves throughput more than any GPU micro-opt until the cache is warm. The
HIP opportunities above target the *compute/sync* portion ("other" + attention
+ expert-matmul ≈ 81s of CPU/GPU work), which is what the GPU can actually own.

---

## Key code locations

| What | File:line |
|---|---|
| Shared expert (3× matmul_qt) | `glm.c:2074-2080` |
| `dense_mlp` (3× matmul_qt) | `glm.c:2086` |
| `matmul_qt` GPU dispatch | `glm.c:710-722` |
| Attention GPU gate (`COLI_CUDA_ATTN`) | `glm.c:1645-1658` |
| `coli_cuda_expert_mlp` (fused, exists) | `backend_hip.cpp` / `backend_cuda.h` |
| `coli_cuda_matmul` (per-call H2D/D2H) | `backend_hip.cpp` |
| `quant_matmul` kernel (int4 double-read) | `backend_hip.cpp` |
| `attention_absorb_kernel` (H blocks, serial softmax) | `backend_hip.cpp` |
| `grouped_hidden_w4_dual` (no x caching) | `backend_hip.cpp` |
| PROFILE accounting (`other = elapsed - accounted`) | `glm.c:2766-2771` |
