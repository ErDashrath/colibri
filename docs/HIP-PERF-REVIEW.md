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

**Correction (measured 2026-07-14):** `other` is **almost entirely disk-I/O
overlap**, NOT shared-expert sync. In a controlled A/B (`COLI_CUDA_FUSE_MLP=1` vs
`=0`, same warm cache), `other − expert-disk` was **~7s in both runs** — i.e. the
non-disk portion of "other" is small and unchanged by fusing the shared expert.
`other ≈ expert-disk service` because disk service overlaps wall time but is not
in `accounted = ewait + emm + attn + head`.

The shared expert is small (sI = `moe_intermediate × n_shared` = 2048 × 1 →
~6 MB weights/triple) and runs resident (no disk). Its 3→1 fusion saves <0.5s
total — invisible in the noise. **Item A (fuse shared expert) was measured to
have no benefit and reverted.** The real non-disk costs are `expert-matmul`
(routed experts, grouped) and `attention` (CPU by default).

The uninstrumented GPU work in "other" is:

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

#### A. ~~Fuse shared-expert + dense MLP through `coli_cuda_expert_mlp`~~ (REVERTED — no benefit)
Measured A/B (`COLI_CUDA_FUSE_MLP=1` vs `0`, warm cache, 12 tokens): wall
215s vs 190s (disk noise), `other − disk` identical (~7s) in both. The shared
expert is tiny (~6 MB/triple, resident) so 3→1 fusion saves <0.5s. "other" is
disk overlap, not shared-expert sync. **Reverted; glm.c unchanged.**

#### B. ~~Enable `COLI_CUDA_ATTN` for HIP~~ (NO BENEFIT — not enabled)
Measured A/B (`COLI_CUDA_ATTN=1` vs `0`, warm): attention bucket ~10s either
way (swamped by disk variance). The GPU attention kernel (H=64 blocks, serial
softmax, per-call H2D/D2H/sync) cannot beat the fast Zen5 CPU for single-token
decode — the score-softmax-value core was ~1.7s on GPU vs ~0.5–2s on CPU.
**Left off by default; attention stays on CPU.** Not worth a kernel rework: the
bucket is ~5–10s vs ~80–155s disk.

### Tier 0 — the actual win (implemented)

#### ★ `CUDA_DENSE=0` for HIP: don't put dense matmuls on the GPU
The engine default is `CUDA_DENSE=0` (dense on CPU), but the `coli` launcher
force-set `CUDA_DENSE=1` whenever `--gpu` was passed. On this APU that **hurts**:
per-call H2D/D2H/sync on unified memory makes the GPU slower than the Zen5 CPU
for small decode matmuls. Measured (warm, 12 tokens):

| config | wall | attention | expert-matmul | lm_head |
|---|---|---|---|---|
| `CUDA_DENSE=1` (dense GPU) | 161–196s | 10.5s | 19–27s | 0.5–0.9s |
| `CUDA_DENSE=0` (dense CPU) | **90–106s** | **4.0s** | **12.1s** | **0.14s** |

~2× faster. **Fix:** `coli` now defaults `CUDA_DENSE=1` only for the CUDA
backend (`gpu_backend()=="cuda"`); for HIP it leaves the engine default (0).
Users can override with `CUDA_DENSE=1` (e.g. AMD dGPUs with fast VRAM may want
it). The GPU's correct role on this APU is the **expert tier only** (grouped /
batched), matching the README's finding that the GPU "buys ≈0%" on small decode
matmuls when the CPU is fast.

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

1. ✅ **`CUDA_DENSE=0` for HIP** (the real win, ~2× faster on APU) — implemented in
   `coli`.
2. ~~Fuse shared expert (A)~~ — measured no benefit; reverted.
3. ~~Attention on GPU (B)~~ — measured no benefit; left on CPU.
4. The remaining wall time is **disk-I/O-bound** (80–155s expert-disk). The GPU
   can't help; raise the expert hit rate (more `--ram`, more usage history → more
   pinning, `--topp 0.7`). On this APU + fast Zen5 CPU, single-token decode is
   CPU+disk bound, not GPU-compute bound.
5. If a GPU-compute win is still wanted: **item E (rocWMMA tensor cores)** for the
   grouped expert matmul (12s bucket) is the only remaining GPU lever — but it's
   small vs disk, and high effort.

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
