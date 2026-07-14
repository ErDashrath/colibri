# AGENTS.md — colibrì

Guidance for AI coding agents working on colibrì. Keep this file updated when
the architecture, build, or backends change.

## 1. Project description

colibrì is a **single-file C inference engine** for the **GLM-5.2** family of
MoE models (HuggingFace architecture `glm_moe_dsa`: Mixture-of-Experts with
Multi-head Latent Attention, RoPE, a shared expert, and a DeepSeek-V3-style MTP
speculative head). Zero runtime dependencies: no BLAS, no PyTorch, no GPU
required. Weights are quantised (int8/int4/int2 + per-row float scale) and
dequantised on use; streaming experts are paged from NVMe and cached in RAM
(LRU/LFRU); resident dense tensors stay quantised in RAM.

The engine is `c/glm.c` (~3,900 lines) plus small headers. A Python launcher
`c/coli` wraps build/run/serve/doctor/plan/convert. `c/openai_server.py` is an
OpenAI-compatible HTTP server. Optional GPU backends accelerate resident
tensors; the default build is pure C.

## 2. Design decisions

- **Backend-agnostic GPU contract.** The host engine never calls a GPU runtime
  directly. It calls the `extern "C"` `coli_cuda_*` ABI in `c/backend_cuda.h`
  and stores an **opaque** `ColiCudaTensor*`. This lets different backends drop
  in with zero host changes — the host code is identical for CUDA, HIP, and
  Metal. The "cuda" in the symbol names is a historical label for "the
  resident-GPU backend".
- **Opt-in backends.** `make` is pure C. `CUDA=1`, `HIP=1`, `METAL=1` add GPU
  backends; default runtime behaviour is unchanged. Each is behind `#ifdef`.
- **Lazy resident upload.** `coli_cuda_matmul` / `expert_*` upload weights on
  first use and reuse them, so capacity failures surface at model start, not
  mid-run. Weights stay resident; activations are H2D/D2H per call (the
  documented "PCIe copy tax").
- **W4A32 packed path is the default expert path** (FP32 activations × signed
  int4 weights, one byte unpacked once). The int4 tensor-core (WMMA) path is
  opt-in (`COLI_CUDA_TC_INT4`) and NVIDIA-only.
- **`glm.c` is shared by all backends.** GPU integration is exclusively behind
  `#ifdef COLI_CUDA` (CUDA and HIP both define this) / `#ifdef COLI_METAL`.

## 3. Implementation details

### Backends
| Backend | Source | Build | Host define | Linked lib |
|---|---|---|---|---|
| CUDA (NVIDIA) | `c/backend_cuda.cu` | `make CUDA=1` (nvcc) | `-DCOLI_CUDA` | `-lcudart` |
| **HIP (AMD ROCm)** | `c/backend_hip.cpp` | `make HIP=1` (hipcc) | `-DCOLI_CUDA` | `-lamdhip64` |
| Metal (Apple) | `c/backend_metal.mm` | `make METAL=1` | `-DCOLI_METAL` | `-framework Metal` |
| Windows CUDA DLL | `c/backend_loader.c` | `make CUDA_DLL=1 cuda-dll` | `-DCOLI_CUDA` | runtime `LoadLibrary` |

`HIP=1` and `CUDA=1` are mutually exclusive. The HIP backend exports the same
`coli_cuda_*` symbols as the CUDA backend, so `glm.c` is compiled with
`-DCOLI_CUDA` and links `backend_hip.o` instead of `backend_cuda.o`.

### HIP-specific build note
Device kernels are compiled with `hipcc -c` into `backend_hip.o`, then linked
into the gcc-built `glm` with `-lamdhip64 -lstdc++`. On some toolchains
(notably CachyOS / glibc-2.43) `hipcc`'s own `--hip-link` driver crashes inside
`libamdhip64` static init; the gcc-link path (matching llama.cpp's ROCm build)
avoids it. `hip-test` follows the same split.

### CLI tooling is GPU-backend-aware
`c/coli` (`gpu_backend()`), `c/doctor.py` (`cuda_linkage()`), and
`c/resource_plan.py` (`discover_gpus()`) detect **both** `libcudart` (CUDA) and
`libamdhip64` (HIP), and fall back to `rocm-smi` for AMD GPU discovery. The
`accelerator.cuda` doctor check id and the `linkage` dict shape are preserved
for schema compatibility.

### APU caveats (Radeon 8060S / gfx1151, unified memory)
- `rocm-smi --showmeminfo vram` reports only the 512 MB carved-out partition;
  `hipMalloc` actually uses the full ~128 GB unified pool (`hipMemGetInfo`).
  nvtop/`rocm-smi` therefore **cannot see** `hipMalloc` allocations — monitor
  with the engine's `[CUDA] resident set: … GB VRAM` log line instead.
- Don't use `--auto-tier` on the APU (it clamps VRAM to ~0 from the 512 MB
  figure). Pass `--vram N` directly. The VRAM expert tier is bounded by
  `PIN_GB` (candidate count) as well as `CUDA_EXPERT_GB` (byte budget); raise
  both, with an explicit `PIN=<usagefile>` to bypass AUTOPIN's history
  confidence cap, and `CUDA_RELEASE_HOST=1` for disjoint placement.

### Verification
- `make hip-test` — HIP backend correctness (q8/q4/q2/f32, expert MLP, grouped
  W4A32, MLA attention, placement, accounting).
- `make test-c` / `python3 -m unittest discover -s tests` — CPU + Python suite.
- End-to-end: `coli run|chat|serve --gpu 0 --vram N` on the real GLM-5.2 model.

## 4. Changelog

### v1.1.1 — 2026-07-14 — HIP decode tuning (CUDA_DENSE)
- **`coli` no longer force-enables `CUDA_DENSE=1` for the HIP backend.** On AMD
  APUs (unified memory), putting resident dense matmuls on the GPU adds a
  per-call H2D/D2H/sync tax that makes the GPU slower than the fast Zen CPU for
  single-token decode. Measured ~2× faster decode with dense on CPU
  (161–196s → 90–106s; attention 10.5→4.0s, expert-matmul 19–27→12s). The CUDA
  backend still defaults `CUDA_DENSE=1` (helps on dGPUs); HIP uses the engine
  default (0). Override with `CUDA_DENSE=1/0`.
- The GPU's correct role on this APU is the **expert tier only** (grouped /
  batched). Confirmed via `COLI_CUDA_PROFILE`: single-token decode is CPU +
  disk bound (`expert-disk` service dominates, `wait≈0` — I/O already
  overlapped via the PIPE thread pool + readahead).
- Investigated and **rejected** two Tier-1 candidates (documented in
  `docs/HIP-PERF-REVIEW.md`): fusing the shared expert (A) — no benefit (shared
  expert is tiny, ~6 MB/triple); attention on GPU (B) — no benefit (GPU kernel
  slower than CPU for decode). Both leave `glm.c` unchanged.

### v1.1.0 — 2026-07-14 — ROCm/HIP backend
- **New HIP backend** (`c/backend_hip.cpp`): faithful 1:1 port of
  `backend_cuda.cu` — general WxA32 GEMM, fused expert MLP, grouped expert
  dispatch (async pinned staging + event profiling), MLA weight-absorption
  attention, int4 `0x88` offset→signed conversion. The NVIDIA-only int4 WMMA
  tensor-core path is guarded out; the production W4A32 packed path is always
  used. `COLI_CUDA_TC_INT4` is accepted but a no-op on HIP.
- **`make HIP=1`** build path + `make hip-test` + `c/tests/test_backend_hip.cpp`.
- **CLI tooling HIP-aware**: `coli` (`gpu_backend()`, `coli build --hip`),
  `doctor.py` (`cuda_linkage()` detects `libamdhip64`), `resource_plan.py`
  (`discover_gpus()` ROCm/`rocm-smi` fallback). `glm.c` unchanged.
- **Docs**: `docs/CODE-REVIEW.md`, `docs/HIP-PORT-PLAN.md`,
  `docs/HIP-PORT-FINAL-REPORT.md`, `docs/HIP-RUN-GUIDE.md`,
  `docs/HIP-PERF-REVIEW.md`; README "ROCm/HIP backend" section.
- Verified end-to-end on AMD Ryzen AI MAX+ 395 (Radeon 8060S, gfx1151): full
  744B GLM-5.2 int4 generation via `coli run`/`serve` with the HIP backend.

### v1.0 — baseline
- Single-file C GLM-5.2 engine, streaming experts, MTP speculation, optional
  CUDA/Metal backends, `coli` CLI, OpenAI server.
