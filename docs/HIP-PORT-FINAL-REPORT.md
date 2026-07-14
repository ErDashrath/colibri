# colibrì — ROCm/HIP Backend: Final Report

**Date:** 2026-07-14
**Repository:** `/home/ugo/software/colibri` (branch `main`, commit `6d3ed7e`)
**Target machine:** AMD Ryzen AI MAX+ 395, integrated Radeon 8060S Graphics
(ISA `gfx1151`, RDNA 3.5), 128 GB unified LPDDR5x. ROCm 7.13 / HIP 7.13
(`/opt/rocm`, `hipcc` at `/opt/rocm/bin/hipcc`, `libamdhip64.so` at
`/opt/rocm/lib`).
**Model used for end-to-end validation:** the full GLM-5.2 int4 + int8-MTP
checkpoint (384 GB, 144 shards) at
`models/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`.

This report consolidates the whole effort: a code review, the porting
research/plan, the implementation, and the end-to-end verification that every
colibrì CLI subcommand runs on the HIP backend.

---

## 1. Objective

Add a working, opt-in **ROCm/HIP** GPU backend that accelerates resident
(dense + expert) tensors on AMD GPUs, mirroring the existing CUDA backend's
host contract so the core engine (`glm.c`) needs **zero changes**, and make
the project's own CLI tooling (`coli`, `doctor.py`, `resource_plan.py`,
`openai_server.py`) HIP-aware so all subcommands run with HIP support.

---

## 2. Background & code review (full text: `docs/CODE-REVIEW.md`)

colibrì is a single-file C inference engine for GLM-5.2 (`glm_moe_dsa`:
MoE + Multi-head Latent Attention + RoPE + shared expert + DeepSeek-V3-style
MTP head). Zero runtime dependencies; weights are quantised (int8/int4/int2 +
per-row scale) and dequantised on use. Streaming experts are paged from NVMe
and cached in RAM (LRU/LFRU); resident dense tensors stay quantised in RAM.

The GPU backend contract (`c/backend_cuda.h`) is a small `extern "C"` ABI
(`coli_cuda_init`, `coli_cuda_matmul`, `coli_cuda_expert_mlp`,
`coli_cuda_expert_group`, `coli_cuda_attention_absorb`, …) with an **opaque**
`ColiCudaTensor*` handle. The host stores the handle but never dereferences
it — this is what lets a different backend drop in with no host changes.

`backend_cuda.cu` implements: a general WxA32 GEMM (`quant_matmul`), fused
expert MLP, grouped expert dispatch (async pinned staging + event profiling),
the MLA weight-absorption attention core, an int4 on-disk offset→signed
conversion (`offset_to_signed_s4`, XOR `0x88` at upload), and one
NVIDIA-only int4 tensor-core (WMMA) path (`grouped_s4_wmma`, opt-in via
`COLI_CUDA_TC_INT4`, off by default). The production default is the **W4A32
packed path** (`grouped_hidden_w4_dual` / `grouped_down_w4`).

---

## 3. Porting research & plan (full text: `docs/HIP-PORT-PLAN.md`)

**Design decision:** the HIP backend exports the **same** `coli_cuda_*`
symbols. `glm.c` is compiled with `-DCOLI_CUDA` (reusing its existing
`#ifdef COLI_CUDA` integration verbatim) and links `backend_hip.o` instead of
`backend_cuda.o`. The `cuda` in the symbol names becomes a historical label
for "the resident-GPU backend". Zero host changes, zero regression risk.

**API mapping (verified against `/opt/rocm/include/hip`):** every CUDA runtime
call has a 1:1 HIP equivalent (`cudaMalloc→hipMalloc`,
`cudaMemcpy{,Async}→hipMemcpy{,Async}`, `cudaMallocHost/FreeHost→hipMallocHost
/FreeHost`, `cudaStream_t→hipStream_t`, `cudaEvent*→hipEvent*`,
`cudaGetDeviceProperties→hipGetDeviceProperties` — same `name`/
`totalGlobalMem`/`major`/`minor` fields, `cudaMemGetInfo→hipMemGetInfo`,
`cudaSuccess→hipSuccess`, …). Kernel language (`__global__`, `__shared__`,
`__syncthreads`, `blockIdx`/`threadIdx`, `extern __shared__`, `<<<>>>`
launch) is identical.

**The one NVIDIA-specific kernel:** `grouped_s4_wmma` uses
`nvcuda::wmma` `experimental::precision::s4` fragments (sm_≥750), with no
header-level HIP equivalent (ROCm's WMMA lives in the separate `rocwmma`
library). Decision: ship **without** the int4 tensor-core path on HIP —
`coli_cuda_expert_group` forces `tc=0`, so the production W4A32 packed path is
always used (it is the CUDA default too). `COLI_CUDA_TC_INT4` is accepted but
logged as ignored. This is a documented capability gap, not a placeholder.

---

## 4. Implementation

### 4.1 New files

| File | Lines | Purpose |
|---|---|---|
| `c/backend_hip.cpp` | 561 | Faithful 1:1 port of `backend_cuda.cu`: general WxA32 GEMM, fused expert MLP, grouped expert dispatch (async pinned staging + event profiling), MLA attention-absorb core, int4 `0x88` conversion. WMMA kernel guarded out; `tc=0` forced. |
| `c/tests/test_backend_hip.cpp` | 156 | Correctness oracle (q8/q4/q2/f32 matmul, expert MLP, grouped W4A32 path, MLA attention, device placement, resident-set accounting). WMMA section dropped; group-stats adjusted. |
| `docs/CODE-REVIEW.md` | 374 | Detailed code review / architecture report. |
| `docs/HIP-PORT-PLAN.md` | 208 | Research, API mapping, plan, risks. |

### 4.2 Modified files

- **`c/Makefile`** — new `HIP=1` branch (Linux direct-link, mutually exclusive
  with `CUDA=1`/`CUDA_DLL=1`); `backend_hip.o` recipe (`hipcc -c`);
  `hip-test` target; `clean` entries. Adds `HIPHOST`, `HIP_HOME`, `HIP_ARCH`
  (default `native` → auto-detects `gfx1151`).
- **`c/coli`** — `gpu_backend()` returns `'cuda'`/`'hip'`/`None` (detects
  `libamdhip64`); `cuda_binary()` wraps it; `coli build --hip` / `COLI_HIP=1`
  builds `make HIP=1`; `--gpu`/`--vram` error messages mention both backends.
- **`c/doctor.py`** — `cuda_linkage()` detects both libs, adds a `backend`
  field (preserves the `linked`/`missing` shape and the test-pinned "CUDA
  runtime library is missing" string); pass/warn labels are backend-aware.
- **`c/resource_plan.py`** — `discover_gpus()` falls back to `rocm-smi --json`
  (VRAM + `Card Series` name) when `nvidia-smi` is absent.
- **`c/.gitignore`** — ignores `backend_hip.o`, `backend_hip_test`,
  `tests/test_backend_hip.o`, etc.
- **`README.md`** — new "Experimental ROCm/HIP backend (AMD GPUs)" section.

`glm.c`, `backend_cuda.*`, and `backend_loader.c` were **not modified**.

### 4.3 Key build finding

On this CachyOS / glibc-2.43 host, linking a HIP program with `hipcc`'s own
`--hip-link` driver crashes inside `libamdhip64`'s `pthread_once` static
initialisation on the first `hipGetDeviceCount` (`std::locale::~locale()` →
`free(): invalid size`). A minimal `hipcc`-linked program reproduces it; the
same ROCm install runs `llama.cpp`'s ROCm build and a gcc-linked binary fine.
**Fix (matching llama.cpp's ROCm build):** compile device code with
`hipcc -c`, link with `gcc` + `-lamdhip64 -lstdc++ -lm`. The `glm` link already
uses `gcc` (`$(CC)`), so `make HIP=1` is unaffected; only `hip-test` needed
this split.

---

## 5. Verification

### 5.1 Unit correctness
- `make hip-test` → **passes**: `hip backend: q8/q4/q2/f32 correctness ok on 1
  device(s)` (int8/int4/int2/f32 matmul, fused expert MLP, grouped W4A32 path,
  MLA attention absorb, device placement, resident-set accounting).

### 5.2 Build & linkage
- `make HIP=1` → `glm` linked to `libamdhip64.so.7`, exports `coli_cuda_init`.
- `gpu_backend()` → `'hip'`; `cuda_binary()` → `True`.

### 5.3 No regression
- Default `make` + `make test-c` → all C tests pass.
- `python3 -m unittest discover` → **59/59 Python tests pass** (doctor +
  resource_plan schema preserved).

### 5.4 CLI on the HIP backend (real 744B GLM-5.2 model)

| Command | Result |
|---|---|
| `coli build --hip` | Built `glm` linked to `libamdhip64`. |
| `coli info` | model/shards/RAM/engine reported. |
| `coli doctor --gpu 0` | `[ok] accelerator.cuda HIP engine and devices are available`. |
| `coli plan --gpu 0 --vram 8` | Plan produced. |
| `coli run --gpu 0 --vram 8 "Say hello…"` | Full end-to-end generation on the AMD GPU. |

`coli run` log highlights:
```
[HIP] device 0: Radeon 8060S Graphics, 131.0 GB VRAM, gfx115
loaded in 5.00s | resident dense: 9912.75 MB | layers=78 experts=256
[CUDA] resident set: 1891 tensors, 17.15 GB VRAM
[CUDA] expert groups: 605 call, 1423 expert, 2129 righe (2.35 expert/call)
CUDA expert tier: 422 resident experts (7.98 GB) | 1423 calls served from VRAM
Hello, how can I help you today?
9 tokens in 81.87s (0.11 tok/s) | expert hit rate 50.6% | RSS 79.50 GB
```

`coli serve --gpu 0 --vram 8` (OpenAI-compatible server) on HIP:
```
[HIP] device 0: Radeon 8060S Graphics, 131.0 GB VRAM, gfx115
[CUDA] hot expert tier: 422/2100 experts, VRAM 7.98 GB
GET  /v1/models            → {"id":"glm-5.2-colibri",...}
POST /v1/chat/completions  → {"content":"hi"}  (correct)
```

**Every CLI subcommand** (`build`, `info`, `plan`, `doctor`, `run`, `serve`,
`chat`) runs with HIP support.

---

## 6. Known limitations / future work

1. **int4 tensor-core (WMMA) path unavailable on HIP.** The production W4A32
   packed path is used instead (the CUDA default too). A rocWMMA
   implementation could fill this gap behind the same `COLI_CUDA_TC_INT4`
   flag.
2. **APU VRAM reporting.** `rocm-smi` reports only the 512 MB carved-out VRAM
   partition on the Radeon 8060S, so `coli plan --auto-tier` clamps the VRAM
   tier to ~0. The engine's `hipMalloc` can use the full ~128 GB unified pool
   (`hipMemGetInfo` reports 131 GB), so on this APU pass `--vram N` **without**
   `--auto-tier`. Future improvement: have `discover_gpus` query
   `hipMemGetInfo` for an accurate unified-memory figure.
3. **`hipcc` link-driver crash** on glibc-2.43 — worked around by gcc-link
   (§4.3). Environment-specific.
4. **`hipMallocHost` deprecation** warning (ROCm prefers `hipHostMalloc`) —
   left as-is for port fidelity; works correctly.
5. **Performance.** 0.11 tok/s on a cold cache is disk-bound (50.6% expert hit
   rate), as expected for a first run of a 384 GB model — not a HIP
   regression. The HIP dense + expert tier executed correctly
   (1423 expert calls served from VRAM).

---

## 7. How to use

```bash
cd c
export PATH=/opt/rocm/bin:$PATH
make hip-test                       # correctness
make HIP=1                          # build glm linked to libamdhip64
# or: coli build --hip

M=/path/to/glm52_i4
COLI_CUDA=1 COLI_GPU=0 CUDA_DENSE=1 SNAP=$M ./glm 64 4 4          # direct
./coli run    --model $M --gpu 0 --vram 8 --ngen 32 "prompt"      # CLI one-shot
./coli serve  --model $M --gpu 0 --vram 8 --port 8000             # OpenAI server
./coli doctor --model $M --gpu 0                                  # diagnostics
```

Env vars are unchanged from CUDA (`COLI_CUDA=1`, `COLI_GPU=0`/`COLI_GPUS`,
`CUDA_DENSE=1`, `CUDA_EXPERT_GB=…`, `COLI_CUDA_ATTN=1`,
`COLI_CUDA_PROFILE=1`, …); the HIP backend interprets them identically.

---

## 8. Files changed

```
Modified:
  README.md            +43   new "ROCm/HIP backend" section
  c/.gitignore         +3    ignore HIP build artifacts
  c/Makefile           +55   HIP=1 branch, hip-test, clean
  c/coli               +28   gpu_backend(), coli build --hip, HIP-aware errors
  c/doctor.py          +24   cuda_linkage() detects HIP, backend-aware labels
  c/resource_plan.py   +39   discover_gpus() ROCm/rocm-smi fallback

New:
  c/backend_hip.cpp          561 lines — HIP backend (ports backend_cuda.cu)
  c/tests/test_backend_hip.cpp  156 lines — correctness test
  docs/CODE-REVIEW.md        374 lines — architecture/code review
  docs/HIP-PORT-PLAN.md      208 lines — research & plan
```

Untouched (no regression): `c/glm.c`, `c/backend_cuda.{h,cu}`,
`c/backend_loader.c`, `c/backend_metal.*`.
