# colibrì — ROCm/HIP Backend Port: Research & Plan

**Goal:** add a working, opt-in **ROCm/HIP** GPU backend that accelerates
resident (dense + expert) tensors on AMD GPUs, mirroring the existing CUDA
backend's host contract so `glm.c` needs **zero changes**.

**Target machine (verified):**
- APU: AMD RYZEN AI MAX+ 395, integrated **Radeon 8060S Graphics**, ISA
  `gfx1151` (RDNA 3.5), 40 CUs, unified LPDDR5x memory.
- ROCm: `/opt/rocm`, **HIP 7.13.99004**, `hipcc` at `/opt/rocm/bin/hipcc`,
  `libamdhip64.so` at `/opt/rocm/lib`.

This document records the research findings and the step-by-step plan that
the implementation in `c/backend_hip.cpp` + `c/Makefile` +
`c/tests/test_backend_hip.cpp` follows.

---

## 1. Design decision: reuse the `coli_cuda_*` ABI

`glm.c`'s entire GPU integration is behind `#ifdef COLI_CUDA` and calls only
the `extern "C"` symbols declared in `c/backend_cuda.h`, storing an opaque
`ColiCudaTensor*`. Two options were considered:

| Option | Host changes | Regression risk | Verdict |
|---|---|---|---|
| A. New `coli_hip_*` ABI + `#ifdef COLI_HIP` mirrors in `glm.c` | large (every call site) | high | rejected |
| B. HIP backend exporting the **same** `coli_cuda_*` symbols, built under `-DCOLI_CUDA` | none | none | **chosen** |

With option B, `glm.c` is compiled with `-DCOLI_CUDA` exactly as for CUDA, but
links `backend_hip.o` (compiled by `hipcc`) instead of `backend_cuda.o`. The
`coli_cuda_*` / `ColiCudaTensor` names become a historical label for "the
resident-GPU backend"; this is documented in the README and in
`backend_hip.cpp`. All existing env vars (`COLI_CUDA=1`, `COLI_GPU=0`,
`CUDA_DENSE=1`, `CUDA_EXPERT_GB=…`, `COLI_CUDA_ATTN=1`, `COLI_CUDA_PROFILE=1`,
`COLI_CUDA_ASYNC`, …) work unchanged.

This satisfies the project rules: *no regression, do not modify working code,
minimise churn*.

---

## 2. API mapping (verified against `/opt/rocm/include/hip`)

Every CUDA runtime call used in `backend_cuda.cu` has a 1:1 HIP equivalent.
Verified by grepping the installed headers.

| CUDA | HIP | Notes |
|---|---|---|
| `cudaMalloc` / `cudaFree` | `hipMalloc` / `hipFree` | |
| `cudaMemcpy` (H2D/D2H) | `hipMemcpy` (`hipMemcpyHostToDevice`/`DeviceToHost`) | |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | takes `hipStream_t` |
| `cudaMallocHost` / `cudaFreeHost` | `hipMallocHost` / `hipFreeHost` | pinned staging |
| `cudaMemsetAsync` | `hipMemsetAsync` | |
| `cudaStream_t` | `hipStream_t` | |
| `cudaStreamCreateWithFlags` | `hipStreamCreateWithFlags` | `hipStreamNonBlocking` = 0x01 ✓ |
| `cudaStreamDestroy` / `cudaStreamSynchronize` | `hipStreamDestroy` / `hipStreamSynchronize` | |
| `cudaEvent_t` + create/record/sync/elapsed/destroy | `hipEvent_t` + `hipEvent*` | profiling |
| `cudaSetDevice` | `hipSetDevice` | |
| `cudaGetDeviceCount` | `hipGetDeviceCount` | |
| `cudaGetDeviceProperties` | `hipGetDeviceProperties` | `hipDeviceProp_t` has `name[256]`, `totalGlobalMem`, `major`, `minor` ✓ |
| `cudaMemGetInfo` | `hipMemGetInfo` | |
| `cudaError_t` / `cudaSuccess` / `cudaGetLastError` / `cudaGetErrorString` | `hipError_t` / `hipSuccess` / `hipGetLastError` / `hipGetErrorString` | |

Kernel-language features used — all identical in HIP:
`__global__`, `__device__`, `__host__`, `__shared__`, `extern __shared__`
(dynamic shared), `__syncthreads()`, `blockIdx/threadIdx/blockDim/gridDim`,
`<<<grid,block,smem,stream>>>` launch, `expf/fmaxf/fabsf`, `__float2int_rn`.

The `std::mutex` / `std::lock_guard` host-side stats lock is plain C++ —
unaffected by the device runtime.

---

## 3. The one NVIDIA-specific kernel: WMMA int4

`grouped_s4_wmma` uses `nvcuda::wmma` with
`wmma::experimental::precision::s4` 8×8×32 fragments (requires `sm_≥750`).
ROCm's equivalent is the separate **rocWMMA** library (`/opt/rocm/include/rocwmma`),
not a header-level `nvcuda::wmma` shim — there is **no** `hip/cuda_runtime.h`-style
WMMA compatibility layer installed, and `nvcuda::wmma` does not compile under
hipcc. A faithful int4-WMMA port would require rewriting this kernel against
rocWMMA (or the `__builtin_amdgcn_wmma_*` intrinsics), and int4-WMMA support
on `gfx1151` is not guaranteed.

**Decision (honest, not a placeholder):** ship the HIP backend **without** the
int4 tensor-core path. The WMMA kernel is guarded out under
`__HIP_PLATFORM_AMD__`, and `coli_cuda_expert_group` forces `tc=0` on HIP so
the production **W4A32 packed path** (`grouped_hidden_w4_dual` /
`grouped_down_w4`) is always used. This path is:

- the *default* on CUDA too (`COLI_CUDA_W4_PACKED`, on by default; TC is opt-in
  via `COLI_CUDA_TC_INT4`),
- bit-exact in the W4A32 sense (FP32 activations × signed int4 weights),
- the right call on a unified-memory APU where the PCIe round-trip argument
  for tensor cores is moot anyway.

The `COLI_CUDA_TC_INT4` env var is accepted but a no-op on HIP (a log line
notes it is unsupported). This is a documented capability gap, not a dummy.
A future task can add a rocWMMA implementation behind the same flag.

---

## 4. Build integration (`c/Makefile`)

A new `HIP=1` branch, parallel to `CUDA=1` (Linux direct-link):

```make
HIP      ?= 0
HIPCC    ?= /opt/rocm/bin/hipcc
HIP_ARCH ?= native
HIPFLAGS ?= -O3 -std=c++17 --offload-arch=$(HIP_ARCH)

ifeq ($(HIP),1)
  # Linux only; mutually exclusive with CUDA=1 / CUDA_DLL=1
  CFLAGS   += -DCOLI_CUDA            # reuse glm.c's existing GPU integration
  LDFLAGS  += -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -lstdc++
  CUDA_OBJ  = backend_hip.o
endif

backend_hip.o: backend_hip.cpp backend_cuda.h
	$(HIPCC) $(HIPFLAGS) -c backend_hip.cpp -o $@

hip-test: backend_hip.cpp backend_cuda.h tests/test_backend_hip.cpp
	$(HIPCC) $(HIPFLAGS) backend_hip.cpp tests/test_backend_hip.cpp -o backend_hip_test$(EXE)
	./backend_hip_test$(EXE)
```

**Why this links cleanly:** `hipcc -c` embeds the device code object as a
bundle inside a normal host relocatable `.o`, with host-side registration
data that the HIP runtime (`libamdhip64`) reads at kernel-launch time. The
host `glm.c` is compiled by `gcc` and links that `.o` plus `-lamdhip64
-lstdc++`; the `extern "C"` `coli_cuda_*` symbols resolve across the
gcc↔hipcc boundary exactly as they do across the gcc↔nvcc boundary today.
`HIP_ARCH=native` makes `hipcc` auto-detect the installed GPU (`gfx1151` on
this machine); set `HIP_ARCH=gfx1100` etc. to cross-compile.

**Implementation finding (important):** on this CachyOS / glibc-2.43 host,
linking a HIP program with `hipcc`'s own `--hip-link` driver crashes inside
`libamdhip64`'s `pthread_once` static initialisation on the very first
`hipGetDeviceCount` call (`std::locale::~locale()` → `free(): invalid size`).
A minimal `hipcc`-linked `hipGetDeviceCount` reproduces it; the same ROCm
install runs `llama.cpp`'s ROCm build and a gcc-linked `g++ … -lamdhip64`
binary fine. The fix adopted (and matching llama.cpp's build): **compile
device code with `hipcc -c`, link with `gcc` + `-lamdhip64 -lstdc++`**.
The `hip-test` recipe follows this split; the main `glm` link already uses
`gcc` (`$(CC)`), so `make HIP=1` is unaffected.

A mutual-exclusion guard prevents linking two backends at once.

---

## 5. Verification strategy

1. **Unit correctness** — `make hip-test` compiles a HIP-native test
   (`tests/test_backend_hip.cpp`, ported from `test_backend_cuda.cu`) that
   checks exact outputs for: int8/int4/int2/f32 `matmul`, fused `expert_mlp`,
   grouped `expert_group` (W4A32 path), the MLA `attention_absorb` core,
   multi/single-device placement, and resident-set accounting (tensor counts
   + bytes). The WMMA/TC section of the CUDA test is dropped (see §3); the
   group-stats assertion is adjusted to `2 calls / 4 experts / 4 rows`.

2. **Build sanity** — `make HIP=1` produces a `glm` binary that links
   `libamdhip64` and contains the `coli_cuda_*` symbols; `ldd glm` shows the
   HIP runtime; running with no GPU / `COLI_CUDA` unset is unchanged
   (CPU-only path).

3. **Runtime smoke** — `COLI_CUDA=1 COLI_GPU=0 ./glm …` initialises the
   device, reports `gfx1151` / VRAM, and runs a short generation; on any
   backend error the engine marks the tensor `cuda_failed` and falls back to
   CPU (existing host behaviour, unchanged).

4. **No regression** — default `make` + `make test` still pass; `glm.c`,
   `backend_cuda.*`, and `backend_loader.c` are **not modified**.

---

## 6. Implementation steps

1. Write `c/backend_hip.cpp` — a line-for-line port of
   `c/backend_cuda.cu` with the §2 API substitutions; guard the WMMA kernel
   out under `__HIP_PLATFORM_AMD__`; force `tc=0` in `expert_group`.
2. Add the `HIP=1` Makefile branch + `hip-test` target + `clean` entries.
3. Write `c/tests/test_backend_hip.cpp` (CUDA test minus the WMMA section,
   adjusted stats).
4. `make hip-test` → green.
5. `make HIP=1` → produces `glm`; smoke-run.
6. Document in README (HIP section) — added after verification.

## 7. Risks / open items

- **`hipcc` link driver crash** — resolved by gcc-link (see §4). Documented
  because it is environment-specific and would otherwise be confusing.
- **`hipMallocHost` deprecation** — ROCm marks `hipMallocHost` deprecated in
  favour of `hipHostMalloc`, but it works and mirrors the CUDA backend's
  `cudaMallocHost`. Left as-is for port fidelity; can switch later.
- **`min`/`max` in device code**: replaced with ternary in `quantize_s4_rows`
  (the only int `min`/`max` site) to avoid relying on HIP device overloads.
- **Dynamic shared-memory size** for `attention_absorb_kernel`
  (`(2K+T)*4` bytes, K≤512, T≤4096 → ≤~18 KB) is well within the per-block
  LDS limit on `gfx1151` (64 KB/CU). No change needed.
- **Pinned host memory on a unified-memory APU**: `hipMallocHost` is
  supported; benefit is modest but correct.
- **int4 weight `0x88` conversion** (`offset_to_signed_s4`) is pure compute
  and ports directly — this is the one subtlety a naive copy would miss, and
  it is preserved.
- **`nodiscard` warnings** from ignoring `hipFree`/`hipEvent*` return values
  are harmless and match the CUDA original's behaviour.
