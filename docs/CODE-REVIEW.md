# colibrì — Code Review & Architecture Report

**Scope:** repository at `/home/ugo/software/colibri` (commit `6d3ed7e`, branch `main`).
**Goal of this document:** a precise, implementation-level description of what
colibrì is, how it works end-to-end, and how its optional GPU backends are
structured — with special attention to the CUDA backend, which is the
porting target for the ROCm/HIP work tracked in
[`docs/HIP-PORT-PLAN.md`](HIP-PORT-PLAN.md).

---

## 1. What colibrì is

colibrì is a **single-file C inference engine** for the **GLM-5.2** family of
models (HuggingFace architecture `glm_moe_dsa`: Mixture-of-Experts with
Multi-head Latent Attention, RoPE, a shared expert, and a DeepSeek-V3-style
MTP speculative head). Its defining constraint, stated in the README and the
header of `c/glm.c`, is **zero runtime dependencies**: no BLAS, no PyTorch, no
GPU required. The whole transformer forward — tokenisation, embedding, MLA
attention, MoE routing, RMSNorm, RoPE, sampling, speculative decoding — lives
in `c/glm.c` (~3,900 lines) plus a handful of small headers.

The trade-off that makes a ~17B-active-parameter MoE model fit on a laptop:
every weight is **quantised on load** to int8 / int4 / int2 with a per-row
float scale, and dequantised *on use* inside the integer dot-product kernels.
Streaming experts are paged in from NVMe on demand and cached in RAM with an
LRU/LFRU policy; resident dense tensors (attention, lm_head, embed, dense MLP,
shared expert) stay quantised in RAM.

A CUDA backend (`c/backend_cuda.cu`) and a Metal backend
(`c/backend_metal.mm`) are **opt-in accelerators** for resident tensors. They
are *not* required to run, and the default build (`make`) is pure C.

### 1.1 Repository layout (relevant parts)

```
c/
├── glm.c                 # the engine (single translation unit)
├── st.h                  # shard store: mmap/pread weight container
├── tok.h, tok_unicode.h  # BPE tokenizer + tables
├── json.h                # minimal JSON reader/writer
├── compat.h              # POSIX/Windows portability shims
├── grammar.h             # grammar-constrained drafting (method F)
├── tier.h                # hardware tier heuristics
├── decode_batch.h        # continuous-batching bookkeeping
├── backend_cuda.{h,cu}   # opt-in CUDA backend      <-- port target's sibling
├── backend_loader.c      # Windows runtime DLL loader for coli_cuda.dll
├── backend_metal.{h,mm}  # opt-in Apple Metal backend
├── Makefile              # build system (OS/CUDA/Metal detection)
├── coli                  # Python launcher (wraps glm, doctor, plan, serve)
├── openai_server.py      # OpenAI-compatible HTTP server
├── doctor.py, resource_plan.py
├── tools/                # converters, oracle generator, benchmarks
└── tests/                # C unit tests + test_backend_cuda.cu
```

---

## 2. The engine (`c/glm.c`)

### 2.1 Model representation

`Cfg` (line ~70) mirrors the HuggingFace config: `hidden`, `n_layers`,
`n_heads`, `n_experts`, `topk`, `q_lora`, `kv_lora`, `qk_nope`, `qk_rope`,
`qk_head`, `v_head`, `first_dense`, `moe_inter`, `dense_inter`,
`routed_scale`, `theta`, `attn_scale`, `index_topk` (DSA lightning
indexer), and the MTP head flag.

The central weight container is `QT` (line ~98):

```c
typedef struct {
    int fmt;            // 0=F32, 1=INT8, 2=INT4 (2/byte), 3=INT2 (4/byte)
    float *qf;          // fmt==0
    int8_t *q8;         // fmt==1
    uint8_t *q4;        // fmt==2 or 3 (packed)
    float *s;           // per-row scale [O]
    int O, I;
#ifdef COLI_CUDA
    ColiCudaTensor *cuda;          // resident device copy (opaque)
#endif
    int cuda_eligible, cuda_failed, cuda_device;
} QT;
```

`qt_bytes()` computes resident footprint; the four formats share one struct
and dispatch on `fmt` everywhere. `Layer` holds the MLA projections
(`q_a,q_b,kv_a,kv_b,o` + their layernorms), and either a dense MLP
(`gate_proj/up_proj/down_proj`) or a router + shared expert
(`sh_gate/sh_up/sh_down`) for sparse layers.

### 2.2 Forward pipeline

`model_init` (line ~992) loads config + shard store, then `qt_load`s every
resident tensor through the shard store (`st.h`). Experts are *not* loaded
here; they live in on-disk shards and are paged per-layer/per-token.

The main loop (bottom of `glm.c`, `main`) runs:

1. **Tokenise** prompt → ids.
2. **Prefill** (`step_all`): batched forward over the prompt, filling the
   compressed MLA KV cache (`Lc`/`Rc` — only the latent + rope parts per
   token, not the full 64-head KV).
3. **Decode** (`step`): single-token forward; router selects `topk` experts
   per layer, loads them (cache hit / NVMe miss), runs the MoE, attention
   with the compressed cache.
4. **MTP speculative decoding** (optional, `g_draft`): the MTP head drafts
   tokens that are verified in one batched forward; acceptance measured.
5. **Sample** (greedy / top-p / grammar-constrained) → emit token.

### 2.3 Quantised matmul kernels

`matmul`, `matmul_q` (int8), `matmul_i4` (int4), and the IDOT
(`idot`) variants use hand-written SIMD: AVX2/AVX-512 (with the AVX-VNNI
`VPDPBUSD` int8/int4 dot on Alder Lake+), ARM NEON/SDOT, and POWER VSX. A
portable scalar tail handles the remainder. `matmul_qt` (line ~703) is the
dispatcher: it tries Metal, then CUDA, then the CPU integer kernels based on
shape/format thresholds (`g_i4s`, `g_idot`).

### 2.4 Expert streaming & caching

`ESlot` (line ~145) is a reusable expert slot holding `g,u,d` QTs backed by a
single coalesced `slab` buffer (one `pread` per expert). The engine keeps a
per-layer LRU cache (`ecache`/`ecn`), a hot-store of pinned experts
(`pin`/`npin`), persistent usage counters (`eusage`), and recency heat
(`eheat`/`elast`) for live promotion/demotion. `COLI_MMAP` optionally mmaps
shards instead of pread.

---

## 3. The GPU backend contract (`c/backend_cuda.h`)

The host never calls CUDA/HIP runtime APIs directly. It calls a small
`extern "C"` ABI defined in `backend_cuda.h`. This is the contract any GPU
backend must implement:

```c
typedef struct ColiCudaTensor ColiCudaTensor;   // opaque, device-resident

int  coli_cuda_init(const int *devices, int count);
void coli_cuda_shutdown(void);
int  coli_cuda_device_count(void);
int  coli_cuda_device_at(int index);
int  coli_cuda_mem_info(int device, size_t *free, size_t *total);
void coli_cuda_stats(int device, size_t *count, size_t *bytes);
void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);

/* upload (lazy, idempotent) a resident [O,I] quantised weight tensor */
int  coli_cuda_tensor_upload(ColiCudaTensor **t, const void *weights,
                             const float *scales, int fmt, int I, int O, int device);

/* y[S,O] = x[S,I] @ W[O,I]^T ; first call uploads W, later calls reuse */
int  coli_cuda_matmul(ColiCudaTensor **t, float *y, const float *x,
                      const void *weights, const float *scales,
                      int fmt, int S, int I, int O, int device);

/* fused expert MLP: y = down(silu(gate(x)) * up(x)) */
int  coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                          ColiCudaTensor *down, float *y, const float *x, int S);

/* packed group of same-shaped experts (one H2D/D2H round-trip) */
int  coli_cuda_expert_group(ColiCudaTensor *const *gates,
                            ColiCudaTensor *const *ups,
                            ColiCudaTensor *const *downs,
                            const int *rows, int count, float *y, const float *x);

/* decode-only MLA weight-absorption core for one token */
int  coli_cuda_attention_absorb(ColiCudaTensor *kv_b, float *ctx, const float *q,
                                const float *latent, const float *rope,
                                int H,int Q,int R,int V,int K,int T,float scale);

void  coli_cuda_tensor_free(ColiCudaTensor *t);
size_t coli_cuda_tensor_bytes(const ColiCudaTensor *t);
int    coli_cuda_tensor_device(const ColiCudaTensor *t);
```

Key design points:

- **`fmt` matches `QT.fmt`**: 0=f32, 1=int8, 2=int4 (2/byte, low nibble first),
  3=int2 (4/byte). Scales are per-output-row floats.
- **Lazy upload**: `coli_cuda_matmul` / `expert_*` call `tensor_upload` on the
  opaque handle; the first call allocates+copies device memory, later calls
  reuse it. This is why capacity failures surface at model start, not mid-run.
- **Opaque handle**: the host stores `ColiCudaTensor*` but never dereferences
  it — that is what lets the Windows DLL split (MSVC `nvcc` ↔ MinGW `gcc`)
  work, and what lets a HIP backend drop in with **zero host changes**.
- **Synchronous semantics per device**: each `DeviceContext` owns one stream;
  `matmul`/`expert_mlp`/`attention_absorb` are synchronous (upload → launch →
  download). Only `expert_group` uses async pinned staging + profiling events
  (gated by `COLI_CUDA_ASYNC`, default on).

### 3.1 Host-side integration (`glm.c`)

All host touch-points are behind `#ifdef COLI_CUDA`:

- **Init / CLI** (line ~3738): `COLI_CUDA=1` enables the backend;
  `COLI_GPU=0` / `COLI_GPUS=0,1,2` select ordinals; `CUDA_DENSE=1` also
  places resident dense tensors on-device (round-robin); `CUDA_EXPERT_GB`
  sets a per-tier VRAM budget for expert residency; `CUDA_RELEASE_HOST=1`
  frees the RAM copy after a successful upload (multi-GPU default).
- **`matmul_qt`** (line ~710): for resident `cuda_eligible` tensors, tries
  `coli_cuda_matmul` first; on failure marks `cuda_failed` and falls back to
  CPU. Nested OpenMP calls stay on CPU (single stream per device).
- **`qt_load`** (line ~987): with `CUDA_DENSE`, tags each resident tensor
  `cuda_eligible` and round-robin-assigns `cuda_device`, projecting VRAM use.
- **MoE dispatch** (line ~1982): routed experts whose `g/u/d` are all
  `cuda_eligible` are batched per-device into `coli_cuda_expert_group`
  (one H2D/D2H round-trip for the whole group); fall-back to per-expert
  `coli_cuda_expert_mlp` then CPU. Multi-device groups run in an OpenMP
  parallel loop.
- **Attention** (line ~1643): `COLI_CUDA_ATTN=1` (opt-in, S≤4) routes the
  MLA weight-absorption core to `coli_cuda_attention_absorb`.
- **Stats** (line ~196 / ~2804): `cuda_stats_print` reports resident set +
  group-call timing (`COLI_CUDA_PROFILE`).

---

## 4. The CUDA backend (`c/backend_cuda.cu`) — detailed

This is the file the HIP port translates. It is ~700 lines of C++17 + CUDA.

### 4.1 State

```c
struct ColiCudaTensor { void *weights; float *scales; size_t weight_bytes;
                        int fmt, I, O, device, tracked; };

typedef struct {
    int device;
    float *x,*y,*gate,*up;     size_t x_cap,y_cap,gate_cap,up_cap;   // device scratch
    uint8_t *qx; float *qscale; size_t qx_cap,qscale_cap;            // int4 act scratch
    float *host_x,*host_y;     size_t host_x_cap,host_y_cap;         // pinned staging
    float *aq,*al,*ar,*ac;     size_t aq_cap,al_cap,ar_cap,ac_cap;   // attention scratch
    cudaStream_t stream;
    void *group_desc; size_t group_desc_cap;                          // device GroupDesc[]
    size_t tensor_count, tensor_bytes;
} DeviceContext;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];   // up to 16 devices
static int g_nctx;
```

`GroupDesc` packs a grouped expert's three weight pointers + scales + formats +
row count + offset, copied to device once per `expert_group` call.

### 4.2 Weight packing helpers

- `row_bytes(fmt,I)`: bytes per weight row. f32=`I*4`, int8=`I`,
  int4=`(I+1)/2`, int2=`(I+3)/4`.
- `weight_at(weights,fmt,row,i)` (`__device__`): returns the dequantised
  *integer* value (not yet scaled) of element `i` of row `row` (row given in
  **bytes**). int4 unpacks low nibble first with two's-complement sign
  extension; int2 maps the 2-bit code `c` → `c-2`.
- `offset_to_signed_s4`: a one-time `__global__` pass that XORs every int4
  weight byte with `0x88`, converting the *offset* encoding used on disk
  (0..15, where 8==0) into two's-complement signed nibbles (-8..7). This lets
  the W4A32 kernels use a single `unpack_s4` that works on signed data, and
  is what the WMMA s4 path also expects. **Run once at upload for fmt==2.**

### 4.3 Kernels

1. **`quant_matmul`** — the general WxA32 GEMM. Grid `(O,S)`, block 256.
   Each block computes one `y[s,o]`: threads strided over `I` accumulate
   `x[s,i]*weight_at(...)`, then a 256-wide shared-memory tree reduction,
   then `y = sum * (fmt ? scales[o] : 1)`. Used by `matmul` and `expert_mlp`.

2. **`silu_mul`** — elementwise `gate = silu(gate)*up`. Block 256.

3. **`quantize_s4_rows`** — per-row absmax int4 quantisation of activations
   (W4A4 path). Block 256, shared-memory absmax reduction, writes packed
   nibbles + per-row scale. Only used by the tensor-core path.

4. **`grouped_s4_wmma`** — the **int4 tensor-core** (WMMA) path, guarded by
   `__CUDA_ARCH__ >= 750` (Turing+). Uses `nvcuda::wmma` 8×8×32
   `experimental::precision::s4` fragments for both A (activations) and B
   (weights), `int` accumulator, 8 warps/block tiling `O` in tiles of 64.
   Opt-in via `COLI_CUDA_TC_INT4` (+ `COLI_CUDA_TC_MIN_ROWS`), and only when
   all experts are s4 and `D%32==I%32==0`. **This is the only NVIDIA-specific
   kernel**; everything else is portable compute.

5. **`grouped_hidden` / `grouped_down`** — generic dequant GEMM for the
   grouped path (any fmt), grid `(O,max_rows,count)`, block 256, shared-mem
   reduction. Reads weights via `weight_at`.

6. **`grouped_hidden_w4` / `grouped_hidden_w4_dual` / `grouped_down_w4`** —
   the **exact W4A32 packed path** for int4 experts. Same grid/shape as above
   but consume each packed byte once via `unpack_s4` (both nibbles at once),
   preserving FP32 activations. `_dual` fuses gate+up projection in one pass
   over the activation row (halves memory traffic). This is the *default*
   grouped path (`COLI_CUDA_W4_PACKED`, default on) and is what makes int4
   MoE fast without tensor cores.

7. **`attention_absorb_kernel`** — MLA weight-absorption for one token.
   Grid `H` (one block per head), block 256, dynamic shared mem
   `(2K+T)*sizeof(float)` holding `qa[K]`, `cl[K]`, `scores[T]`. Phases:
   (a) build absorbed query `qa[k]=Σ_d q[d]*W[rbase+d,k]*scale`; (b) score
   every token `scores[t]=Σ_k qa[k]*latent[t,k] + Σ_d q[Q+d]*rope[t,d]`,
   times `scale`; (c) softmax (single-threaded over T≤4096); (d) context
   `cl[k]=Σ_t scores[t]*latent[t,k]`; (e) value projection
   `ctx[h,v]=Σ_k cl[k]*W[rbase+Q+v,k]*scale`. Constraints: `K≤512`,
   `T≤4096` (enforced by the host caller).

### 4.4 Host functions

- **`coli_cuda_init`**: validates ordinals, creates one non-blocking
  `cudaStream` per device, prints `name/VRAM/sm_xx`.
- **`tensor_upload`**: `cudaMalloc`+`cudaMemcpy` weights; for fmt==2 runs
  `offset_to_signed_s4`; for fmt!=0 also uploads `O` scales. Idempotent
  (returns existing tensor if shape/fmt/device match). Tracks count+bytes.
- **`matmul`**: reserve x/y scratch, H2D x, launch `quant_matmul`, D2H y.
- **`expert_mlp`**: three `quant_matmul` launches + `silu_mul`, one H2D/D2H.
- **`expert_group`**: the performance-critical path. Builds `GroupDesc[64]`,
  copies to device, then selects one of three strategies:
  - **TC int4** (`COLI_CUDA_TC_INT4`): quantise activations s4, run
    `grouped_s4_wmma` for gate/up, `silu_mul`, re-quantise, `grouped_s4_wmma`
    for down.
  - **W4A32 packed** (default, all-s4): `grouped_hidden_w4_dual` (or
    separate) + `silu_mul` + `grouped_down_w4`.
  - **generic**: `grouped_hidden`×2 + `silu_mul` + `grouped_down`.
  Uses async pinned staging (`COLI_CUDA_ASYNC`, default on) and optional
  `cudaEvent` profiling (`COLI_CUDA_PROFILE`).
- **`attention_absorb`**: async H2D of q/latent/rope, launch, async D2H,
  sync.
- **`tensor_free`**: decrements stats, `cudaFree` weights+scales.

### 4.5 Build & test (`c/Makefile`)

- `make` → pure-C `glm` (default).
- `make CUDA=1` → compiles `backend_cuda.cu` with `nvcc` (`-arch=$(CUDA_ARCH)`,
  default `native`) into `backend_cuda.o`, links it directly into `glm` with
  `-lcudart -lstdc++`, and adds `-DCOLI_CUDA` so `glm.c`'s `#ifdef COLI_CUDA`
  integration activates. **Linux only.**
- Windows: `make CUDA_DLL=1 cuda-dll` builds `coli_cuda.dll` (nvcc+MSVC) and
  `make glm.exe CUDA_DLL=1` links `backend_loader.o` (runtime
  `LoadLibrary`/`GetProcAddress`).
- `make cuda-test` → compiles `backend_cuda.cu` + `tests/test_backend_cuda.cu`
  and runs q8/q4/q2/f32 + expert-group + attention + WMMA correctness.
- `make cuda-bench` → tensor-core micro-benchmark.

`tests/test_backend_cuda.cu` is the correctness oracle: it checks exact
outputs for every format, multi-device placement, the fused expert MLP, the
grouped expert path, the MLA absorb core, the WMMA path (relative RMS ≤0.30
vs the W4A32 path), and the resident-set accounting (tensor counts + bytes
per device).

---

## 5. Why a HIP port is straightforward

1. **The host contract is backend-agnostic.** `glm.c` only ever calls the
   `coli_cuda_*` ABI and stores an opaque `ColiCudaTensor*`. A HIP backend
   implementing the *same symbols* links in with **zero changes to `glm.c`**
   — it just needs `-DCOLI_CUDA` (reusing the existing integration) and a
   different object.
2. **Only one kernel is NVIDIA-specific** (`grouped_s4_wmma`, via
   `nvcuda::wmma`). It is *opt-in* (`COLI_CUDA_TC_INT4`, off by default). The
   production path is W4A32 packed compute, which is fully portable.
3. **HIP is a near drop-in for the CUDA runtime API** used here:
   `cudaMalloc→hipMalloc`, `cudaMemcpy{,Async}→hipMemcpy{,Async}`,
   `cudaMallocHost/FreeHost→hipMallocHost/FreeHost`, `cudaStream_t→hipStream_t`,
   `cudaEvent*→hipEvent*`, `cudaGetDeviceProperties→hipGetDeviceProperties`
   (same `name`/`totalGlobalMem`/`major`/`minor` fields), `cudaMemGetInfo→
   hipMemGetInfo`, `cudaSetDevice→hipSetDevice`, `cudaSuccess→hipSuccess`,
   `cudaGetErrorString→hipGetErrorString`. The `<<<grid,block,smem,stream>>>`
   launch syntax, `__global__`/`__device__`/`__shared__`/`__syncthreads`/
   `blockIdx`/`threadIdx`/`extern __shared__` are all identical in HIP.
4. **The build is already structured for it**: the Makefile has a clean
   `CUDA=1` direct-link branch; a parallel `HIP=1` branch that compiles a
   `backend_hip.cpp` with `hipcc` and links `-lamdhip64` is a small,
   isolated addition.

The detailed plan, risks, and verification strategy are in
[`docs/HIP-PORT-PLAN.md`](HIP-PORT-PLAN.md).
