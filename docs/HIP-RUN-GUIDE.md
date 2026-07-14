# Running colibrì CLI with HIP support

## Build the HIP engine (once)

```bash
cd /home/ugo/software/colibri/c
export PATH=/opt/rocm/bin:$PATH          # so hipcc is found
make HIP=1                               # builds glm linked to libamdhip64
```

## Set the model + GPU env

```bash
M="/home/ugo/software/colibri/models/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp"
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
```

> Note: with `coli`, put `--model` **after** the subcommand (argparse quirk).

## Interactive chat (HIP)

```bash
./coli chat --model "$M" --gpu 0 --vram 8 --ram 96
```

- `--gpu 0` → enables the HIP backend (`COLI_CUDA=1 COLI_GPUS=0`)
- `--vram 8` → 8 GB expert tier on the GPU (the engine uses `hipMalloc`, which sees the full ~128 GB unified pool)
- `--ram 96` → RAM budget for the CPU expert cache
- type a message, Enter to send · `:q` to quit · `:reset` to clear memory

## OpenAI-compatible server (HIP)

```bash
./coli serve --model "$M" --gpu 0 --vram 8 --ram 96 --port 8000
```

Then from another terminal:

```bash
# list models
curl -s http://127.0.0.1:8000/v1/models

# chat completion
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2-colibri","messages":[{"role":"user","content":"Say hello"}],"max_tokens":16}'
```

It's also compatible with any OpenAI client (set `base_url=http://127.0.0.1:8000/v1`).

## Confirm it's really using HIP

Look for these lines in the output:

```
[HIP] device 0: Radeon 8060S Graphics, 131.0 GB VRAM, gfx115
[CUDA] resident set: 1891 tensors, 17.15 GB VRAM
[CUDA] hot expert tier: 422/2100 experts, VRAM 7.98 GB
```

(The `[CUDA]` labels are historical — the symbols are named `coli_cuda_*`; the actual backend is HIP.)

## One-shot generation (bonus)

```bash
./coli run --model "$M" --gpu 0 --vram 8 --ram 96 --ngen 32 "your prompt"
```

## APU caveat

`rocm-smi` reports only the 512 MB carved-out VRAM partition on this Radeon 8060S, so **don't use `--auto-tier`** (it clamps VRAM to ~0). Use `--vram N` directly as shown above — the engine validates the budget against `hipMemGetInfo` (131 GB available) and it works.

## Filling the GPU / reading memory usage

Two things trip people up on this APU:

**1. nvtop cannot see the engine's GPU allocations.** On the Radeon 8060S,
`hipMalloc` allocates from unified system memory, which ROCm reports *outside*
the dedicated-VRAM counter that nvtop/`rocm-smi --showmeminfo vram` read (that
counter is stuck at the 512 MB carveout). Verified: a 32 GB `hipMalloc` drops
`hipMemGetInfo` free by ~34 GB but leaves `rocm-smi` VRAM unchanged. **Monitor
with the engine log line instead** (it is backed by `hipMemGetInfo`):
```
[CUDA] resident set: 8212 tensors, 57.15 GB VRAM
```

**2. `--vram` alone does not fill the GPU.** The VRAM expert tier can only place
experts that are *pinned* (loaded as placement candidates). The pin count comes
from AUTOPIN (default), whose budget is capped by your usage-history confidence:
`pin_gb = expert_avail × 0.5 × (history/200000)`. With only ~16k selections
recorded, confidence is ~8%, so AUTOPIN pins just a few GB — raising `--vram`
then has no candidates to spend it on. `--vram` (`CUDA_EXPERT_GB`) is the
byte-budget ceiling; `PIN_GB` is the candidate-count ceiling — raise **both**,
and set an explicit `PIN=<usagefile>` to bypass the AUTOPIN confidence cap.
`CUDA_RELEASE_HOST=1` makes placement disjoint (VRAM experts release their RAM
copy, so memory isn't doubled).

**Command that actually fills the GPU** (≈48 GB of experts + ~10 GB dense on the GPU):
```bash
M="/home/ugo/software/colibri/models/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp"
PIN="$M/.coli_usage" PIN_GB=48 CUDA_RELEASE_HOST=1 \
  ./coli run --model "$M" --gpu 0 --vram 48 --ram 96 "your prompt"
```
Expected log:
```
[CUDA] hot expert tier: 2529/2537 experts, VRAM 47.99 GB (total budget 48.0 GB)
[CUDA] resident set: 8212 tensors, 57.15 GB VRAM
```
Scale `48` up (e.g. 64/80) as long as `hipMemGetInfo` free allows. The same
`PIN=`/`PIN_GB`/`CUDA_RELEASE_HOST=1` env vars work with `coli chat` and
`coli serve`. AUTOPIN confidence grows with usage (full at 200k selections),
so over time it will pin more on its own — use the explicit `PIN=` form to
force it now.
