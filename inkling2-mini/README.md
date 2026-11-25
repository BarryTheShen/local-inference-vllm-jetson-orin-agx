# Inkling 2 Mini on the Jetson AGX Orin

This folder holds the production setup that replaced Qwen3.6-35B-A3B on the Jetson on 2026-11-23. Inkling 2 Mini is not supported by released vLLM on GPUs older than Hopper, so PyTorch, Triton, and vLLM are built from source for the Orin (SM 8.7) and patched.

## Results (Jetson AGX Orin 64 GB, 2026-11-23)

| Measurement | Untuned | This setup |
|---|---|---|
| One request, decode | 25.7 tokens/s | 40.5 tokens/s |
| 8 parallel requests, total | 75.5 tokens/s | 108.9 tokens/s |
| 128 parallel requests, total | – | 551.7 tokens/s |
| Perplexity (wikitext) | 7.34 | 7.34 |
| GSM8K, 5-shot, strict | 0.832 (250 questions) | 0.82–0.85 (100 questions, several runs) |

## Files

- `docker-compose.yml`: the production service (vLLM on port 8000, model name `inkling-2-mini`).
- `humming-orin.json`: measured expert-kernel tiles for the Orin, read through `VLLM_INKLING_HUMMING_TUNING`. The same tiles are proposed upstream in [vllm-project/humming#223](https://github.com/vllm-project/humming/pull/223).
- `patches/`: the changes to vLLM that production runs, in the order they apply:
  1. `upstream-fdda4ee-sconv-fix.patch`, `upstream-fdda4ee-tests.patch`: a later fix from vLLM PR #69165.
  2. `w8-int8-dense.patch`, `w8-retune-real-shapes.patch`: int8 copies of the attention and LM-head weights (`VLLM_INKLING_W8=attn,head`). The kernels are adapted from [lulakien/inkling2-mini-3090-recipe](https://github.com/lulakien/inkling2-mini-3090-recipe) (Apache-2.0).
  3. `humming-tuning-hook.patch`: loads `humming-orin.json`.
  4. `attn-prefill-launch.patch`: prefill attention launch for the Orin (`VLLM_INKLING_ATTN_PREFILL=32,4,2`).
  5. `fp8-kv-sm87.patch`: allows an fp8 KV cache on the Orin (not used by default).
  6. `router-stages-sm87.patch`: three pipeline stages for the router GEMM on the Orin.
- `build/`: the source build of PyTorch 2.14, Triton 3.8, and vLLM (PR #69165 at `aade5d0`) inside `nvcr.io/nvidia/l4t-jetpack:r36.4.0`.

## Setup

1. Copy `build/` and `patches/` to `~/inkling-build` on the Jetson and run `./run-build.sh` (several hours). Set `BUILD_PROXY` if downloads need a proxy.
2. Save the build container as the image: `docker commit inkling-build inkling-jetson:built`.
3. Apply the patches in the order above to the installed package (the tests patch applies only to the source checkout in `src/vllm`). The venv is owned by root, so run it in the image, for example: `docker run --rm -v ~/inkling-build:/build --entrypoint bash inkling-jetson:built -c 'cd /build/venv/lib/python3.12/site-packages && patch -p1 < /build/patches/w8-int8-dense.patch'`.
4. Copy `humming-orin.json` to `~/inkling-build/tuning/` and the model to `~/models/Inkling-2-Mini`.
5. Start it with `docker compose -f docker-compose.yml up -d`.

`HUMMING_COMPILER=nvcc` is required: JetPack's NVRTC 12.6 compiles bf16 arithmetic for SM 8.7 into traps (reported in [vllm-project/humming#222](https://github.com/vllm-project/humming/issues/222)).
