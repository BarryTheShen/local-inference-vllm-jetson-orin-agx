# Qwen3.6 35B-A3B INT8 on Jetson AGX Orin

This repository contains the reproducible vLLM deployment and benchmark tooling for the local **Qwen3.6-35B-A3B INT8 AutoRound** checkpoint on a Jetson AGX Orin 64GB.

## Scope and constraints

- Inference engine: **vLLM only**.
- Weight precision: **INT8/W8A16 AutoRound** or higher. The deployed checkpoint is not replaced with Q4 weights.
- KV cache: FP8 is the supported compressed-cache path in the pinned Jetson image. Q4/TurboQuant is not enabled because the installed vLLM 0.19.0 parser rejects those dtypes and the Qwen3.6 model is a hybrid Gated DeltaNet + full-attention model.
- Hardware target: Jetson AGX Orin, SM 8.7, JetPack/L4T R36.4.3, CUDA 12.6.
- Startup expectation: cold model load and compilation can take 10–15 minutes. Do not restart between benchmark requests.

## Current deployment

```bash
docker compose -f qwen3.6_35b-int8/docker-compose.yml up
```

The compose file is restored to the original NVIDIA AI-IoT Jetson Orin INT8 stack at the user's request: `ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin`. The weight checkpoint and original serving flags are retained:

| Setting | Value | Reason |
| --- | --- | --- |
| Weight checkpoint | `/models/Qwen3.6-35B-A3B-INT8-AutoRound` | 8-bit AutoRound, group size 128 |
| `dtype` | `bfloat16` | W8A16 activations; matches the known Orin path |
| `max-model-len` | `262144` | Required full Qwen3.6 context window |
| `gpu-memory-utilization` | `0.75` | User-approved target; Fun-ASR is stopped during model tests |
| `max-num-seqs` | `8` | Original INT8 profile; restores the user's previous scheduler setting |
| `max-num-batched-tokens` | `4096` | Established prefill budget and compile range |
| `kv-cache-dtype` | `fp8` | Supported compressed KV cache in vLLM 0.19.0 |
| Attention backend | vLLM default/auto-selected | Original profile; no experimental backend override |
| Prefix caching | enabled | Helps repeated identical prefixes; does not speed unrelated prompts |
| Chunked prefill | vLLM default/enabled | Balances long prefills with decode work |

The model's quantization metadata is read from the checkpoint. Do **not** add `--quantization=auto_round`; this image accepts the checkpoint metadata and reports `quantization=inc` internally.

## Original image/profile restored

The production compose intentionally uses the original floating `latest-jetson-orin` tag rather than the experimental digest pin. The tested digest and newer-runtime experiments remain documented for reference, but are not active in the user's stack. The original INT8 profile uses `max-num-seqs=8`, `max-num-batched-tokens=4096`, FP8 KV, and 0.75 memory utilization.

A future upgrade should be treated as a separate porting project: validate the JetPack/CUDA ABI, SM 8.7 kernels, AutoRound loading, FP8 KV backend, hybrid Qwen3.6 execution, and the benchmark suite before changing the production image.

## KV-cache decision

The installed vLLM 0.19.0 CLI accepts only:

```text
auto, bfloat16, float16, fp8, fp8_ds_mla, fp8_e4m3, fp8_e5m2, fp8_inc
```

`turboquant_*`, `tq-*`, `int4`, `q4`, and `--kv-cache-compression` are not valid in this image. TurboQuant was added after v0.19.0 and later hybrid-model support still requires a newer stack than the pinned Jetson image. Do not silently substitute a fork or a lower-precision weight checkpoint.

The safe experiment order is:

1. Keep the current `fp8` + `FLASHINFER` profile as the production control.
2. Compare `fp8_e4m3` only if an explicit dtype is needed; retain the same backend and workload.
3. Treat `fp8_e5m2` as diagnostic only. vLLM 0.19.0 has known quantized-checkpoint/backend restrictions and it must not be promoted without a clean startup, correct output, and benchmark evidence.
4. If FP8 output quality is unacceptable, revert to `bfloat16` KV cache. Do not add `--calculate-kv-scales`; the v0.19 hybrid-cache path has known corruption reports for that option.

No KV experiment may use more memory than the FP8 control. A candidate that allocates more KV memory, fails startup, emits incoherent/repeating output, or regresses TTFT/decode speed is rejected.

## Benchmarking

Wait until the container healthcheck is healthy, then run one benchmark command. The script does not intentionally restart vLLM, but a failed engine experiment can still trigger the compose `unless-stopped` restart policy:

```bash
scripts/benchmark_server.sh
```

For a warmed steady-state run:

```bash
NUM_WARMUPS=2 RESULT_FILENAME=warmed.json scripts/benchmark_server.sh
```

For a warmup curve without restarting the model:

```bash
NUM_PROMPTS=8 NUM_WARMUPS=0 RESULT_FILENAME=curve.json scripts/benchmark_server.sh
python3 scripts/summarize_benchmark.py benchmarks/curve.json
```

For a lightweight direct streaming smoke test that does not start a second vLLM benchmark process inside the memory-constrained container:

```bash
python3 scripts/measure_stream.py \
  --warmups 1 --requests 3 \
  --output benchmarks/direct_stream.json
```

This reports TTFT and decode tok/s from the streamed API response. It is useful for validating the 262K profile after startup; keep the exact random-length `bench serve` files for control comparisons.

The benchmark uses vLLM's online `bench serve` client with one concurrent request, deterministic random lengths, detailed per-request output, and the served API model name. Report **TTFT**, **TPOT/decode tok/s**, and aggregate throughput separately. Aggregate output throughput is not single-session decode speed.

## Measured results and rejected experiments
The concurrency measurements below are archived experiments from the temporary `max-num-seqs=3` profile. The runnable production compose has been restored to the original `max-num-seqs=8` INT8 setting; these results are not claimed as an uplift over the original stack.

- Exact 256-input/128-output control with two warmups: TTFT 221.10 ms, TPOT 34.97 ms, decode 28.59 tok/s, aggregate 27.45 tok/s (`benchmarks/optimized_warm.json`).
- Direct 15-token prompt on the healthy 262K profile: three 128-token requests measured 212.39–214.67 ms TTFT and 28.47–28.50 tok/s (`benchmarks/direct_stream_final_script.json`).
- Native MTP-1 failed during EngineCore initialization; an `--enforce-eager` retry also failed after KV setup. `--language-model-only` likewise failed during engine initialization. These overlays remain experimental and are not enabled by the production compose.
- Final concurrency-2: TTFT 430–439 ms/request, 25.63–25.75 tok/s/request, 5.44 s wall for 256 output tokens (`benchmarks/final_concurrency2.json`).
- Final concurrency-3: TTFT 780–816 ms/request, 22.67–22.72 tok/s/request, 6.45 s wall for 384 output tokens (`benchmarks/final_concurrency3.json`).
- Q4/TurboQuant was rejected by the vLLM 0.19.0 CLI and is not enabled.

- The 262K setting starts, but tokenizer-verified 260,000- and 262,143-token generation requests abort in vLLM 0.19.0's hybrid Qwen3.6 EngineCore (the eager variants take roughly 84 seconds; the graph variant aborts immediately). Do not claim end-to-end 262K support from the startup line alone; see `docs/optimization-log.md` and `benchmarks/full_context_failures.json`.
- Native KV offload and 8 GiB CPU weight offload were tested and rejected by vLLM 0.19.0 hybrid-cache/input-batch errors. Fun-ASR is intentionally stopped for all model experiments.

## Evidence and history

- `docs/optimization-log.md` records the investigation, commands, observed startup failure, source-backed decisions, and pending experiments.
- `qwen3.6_35b-int8/docker-compose.yml` is the runnable deployment.
- `scripts/benchmark_server.sh` runs reproducible online measurements.
- `scripts/summarize_benchmark.py` computes per-request TTFT and decode tok/s from detailed JSON.
- `vLLM docs.zip` is the supplied prior optimization notebook export retained as research input; its conclusions are not treated as new measurements.
- `scripts/measure_stream.py` validates direct streaming without creating another vLLM benchmark process.

## References

- [NVIDIA AI-IoT vLLM container versions](https://github.com/orgs/NVIDIA-AI-IOT/packages/container/vllm/versions?filters%5Bversion_type%5D=tagged)
- [vLLM 0.19.0 serve CLI](https://docs.vllm.ai/en/v0.19.0/cli/serve/)
- [vLLM 0.19.0 quantized KV cache](https://docs.vllm.ai/en/v0.19.0/features/quantization/quantized_kvcache.html)
- [vLLM 0.19.0 online benchmark CLI](https://docs.vllm.ai/en/v0.19.0/cli/bench/serve)
- [vLLM 0.20.0 release: TurboQuant introduction](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
- [vLLM TurboQuant study](https://vllm.ai/blog/2026-05-11-turboquant)
- [Qwen3.6 model repository](https://github.com/QwenLM/Qwen3.6)

- [Qwen3.6 architecture/config](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B/resolve/master/config.json)
- [SGLang AutoRound W8A16](https://www.lmsys.org/blog/2025-11-13-AutoRound)
- [SGLang Jetson guide](https://docs.sglang.ai/docs/hardware-platforms/nvidia_jetson)
- [llama.cpp CUDA build](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md)
- [llama.cpp Jetson MoE issue](https://github.com/ggml-org/llama.cpp/issues/19219)