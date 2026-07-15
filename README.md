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

The compose file pins the NVIDIA AI-IoT Jetson Orin image at the published R36.4/CUDA 12.6 build (`vLLM 0.19.0`) instead of using a floating `latest` tag. The current serving profile is:

| Setting | Value | Reason |
| --- | --- | --- |
| Weight checkpoint | `/models/Qwen3.6-35B-A3B-INT8-AutoRound` | 8-bit AutoRound, group size 128 |
| `dtype` | `bfloat16` | W8A16 activations; matches the known Orin path |
| `max-model-len` | `32768` | Practical context budget; avoids the previous 262K reservation |
| `gpu-memory-utilization` | `0.65` | Leaves headroom for unified-memory host services |
| `max-num-seqs` | `1` | Single-session latency profile and small CUDA-graph capture |
| `max-num-batched-tokens` | `4096` | Prefill budget; compare with 2048/8192 using the benchmark tools |
| `kv-cache-dtype` | `fp8` | Supported compressed KV cache in vLLM 0.19.0 |
| `attention-backend` | `FLASHINFER` | Explicit FP8-capable attention backend for Orin SM 8.7 |
| Prefix caching | enabled | Helps repeated identical prefixes; does not speed unrelated prompts |
| Chunked prefill | vLLM default/enabled | Balances long prefills with decode work |

The model's quantization metadata is read from the checkpoint. Do **not** add `--quantization=auto_round`; this image accepts the checkpoint metadata and reports `quantization=inc` internally.

## Why the image is pinned at vLLM 0.19.0

The NVIDIA AI-IoT package page currently lists the Orin R36.4/CUDA 12.6 tags as vLLM 0.19.0. The newer generic vLLM releases are published with newer CUDA/PyTorch baselines, while the Orin-specific package does not expose a validated >0.19 Orin tag. A newer version is therefore not automatically faster on this board: losing the Jetson-compatible kernels or CUDA ABI is a larger risk than the version number suggests.

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

Wait until the container healthcheck is healthy, then run one benchmark command. The script does not restart vLLM:

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

The benchmark uses vLLM's online `bench serve` client with one concurrent request, deterministic random lengths, detailed per-request output, and the served API model name. Report **TTFT**, **TPOT/decode tok/s**, and aggregate throughput separately. Aggregate output throughput is not single-session decode speed.

## Evidence and history

- `docs/optimization-log.md` records the investigation, commands, observed startup failure, source-backed decisions, and pending experiments.
- `qwen3.6_35b-int8/docker-compose.yml` is the runnable deployment.
- `scripts/benchmark_server.sh` runs reproducible online measurements.
- `scripts/summarize_benchmark.py` computes per-request TTFT and decode tok/s from detailed JSON.
- `vLLM docs.zip` is the supplied prior optimization notebook export retained as research input; its conclusions are not treated as new measurements.

## References

- [NVIDIA AI-IoT vLLM container versions](https://github.com/orgs/NVIDIA-AI-IOT/packages/container/vllm/versions?filters%5Bversion_type%5D=tagged)
- [vLLM 0.19.0 serve CLI](https://docs.vllm.ai/en/v0.19.0/cli/serve/)
- [vLLM 0.19.0 quantized KV cache](https://docs.vllm.ai/en/v0.19.0/features/quantization/quantized_kvcache.html)
- [vLLM 0.19.0 online benchmark CLI](https://docs.vllm.ai/en/v0.19.0/cli/bench/serve)
- [vLLM 0.20.0 release: TurboQuant introduction](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
- [vLLM TurboQuant study](https://vllm.ai/blog/2026-05-11-turboquant)
- [Qwen3.6 model repository](https://github.com/QwenLM/Qwen3.6)
