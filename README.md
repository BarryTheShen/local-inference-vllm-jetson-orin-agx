# LLM inference on a Jetson AGX Orin

This repository holds the setups and benchmarks for the models served on a Jetson AGX Orin 64 GB (L4T R36.4.3, CUDA 12.6, SM 8.7) with vLLM.

## Current: Inkling 2 Mini

Since 2026-11-23 the Jetson serves Inkling 2 Mini. Released vLLM does not support it on GPUs older than Hopper, so PyTorch, Triton, and vLLM are built from source for the Orin and patched (int8 attention and LM-head weights, Orin-tuned expert kernels, speculative decoding). The full setup is in [`inkling2-mini/`](inkling2-mini/).

| Measurement | Untuned | Tuned |
|---|---|---|
| One request, decode | 25.7 tokens/s | 40.5 tokens/s |
| 8 parallel requests, total | 75.5 tokens/s | 108.9 tokens/s |
| 128 parallel requests, total | – | 551.7 tokens/s |
| Perplexity (wikitext) | 7.34 | 7.34 |

The Orin kernel tiles are proposed upstream in [vllm-project/humming#223](https://github.com/vllm-project/humming/pull/223).

## Earlier: Qwen3.6-35B-A3B INT8

Docker Compose deployment and benchmark scripts for serving the Qwen3.6-35B-A3B INT8 AutoRound checkpoint with vLLM on the same Jetson, before Inkling replaced it.

Stack: vLLM 0.19.0 (NVIDIA AI-IoT Jetson image), Docker Compose, Qwen3.6-35B-A3B AutoRound W8A16.

### Results

Benchmark runs from the JSON files in `benchmarks/`. Decode is tokens per second for one request. Aggregate is total output tokens per second across all requests; it is not single-session speed.

| Run | Requests | TTFT | Decode tok/s per request | Other |
| --- | --- | --- | --- | --- |
| 256 in / 128 out, 2 warmups (`optimized_warm.json`) | 1 | 221.10 ms | 28.59 | TPOT 34.97 ms; aggregate 27.45 tok/s |
| 256 in / 128 out, first request after load (`optimized_control.json`) | 1 | 2444.54 ms | 28.62 | TPOT 34.94 ms; aggregate 18.60 tok/s |
| 15-token prompt, 128 out, streamed from the host (`direct_stream_final_script.json`) | 3 | 212.39–214.67 ms | 28.47–28.50 | 262K profile |
| 256 in / 128 out, concurrency 2 (`final_concurrency2.json`) | 2 | 430.29–439.17 ms | 25.63–25.75 | batch wall 5.44 s |
| 256 in / 128 out, concurrency 3 (`final_concurrency3.json`) | 3 | 780.49–815.90 ms | 22.67–22.72 | batch wall 6.45 s |
| Concurrency 3 after an automatic engine restart (`final_concurrency3_postrestart.json`) | 3 | 755.25–788.37 ms | 21.73–21.77 | batch wall 6.67 s |
| Full context: 260,000 and 262,143 prompt tokens (`full_context_failures.json`) | 4 attempts | no completion | none | `finish_reason=abort` in all four; 83–85 s with eager profiles, 3.9 s with graphs |

The concurrency 2 and 3 rows use the `max-num-seqs=3` overlay.

**Production server, chat prompts, 2026-11-23** (`production_chat_2026-11-23.json`; `max-num-seqs=12`):

| Requests at once | Total tok/s | Per request tok/s | First token (median) |
| --- | --- | --- | --- |
| 1 | 28.2 | 28.5 | 0.12 s |
| 8 | 116.6 | 14.8 | 0.37 s |
| 16 | 108.8 | 12.9 | 0.39 s |
| 32 | 124.0 | 11.7 | 24.6 s (requests past 12 wait in a queue) |

Prompt reading runs at about 2,000–2,400 tokens per second from 1K to 17K tokens (`production_prefill_2026-11-23.json`).

### Quickstart

Requirements: a Jetson AGX Orin 64GB with Docker and the NVIDIA container runtime. The profile reserves 0.75 of unified memory (46.03 GiB of 61.37 GiB on the test device), so stop other GPU workloads first.

```bash
mkdir -p ~/models
docker compose -f qwen3.6_35b-int8/docker-compose.yml config --quiet
docker compose -f qwen3.6_35b-int8/docker-compose.yml up -d
```

The two `download-*` services run first. They fetch `Minachist/Qwen3.6-35B-A3B-INT8-AutoRound` (revision `w8a16-gs128`) and the tokenizer from `Qwen/Qwen3.6-35B-A3B` into `~/models`, through `HF_ENDPOINT` (default `https://hf-mirror.com`). The server then starts with `HF_HUB_OFFLINE=1` and loads the weights from `/models`. Startup takes several minutes; the health check allows 30 minutes.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.6-35b-a3b-int8", "prompt": "Say hello.", "max_tokens": 16}'
```

If the downloads need a proxy, put `HF_PROXY=http://<proxy-host>:<port>` in `qwen3.6_35b-int8/.env`. Compose reads that file from the directory of the compose file, and `.env` files are git-ignored. Without it, no proxy is used.

**Security:** port 8000 is published on all host interfaces, and vLLM runs without an API key. Anyone who can reach the host can send requests to the model. Put the host behind a firewall, or change the publish line in `qwen3.6_35b-int8/docker-compose.yml` to `"127.0.0.1:8000:8000"` if only local clients need access.

### Production profile

Settings from `qwen3.6_35b-int8/docker-compose.yml`:

| Setting | Value | Notes |
| --- | --- | --- |
| Weights | `/models/Qwen3.6-35B-A3B-INT8-AutoRound` | AutoRound, 8 bits, group size 128 |
| Tokenizer | `/models/Qwen3.6-35B-A3B-tokenizer` | Official Qwen tokenizer files; AutoRound checkpoints can fail to load their own tokenizer (`TokenizersBackend does not exist`) |
| `dtype` | `bfloat16` | 8-bit weights, bfloat16 activations |
| `max-model-len` | `262144` | The server starts with this context, but full-length generation aborts (see Known limitations) |
| `gpu-memory-utilization` | `0.75` | Unified-memory budget for vLLM |
| `max-num-seqs` | `8` | Scheduler limit; not measured under concurrency |
| `max-num-batched-tokens` | `4096` | Prefill token budget; `512` is rejected for this model |
| `kv-cache-dtype` | `fp8` | Supported compressed KV cache on SM 8.7 in vLLM 0.19.0 |
| Attention backend | not set (vLLM default) | The experiment overlays set `FLASHINFER` |
| Prefix caching | enabled | Helps repeated identical prefixes only |
| Reasoning parser | `qwen3` | Thinking output goes to `reasoning_content` |
| Tool calls | auto tool choice, `qwen3_xml` | `qwen3_coder` is not used (vllm#39056) |

Do not add `--quantization=auto_round`. The checkpoint metadata selects the AutoRound path, and the flag is an invalid choice in this image.

The image tag `latest-jetson-orin` floats. On 2026-07-15 it pointed to the same published image as `0.19.0-r36.4.tegra-aarch64-cu126-22.04`. Check the tag after each pull.

### KV cache

vLLM 0.19.0 in this image accepts `--kv-cache-dtype` values `auto, bfloat16, float16, fp8, fp8_ds_mla, fp8_e4m3, fp8_e5m2, fp8_inc`. TurboQuant (`turboquant_*`), `int4` and `q4` are rejected by the CLI. TurboQuant first appears in vLLM v0.20.0, and hybrid-model support needs a newer stack than this image provides. `--calculate-kv-scales` is not used: it is deprecated in 0.19.0, and hybrid-cache corruption has been reported with it.

If the FP8 cache must change, test in this order: `fp8_e4m3` only if an explicit dtype is needed; `fp8_e5m2` for diagnosis only; `bfloat16` if FP8 output quality is unacceptable. A candidate is rejected if it fails startup, allocates more KV memory than the FP8 control, produces incoherent or repeating output, or slows TTFT or decode.

### Experiments

These overlays are not used by the production file. Stop the running server before applying one: `docker compose -f docker-compose.yml -f docker-compose.<name>.yml up`, run from `qwen3.6_35b-int8/`.

| Overlay | What it tested | Outcome |
| --- | --- | --- |
| `concurrency3` | `max-num-seqs=3`, FLASHINFER | Measured; see Results |
| `eager-262k` | 262K context without CUDA graphs | Starts; 260,000- and 262,143-token generation aborts |
| `cpu-offload` | 8 GiB CPU weight offload | Rejected: vLLM asserts during input-batch re-initialization |
| `kv-offload` | Native 8 GiB KV offload | Rejected: the connector does not support the hybrid cache |
| `mtp1` | MTP-1 speculative decoding, text only, eager | EngineCore died before serving |
| `text-only` | `--language-model-only`, 32K | EngineCore died during initialization |

`Dockerfile.vllm-source` is an unfinished recipe for building vLLM v0.21.0 on the pinned base image. The compile was stopped before an image was produced.

### Known limitations

- 262K: the server starts with `max-model-len=262144`, but generation at 260,000 and 262,143 prompt tokens aborted in every test (`benchmarks/full_context_failures.json`). Do not treat 262K as supported end to end.
- Concurrency was measured only with `max-num-seqs=3`.
- Each row is one benchmark run; the ranges show the spread across the requests in that run.
- No API authentication (see Security above).

### Benchmarking

Wait until the container health check reports healthy, then run one benchmark at a time. The scripts do not restart vLLM, but a failed experiment can still trigger the compose `unless-stopped` restart policy.

```bash
scripts/benchmark_server.sh
NUM_WARMUPS=2 RESULT_FILENAME=warmed.json scripts/benchmark_server.sh
NUM_PROMPTS=8 NUM_WARMUPS=0 RESULT_FILENAME=curve.json scripts/benchmark_server.sh
python3 scripts/summarize_benchmark.py benchmarks/curve.json
python3 scripts/measure_stream.py --warmups 1 --requests 3 --output benchmarks/direct_stream.json
```

`benchmark_server.sh` runs `vllm bench serve` inside the server container with fixed random lengths (256 in, 128 out by default), `--seed 0` and `--temperature 0`. `measure_stream.py` is a host-side smoke test; use the `bench serve` files for control comparisons. Report TTFT, TPOT or decode tok/s, and aggregate throughput separately.

## Repository layout

- `inkling2-mini/`: Inkling 2 Mini compose file, Orin tuning table, vLLM patches, source build.
- `qwen3.6_35b-int8/`: Qwen compose file, experiment overlays, `Dockerfile.vllm-source`.
- `scripts/`: `benchmark_server.sh`, `measure_stream.py`, `summarize_benchmark.py`.
- `benchmarks/`: result JSON files.
- `docs/optimization-log.md`: investigation log with commands, observed errors and decisions.

## References

- [NVIDIA AI-IoT vLLM image tags](https://github.com/orgs/NVIDIA-AI-IOT/packages/container/vllm/versions?filters%5Bversion_type%5D=tagged)
- [vLLM 0.19.0 serve CLI](https://docs.vllm.ai/en/v0.19.0/cli/serve/)
- [vLLM 0.19.0 quantized KV cache](https://docs.vllm.ai/en/v0.19.0/features/quantization/quantized_kvcache.html)
- [vLLM 0.19.0 online benchmark CLI](https://docs.vllm.ai/en/v0.19.0/cli/bench/serve)
- [vLLM 0.20.0 release notes (TurboQuant)](https://github.com/vllm-project/vllm/releases/tag/v0.20.0)
- [Qwen3.6 model repository](https://github.com/QwenLM/Qwen3.6)

Other sources are listed in `docs/references.md`.
