# Optimization log

Date: 2026-07-15  
Target: Jetson AGX Orin 64GB, Qwen3.6-35B-A3B INT8 AutoRound, vLLM

## 1. Repository and source material

- The working directory was not initially a Git worktree. It was initialized and connected to `git@github.com:BarryTheShen/local-inference-vllm-jetson-orin-agx.git`.
- The remote contained only its initial `LICENSE`; the local baseline deployment and supplied `vLLM docs.zip` were committed as `c45b792`, then merged with the remote license and pushed as `6ea9596`.
- `vLLM docs.zip` contains a nested Notion export. It was extracted outside the repository and reviewed. The prior notes report approximately 26.5–30 tok/s for the earlier 35B Q4 GPTQ-Marlin profile, recommend real INT8 AutoRound for quality, and recommend FP8 KV cache rather than TurboQuant on this hybrid model.
- Model files already exist outside this repository at `/home/nvidia/models/Qwen3.6-35B-A3B-INT8-AutoRound` and `/home/nvidia/models/Qwen3.6-35B-A3B-tokenizer`; no model weights are copied into Git.

## 2. Host and model facts collected

Commands and observed facts:

```text
cat /etc/nv_tegra_release
# R36 (release), REVISION: 4.3, ... BOARD: generic, EABI: aarch64

uname -a
# ... aarch64 ... 5.15.148-tegra

docker version --format '{{.Server.Version}}'
# 28.3.3

docker run --rm --entrypoint python3 ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin \
  -c 'import vllm,torch; print(vllm.__version__, torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())'
# vllm 0.19.0, torch 2.10.0, CUDA 12.6, capability (8, 7)
```

The checkpoint metadata was inspected directly:

- Architecture: `Qwen3_5MoeForConditionalGeneration`
- Model type: `qwen3_5_moe`
- AutoRound: `bits=8`, `group_size=128`, `quant_method=auto-round`, `packing_format=auto_round:auto_gptq`
- The index contains 19 `mtp.*` keys; the quantization config keeps MTP weights at 16-bit.
- The model is hybrid: Gated DeltaNet/linear-attention layers plus periodic full-attention layers. FP8 KV applies to the full-attention cache; the recurrent GDN state has separate cache geometry.

## 3. Image-version investigation

The official NVIDIA AI-IoT GHCR versions page was checked. The Orin R36.4/CUDA 12.6 entries are:

- `r36.4.tegra-aarch64-cu126-22.04`
- `0.19.0-r36.4.tegra-aarch64-cu126-22.04`
- `latest-jetson-orin`

They refer to the same published Orin image. No validated NVIDIA AI-IoT Orin tag newer than vLLM 0.19.0 was found. vLLM 0.20.0 and 0.21.0 publish generic aarch64 CUDA 12.9 wheels and newer CUDA/PyTorch requirements; those are not a drop-in match for this JetPack R36.4.3/CUDA 12.6 host.

Decision: pin the known Orin-compatible v0.19.0 image. Do not switch to a generic NGC/x86 image or invent a nonexistent `21.0-jetson-orin` tag. A future v0.20+ port needs a separate CUDA ABI/kernel/quality benchmark before promotion.

## 4. KV-cache investigation

The pinned image's `vllm serve --help=CacheConfig` reports only:

```text
auto, bfloat16, float16, fp8, fp8_ds_mla, fp8_e4m3, fp8_e5m2, fp8_inc
```

The following preflight was run and failed immediately as expected:

```bash
docker run --rm --entrypoint vllm \
  ghcr.io/nvidia-ai-iot/vllm:0.19.0-r36.4.tegra-aarch64-cu126-22.04 \
  serve /models/unused --kv-cache-dtype turboquant_4bit_nc
```

Observed result: argparse rejected `turboquant_4bit_nc` and printed the allow-list above. TurboQuant first appears in the v0.20.0 release notes. Later upstream documentation/issues also identify hybrid Qwen3.6 + TurboQuant and Ampere/SM8.7 FP8 kernel/page-size combinations as unsafe. Therefore no TurboQuant/q4 candidate is wired into the deployment.

`--calculate-kv-scales` is intentionally absent. vLLM 0.19.0 marks it deprecated, and the hybrid-cache issue reports corruption with that path. The supported FP8 control remains the only cache profile promoted here.

## 5. First startup attempt: reproduce the previous compose

The original compose requested `--gpu-memory-utilization=0.75`, `--max-model-len=262144`, `--max-num-seqs=8`, and FP8 KV. The model and tokenizer download checks completed successfully, but the engine failed before loading weights:

```text
ValueError: Free memory on device cuda:0 (42.21/61.37 GiB) on startup is less than desired GPU memory utilization (0.75, 46.03 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
```

This is a host unified-memory budget failure, not a model or quantization failure. Fun-ASR and LazyCat host services were running concurrently.

## 6. Optimized profile applied

`qwen3.6_35b-int8/docker-compose.yml` was changed to:

- pin `0.19.0-r36.4.tegra-aarch64-cu126-22.04` rather than floating `latest-jetson-orin`;
- reduce `max-model-len` to 32768 for the practical latency profile;
- reduce `gpu-memory-utilization` to 0.65 so startup fits beside co-hosted services;
- set `max-num-seqs=1` for the single-session target, reducing CUDA-graph capture sizes from `[1,2,4,8,16]` to `[1,2]`;
- keep `max-num-batched-tokens=4096` and chunked prefill;
- keep FP8 KV and explicitly select `FLASHINFER` for the SM8.7 FP8 path;
- retain prefix caching, Qwen3 reasoning parsing, and tool-calling parser.

The edited compose passed `docker compose ... config --quiet`.

## 7. Optimized startup observations

The optimized server was launched once with a supervised Docker Compose process. No second server was started. Startup log evidence:

```text
version 0.19.0
quantization=inc
dtype=torch.bfloat16
max_seq_len=32768
kv_cache_dtype=fp8
attention_backend=FLASHINFER
max_num_seqs=1
max_num_batched_tokens=4096
Model loading took 36.11 GiB memory
Loading weights took 46.63 seconds
Compiling a graph for compile range (1, 4096) takes 85.42 s
torch.compile took 104.04 s in total
```

The engine then reached the `GPU KV cache size` readiness log while this log was written. The benchmark must still verify `/health`, one successful request, correctness, TTFT, and decode tok/s before this profile is called final.

## 8. Benchmark protocol

The online server benchmark is `vllm bench serve`, not the offline throughput benchmark. The benchmark script uses:

- the served model name `qwen3.6-35b-a3b-int8`;
- one request and `max-concurrency=1` for single-session numbers;
- deterministic random input/output lengths (`seed=0`, `random-range-ratio=0`);
- `--save-result --save-detailed` so per-request TTFT/ITL/output lengths are retained;
- separate reporting of TTFT, TPOT/decode tok/s, and aggregate output throughput.

Warmup runs are explicitly excluded from measured duration. The first request after startup is not treated as a steady-state number; use a warmup curve or two warmups and report the exact result file.

## 9. Planned experiments after the first healthy request

Each server restart is a separate cold experiment and should be allowed 10–15 minutes:

1. FP8 + FlashInfer, current profile: correctness, TTFT, TPOT, decode tok/s.
2. Same profile with `max-num-batched-tokens=2048` and `8192`: TTFT/prefill trade-off.
3. Text-only `--language-model-only` profile: remove unused vision work for coding/tool use, then repeat the same workload.
4. MTP-1 using the preserved `mtp.*` checkpoint weights, only if the v0.19.0 speculative configuration starts cleanly and accepted throughput is positive.
5. MTP-2 only if MTP-1 has healthy acceptance and improves wall-clock latency.
6. Optional explicit `fp8_e4m3` cache spelling; reject on any SM8.7 backend error, quality issue, or regression.

No candidate is promoted solely because vLLM prints a high aggregate generation number. The acceptance criteria are clean startup, healthy API, coherent output, no OOM, no preemption, KV memory no larger than FP8 control, and improved measured single-session TTFT/decode behavior.

## 10. Git checkpoints

- `c45b792`: capture existing INT8 compose and supplied docs archive.
- `6ea9596`: merge the remote repository license and push the baseline.
- The optimized compose, benchmark tooling, and this log are the next checkpoint after the first healthy smoke test.
