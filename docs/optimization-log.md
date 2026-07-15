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

`qwen3.6_35b-int8/docker-compose.yml` is the production candidate and is pinned to the exact NVIDIA image digest `sha256:817f0f940d2d9c9067d861d2118d7bf58c40873598f0c35e19c8516269ebc4bd` (vLLM 0.19.0, JetPack R36.4/CUDA 12.6).

The serving profile keeps the requested INT8/W8A16 AutoRound weights and vLLM-only runtime:

- `max-model-len=262144` — mandatory full Qwen3.6 context setting.
- `gpu-memory-utilization=0.75` — user-approved target; Fun-ASR is stopped during experiments.
- `max-num-seqs=3` — production candidate for two-to-three concurrent short requests; single-request decode remains near the 32K control.
- `max-num-batched-tokens=4096` — required by the model's Mamba alignment (`block_size=2096`); 512 is rejected.
- `kv-cache-dtype=fp8` — supported compressed cache path on SM 8.7.
- `attention-backend=FLASHINFER` — explicit FP8-capable backend.
- prefix caching, chunked prefill, Qwen3 reasoning parsing, and XML tool parsing remain enabled.

The compose file passes `docker compose -f qwen3.6_35b-int8/docker-compose.yml config --quiet`. The image digest is pinned for all download and serve services; no floating `latest` tag remains in the runnable profile.

The earlier 32K latency profile was only an exploratory control. It is not the final deployment because the user requires 262K context.

## 7. 262K startup and request results

Startup itself accepts `max_model_len=262144`. With the graph-free profile (`--enforce-eager`) and Fun-ASR stopped:

```text
Available KV cache memory: 11.33 GiB
GPU KV cache size: 295,536 tokens
Maximum concurrency for 262,144 tokens per request: 4.29x
init engine (profile, create kv cache, warmup model) took 292.22 seconds
Application startup complete
```

The same profile accepted tokenizer-verified prompts of 262,143 and 260,000 tokens, but one-token generation aborted after approximately 83–85 seconds with `finish_reason=abort`; EngineCore then shut down. The 262,143-token request had `prompt_tokens=262143`, and the 260,000-token request had `prompt_tokens=260000`. This is a failed full-context end-to-end result even though startup advertises the max length.

The graph-free 262K profile is therefore not promoted as a completed full-context solution. The failure is inside the pinned vLLM 0.19.0 hybrid Qwen3.6 path, not an HTTP client timeout. The regular graph profile also starts, but its smaller cache budget cannot improve this boundary case.

The three-request graph profile started cleanly with Fun-ASR stopped:

```text
max_num_seqs=3
Available KV cache memory: 11.79 GiB
GPU KV cache size: 308,112 tokens
Maximum concurrency for 262,144 tokens per request: 4.47x
Application startup complete
```

The reusable direct benchmark then completed both two- and three-request tests:

- concurrency 2: TTFT 415–425 ms/request, decode 25.68–25.80 tok/s/request, wall 5.41 s for 256 output tokens;
- concurrency 3: TTFT 416–444 ms/request, decode 22.74–22.76 tok/s/request, wall 6.07 s for 384 output tokens;
- evidence: `benchmarks/concurrency2_script.json` and `benchmarks/concurrency3_script.json`.

This profile supports the requested 2–3 parallel short-request behavior. The 262K single-request boundary failure remains independent of scheduler concurrency.

Memory experiments:

- `gpu-memory-utilization=0.65` with Fun-ASR running: 6.03 GiB KV, 157,200 tokens; healthy startup, full request aborted.
- `gpu-memory-utilization=0.75` with Fun-ASR running: 4.58 GiB KV, 119,472 tokens; healthy startup.
- `gpu-memory-utilization=0.75` with Fun-ASR stopped and page cache reclaimed: 3.88 GiB KV, 100,608 tokens; healthy startup.
- `--kv-offloading-size=8` raised the reported budget to 11.8 GiB/308,112 tokens, but the native connector rejected Qwen3.6 hybrid attention (`OffloadingConnector does not support HMA`); disabling HMA then failed to unify the hybrid KV specs.
- `--cpu-offload-gb=8` raised the profile to 9.75 GiB/253,616 tokens, but vLLM 0.19.0 asserts during input-batch reinitialization, both with and without CUDA graphs.

The host already had `/ssd/swapfile.swap` before this work: 32 GiB configured, approximately 2 GiB used under load. No swapfile was created or enabled by this optimization. Page-cache reclamation, when needed, is:

```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
```
## 8. Optimized startup observations

The earlier 32K control reached the same vLLM 0.19.0 execution path with:

```text
quantization=inc
dtype=torch.bfloat16
max_seq_len=32768
kv_cache_dtype=fp8
attention_backend=FLASHINFER
max_num_seqs=1
max_num_batched_tokens=4096
Model loading took 36.11 GiB memory
torch.compile took 102.48 s
CUDA graph capture completed
```

The 32K profile was used only to establish a fast, repeatable single-session baseline before the mandatory context requirement was supplied.
## 9. Benchmark protocol

The exact control measurement used `vllm bench serve` with one request, `max-concurrency=1`, `seed=0`, random input length 256, and output length 128. Results are stored with `--save-result --save-detailed`:

- `benchmarks/optimized_control.json`: cold first request, TTFT 2444.54 ms; decode 28.62 tok/s.
- `benchmarks/optimized_warm.json`: two warmups, measured TTFT 221.10 ms; TPOT 34.97 ms; decode 28.59 tok/s; aggregate output throughput 27.45 tok/s.

The first `vllm bench serve` attempt after restarting the final profile exited with status 137 and the server performed a clean EngineCore shutdown/restart. The API was healthy afterward. To avoid adding another model-aware benchmark process inside the memory-constrained container, `scripts/measure_stream.py` performs a lightweight direct streaming test from the host.

The direct smoke test completed three 128-token requests after one warmup against the 262K profile:

- prompt: 15 tokens;
- measured TTFT: 212.39–214.67 ms;
- measured decode speed: 28.47–28.50 tok/s;
- evidence: `benchmarks/direct_stream_final_script.json`.

These direct measurements are a smoke test, not a replacement for the exact random 256/128 control. Aggregate output throughput is never reported as single-session decode speed.

The final production candidate uses `max-num-seqs=3` and the reusable script validates parallel service:

- final concurrency 2: TTFT 430–439 ms/request, decode 25.63–25.75 tok/s/request, wall 5.44 s for 256 output tokens (`benchmarks/final_concurrency2.json`);
- final concurrency 3: TTFT 780–816 ms/request, decode 22.67–22.72 tok/s/request, wall 6.45 s for 384 output tokens (`benchmarks/final_concurrency3.json`);
- single-request warmup in the same runs remains about 28.5 tok/s decode.
## 10. Experiments and rejected candidates

### MTP-1, normal CUDA graphs

The MTP overlay used the preserved `mtp.*` checkpoint weights and vLLM's native `{"method":"mtp","num_speculative_tokens":1}` configuration. It loaded the target and draft model, but EngineCore died after KV allocation and before serving.

### MTP-1, eager mode

The same MTP overlay was retried with `--enforce-eager` to remove CUDA graph capture as a possible cause. It loaded successfully, reported 7.64 GiB available KV and 181,632 KV tokens, then EngineCore died before serving. This rules out a simple CUDA-graph-only workaround.

### `--language-model-only`

The text-only overlay was tested to remove unused multimodal encoder work. vLLM reached model load but EngineCore died during/after CUDA graph initialization. The default multimodal configuration remains the stable path for this checkpoint.

### Lower KV cache dtypes

The v0.19.0 parser rejects `turboquant_4bit_nc`; its accepted list contains FP8/BF16/FP16 spellings only. No Q4/TurboQuant setting was silently substituted. The deployed FP8 cache is the supported SM8.7 path.

### KV offloading

Native `--kv-offloading-size=8` is incompatible with this hybrid Qwen3.6 model in vLLM 0.19.0. The connector reports `OffloadingConnector does not support HMA`; adding `--disable-hybrid-kv-cache-manager` then reports that the hybrid specs cannot be unified.

### CPU weight offloading

`--cpu-offload-gb=8` increases the apparent KV budget, but the v0.19.0 model runner asserts `Cannot re-initialize the input batch when CPU weight offloading is enabled`, both with CUDA graphs and with `--enforce-eager`.

### Chunk-size tuning

`--max-num-batched-tokens=512` is rejected before startup because the model's Mamba alignment requires `block_size=2096`. `2096` starts, but the 260K/262K full-context request still aborts. The stable control remains 4096.
## 11. Alternative runtime research

The read-only runtime survey ranked candidates for Jetson AGX Orin 64GB (SM87, JetPack 6.1/L4T R36.4, CUDA 12.6.10):

1. **Custom newer vLLM source build — best practical path.** Upstream vLLM's model table exposes the matching Qwen3.5 MoE architecture, but no explicit Qwen3.6 row; Qwen3.6's official config is `Qwen3_5MoeForConditionalGeneration` with 262,144 max positions and hybrid full/linear attention. Upstream wheels target newer CUDA, so a JetPack-matched source build is required. Sources: [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/), [vLLM installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/), [Qwen3.6 config](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B/resolve/master/config.json).
2. **SGLang — strongest alternate experiment.** SGLang >=v0.5.4.post2 supports direct AutoRound W8A16 loading in principle, and its Jetson guide supports building with Jetson Containers. The official Jetson example is only 8K context and TorchAO int4, so 262K/W8A16/SM87 remains unvalidated. Sources: [SGLang AutoRound](https://www.lmsys.org/blog/2025-11-13-AutoRound), [SGLang Jetson](https://docs.sglang.ai/docs/hardware-platforms/nvidia_jetson).
3. **llama.cpp — concurrency-capable but format-changing.** Its server supports parallel slots, continuous batching, KV offload/quantization, and CUDA SM87 builds, but this HF AutoRound checkpoint is not GGUF. Conversion would not preserve the exact W8A16 artifact. A Jetson MoE hang is documented for current software stacks; `CUDA_SCALE_LAUNCH_QUEUES=1x` is a cautious workaround. Sources: [llama.cpp build](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md), [server options](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md), [Orin MoE issue](https://github.com/ggml-org/llama.cpp/issues/19219).
4. **TensorRT-LLM/Triton — reject for this contract.** Official support omits SM87 Orin and Qwen3.6/W8A16 AutoRound; the Qwen guide targets newer datacenter GPUs.
5. **Transformers native server — reject for now.** Continuous-batching has a known Qwen3.5 multimodal crash, and the ordinary Transformers cache is not vLLM-native MTP serving.

The source survey does not replace hardware smoke tests. Any alternate must preserve INT8-or-better weights, validate `max_model_len=262144`, and run three simultaneous API requests before it can replace the pinned profile.
## 12. Custom vLLM 0.21 source build in progress

The host clone of upstream vLLM tag `v0.21.0` completed at source commit `ad7125a431e176d4161099480a66f0169609a690`. `qwen3.6_35b-int8/Dockerfile.vllm-source` records the reproducible recipe: pinned Jetson vLLM base, existing JetPack PyTorch, `TORCH_CUDA_ARCH_LIST=8.7`, one compile job, and no test suite.

The first Dockerfile clone failed because the container could not complete the GitHub TLS handshake. A host clone through the configured proxy succeeded. The local-source image build is compiling now. CMake has confirmed CUDA 12.6.85 and SM87 target selection, but reports:

- QuTLASS requires CUDA 12.8+ and is skipped;
- FlashAttention's supported target list exposes 8.6 rather than 8.7;
- the source build warns that its expected PyTorch version differs from the base image.

These warnings make this an experiment, not a promotion. If the image completes, it must pass model load, 262K request, short-request TTFT, and two/three-request concurrency before the production compose changes.

## 13. Git checkpoints and current progress

- `c45b792`: capture existing INT8 compose and supplied docs archive.
- `6ea9596`: merge the remote repository license and push the baseline.
- `2c1271a`: commit exact 32K control evidence and initial MTP overlay.
- Current uncommitted checkpoint: digest-pinned 262K compose, direct streaming benchmark, reusable benchmark script, context/offload overlays, and this log.

The production compose is configured for `max_model_len=262144`, FP8 KV, FlashInfer, and the user-approved 0.75 memory target, with Fun-ASR stopped during tests. Startup and normal short requests are validated; exact 260K/262K generation aborts in the pinned vLLM 0.19.0 hybrid path. A custom newer vLLM build or SGLang experiment is the credible next path for end-to-end full-context support.
