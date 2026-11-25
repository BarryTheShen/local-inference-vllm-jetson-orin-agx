#!/usr/bin/env bash
# Builds the Inkling 2 Mini toolchain for Jetson AGX Orin (SM 8.7, L4T R36.4, CUDA 12.6) as wheels in /build/wheels.
# Runs inside nvcr.io/nvidia/l4t-jetpack:r36.4.0 with /build mounted (see run-build.sh). Each step leaves a marker
# in /build/done/ and is skipped when the marker exists, so a failed run resumes where it stopped.
set -uo pipefail

B=/build
mkdir -p "$B/src" "$B/wheels" "$B/done" "$B/logs" "$B/ccache"
export CCACHE_DIR=$B/ccache
export TORCH_CUDA_ARCH_LIST=8.7
export CUDA_HOME=/usr/local/cuda
export PATH=$B/venv/bin:$CUDA_HOME/bin:/root/.local/bin:$PATH
export MAX_JOBS=${MAX_JOBS:-8}           # 64 GB shared with the GPU: 8 compile jobs leave headroom
export PY=$B/venv/bin/python
# Optional proxy for GitHub/Hugging Face downloads: set BUILD_PROXY (e.g. http://proxy-host:port). Unset = direct.
# NO_PROXY keeps LAN and local traffic direct.
export HTTPS_PROXY=${BUILD_PROXY:-} HTTP_PROXY=${BUILD_PROXY:-}
export https_proxy=$HTTPS_PROXY http_proxy=$HTTP_PROXY NO_PROXY=localhost,127.0.0.1,192.168.0.0/16 no_proxy=localhost,127.0.0.1,192.168.0.0/16
git config --global http.lowSpeedLimit 1000 && git config --global http.lowSpeedTime 120

TORCH_TAG=v2.14.0
VISION_TAG=v0.29.2
TRITON_BRANCH=release/3.8.x
VLLM_PR=69165
VLLM_SHA=aade5d062e27db3625828d14ab6b8c7cd4998314   # PR #69165 head, 2026-11-22
HUMMING="humming-kernels[cu12]==0.1.20"   # the CUDA 12 extra (JetPack 6); vLLM main pins the cu13 one

step() { local name=$1; shift
  [ -f "$B/done/$name" ] && { echo "skip $name"; return 0; }
  echo "=== $name $(date -Is)"
  # Each step runs in a subshell from /build: a `cd` in one step must not leak into the next (an import check run
  # inside the PyTorch source tree imports the source, not the build).
  if (cd "$B" && "$@") > "$B/logs/$name.log" 2>&1; then touch "$B/done/$name"; echo "ok $name $(date -Is)"
  else echo "FAILED $name: tail of $B/logs/$name.log"; tail -40 "$B/logs/$name.log"; exit 1; fi
}

s_apt() {
  apt-get update && apt-get install -y --no-install-recommends git ccache ninja-build cmake curl ca-certificates \
    libopenblas-dev libjpeg-dev libpng-dev zlib1g-dev patchelf pkg-config
}
s_python() {
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv python install 3.12
  uv venv --python 3.12 "$B/venv"
  uv pip install --python "$PY" -U pip setuptools wheel "setuptools<81" setuptools-scm setuptools-rust ninja cmake \
    numpy pyyaml typing_extensions requests packaging
}
s_torch() {
  cd "$B/src"; [ -d pytorch/.git ] || git clone --depth 1 --branch $TORCH_TAG https://github.com/pytorch/pytorch
  # Submodules separately, resumable: a rerun reuses the ones already cloned and retries the rest.
  ( cd pytorch && for t in 1 2 3 4 5; do
      # --force re-checks-out submodules left with only a .git pointer by an interrupted clone.
      git submodule update --init --recursive --force --depth 1 --jobs 4 && break
      echo "submodule update failed (try $t), retrying"; sleep 20; done )
  ( cd pytorch && git submodule status --recursive | grep -q '^-' ) && { echo "submodules missing"; return 1; }
  empty=$(cd pytorch && git submodule foreach --quiet --recursive '[ "$(ls -A | grep -v "^.git$" | head -1)" ] || echo "$displaypath"')
  [ -n "$empty" ] && { echo "submodules with no files: $empty"; return 1; }
  cd pytorch && uv pip install --python "$PY" -r requirements.txt
  # No NCCL (Tegra has none; vLLM runs TP=1), no FlashAttention in torch (vLLM brings its own), cuDNN from JetPack.
  USE_NCCL=0 USE_DISTRIBUTED=1 USE_GLOO=1 USE_MPI=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=1 USE_CUDNN=1 \
  USE_CUSPARSELT=0 USE_CUDSS=0 USE_FBGEMM=0 USE_KINETO=0 BUILD_TEST=0 USE_PRIORITIZED_TEXT_FOR_LD=1 \
  USE_XPU=0 USE_ROCM=0 USE_VULKAN=0 USE_MPS=0 USE_NNPACK=0 USE_QNNPACK=0 \
  PYTORCH_BUILD_VERSION=2.14.0 PYTORCH_BUILD_NUMBER=1 BUILD_IGNORE_SVE_UNAVAILABLE=1 \
    "$PY" -m pip wheel . --no-build-isolation --no-deps -w "$B/wheels" -v || return 1
  uv pip install --python "$PY" "$B"/wheels/torch-2.14.0*.whl
}
s_torch_check() {
  "$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "no CUDA"
assert torch.cuda.get_device_capability() == (8, 7), torch.cuda.get_device_capability()
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
print(torch.__version__, torch.version.cuda, (a @ a).float().abs().mean().item())
EOF
}
s_vision() {
  cd "$B/src"; [ -d vision ] || git clone --depth 1 --branch $VISION_TAG https://github.com/pytorch/vision
  cd vision && FORCE_CUDA=1 "$PY" -m pip wheel . --no-build-isolation --no-deps -w "$B/wheels" -v
  uv pip install --python "$PY" "$B"/wheels/torchvision-*.whl
}
s_triton() {
  cd "$B/src"; [ -d triton ] || git clone --depth 1 --branch $TRITON_BRANCH https://github.com/triton-lang/triton
  cd triton && uv pip install --python "$PY" -r python/requirements.txt 2>/dev/null || true
  "$PY" -m pip wheel . --no-build-isolation --no-deps -w "$B/wheels" -v
  uv pip install --python "$PY" "$B"/wheels/triton-*.whl
}
s_triton_check() {
  # The JetPack ptxas (12.6) compiles for the installed driver; Triton's bundled ptxas may be newer than it loads.
  # Triton reads a kernel's source from its file, so the check is written to one (stdin fails).
  cat > "$B/logs/triton_check.py" <<'EOF'
import torch, triton, triton.language as tl
@triton.jit
def add(x, y, o, n, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B); m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)
x = torch.randn(4096, device="cuda"); y = torch.randn_like(x); o = torch.empty_like(x)
add[(4,)](x, y, o, 4096, B=1024)
assert torch.allclose(o, x + y); print("triton", triton.__version__, "ok")
EOF
  TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas "$PY" "$B/logs/triton_check.py"
}
s_vllm() {
  # Shallow: just the PR head (the full history is slow over the home line).
  cd "$B/src"; if [ ! -d vllm/.git ]; then mkdir -p vllm && cd vllm && git init -q \
    && git remote add origin https://github.com/vllm-project/vllm && git fetch --depth 1 origin $VLLM_SHA \
    && git checkout -q FETCH_HEAD || return 1; else cd vllm; fi
  # vLLM builds a Rust frontend (setuptools-rust); a minimal toolchain under /build survives container restarts.
  export CARGO_HOME=$B/rust/cargo RUSTUP_HOME=$B/rust/rustup PATH=$B/rust/cargo/bin:$PATH
  command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path || return 1
  "$PY" tools/use_existing_torch.py || return 1
  uv pip install --python "$PY" -r requirements/build/cuda.txt || return 1
  CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13 VLLM_TARGET_DEVICE=cuda SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+inkling.sm87 \
    "$PY" -m pip wheel . --no-build-isolation --no-deps -w "$B/wheels" -v || return 1
}
s_vllm_install() {
  cd "$B/src/vllm"
  # torch, torchvision and triton stay the local sm_87 builds; humming gets its CUDA 12 extra. The filtered file sits
  # next to the originals so its `-r common.txt` still resolves.
  grep -vE '^(torch|torchvision|torchaudio|triton|nvidia-|humming)' requirements/cuda.txt > requirements/cuda-jetson.txt
  uv pip install --python "$PY" "$B"/wheels/vllm-*.whl -r requirements/cuda-jetson.txt "$HUMMING" \
    "torch==2.14.0" "torchvision==0.29.2" "triton==3.8.*" || return 1
  # humming's [cu12] extra pulls pip CUDA 12.9 (nvrtc, nvcc, runtime, cccl) and humming prefers the newest 12.x; the
  # Jetson's driver is CUDA 12.6 and its cubins must come from JetPack's own toolkit, so the pip copies go.
  uv pip uninstall --python "$PY" nvidia-cuda-nvrtc-cu12 nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-cuda-cccl-cu12 || true
  "$PY" -c "import torch; assert torch.version.cuda.startswith('12.6'), torch.version.cuda"
  "$PY" -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
}
s_tiny_tests() {
  # tests/conftest.py imports tblib and anthropic; the async tests need pytest-asyncio.
  cd "$B/src/vllm"; uv pip install --python "$PY" pytest tblib anthropic pytest-asyncio pytest-timeout || return 1
  # pytest puts the checkout on sys.path, so `import vllm` would load the uncompiled source package: hide it for the
  # run (the tests import the installed build) and always put it back.
  mv vllm vllm.src || return 1
  # HF_PROXY: optional proxy for Hugging Face downloads during the tests.
  # HUMMING_COMPILER=nvcc: JetPack's NVRTC 12.6 compiles bf16 arithmetic for sm_87 into BPT.TRAP (every expert GEMM
  # died with "unspecified launch failure"); nvcc 12.6 compiles the same code correctly.
  HTTPS_PROXY=${HF_PROXY:-$HTTPS_PROXY} HTTP_PROXY=${HF_PROXY:-$HTTP_PROXY} \
  HUMMING_COMPILER=nvcc TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas "$PY" -m pytest tests/models/inkling2 -q -rs -p no:cacheprovider
  rc=$?
  mv vllm.src vllm
  return $rc
}
s_gcc13() {
  # vLLM main needs GCC >= 13 (DeepGEMM/DeepSelect host code uses C++20 <format>); JetPack 6 (Ubuntu 22.04) has 11.4.
  # CUDA 12.6's nvcc accepts GCC 13 as host compiler. apt reaches the PPA through the proxy exported above.
  apt-get install -y --no-install-recommends software-properties-common gpg-agent || return 1
  add-apt-repository -y ppa:ubuntu-toolchain-r/test || return 1
  apt-get update && apt-get install -y --no-install-recommends gcc-13 g++-13 || return 1
  g++-13 --version | head -1
}

step apt s_apt
step python s_python
step torch s_torch
step torch_check s_torch_check
step vision s_vision
step triton s_triton
step triton_check s_triton_check
step gcc13 s_gcc13
step vllm s_vllm
step vllm_install s_vllm_install
step tiny_tests s_tiny_tests
echo "ALL DONE $(date -Is)"
