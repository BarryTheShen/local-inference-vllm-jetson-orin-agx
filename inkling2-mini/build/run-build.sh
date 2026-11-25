#!/usr/bin/env bash
# Run on the Jetson from ~/inkling-build. Starts the build container detached; follow with: docker logs -f inkling-build.
# Re-running resumes from the last finished step. An existing container is restarted, not recreated: its filesystem
# holds the apt packages and uv's Python that /build/venv points to (`FRESH=1` recreates it and redoes apt/python).
set -u
cd "$(dirname "$0")"
if [ "${FRESH:-0}" = 1 ]; then docker rm -f inkling-build 2>/dev/null; rm -f done/apt done/python; fi
if docker inspect inkling-build >/dev/null 2>&1; then
  docker start inkling-build
else
  docker run -d --name inkling-build --runtime nvidia --network host --ipc host \
    -e MAX_JOBS="${MAX_JOBS:-8}" -v "$PWD":/build nvcr.io/nvidia/l4t-jetpack:r36.4.0 bash /build/steps.sh
fi
