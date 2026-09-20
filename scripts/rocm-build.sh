#!/usr/bin/env bash
# Runs a command inside the ROCm build container (docker/Dockerfile.rocm) with this host's
# AMD GPU passed through, from the repository root.
#
#   scripts/rocm-build.sh 'cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Debug -DEXAI_ENABLE_HIP=ON'
#   scripts/rocm-build.sh 'cmake --build build-hip -j"$(nproc)"'
#   scripts/rocm-build.sh './build-hip/tests/exai_tests'
#
# Runs as the invoking uid/gid so build artifacts stay owned by you rather than root. This
# also keeps /dev/kfd reachable where access is granted by a logind seat ACL (which is
# uid-based) rather than by render/video group membership.
set -euo pipefail

IMAGE="${EXAI_ROCM_IMAGE:-exai-rocm:7.2.4}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building $IMAGE from docker/Dockerfile.rocm (first run only)..." >&2
    docker build -t "$IMAGE" -f "$REPO_ROOT/docker/Dockerfile.rocm" "$REPO_ROOT/docker"
fi

render_gid="$(getent group render | cut -d: -f3)"
video_gid="$(getent group video  | cut -d: -f3)"

exec docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    ${render_gid:+--group-add "$render_gid"} \
    ${video_gid:+--group-add "$video_gid"} \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -v "$REPO_ROOT:/work" \
    -w /work \
    "$IMAGE" bash -lc "$*"
