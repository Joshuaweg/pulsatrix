#!/usr/bin/env bash
# Runs a command inside the ROCm build container (docker/Dockerfile.rocm) with this host's
# AMD GPU passed through, from the repository root.
#
#   scripts/rocm-build.sh 'cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Debug -DPULSATRIX_ENABLE_HIP=ON'
#   scripts/rocm-build.sh 'cmake --build build-hip -j"$(nproc)"'
#   scripts/rocm-build.sh './build-hip/tests/pulsatrix_tests'
#   PULSATRIX_ROCM_VERSION=7.2.4 scripts/rocm-build.sh '...'    # the previous pin
#
# Runs as the invoking uid/gid so build artifacts stay owned by you rather than root. This
# also keeps /dev/kfd reachable where access is granted by a logind seat ACL (which is
# uid-based) rather than by render/video group membership.
set -euo pipefail

# PULSATRIX_ROCM_VERSION picks the ROCm release: 10.0.0 (the default since HIP-9) or 7.2.4 (the
# previous pin). PULSATRIX_ROCM_IMAGE overrides the image name outright. Keep each version's
# build directories separate: the two releases ship different compilers.
ROCM_VERSION="${PULSATRIX_ROCM_VERSION:-10.0.0}"
case "$ROCM_VERSION" in
    7.2.4) ROCM_BASE="rocm/dev-ubuntu-24.04:7.2.4-complete" ;;
    10.0.0) ROCM_BASE="rocm/dev-ubuntu-24.04:10.0.0-full" ;;
    *) echo "rocm-build.sh: PULSATRIX_ROCM_VERSION must be 7.2.4 or 10.0.0, not $ROCM_VERSION" >&2; exit 2 ;;
esac
IMAGE="${PULSATRIX_ROCM_IMAGE:-pulsatrix-rocm:$ROCM_VERSION}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Known gfx1151 crashes depend on the host kernel as well as on ROCm (docs/gpu-profiling.md).
"$REPO_ROOT/scripts/check_host_kernel.sh" --quiet || true

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building $IMAGE from docker/Dockerfile.rocm (first run only)..." >&2
    docker build -t "$IMAGE" --build-arg "ROCM_BASE=$ROCM_BASE" -f "$REPO_ROOT/docker/Dockerfile.rocm" "$REPO_ROOT/docker"
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
