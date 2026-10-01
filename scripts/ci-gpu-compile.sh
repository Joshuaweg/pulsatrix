#!/usr/bin/env bash
# Compile-only check of the CUDA or HIP backend inside the vendor's own toolchain image.
# Needs Docker, not a GPU: GitHub-hosted runners have none, so this proves the GPU code and
# its tests compile and link -- it does not run them. Running them still needs real hardware
# (scripts/rocm-build.sh for HIP; an NVIDIA machine for CUDA).
#
#   scripts/ci-gpu-compile.sh cuda
#   scripts/ci-gpu-compile.sh hip
#
# The repository is mounted read-only and copied inside the container, so no root-owned
# build artifacts land in the working tree.
set -euo pipefail

backend="${1:?usage: $0 cuda|hip}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "$backend" in
    cuda)
        image="nvidia/cuda:12.6.3-devel-ubuntu24.04"
        extra_packages=""
        # "native" (the CMake default here) needs a visible GPU to resolve; sm_86 is the
        # project's verified RTX 3060 target.
        configure_args="-DPULSATRIX_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86"
        ;;
    hip)
        # The slim image, not -complete (29 GB): hipBLAS is the only library this project
        # links, and installing it alone keeps the job inside a hosted runner's disk budget.
        image="rocm/dev-ubuntu-24.04:7.2.4"
        extra_packages="hipblas-dev"
        configure_args="-DPULSATRIX_ENABLE_HIP=ON"
        ;;
    *)
        echo "unknown backend '$backend' (expected cuda or hip)" >&2
        exit 2
        ;;
esac

docker run --rm -v "$REPO_ROOT:/src:ro" "$image" bash -euo pipefail -c "
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        cmake git ca-certificates $extra_packages >/dev/null
    cp -r /src /work
    cd /work
    rm -rf build build-*
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release $configure_args
    cmake --build build -j\"\$(nproc)\"
"
