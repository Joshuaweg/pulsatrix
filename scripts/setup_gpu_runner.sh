#!/usr/bin/env bash
# Sets this machine up as pulsatrix's self-hosted GPU runner (roadmap KS-8): a dedicated user, the
# GitHub Actions runner (version pinned, checksum verified), registered with the gfx1151 label and
# installed as a service. See docs/gpu-ci.md first: it covers the repository settings this needs.
#
#   sudo scripts/setup_gpu_runner.sh TOKEN
#
# TOKEN: from https://github.com/Joshuaweg/pulsatrix/settings/actions/runners/new (valid one hour).
# Run it again to re-register (for example after removing the runner on GitHub).
#
# The runner user gets no sudo. Its only extra group is docker, which the tests need for the ROCm
# container (scripts/rocm-build.sh). Docker access is close to root access on this machine, which
# is why the workflow (.github/workflows/gpu.yml) runs only code pushed to this repository.
set -euo pipefail

REPO_URL="https://github.com/Joshuaweg/pulsatrix"
RUNNER_VERSION="2.338.0"
RUNNER_SHA256="af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32"
RUNNER_USER="pulsatrix-runner"
LABELS="gfx1151"

if [ "$(id -u)" -ne 0 ]; then
    echo "setup_gpu_runner.sh: run with sudo" >&2
    exit 2
fi
if [ $# -ne 1 ] || [ -z "$1" ]; then
    sed -n '2,9p' "$0" >&2
    exit 2
fi
TOKEN="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The checks the jobs would otherwise fail on, before anything is changed.
"$REPO_ROOT/scripts/check_host_kernel.sh"
command -v docker >/dev/null || { echo "setup_gpu_runner.sh: docker isn't installed" >&2; exit 1; }
getent group docker >/dev/null || { echo "setup_gpu_runner.sh: there is no docker group" >&2; exit 1; }
docker image inspect pulsatrix-rocm:10.0.0 >/dev/null 2>&1 ||
    echo "note: the pulsatrix-rocm:10.0.0 image isn't built yet; the first CI run builds it (about 20 minutes)"

if ! id "$RUNNER_USER" >/dev/null 2>&1; then
    useradd --create-home --shell /bin/bash "$RUNNER_USER"
fi
usermod -aG docker "$RUNNER_USER"
HOME_DIR="$(getent passwd "$RUNNER_USER" | cut -d: -f6)"
DIR="$HOME_DIR/actions-runner"

if [ -f "$DIR/.runner" ]; then
    # Re-registering: stop the service and forget the local registration; --replace below takes
    # over the runner's entry on GitHub.
    echo "Replacing the previous registration..."
    (cd "$DIR" && { ./svc.sh stop || true; ./svc.sh uninstall || true; })
    rm -f "$DIR/.runner" "$DIR/.credentials" "$DIR/.credentials_rsaparams" "$DIR/.service"
fi

sudo -u "$RUNNER_USER" mkdir -p "$DIR"
TARBALL="actions-runner-linux-x64-$RUNNER_VERSION.tar.gz"
if [ ! -x "$DIR/config.sh" ]; then
    curl -fsSL -o "$DIR/$TARBALL" "https://github.com/actions/runner/releases/download/v$RUNNER_VERSION/$TARBALL"
    echo "$RUNNER_SHA256  $DIR/$TARBALL" | sha256sum -c -
    sudo -u "$RUNNER_USER" tar -xzf "$DIR/$TARBALL" -C "$DIR"
    rm -f "$DIR/$TARBALL"
fi

sudo -u "$RUNNER_USER" bash -c "cd '$DIR' && ./config.sh --unattended --url '$REPO_URL' --token '$TOKEN' \
    --name '$(hostname)-gfx1151' --labels '$LABELS' --work _work --replace"
(cd "$DIR" && ./svc.sh install "$RUNNER_USER" && ./svc.sh start)

echo
echo "Done. The runner shows as Idle at $REPO_URL/settings/actions/runners."
echo "Optional: published ResNet18/VGG16 checks, see docs/gpu-ci.md (PULSATRIX_TORCHVISION_DIR)."
