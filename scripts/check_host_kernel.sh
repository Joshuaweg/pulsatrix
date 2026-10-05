#!/usr/bin/env bash
# Checks that this host's kernel can run pulsatrix's HIP backend on gfx1151 (Strix Halo), roadmap
# HIP-9. A wrong VGPR count crashed gfx1151 until a kernel-mode fix: AMD asks for Linux 6.18.4 or
# newer, or Ubuntu's OEM kernel at ABI 1018 or newer (6.14.0-1018-oem, for example), which
# carries the fix. Also checks that the amdgpu driver and /dev/kfd are present.
#
#   scripts/check_host_kernel.sh            # report, exit 1 if the kernel is too old
#   scripts/check_host_kernel.sh --quiet    # print only problems (scripts/rocm-build.sh uses this)
#   scripts/check_host_kernel.sh --kernel 6.14.0-1018-oem   # check a release string instead of uname -r
#
# It doesn't change anything. To keep a good kernel from being replaced, hold its packages,
# for example: sudo apt-mark hold linux-image-$(uname -r)
set -euo pipefail

quiet=0
release="$(uname -r)"
check_devices=1
while [ $# -gt 0 ]; do
    case "$1" in
        --quiet) quiet=1 ;;
        --kernel) release="$2"; check_devices=0; shift ;;
        *) echo "check_host_kernel.sh: unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

say() { [ "$quiet" = 1 ] || echo "$@"; }
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

status=0
base="${release%%-*}"                      # 6.14.0
rest="${release#"$base"}"                  # -1018-oem
abi="$(echo "$rest" | sed -n 's/^-\([0-9]*\).*/\1/p')"
if version_ge "$base" "6.18.4"; then
    say "kernel $release: ok (6.18.4 or newer)"
elif [[ "$release" == *-oem ]] && [ -n "$abi" ] && [ "$abi" -ge 1018 ]; then
    say "kernel $release: ok (Ubuntu OEM kernel at ABI 1018 or newer)"
else
    echo "warning: kernel $release predates the gfx1151 VGPR fix; use Linux 6.18.4+ or Ubuntu's OEM kernel at ABI 1018+ (TheRock #2991)" >&2
    status=1
fi

if [ "$check_devices" = 1 ]; then
    if [ -d /sys/module/amdgpu ]; then
        say "amdgpu driver: loaded"
    else
        echo "warning: the amdgpu kernel module is not loaded" >&2
        status=1
    fi
    if [ -e /dev/kfd ]; then
        say "/dev/kfd: present"
    else
        echo "warning: /dev/kfd is missing, so no ROCm program can reach the GPU" >&2
        status=1
    fi
fi
exit "$status"
