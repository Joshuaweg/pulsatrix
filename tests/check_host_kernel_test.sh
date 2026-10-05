#!/usr/bin/env bash
# CTest: scripts/check_host_kernel.sh accepts kernels with the gfx1151 VGPR fix and rejects older
# ones (roadmap HIP-9).
set -u
script="$(dirname "$0")/../scripts/check_host_kernel.sh"
failures=0
expect() {
    "$script" --kernel "$2" --quiet 2>/dev/null
    local got=$?
    if [ "$got" != "$1" ]; then
        echo "FAIL: $2 exited $got, expected $1"
        failures=$((failures + 1))
    fi
}
expect 0 6.18.4-generic
expect 0 6.19.0
expect 0 7.0.0-34-generic
expect 0 6.14.0-1018-oem
expect 0 6.14.0-1027-oem
expect 1 6.18.3-generic
expect 1 6.14.0-1017-oem
expect 1 6.11.0-29-generic
expect 1 6.14.0-1018-generic   # ABI 1018 counts only for the OEM flavour
[ "$failures" = 0 ] && echo "check_host_kernel.sh: all cases pass"
exit "$failures"
