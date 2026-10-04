#!/usr/bin/env bash
# Profiles a command's GPU kernels with rocprofv3 and summarizes kernel time per op type
# (roadmap HIP-1): the baseline, and the falsifier for every HIP performance item.
#
#   scripts/profile_hip.sh <ops.csv> -- <command> [args...]
#
# Writes <ops.csv> (op, calls, total_us, mean_us, percent) and prints the GPU's busy time over
# the trace's wall span. Needs rocprofv3, so on this node run it in the ROCm container:
#
#   scripts/rocm-build.sh 'scripts/profile_hip.sh profile/cnn.csv -- ./build-hip-release/hip_profile_workloads cnn 20'
set -euo pipefail

if [ $# -lt 3 ] || [ "$2" != "--" ]; then
    echo "usage: $0 <ops.csv> -- <command> [args...]" >&2
    exit 2
fi
out="$1"
shift 2

if ! command -v rocprofv3 >/dev/null; then
    echo "rocprofv3 not found: run this inside the ROCm container (scripts/rocm-build.sh)" >&2
    exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$(dirname "$out")"

if ! rocprofv3 --kernel-trace --output-format csv -d "$tmp" -o run -- "$@" >"$tmp/command.log" 2>&1; then
    echo "profiled command failed; last lines of its output:" >&2
    tail -20 "$tmp/command.log" >&2
    exit 1
fi
trace="$(find "$tmp" -name '*kernel_trace.csv' | head -1)"
if [ -z "$trace" ]; then
    echo "rocprofv3 wrote no kernel trace; its output:" >&2
    tail -20 "$tmp/command.log" >&2
    exit 1
fi
grep -E '^(step|tagger)' "$tmp/command.log" | tail -3 || true
python3 "$(dirname "${BASH_SOURCE[0]}")/rocprof_summary.py" "$trace" "$out"
