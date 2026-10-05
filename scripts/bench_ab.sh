#!/usr/bin/env bash
# Compares two builds with the benchmark suite (roadmap KS-2), interleaved in ABBA order.
# Both builds see the same machine load in every round, which matters on a shared GPU where
# other programs can triple raw timings (docs/benchmarks.md). Exits 1 if the new build regressed.
#
#   scripts/bench_ab.sh OLD_BUILD_DIR NEW_BUILD_DIR [ROUNDS] [-- pulsatrix_bench run options]
#   scripts/bench_ab.sh build-master build-feature 5 -- --device hip --filter train
#
# Reports go to bench-ab/ (gitignored): old-N.json, new-N.json.
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,9p' "$0" >&2
    exit 2
fi
OLD="$1/pulsatrix_bench"
NEW="$2/pulsatrix_bench"
shift 2
ROUNDS=5
if [ $# -gt 0 ] && [ "$1" != "--" ]; then
    ROUNDS="$1"
    shift
fi
[ $# -gt 0 ] && [ "$1" = "--" ] && shift
for bin in "$OLD" "$NEW"; do
    [ -x "$bin" ] || { echo "bench_ab.sh: $bin not found; build the pulsatrix_bench target first" >&2; exit 2; }
done

OUT=bench-ab
mkdir -p "$OUT"
rm -f "$OUT"/old-*.json "$OUT"/new-*.json
# ABBA order: odd rounds run old then new, even rounds new then old, so a drift in machine state
# (clock frequency, heat, a background job) lands on both builds equally.
run_side() {
    echo "round $2/$ROUNDS: $1" >&2
    local bin="$OLD"
    [ "$1" = new ] && bin="$NEW"
    "$bin" run --label "$1" "${ARGS[@]}" -o "$OUT/$1-$2.json" >/dev/null
}
ARGS=("$@")
for ((round = 1; round <= ROUNDS; round++)); do
    if ((round % 2 == 1)); then
        run_side old "$round"
        run_side new "$round"
    else
        run_side new "$round"
        run_side old "$round"
    fi
done
exec "$NEW" compare --baseline "$OUT"/old-*.json --candidate "$OUT"/new-*.json
