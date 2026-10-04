#!/usr/bin/env python3
"""Summarizes a rocprofv3 kernel trace by op type (roadmap HIP-1).

    python3 scripts/rocprof_summary.py <run_kernel_trace.csv> <ops.csv>

Writes one CSV row per op type (calls, total and mean time, share of kernel time) and prints
the GPU's busy time over the trace's wall span: the falsifier for HIP-2, HIP-4 and HIP-5.
"""
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class OpRow:
    op: str
    calls: int
    total_ns: int
    percent: float

    @property
    def mean_ns(self):
        return self.total_ns / self.calls


@dataclass
class Summary:
    ops: list = field(default_factory=list)
    span_ns: int = 0
    busy_ns: int = 0
    kernel_ns: int = 0

    @property
    def busy_fraction(self):
        return self.busy_ns / self.span_ns if self.span_ns else 0.0


def op_type(kernel_name):
    """The op a kernel belongs to: our kernels by function name, hipBLAS's Tensile kernels as one
    GEMM op, and HIP runtime helpers (copies, fills) under "runtime:"."""
    if kernel_name.startswith("Cijk_"):
        return "gemm (hipBLAS)"
    if kernel_name.startswith("__amd_rocclr_"):
        return "runtime: " + kernel_name[len("__amd_rocclr_"):]
    name = kernel_name.replace("(anonymous namespace)", "anon")
    name = name.split("(", 1)[0]                 # drop the argument list
    name = re.sub(r"<[^<>]*>", "", name)         # drop template arguments
    name = name.split()[-1]                      # drop a leading return type ("void ...")
    name = name.split("::")[-1]                  # drop namespaces
    return name[: -len("_kernel")] if name.endswith("_kernel") else name


def summarize(trace_path):
    """Reads a rocprofv3 *_kernel_trace.csv. Raises ValueError if it holds no kernels."""
    intervals = []
    totals = defaultdict(lambda: [0, 0])  # op -> [calls, total_ns]
    with open(trace_path, newline="") as f:
        for row in csv.DictReader(f):
            start, end = int(row["Start_Timestamp"]), int(row["End_Timestamp"])
            intervals.append((start, end))
            entry = totals[op_type(row["Kernel_Name"])]
            entry[0] += 1
            entry[1] += end - start
    if not intervals:
        raise ValueError(f"no kernel dispatches in {trace_path}")

    # Busy time is the union of the kernels' intervals (they can overlap across streams).
    intervals.sort()
    busy = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            busy += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    busy += cur_end - cur_start

    kernel_ns = sum(total for _, total in totals.values())
    ops = [OpRow(op, calls, total, 100.0 * total / kernel_ns if kernel_ns else 0.0)
           for op, (calls, total) in totals.items()]
    ops.sort(key=lambda r: (-r.total_ns, r.op))
    span = max(e for _, e in intervals) - intervals[0][0]
    return Summary(ops=ops, span_ns=span, busy_ns=busy, kernel_ns=kernel_ns)


def write_csv(summary, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op", "calls", "total_us", "mean_us", "percent"])
        for r in summary.ops:
            w.writerow([r.op, r.calls, f"{r.total_ns / 1000:.3f}", f"{r.mean_ns / 1000:.3f}", f"{r.percent:.2f}"])


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    s = summarize(argv[1])
    write_csv(s, argv[2])
    print(f"kernels: {sum(r.calls for r in s.ops)} dispatches, {s.kernel_ns / 1e6:.3f} ms of kernel time")
    print(f"GPU busy: {s.busy_ns / 1e6:.3f} ms of a {s.span_ns / 1e6:.3f} ms span ({100 * s.busy_fraction:.1f}%)")
    for r in s.ops[:10]:
        print(f"  {r.op:<28} {r.calls:>8} calls {r.total_ns / 1e3:>12.1f} us {r.percent:>6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
