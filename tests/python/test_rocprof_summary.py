"""Tests for scripts/rocprof_summary.py, the post-processing half of scripts/profile_hip.sh (HIP-1)."""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import rocprof_summary as rs  # noqa: E402

FILL = "pulsatrix::(anonymous namespace)::gpu::fill_kernel(float*, float, unsigned long)"
SOFTMAX = "pulsatrix::(anonymous namespace)::gpu::softmax_rows_kernel(float const*, float*, unsigned long, unsigned long)"
TENSILE = "Cijk_Ailk_Bljk_SB_MT32x32x8_SN_1LDSB0_ISA1151_WG16_16_1_WGM8"
COPY = "__amd_rocclr_copyBuffer"
TEMPLATE = "void pulsatrix::gpu::reduce_kernel<256u>(float const*, float*, unsigned long)"


def test_op_type_names_our_kernels_tensile_and_runtime():
    assert rs.op_type(FILL) == "fill"
    assert rs.op_type(SOFTMAX) == "softmax_rows"
    assert rs.op_type(TEMPLATE) == "reduce"
    assert rs.op_type(TENSILE) == "gemm (hipBLAS)"
    assert rs.op_type(COPY) == "runtime: copyBuffer"


def write_trace(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["Kind", "Kernel_Name", "Start_Timestamp", "End_Timestamp"])
        for name, start, end in rows:
            w.writerow(["KERNEL_DISPATCH", name, start, end])


def test_summary_groups_by_op_and_measures_busy_time(tmp_path):
    trace = tmp_path / "run_kernel_trace.csv"
    # Kernels at [0,100), [150,250), [250,300) and an overlapping [280,320): busy time is the
    # union, 100 + 170 = 270 ns, over a 320 ns span.
    write_trace(trace, [(FILL, 0, 100), (TENSILE, 150, 250), (FILL, 250, 300), (SOFTMAX, 280, 320)])
    s = rs.summarize(str(trace))
    assert s.span_ns == 320
    assert s.busy_ns == 270
    assert abs(s.busy_fraction - 270 / 320) < 1e-12
    by_op = {r.op: r for r in s.ops}
    assert by_op["fill"].calls == 2 and by_op["fill"].total_ns == 150
    assert by_op["gemm (hipBLAS)"].total_ns == 100
    assert [r.op for r in s.ops] == ["fill", "gemm (hipBLAS)", "softmax_rows"]  # by total time, descending
    assert abs(sum(r.percent for r in s.ops) - 100.0) < 1e-9


def test_writes_the_op_csv(tmp_path):
    trace = tmp_path / "t.csv"
    write_trace(trace, [(FILL, 0, 1000), (FILL, 2000, 4000)])
    out = tmp_path / "ops.csv"
    rs.write_csv(rs.summarize(str(trace)), str(out))
    rows = list(csv.DictReader(open(out)))
    assert rows == [{"op": "fill", "calls": "2", "total_us": "3.000", "mean_us": "1.500", "percent": "100.00"}]


def test_empty_trace_is_an_error(tmp_path):
    trace = tmp_path / "t.csv"
    write_trace(trace, [])
    try:
        rs.summarize(str(trace))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a trace with no kernels")
