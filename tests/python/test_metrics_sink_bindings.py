# Phase 5 Mission 1, Objective 4: MetricsSink binding + PyMetricsSink trampoline.
# MetricsSink is the one class in this mission with a genuine polymorphic Python-
# subclassing surface (charter intent: TensorBoard/W&B-style writers belong in Python).
# _drive_metrics_sink_for_test mirrors metrics_sink_test.cpp's own
# CallableThroughBasePointer test -- calling both methods through a base MetricsSink&,
# proving the trampoline is actually reachable from a real (non-Python) call site.
import exai_py


class RecordingSink(exai_py.MetricsSink):
    def __init__(self):
        super().__init__()
        self.scalars = []
        self.histograms = []

    def log_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def log_histogram(self, tag, values, step):
        self.histograms.append((tag, values.at([0]), values.at([1]), step))


def test_python_subclassed_metrics_sink_receives_calls_from_cpp():
    sink = RecordingSink()

    exai_py._drive_metrics_sink_for_test(sink)

    assert sink.scalars == [("loss", 0.5, 3)]
    assert sink.histograms == [("weights", 1.0, 2.0, 3)]


def test_no_op_metrics_sink_does_not_raise():
    sink = exai_py.NoOpMetricsSink()

    # Must not raise -- exercises the one concrete C++ writer the charter ships.
    exai_py._drive_metrics_sink_for_test(sink)
