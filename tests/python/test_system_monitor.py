# SystemMonitor binding: samples are plain dicts, missing readings are None (never 0), the
# monitor works as a context manager, and the log is written live while it runs.
import json
import sys
import time

import pytest

import pulsatrix_py

SAMPLE_KEYS = {
    "timestamp", "timestamp_unix", "elapsed_seconds", "label", "cpu_utilization_percent",
    "process_cpu_percent", "memory_used_bytes", "memory_total_bytes", "process_rss_bytes",
    "cpu_temperature_c", "gpus",
}


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_sample_now_returns_plain_dict_with_none_or_sane_values():
    monitor = pulsatrix_py.SystemMonitor(interval_ms=50)
    time.sleep(0.05)
    s = monitor.sample_now()
    assert isinstance(s, dict)
    assert SAMPLE_KEYS <= set(s)
    assert s["timestamp"].endswith("Z") and "T" in s["timestamp"]
    for key in ("cpu_utilization_percent", "cpu_temperature_c", "memory_total_bytes"):
        assert s[key] is None or isinstance(s[key], float)
    if s["cpu_utilization_percent"] is not None:
        assert 0.0 <= s["cpu_utilization_percent"] <= 100.0
    if sys.platform.startswith("linux"):
        assert s["memory_total_bytes"] > 0
    for gpu in s["gpus"]:
        assert gpu["vendor"] in ("AMD", "NVIDIA")
        assert gpu["utilization_percent"] is None or 0.0 <= gpu["utilization_percent"] <= 100.0


def test_capabilities_report_every_metric_with_a_detail():
    caps = pulsatrix_py.SystemMonitor().capabilities()
    metrics = {c["metric"] for c in caps}
    assert {"cpu_utilization_percent", "memory_total_bytes", "cpu_temperature_c"} <= metrics
    for c in caps:
        assert isinstance(c["available"], bool)
        assert c["detail"]


def test_context_manager_logs_live_jsonl_with_marks(tmp_path):
    log = tmp_path / "monitor.jsonl"
    with pulsatrix_py.SystemMonitor(interval_ms=20, log_path=str(log)) as monitor:
        assert monitor.running
        assert monitor.latest() is None or isinstance(monitor.latest(), dict)
        monitor.mark('epoch "1"')
        # Read while still running: every record is flushed as it is written.
        assert wait_until(lambda: len(log.read_text().splitlines()) >= 3)
        assert wait_until(lambda: monitor.latest() is not None and monitor.latest()["label"] == 'epoch "1"')
    assert not monitor.running
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert records[0] == {**records[0], "event": "mark", "label": 'epoch "1"'}
    samples = [r for r in records if r["event"] == "sample"]
    assert samples and all(r["label"] == 'epoch "1"' for r in samples)
    assert all(isinstance(r["gpus"], list) for r in samples)
    assert len(monitor.history()) >= 1


def test_csv_format_and_history_capacity(tmp_path):
    log = tmp_path / "monitor.csv"
    monitor = pulsatrix_py.SystemMonitor(interval_ms=20, log_path=str(log), format="csv", history_capacity=2)
    monitor.start()
    assert wait_until(lambda: len(log.read_text().splitlines()) >= 4)
    monitor.stop()
    lines = log.read_text().splitlines()
    assert lines[0].startswith("event,timestamp,elapsed_seconds,label,cpu_utilization_percent")
    assert all(line.count(",") == lines[0].count(",") for line in lines)
    assert len(monitor.history()) == 2


def test_invalid_format_and_unwritable_path_raise(tmp_path):
    with pytest.raises(ValueError):
        pulsatrix_py.SystemMonitor(format="xml")
    monitor = pulsatrix_py.SystemMonitor(log_path=str(tmp_path / "missing_dir" / "log.jsonl"))
    with pytest.raises(RuntimeError):
        monitor.start()
    assert not monitor.running


def test_stop_is_prompt_and_idempotent():
    monitor = pulsatrix_py.SystemMonitor(interval_ms=1000)
    monitor.start()
    start = time.monotonic()
    monitor.stop()
    assert time.monotonic() - start < 0.2
    monitor.stop()
    assert not monitor.running
