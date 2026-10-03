# System Monitoring

`SystemMonitor` ([`include/pulsatrix/system_monitor.hpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/include/pulsatrix/system_monitor.hpp))
records CPU and GPU utilization, memory and temperatures while your code runs. It samples on a
background thread and writes each sample to a log file as soon as it's taken, so you can follow
a training run live with `tail -f`.

## Usage

```cpp
#include "pulsatrix/system_monitor.hpp"

pulsatrix::SystemMonitor::Options opts;
opts.interval = std::chrono::milliseconds(1000);
opts.log_path = "run.jsonl";              // CSV: opts.format = pulsatrix::LogFormat::Csv
pulsatrix::SystemMonitor monitor(opts);

monitor.start();
monitor.mark("epoch 1");                  // labels the samples that follow and logs a "mark" record
// ... train ...
auto last = monitor.latest();             // std::optional<SystemSample>
auto all = monitor.history();             // bounded ring buffer (Options::history_capacity)
monitor.stop();                           // returns within milliseconds
```

Other methods:

- `sample_now()`: take one reading synchronously, without starting the thread.
- `on_sample(callback)`: run a callback for every sample. It runs on the monitor thread; if it
  throws, the exception is caught and reported once.
- `capabilities()`: which metrics are available on this machine, where each one comes from, or
  why it's missing.

From Python, samples come back as dicts:

```python
from pulsatrix_py import SystemMonitor

with SystemMonitor(interval_ms=1000, log_path="run.jsonl", format="jsonl") as m:
    ...  # train
```

To see it run, build and run the `system_monitor_demo` example
([`examples/system_monitor_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/system_monitor_demo.cpp)).

## Log format

JSON Lines, one object per line, with ISO-8601 UTC timestamps to the millisecond:

```json
{"event":"mark","timestamp":"2026-10-02T18:52:48.529Z","elapsed_seconds":0.000,"label":"idle"}
{"event":"sample","timestamp":"2026-10-02T18:52:49.530Z","elapsed_seconds":1.001,"label":"idle","cpu_utilization_percent":4.698,"process_cpu_percent":0,"memory_used_bytes":11615903744,"memory_total_bytes":32723628032,"process_rss_bytes":199770112,"cpu_temperature_c":86.625,"gpus":[{"index":0,"name":"AMD GPU 0x1586 (card1)","vendor":"AMD","utilization_percent":0,"memory_used_bytes":52692738048,"memory_total_bytes":103079215104,"memory_gtt_used_bytes":94412800,"temperature_c":42,"power_watts":34.032}]}
```

CSV is also available (`pulsatrix::LogFormat::Csv` in C++, `format="csv"` in Python).

## Missing metrics are never reported as zero

If the platform can't read a metric, it's reported as missing: `std::nullopt` in C++, `null` in
JSON, an empty cell in CSV and `None` in Python. It's never `0` or an estimate.
`capabilities()` tells you why, for example
`"Windows: CPU temperature not available without WMI/admin"`.

A few things to know when reading the numbers:

- CPU percentages are computed from the change in counters between samples. `start()` takes a
  baseline, so the first background sample arrives one interval later.
- `process_cpu_percent` is relative to one logical CPU, so it can go above 100 on a multi-core
  machine.
- On an AMD APU, the amdgpu power reading covers the whole chip package, not just the GPU.

## Where each metric comes from

| Metric | Linux | Windows | Other OSes |
|---|---|---|---|
| System CPU % | `/proc/stat` | `GetSystemTimes` | -- |
| Process CPU %, RSS | `/proc/self/stat`, `/proc/self/status` | `GetProcessTimes`, `GetProcessMemoryInfo` | -- |
| Memory used/total | `/proc/meminfo` (MemTotal - MemAvailable) | `GlobalMemoryStatusEx` | -- |
| CPU temperature | hwmon `k10temp`/`zenpower`/`coretemp`/`cpu_thermal` (Tctl/Package preferred), `acpitz` last | -- (needs WMI/admin) | -- |
| AMD GPU util/VRAM/GTT/temp/power | amdgpu sysfs (`/sys/class/drm/card<N>/device`) | -- | -- |
| NVIDIA GPU util/memory/temp/power | NVML, `dlopen("libnvidia-ml.so.1")` | NVML, `LoadLibrary("nvml.dll")` | -- |

Everything is detected at runtime. You don't need the NVIDIA or AMD SDK to build. If a driver is
missing, you simply get no GPU entries, and `capabilities()` says why.
