#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "pulsatrix/system_monitor.hpp"
#include "pulsatrix/system_monitor_detail.hpp"

namespace pulsatrix {
namespace {

namespace fs = std::filesystem;
using namespace std::chrono_literals;
using detail::CpuTimes;
using detail::LinuxSources;
using detail::RawReading;

// A unique scratch directory, removed on destruction.
class TempDir {
public:
    TempDir() {
        std::random_device rd;
        path_ = fs::temp_directory_path() / ("pulsatrix_sysmon_" + std::to_string(rd()) + std::to_string(rd()));
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    [[nodiscard]] std::string str() const { return path_.string(); }

    void write(const std::string& relative, const std::string& contents) const {
        const fs::path file = path_ / relative;
        fs::create_directories(file.parent_path());
        std::ofstream(file, std::ios::binary | std::ios::trunc) << contents;
    }

    void mkdir(const std::string& relative) const { fs::create_directories(path_ / relative); }

private:
    fs::path path_;
};

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

bool wait_until(const std::function<bool()>& predicate, std::chrono::milliseconds timeout = 5000ms) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) {
            return true;
        }
        std::this_thread::sleep_for(2ms);
    }
    return predicate();
}

const MetricCapability* find_cap(const std::vector<MetricCapability>& caps, const std::string& metric) {
    for (const MetricCapability& cap : caps) {
        if (cap.metric == metric) {
            return &cap;
        }
    }
    return nullptr;
}

// Deterministic source: the system CPU counter advances 50 busy per 100 total each read
// (exactly 50 %), RSS and CPU temperature are always missing, one GPU with partial data.
class FakeSources : public detail::PlatformSources {
public:
    explicit FakeSources(std::shared_ptr<std::atomic<int>> reads) : reads_(std::move(reads)) {}

    RawReading read() override {
        const int n = ++*reads_;
        RawReading raw;
        raw.system_cpu = CpuTimes{50.0 * n, 100.0 * n};
        raw.process_cpu_seconds = 0.0;
        raw.memory_total_bytes = 1000.0;
        raw.memory_used_bytes = 250.0;
        GpuSample gpu;
        gpu.index = 0;
        gpu.name = "Fake \"GPU\"";
        gpu.vendor = "AMD";
        gpu.utilization_percent = 42.0;
        gpu.power_watts = 12.5;
        raw.gpus.push_back(gpu);
        return raw;
    }

    std::vector<MetricCapability> capabilities() override {
        return {MetricCapability{"cpu_temperature_c", false, "fake: no sensor"}};
    }

private:
    std::shared_ptr<std::atomic<int>> reads_;
};

SystemMonitor make_fake_monitor(SystemMonitor::Options options,
                                std::shared_ptr<std::atomic<int>> reads = std::make_shared<std::atomic<int>>(0)) {
    return SystemMonitor(std::move(options), std::make_unique<FakeSources>(std::move(reads)));
}

SystemMonitor::Options fast_options(const std::string& log_path = "") {
    SystemMonitor::Options options;
    options.interval = 20ms;
    options.log_path = log_path;
    return options;
}

// ------------------------------------------------------------------------------------------
// Parsers on fake /proc and /sys trees
// ------------------------------------------------------------------------------------------

TEST(SystemMonitorParsers, CpuPercentFromKnownDeltas) {
    const auto pct = detail::cpu_percent_from_deltas(CpuTimes{200, 1000}, CpuTimes{300, 1200});
    ASSERT_TRUE(pct.has_value());
    EXPECT_DOUBLE_EQ(*pct, 50.0);
    EXPECT_FALSE(detail::cpu_percent_from_deltas(CpuTimes{200, 1000}, CpuTimes{200, 1000}).has_value());
    EXPECT_DOUBLE_EQ(*detail::cpu_percent_from_deltas(CpuTimes{0, 0}, CpuTimes{120, 100}), 100.0);  // clamped
}

TEST(SystemMonitorParsers, ProcStatAggregateLineAndMonitorDelta) {
    TempDir root;
    // busy = user+nice+system+irq+softirq+steal; idle = idle+iowait.
    root.write("proc/stat", "cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 1 2 3 4 5 6 7 8 0 0\nintr 1 2\n");
    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    const auto times = sources.read_system_cpu_times();
    ASSERT_TRUE(times.has_value());
    EXPECT_DOUBLE_EQ(times->busy, 200.0);
    EXPECT_DOUBLE_EQ(times->total, 1000.0);

    // End to end: the monitor primes its sample_now() baseline at construction.
    SystemMonitor monitor(SystemMonitor::Options{},
                          std::make_unique<LinuxSources>(root.str() + "/proc", root.str() + "/sys", 100));
    root.write("proc/stat", "cpu  150 0 150 800 100 0 0 0 0 0\n");
    const SystemSample s = monitor.sample_now();
    ASSERT_TRUE(s.cpu_utilization_percent.has_value());
    EXPECT_DOUBLE_EQ(*s.cpu_utilization_percent, 50.0);
    // No /proc/self/stat in the fake tree -> no process rate, never a 0.
    EXPECT_FALSE(s.process_cpu_percent.has_value());
}

TEST(SystemMonitorParsers, ProcessCpuSecondsHandlesParenthesesInComm) {
    TempDir root;
    // pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime ...
    root.write("proc/self/stat", "4242 (my (weird) proc) R 1 2 3 4 5 6 7 8 9 10 250 50 0 0 20 0 1 0\n");
    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    const auto seconds = sources.read_process_cpu_seconds();
    ASSERT_TRUE(seconds.has_value());
    EXPECT_DOUBLE_EQ(*seconds, 3.0);
}

TEST(SystemMonitorParsers, MeminfoAndVmRss) {
    TempDir root;
    root.write("proc/meminfo", "MemTotal:       16000000 kB\nMemFree:  1000 kB\nMemAvailable:    4000000 kB\n");
    root.write("proc/self/status", "Name:\tx\nVmPeak:\t 9999 kB\nVmRSS:\t    2048 kB\nThreads: 3\n");
    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    EXPECT_DOUBLE_EQ(*sources.read_memory_total_bytes(), 16000000.0 * 1024);
    EXPECT_DOUBLE_EQ(*sources.read_memory_used_bytes(), 12000000.0 * 1024);
    EXPECT_DOUBLE_EQ(*sources.read_process_rss_bytes(), 2048.0 * 1024);
}

TEST(SystemMonitorParsers, MissingMemAvailableIsNulloptWithReason) {
    TempDir root;
    root.write("proc/meminfo", "MemTotal:       16000000 kB\nMemFree:  1000 kB\n");
    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    EXPECT_TRUE(sources.read_memory_total_bytes().has_value());
    EXPECT_FALSE(sources.read_memory_used_bytes().has_value());
    const auto caps = sources.capabilities();
    const MetricCapability* used = find_cap(caps, "memory_used_bytes");
    ASSERT_NE(used, nullptr);
    EXPECT_FALSE(used->available);
    EXPECT_NE(used->detail.find("MemAvailable"), std::string::npos);
    EXPECT_TRUE(find_cap(caps, "memory_total_bytes")->available);
}

TEST(SystemMonitorParsers, HwmonPrefersK10tempOverAcpitz) {
    TempDir root;
    root.write("sys/class/hwmon/hwmon0/name", "acpitz\n");
    root.write("sys/class/hwmon/hwmon0/temp1_input", "40000\n");
    root.write("sys/class/hwmon/hwmon1/name", "nvme\n");
    root.write("sys/class/hwmon/hwmon1/temp1_input", "33000\n");
    root.write("sys/class/hwmon/hwmon2/name", "k10temp\n");
    root.write("sys/class/hwmon/hwmon2/temp1_label", "Tctl\n");
    root.write("sys/class/hwmon/hwmon2/temp1_input", "55125\n");
    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    ASSERT_TRUE(sources.read_cpu_temperature_c().has_value());
    EXPECT_DOUBLE_EQ(*sources.read_cpu_temperature_c(), 55.125);
}

TEST(SystemMonitorParsers, HwmonPrefersTctlLabelOverTemp1) {
    TempDir root;
    root.write("sys/class/hwmon/hwmon3/name", "k10temp\n");
    root.write("sys/class/hwmon/hwmon3/temp1_label", "Tccd1\n");
    root.write("sys/class/hwmon/hwmon3/temp1_input", "50000\n");
    root.write("sys/class/hwmon/hwmon3/temp2_label", "Tctl\n");
    root.write("sys/class/hwmon/hwmon3/temp2_input", "61000\n");
    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    EXPECT_DOUBLE_EQ(*sources.read_cpu_temperature_c(), 61.0);

    TempDir coretemp;
    coretemp.write("sys/class/hwmon/hwmon0/name", "coretemp\n");
    coretemp.write("sys/class/hwmon/hwmon0/temp2_label", "Core 0\n");
    coretemp.write("sys/class/hwmon/hwmon0/temp2_input", "45000\n");
    coretemp.write("sys/class/hwmon/hwmon0/temp1_label", "Package id 0\n");
    coretemp.write("sys/class/hwmon/hwmon0/temp1_input", "47000\n");
    LinuxSources intel(coretemp.str() + "/proc", coretemp.str() + "/sys", 100);
    EXPECT_DOUBLE_EQ(*intel.read_cpu_temperature_c(), 47.0);
}

TEST(SystemMonitorParsers, AcpitzIsLastResortAndNoSensorIsNulloptWithReason) {
    TempDir root;
    root.write("sys/class/hwmon/hwmon0/name", "acpitz\n");
    root.write("sys/class/hwmon/hwmon0/temp1_input", "40000\n");
    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    EXPECT_DOUBLE_EQ(*sources.read_cpu_temperature_c(), 40.0);

    TempDir empty;
    empty.write("sys/class/hwmon/hwmon0/name", "nvme\n");
    empty.write("sys/class/hwmon/hwmon0/temp1_input", "33000\n");
    LinuxSources none(empty.str() + "/proc", empty.str() + "/sys", 100);
    EXPECT_FALSE(none.read_cpu_temperature_c().has_value());
    const auto caps = none.capabilities();
    const MetricCapability* cap = find_cap(caps, "cpu_temperature_c");
    ASSERT_NE(cap, nullptr);
    EXPECT_FALSE(cap->available);
    EXPECT_NE(cap->detail.find("k10temp"), std::string::npos);
}

TEST(SystemMonitorParsers, AmdDrmDetectionSkipsConnectorsAndOtherVendors) {
    TempDir root;
    const std::string drm = "sys/class/drm/";
    root.write(drm + "card0/device/vendor", "0x10de\n");  // NVIDIA: NVML's job, not sysfs
    root.write(drm + "card1/device/vendor", "0x1002\n");
    root.write(drm + "card1/device/device", "0x1586\n");
    root.write(drm + "card1/device/gpu_busy_percent", "37\n");
    root.write(drm + "card1/device/mem_info_vram_used", "1073741824\n");
    root.write(drm + "card1/device/mem_info_vram_total", "103079215104\n");
    root.write(drm + "card1/device/mem_info_gtt_used", "2147483648\n");
    root.write(drm + "card1/device/hwmon/hwmon5/name", "amdgpu\n");
    root.write(drm + "card1/device/hwmon/hwmon5/temp1_label", "edge\n");
    root.write(drm + "card1/device/hwmon/hwmon5/temp1_input", "48000\n");
    root.write(drm + "card1/device/hwmon/hwmon5/power1_average", "15500000\n");
    root.write(drm + "card1-DP-1/device/vendor", "0x1002\n");  // connector entry
    root.mkdir(drm + "renderD128");
    // A second AMD card that only exposes power1_input and nothing else.
    root.write(drm + "card10/device/vendor", "0x1002\n");
    root.write(drm + "card10/device/hwmon/hwmon9/power1_input", "2000000\n");

    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    const auto devices = sources.find_amd_gpus();
    ASSERT_EQ(devices.size(), 2u);
    EXPECT_EQ(devices[0].card, "card1");
    EXPECT_EQ(devices[1].card, "card10");

    const GpuSample gpu = sources.read_amd_gpu(devices[0], 0);
    EXPECT_EQ(gpu.vendor, "AMD");
    EXPECT_EQ(gpu.name, "AMD GPU 0x1586 (card1)");
    EXPECT_DOUBLE_EQ(*gpu.utilization_percent, 37.0);
    EXPECT_DOUBLE_EQ(*gpu.memory_used_bytes, 1073741824.0);
    EXPECT_DOUBLE_EQ(*gpu.memory_total_bytes, 103079215104.0);
    EXPECT_DOUBLE_EQ(*gpu.memory_gtt_used_bytes, 2147483648.0);
    EXPECT_DOUBLE_EQ(*gpu.temperature_c, 48.0);
    EXPECT_DOUBLE_EQ(*gpu.power_watts, 15.5);  // microwatts -> watts

    const GpuSample sparse = sources.read_amd_gpu(devices[1], 1);
    EXPECT_EQ(sparse.index, 1);
    EXPECT_DOUBLE_EQ(*sparse.power_watts, 2.0);
    EXPECT_FALSE(sparse.utilization_percent.has_value());
    EXPECT_FALSE(sparse.memory_used_bytes.has_value());
    EXPECT_FALSE(sparse.temperature_c.has_value());

    const auto caps = sources.capabilities();
    EXPECT_TRUE(find_cap(caps, "gpu0.utilization_percent")->available);
    const MetricCapability* missing = find_cap(caps, "gpu1.utilization_percent");
    ASSERT_NE(missing, nullptr);
    EXPECT_FALSE(missing->available);
    EXPECT_NE(missing->detail.find("card10"), std::string::npos);
    EXPECT_TRUE(find_cap(caps, "amd_gpus")->available);
}

TEST(SystemMonitorParsers, EmptyTreeGivesNulloptsAndReasonsForEverything) {
    TempDir root;
    LinuxSources sources(root.str() + "/proc", root.str() + "/sys", 100);
    const RawReading raw = sources.read();
    EXPECT_FALSE(raw.system_cpu.has_value());
    EXPECT_FALSE(raw.process_cpu_seconds.has_value());
    EXPECT_FALSE(raw.memory_total_bytes.has_value());
    EXPECT_FALSE(raw.memory_used_bytes.has_value());
    EXPECT_FALSE(raw.process_rss_bytes.has_value());
    EXPECT_FALSE(raw.cpu_temperature_c.has_value());
    EXPECT_TRUE(raw.gpus.empty());
    const auto caps = sources.capabilities();
    ASSERT_FALSE(caps.empty());
    for (const MetricCapability& cap : caps) {
        EXPECT_FALSE(cap.available) << cap.metric;
        EXPECT_FALSE(cap.detail.empty()) << cap.metric;
    }

    // Through the monitor: missing stays missing.
    SystemMonitor monitor(SystemMonitor::Options{},
                          std::make_unique<LinuxSources>(root.str() + "/proc", root.str() + "/sys", 100));
    const SystemSample s = monitor.sample_now();
    EXPECT_FALSE(s.cpu_utilization_percent.has_value());
    EXPECT_FALSE(s.memory_total_bytes.has_value());
    EXPECT_FALSE(s.cpu_temperature_c.has_value());
}

// ------------------------------------------------------------------------------------------
// Log record formatting
// ------------------------------------------------------------------------------------------

TEST(SystemMonitorFormat, Iso8601UtcWithMilliseconds) {
    const std::chrono::system_clock::time_point tp{std::chrono::milliseconds(1700000000123LL)};
    EXPECT_EQ(format_iso8601_utc(tp), "2023-11-14T22:13:20.123Z");
    EXPECT_EQ(format_iso8601_utc(std::chrono::system_clock::time_point{}), "1970-01-01T00:00:00.000Z");
}

TEST(SystemMonitorFormat, JsonEscaping) {
    EXPECT_EQ(detail::json_escape("a\"b\\c\nd\te\x01"), "a\\\"b\\\\c\\nd\\te\\u0001");
    EXPECT_EQ(detail::json_escape("\xc3\xa9psilon"), "\xc3\xa9psilon");  // UTF-8 passes through
}

TEST(SystemMonitorFormat, NumbersAreLocaleFreeAndIntegralWithoutDecimals) {
    EXPECT_EQ(detail::format_number(103079215104.0), "103079215104");
    EXPECT_EQ(detail::format_number(12.3456), "12.346");
    EXPECT_EQ(detail::format_number(-0.0), "0");
    EXPECT_EQ(detail::format_number(std::numeric_limits<double>::quiet_NaN()), "");
}

TEST(SystemMonitorFormat, JsonSampleUsesNullForMissingValues) {
    SystemSample s;
    s.timestamp = std::chrono::system_clock::time_point{std::chrono::milliseconds(1700000000123LL)};
    s.elapsed_seconds = 1.5;
    s.cpu_utilization_percent = 12.5;
    s.memory_total_bytes = 1024.0;
    s.label = "epoch \"3\"\n";
    GpuSample gpu;
    gpu.index = 0;
    gpu.name = "AMD GPU 0x1586 (card1)";
    gpu.vendor = "AMD";
    gpu.utilization_percent = 37.0;
    s.gpus.push_back(gpu);
    EXPECT_EQ(sample_to_json(s),
              "{\"event\":\"sample\",\"timestamp\":\"2023-11-14T22:13:20.123Z\",\"elapsed_seconds\":1.500,"
              "\"label\":\"epoch \\\"3\\\"\\n\",\"cpu_utilization_percent\":12.500,\"process_cpu_percent\":null,"
              "\"memory_used_bytes\":null,\"memory_total_bytes\":1024,\"process_rss_bytes\":null,"
              "\"cpu_temperature_c\":null,\"gpus\":[{\"index\":0,\"name\":\"AMD GPU 0x1586 (card1)\","
              "\"vendor\":\"AMD\",\"utilization_percent\":37,\"memory_used_bytes\":null,"
              "\"memory_total_bytes\":null,\"memory_gtt_used_bytes\":null,\"temperature_c\":null,"
              "\"power_watts\":null}]}");
    EXPECT_EQ(detail::mark_to_json(s.timestamp, 2.0, "a\\b"),
              "{\"event\":\"mark\",\"timestamp\":\"2023-11-14T22:13:20.123Z\",\"elapsed_seconds\":2,"
              "\"label\":\"a\\\\b\"}");
}

TEST(SystemMonitorFormat, CsvHeaderRowsAndEmptyCells) {
    EXPECT_EQ(detail::csv_header(1),
              "event,timestamp,elapsed_seconds,label,cpu_utilization_percent,process_cpu_percent,"
              "memory_used_bytes,memory_total_bytes,process_rss_bytes,cpu_temperature_c,"
              "gpu0_util,gpu0_mem_used,gpu0_mem_total,gpu0_gtt_used,gpu0_temp,gpu0_power");
    SystemSample s;
    s.timestamp = std::chrono::system_clock::time_point{std::chrono::milliseconds(1700000000123LL)};
    s.elapsed_seconds = 1.0;
    s.cpu_utilization_percent = 50.0;
    s.label = "phase, \"two\"";
    EXPECT_EQ(detail::sample_to_csv(s, 1),
              "sample,2023-11-14T22:13:20.123Z,1,\"phase, \"\"two\"\"\",50,,,,,,,,,,,");
    EXPECT_EQ(detail::mark_to_csv(s.timestamp, 1.0, "m", 1), "mark,2023-11-14T22:13:20.123Z,1,m,,,,,,,,,,,,");
    // Every row has as many cells as the header.
    auto cells = [](const std::string& row) { return std::count(row.begin(), row.end(), ',') + 1; };
    EXPECT_EQ(cells(detail::sample_to_csv(SystemSample{}, 1)), cells(detail::csv_header(1)));
    EXPECT_EQ(cells(detail::mark_to_csv(s.timestamp, 1.0, "m", 1)), cells(detail::csv_header(1)));
}

TEST(SystemMonitorLog, JsonLinesAreWrittenLiveOnePerSample) {
    TempDir dir;
    const std::string path = dir.str() + "/monitor.jsonl";
    SystemMonitor monitor = make_fake_monitor(fast_options(path));
    monitor.start();
    monitor.mark("warmup \"phase\"");
    // Read while the monitor is still running: lines must already be flushed.
    ASSERT_TRUE(wait_until([&] { return read_lines(path).size() >= 4; }));
    EXPECT_TRUE(monitor.running());
    const std::vector<std::string> lines = read_lines(path);
    EXPECT_EQ(lines[0].rfind("{\"event\":\"mark\"", 0), 0u);
    EXPECT_NE(lines[0].find("\"label\":\"warmup \\\"phase\\\"\""), std::string::npos);
    for (std::size_t i = 1; i < lines.size(); ++i) {
        EXPECT_EQ(lines[i].rfind("{\"event\":\"sample\"", 0), 0u) << lines[i];
        EXPECT_EQ(lines[i].back(), '}');
        EXPECT_NE(lines[i].find("\"cpu_utilization_percent\":50,"), std::string::npos) << lines[i];
        EXPECT_NE(lines[i].find("\"cpu_temperature_c\":null"), std::string::npos);
        EXPECT_NE(lines[i].find("\"name\":\"Fake \\\"GPU\\\"\""), std::string::npos);
        EXPECT_NE(lines[i].find("\"label\":\"warmup \\\"phase\\\"\""), std::string::npos);
    }
    monitor.stop();
}

TEST(SystemMonitorLog, CsvHasHeaderAndConsistentColumns) {
    TempDir dir;
    const std::string path = dir.str() + "/monitor.csv";
    SystemMonitor::Options options = fast_options(path);
    options.format = LogFormat::Csv;
    SystemMonitor monitor = make_fake_monitor(options);
    monitor.start();
    ASSERT_TRUE(wait_until([&] { return read_lines(path).size() >= 3; }));
    monitor.mark("phase 2");
    monitor.stop();
    const std::vector<std::string> lines = read_lines(path);
    ASSERT_GE(lines.size(), 4u);
    EXPECT_EQ(lines[0], detail::csv_header(1));
    auto cells = [](const std::string& row) { return std::count(row.begin(), row.end(), ',') + 1; };
    for (const std::string& line : lines) {
        EXPECT_EQ(cells(line), cells(lines[0])) << line;
    }
    EXPECT_EQ(lines.back().rfind("mark,", 0), 0u);

    // A restart appends (no second header, no truncation).
    const std::size_t before = lines.size();
    monitor.start();
    ASSERT_TRUE(wait_until([&] { return read_lines(path).size() > before; }));
    monitor.stop();
    const std::vector<std::string> after = read_lines(path);
    EXPECT_EQ(std::count(after.begin(), after.end(), lines[0]), 1);
}

// ------------------------------------------------------------------------------------------
// Lifecycle
// ------------------------------------------------------------------------------------------

TEST(SystemMonitorLifecycle, StartStopAreIdempotent) {
    SystemMonitor monitor = make_fake_monitor(fast_options());
    EXPECT_FALSE(monitor.running());
    monitor.stop();
    monitor.start();
    monitor.start();
    EXPECT_TRUE(monitor.running());
    monitor.stop();
    monitor.stop();
    EXPECT_FALSE(monitor.running());
}

TEST(SystemMonitorLifecycle, StopReturnsPromptlyWithLongInterval) {
    SystemMonitor::Options options;
    options.interval = 1000ms;
    SystemMonitor monitor = make_fake_monitor(options);
    monitor.start();
    std::this_thread::sleep_for(30ms);
    const auto t0 = std::chrono::steady_clock::now();
    monitor.stop();
    const auto took = std::chrono::steady_clock::now() - t0;
    EXPECT_LT(took, 200ms);
}

TEST(SystemMonitorLifecycle, LatestIsEmptyBeforeFirstSample) {
    SystemMonitor::Options options;
    options.interval = 1000ms;
    SystemMonitor monitor = make_fake_monitor(options);
    EXPECT_FALSE(monitor.latest().has_value());
    monitor.start();
    EXPECT_FALSE(monitor.latest().has_value());
    EXPECT_TRUE(monitor.history().empty());
}

TEST(SystemMonitorLifecycle, HistoryIsBoundedAndCallbackRuns) {
    SystemMonitor::Options options = fast_options();
    options.history_capacity = 3;
    SystemMonitor monitor = make_fake_monitor(options);
    std::atomic<int> calls{0};
    monitor.on_sample([&](const SystemSample&) { ++calls; });
    monitor.start();
    ASSERT_TRUE(wait_until([&] { return calls.load() >= 6; }));
    monitor.stop();
    const std::vector<SystemSample> history = monitor.history();
    ASSERT_EQ(history.size(), 3u);
    EXPECT_LT(history[0].elapsed_seconds, history[1].elapsed_seconds);
    EXPECT_LT(history[1].elapsed_seconds, history[2].elapsed_seconds);
    ASSERT_TRUE(monitor.latest().has_value());
    EXPECT_DOUBLE_EQ(monitor.latest()->elapsed_seconds, history[2].elapsed_seconds);
    EXPECT_DOUBLE_EQ(*history[2].cpu_utilization_percent, 50.0);
}

TEST(SystemMonitorLifecycle, ThrowingCallbackDoesNotKillTheThread) {
    SystemMonitor monitor = make_fake_monitor(fast_options());
    std::atomic<int> calls{0};
    monitor.on_sample([&](const SystemSample&) {
        ++calls;
        throw std::runtime_error("callback failure (expected in this test)");
    });
    monitor.start();
    ASSERT_TRUE(wait_until([&] { return calls.load() >= 3; }));
    EXPECT_TRUE(monitor.running());
    monitor.stop();
}

TEST(SystemMonitorLifecycle, StopFromCallbackIsRejected) {
    SystemMonitor monitor = make_fake_monitor(fast_options());
    std::atomic<bool> rejected{false};
    monitor.on_sample([&](const SystemSample&) {
        try {
            monitor.stop();
        } catch (const std::logic_error&) {
            rejected = true;
        }
    });
    monitor.start();
    ASSERT_TRUE(wait_until([&] { return rejected.load(); }));
    monitor.on_sample({});
    monitor.stop();
}

TEST(SystemMonitorLifecycle, StartThrowsOnUnwritableLogPath) {
    TempDir dir;
    SystemMonitor monitor = make_fake_monitor(fast_options(dir.str() + "/no_such_dir/monitor.jsonl"));
    EXPECT_THROW(monitor.start(), std::runtime_error);
    EXPECT_FALSE(monitor.running());
}

TEST(SystemMonitorLifecycle, InvalidIntervalThrows) {
    SystemMonitor::Options options;
    options.interval = 0ms;
    EXPECT_THROW(make_fake_monitor(options), std::invalid_argument);
}

TEST(SystemMonitorLifecycle, MarkTagsSubsequentSamples) {
    SystemMonitor monitor = make_fake_monitor(fast_options());
    EXPECT_EQ(monitor.sample_now().label, "");
    monitor.mark("epoch 3");
    EXPECT_EQ(monitor.sample_now().label, "epoch 3");
    monitor.start();
    ASSERT_TRUE(wait_until([&] { return monitor.latest().has_value(); }));
    EXPECT_EQ(monitor.latest()->label, "epoch 3");
    monitor.mark("explain:lrp");
    ASSERT_TRUE(wait_until([&] { return monitor.latest() && monitor.latest()->label == "explain:lrp"; }));
    monitor.stop();
}

// ------------------------------------------------------------------------------------------
// Live smoke test on the real machine: never asserts a GPU or sensor exists.
// ------------------------------------------------------------------------------------------

TEST(SystemMonitorLive, RealHostSampleIsSane) {
    SystemMonitor::Options options;
    options.interval = 50ms;
    SystemMonitor monitor(options);
    std::this_thread::sleep_for(60ms);  // let the CPU counters advance past one tick
    const SystemSample s = monitor.sample_now();
    EXPECT_GT(s.timestamp.time_since_epoch().count(), 0);
#ifdef __linux__
    ASSERT_TRUE(s.memory_total_bytes.has_value());
    EXPECT_GT(*s.memory_total_bytes, 0.0);
#endif
    auto in_range = [](const std::optional<double>& v, double lo, double hi) { return !v || (*v >= lo && *v <= hi); };
    EXPECT_TRUE(in_range(s.cpu_utilization_percent, 0.0, 100.0));
    EXPECT_TRUE(in_range(s.process_cpu_percent, 0.0, 100.0 * std::thread::hardware_concurrency() + 1.0));
    EXPECT_TRUE(in_range(s.cpu_temperature_c, -20.0, 150.0));
    if (s.memory_used_bytes && s.memory_total_bytes) {
        EXPECT_LE(*s.memory_used_bytes, *s.memory_total_bytes);
    }
    for (const GpuSample& gpu : s.gpus) {
        EXPECT_TRUE(in_range(gpu.utilization_percent, 0.0, 100.0)) << gpu.name;
        EXPECT_TRUE(in_range(gpu.temperature_c, -20.0, 150.0)) << gpu.name;
        EXPECT_TRUE(in_range(gpu.power_watts, 0.0, 2000.0)) << gpu.name;
    }

    std::ostringstream found;
    found << "live sample: " << sample_to_json(s) << "\n";
    for (const MetricCapability& cap : monitor.capabilities()) {
        found << "  " << (cap.available ? "[yes] " : "[no]  ") << cap.metric << ": " << cap.detail << "\n";
    }
    std::cout << found.str();
    ::testing::Test::RecordProperty("gpu_count", static_cast<int>(s.gpus.size()));
    ::testing::Test::RecordProperty("sample_json", sample_to_json(s));

    // And the background thread produces samples on the real host too.
    monitor.start();
    ASSERT_TRUE(wait_until([&] { return monitor.history().size() >= 2; }));
    monitor.stop();
}

}  // namespace
}  // namespace pulsatrix
