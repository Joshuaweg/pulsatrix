/** @file system_monitor.hpp
 *  @brief Live CPU/GPU utilization, memory and temperature monitoring with a streaming log.
 *  @ingroup system_monitoring
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace pulsatrix {

namespace detail {
class PlatformSources;
}  // namespace detail

/**
 * @brief One GPU's readings within a SystemSample.
 * @note Every reading is std::optional: a value the platform/driver cannot provide is
 *       std::nullopt, never 0. SystemMonitor::capabilities() says why a metric is missing.
 * @ingroup system_monitoring
 */
struct GpuSample {
    int index = 0;       ///< Position in SystemSample::gpus (AMD sysfs devices first, then NVML devices).
    std::string name;    ///< Driver-reported name (NVML), or "AMD GPU <pci id> (<card>)" for sysfs devices.
    std::string vendor;  ///< "AMD" or "NVIDIA".
    std::optional<double> utilization_percent;  ///< 0-100, busy fraction of the GPU since the driver's last window.
    std::optional<double> memory_used_bytes;    ///< Dedicated memory (VRAM / APU carve-out) in use.
    std::optional<double> memory_total_bytes;   ///< Dedicated memory capacity.
    /// AMD only: GTT (system memory mapped for the GPU). On APUs this is where most allocations live.
    std::optional<double> memory_gtt_used_bytes;
    std::optional<double> temperature_c;  ///< Die/edge temperature, degrees Celsius.
    std::optional<double> power_watts;    ///< Board/package power draw, watts.
};

/**
 * @brief One point-in-time measurement of the host.
 * @note Rates (cpu_utilization_percent, process_cpu_percent) are computed from the
 *       difference between this reading and the previous one taken on the same path (the
 *       background thread, or sample_now()); they are std::nullopt when no counter advanced
 *       between the two readings (e.g. two calls within the kernel's 10 ms tick).
 * @ingroup system_monitoring
 */
struct SystemSample {
    std::chrono::system_clock::time_point timestamp;  ///< Wall-clock time of the reading.
    double elapsed_seconds = 0.0;  ///< Seconds since SystemMonitor::start() (or construction, before start()).
    std::optional<double> cpu_utilization_percent;  ///< System-wide, 0-100 across all logical CPUs.
    /// This process's CPU use, in percent of ONE logical CPU: a process saturating 8 cores reads ~800.
    std::optional<double> process_cpu_percent;
    std::optional<double> memory_used_bytes;   ///< System RAM in use (total - available).
    std::optional<double> memory_total_bytes;  ///< System RAM capacity.
    std::optional<double> process_rss_bytes;   ///< This process's resident set / working set.
    std::optional<double> cpu_temperature_c;   ///< CPU package/die temperature, degrees Celsius.
    std::vector<GpuSample> gpus;               ///< Every GPU detected (possibly empty).
    std::string label;                         ///< Label set by the most recent SystemMonitor::mark().
};

/**
 * @brief Whether one metric can be read on this machine, and from where -- or why not.
 * @ingroup system_monitoring
 */
struct MetricCapability {
    std::string metric;  ///< e.g. "cpu_temperature_c", "gpu0.power_watts", "gpus".
    bool available = false;
    std::string detail;  ///< Source when available (e.g. "/proc/stat"); the reason when not.
};

/** @brief Log record encoding. @ingroup system_monitoring */
enum class LogFormat {
    JsonLines,  ///< One JSON object per line; missing values are JSON null.
    Csv         ///< Header row then one row per record; missing values are empty cells.
};

/**
 * @brief Samples CPU/GPU utilization, memory use and temperatures on a background thread,
 *        keeps a bounded in-memory history, and streams every sample to a log file.
 *
 * Usage:
 * @code
 *   pulsatrix::SystemMonitor::Options opts;
 *   opts.interval = std::chrono::milliseconds(500);
 *   opts.log_path = "run.jsonl";
 *   pulsatrix::SystemMonitor monitor(opts);
 *   monitor.start();
 *   monitor.mark("epoch 1");   // tags this and later samples
 *   ...train...
 *   monitor.stop();
 * @endcode
 *
 * Sources (all detected at runtime, no link-time dependency):
 *  - Linux: /proc/stat, /proc/self/stat, /proc/meminfo, /proc/self/status, hwmon CPU sensors
 *    (k10temp, zenpower, coretemp, cpu_thermal, then acpitz as a last resort), AMD GPUs via
 *    /sys/class/drm/card<N>/device (amdgpu sysfs).
 *  - Windows: GetSystemTimes, GetProcessTimes, GlobalMemoryStatusEx, GetProcessMemoryInfo.
 *    CPU temperature is not readable without WMI/admin rights and is always std::nullopt.
 *  - NVIDIA GPUs (Linux and Windows): NVML, loaded with dlopen/LoadLibrary when present.
 *  - Other platforms: compiles, every metric std::nullopt with a reason.
 *
 * @note No fabricated values: a metric that cannot be read is std::nullopt in the sample,
 *       `null` in the JSON log and an empty CSV cell -- see capabilities() for the reason.
 * @note Timing: start() takes a baseline reading and the first background sample is taken
 *       one interval later, so every background sample carries CPU rates over a full
 *       interval. latest() is empty until then.
 * @note Thread safety: every public member may be called from any thread. Do not call
 *       stop() or destroy the monitor from inside the on_sample() callback (stop() throws
 *       std::logic_error if called from the monitor thread).
 * @ingroup system_monitoring
 */
class SystemMonitor {
public:
    /** @brief Construction options. */
    struct Options {
        std::chrono::milliseconds interval{1000};  ///< Sampling period of the background thread (> 0).
        std::string log_path = "";                 ///< Log file; empty = no log.
        LogFormat format = LogFormat::JsonLines;   ///< Log record encoding.
        /// Append to an existing log instead of truncating it. A restart (start() after stop())
        /// always appends, so the first run's records are never lost.
        bool append = false;
        std::size_t history_capacity = 3600;  ///< Ring-buffer size for history(); 0 keeps none.
    };

    /** @brief Default options: 1 s interval, no log. */
    SystemMonitor();

    /** @throws std::invalid_argument if options.interval is not positive. */
    explicit SystemMonitor(Options options);

    /**
     * @brief Testing hook: reads from @p sources instead of the host (see system_monitor_detail.hpp).
     * @throws std::invalid_argument if options.interval is not positive or sources is null.
     */
    SystemMonitor(Options options, std::unique_ptr<detail::PlatformSources> sources);

    /** @brief Stops the background thread (if running) and closes the log. */
    ~SystemMonitor();

    SystemMonitor(const SystemMonitor&) = delete;
    SystemMonitor& operator=(const SystemMonitor&) = delete;

    /**
     * @brief Opens the log (if any) and starts the background sampling thread. No-op if running.
     * @throws std::runtime_error if Options::log_path is set but cannot be opened for writing.
     */
    void start();

    /**
     * @brief Stops and joins the background thread, then closes the log. No-op if not running.
     * @note Returns within milliseconds regardless of the interval (the thread waits on a
     *       condition variable, not a sleep) -- unless a sample/callback is mid-flight.
     * @throws std::logic_error if called from the on_sample() callback (monitor thread).
     */
    void stop();

    /** @brief True between start() and stop(). */
    [[nodiscard]] bool running() const;

    /**
     * @brief Takes one synchronous measurement on the calling thread. Works without start().
     * @note Does not touch latest(), history() or the log. Its CPU rates are measured since
     *       the previous sample_now() call (or construction, for the first call).
     */
    [[nodiscard]] SystemSample sample_now();

    /** @brief The most recent background sample, or std::nullopt before the first one. */
    [[nodiscard]] std::optional<SystemSample> latest() const;

    /** @brief Background samples, oldest first, at most Options::history_capacity of them. */
    [[nodiscard]] std::vector<SystemSample> history() const;

    /**
     * @brief Registers a callback run on the monitor thread after each background sample
     *        (after it is logged and stored). Replaces any previous callback; pass {} to clear.
     * @note Keep it short: sampling of the next interval waits for it. An exception thrown
     *       from it is caught, reported once to stderr, and the thread keeps running.
     */
    void on_sample(std::function<void(const SystemSample&)> callback);

    /**
     * @brief Sets the label carried by subsequent samples (e.g. "epoch 3", "explain:lrp"), and
     *        if the log is open writes an immediate `"event":"mark"` record.
     */
    void mark(std::string label);

    /** @brief Per-metric availability on this machine, with the source or the reason it is missing. */
    [[nodiscard]] std::vector<MetricCapability> capabilities() const;

    /** @brief The options this monitor was constructed with. */
    [[nodiscard]] const Options& options() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/** @brief Formats @p tp as ISO-8601 UTC with milliseconds, e.g. "2026-10-02T12:34:56.789Z". */
std::string format_iso8601_utc(std::chrono::system_clock::time_point tp);

/** @brief Encodes @p sample as one JSON object (no trailing newline), the JSON Lines log record. */
std::string sample_to_json(const SystemSample& sample);

}  // namespace pulsatrix
