/** @file system_monitor_detail.hpp
 *  @brief SystemMonitor internals exposed for unit tests: the platform-source interface,
 *         the filesystem-rooted Linux parsers, and the log record encoders.
 *  @note Not part of the stable API. Kept out of system_monitor.hpp so the user-facing header
 *        stays small; the tests include this header to drive the parsers against fake
 *        /proc and /sys trees built in a temporary directory.
 *  @ingroup system_monitoring
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pulsatrix/system_monitor.hpp"

namespace pulsatrix::detail {

/** @brief Aggregate CPU time counters (any consistent unit). busy = total - idle - iowait. */
struct CpuTimes {
    double busy = 0.0;
    double total = 0.0;
};

/** @brief Raw counters/levels from one platform read; rates are derived by SystemMonitor. */
struct RawReading {
    std::optional<CpuTimes> system_cpu;
    std::optional<double> process_cpu_seconds;  ///< user + system CPU time consumed by this process.
    std::optional<double> memory_used_bytes;
    std::optional<double> memory_total_bytes;
    std::optional<double> process_rss_bytes;
    std::optional<double> cpu_temperature_c;
    std::vector<GpuSample> gpus;
};

/** @brief Where SystemMonitor reads from. Implementations must make read() thread-safe. */
class PlatformSources {
public:
    virtual ~PlatformSources() = default;
    virtual RawReading read() = 0;
    virtual std::vector<MetricCapability> capabilities() = 0;
};

/** @brief The sources for the host this binary runs on (Linux/Windows + NVML, or a stub). */
std::unique_ptr<PlatformSources> make_host_sources();

/** @brief Utilization in percent from two counter readings; nullopt if no time elapsed. */
std::optional<double> cpu_percent_from_deltas(const CpuTimes& previous, const CpuTimes& current);

/** @brief One AMD GPU found under <sys_root>/class/drm. */
struct AmdGpuDevice {
    std::string card;        ///< e.g. "card1".
    std::string device_dir;  ///< <sys_root>/class/drm/card1/device
    std::string pci_device;  ///< contents of device/device, e.g. "0x1586" (may be empty).
};

/**
 * @brief Linux /proc + /sys parsers rooted at configurable directories so tests can point
 *        them at a fake tree. Pure file parsing: compiles (and is unit-testable) everywhere.
 */
class LinuxSources : public PlatformSources {
public:
    /**
     * @param proc_root Directory standing in for /proc.
     * @param sys_root Directory standing in for /sys.
     * @param clock_ticks_per_second USER_HZ for /proc/self/stat; 0 = sysconf(_SC_CLK_TCK) on
     *        POSIX (100 elsewhere).
     */
    explicit LinuxSources(std::string proc_root = "/proc", std::string sys_root = "/sys",
                          long clock_ticks_per_second = 0);

    RawReading read() override;
    std::vector<MetricCapability> capabilities() override;

    /** @brief Aggregate "cpu" line of <proc>/stat. */
    [[nodiscard]] std::optional<CpuTimes> read_system_cpu_times() const;
    /** @brief utime + stime of <proc>/self/stat, in seconds. */
    [[nodiscard]] std::optional<double> read_process_cpu_seconds() const;
    /** @brief MemTotal from <proc>/meminfo, bytes. */
    [[nodiscard]] std::optional<double> read_memory_total_bytes() const;
    /** @brief MemTotal - MemAvailable from <proc>/meminfo, bytes (nullopt if MemAvailable is absent). */
    [[nodiscard]] std::optional<double> read_memory_used_bytes() const;
    /** @brief VmRSS from <proc>/self/status, bytes. */
    [[nodiscard]] std::optional<double> read_process_rss_bytes() const;
    /**
     * @brief The temp*_input file chosen for the CPU temperature, or nullopt.
     * @note hwmon names in priority order: k10temp, zenpower, coretemp, cpu_thermal, then
     *       acpitz (a board/ACPI zone, not the die -- last resort). Within the chosen sensor a
     *       labelled channel is preferred: Tctl, Tdie, then "Package id *"; else temp1_input.
     */
    [[nodiscard]] std::optional<std::string> find_cpu_temperature_input() const;
    [[nodiscard]] std::optional<double> read_cpu_temperature_c() const;
    /** @brief AMD (PCI vendor 0x1002) DRM cards, skipping connector entries such as card1-DP-1. */
    [[nodiscard]] std::vector<AmdGpuDevice> find_amd_gpus() const;
    /** @brief Readings of one AMD GPU (index/name/vendor filled, missing files nullopt). */
    [[nodiscard]] GpuSample read_amd_gpu(const AmdGpuDevice& device, int index) const;

private:
    std::string proc_root_;
    std::string sys_root_;
    double clock_ticks_per_second_;
};

/** @brief JSON string-literal body escaping (quotes, backslash, control characters). */
std::string json_escape(const std::string& text);

/** @brief Number as a JSON/CSV token: integral values without decimals, else 3 decimals; non-finite -> "". */
std::string format_number(double value);

/** @brief The `"event":"mark"` JSON Lines record. */
std::string mark_to_json(std::chrono::system_clock::time_point timestamp, double elapsed_seconds,
                         const std::string& label);

/** @brief CSV header for @p gpu_count GPUs (columns gpu<i>_util, gpu<i>_mem_used, ...). */
std::string csv_header(std::size_t gpu_count);

/** @brief One CSV data row for @p sample; GPU columns beyond those present are empty. */
std::string sample_to_csv(const SystemSample& sample, std::size_t gpu_count);

/** @brief One CSV `mark` row (only event, timestamp, elapsed_seconds and label filled). */
std::string mark_to_csv(std::chrono::system_clock::time_point timestamp, double elapsed_seconds,
                        const std::string& label, std::size_t gpu_count);

}  // namespace pulsatrix::detail
