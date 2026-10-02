/** @file system_monitor_demo.cpp
 *  @brief Standalone demo: SystemMonitor logging to system_monitor.jsonl while a gemm workload
 *         runs on the CPU and then on this build's GPU backend, with a live one-line status per
 *         second and mark() calls around each phase.
 *  @note Usage: system_monitor_demo [seconds=10]. Watch the log live from another terminal with
 *        `tail -f system_monitor.jsonl`.
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/system_monitor.hpp"

#if defined(PULSATRIX_DEMO_WITH_HIP)
#include "pulsatrix/hip_backend.hpp"
#elif defined(PULSATRIX_DEMO_WITH_CUDA)
#include "pulsatrix/cuda_backend.hpp"
#endif

namespace {

using namespace std::chrono_literals;

std::string fmt(const std::optional<double>& v, const char* format, double scale = 1.0) {
    if (!v) {
        return "n/a";
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), format, *v / scale);
    return buf;
}

void print_status(const pulsatrix::SystemSample& s) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    std::string line = "[" + fmt(s.elapsed_seconds, "%5.1f") + "s] cpu " + fmt(s.cpu_utilization_percent, "%5.1f%%") +
                       " | proc " + fmt(s.process_cpu_percent, "%6.1f%%") + " | mem " +
                       fmt(s.memory_used_bytes, "%.1f", kGiB) + "/" + fmt(s.memory_total_bytes, "%.1f GiB", kGiB) +
                       " | cpu temp " + fmt(s.cpu_temperature_c, "%.1fC");
    for (const pulsatrix::GpuSample& g : s.gpus) {
        line += " | gpu" + std::to_string(g.index) + " " + fmt(g.utilization_percent, "%.0f%%") + " " +
                fmt(g.temperature_c, "%.0fC") + " " + fmt(g.power_watts, "%.1fW") + " vram " +
                fmt(g.memory_used_bytes, "%.2f", kGiB) + " GiB";
    }
    line += " | " + (s.label.empty() ? std::string("-") : s.label);
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
}

// Repeated n x n gemm on @p backend until @p deadline. Returns the number of gemms run.
int gemm_until(pulsatrix::DeviceBackend& backend, std::size_t n, std::chrono::steady_clock::time_point deadline) {
    const std::size_t bytes = n * n * sizeof(float);
    float* a = static_cast<float*>(backend.allocate(bytes));
    float* b = static_cast<float*>(backend.allocate(bytes));
    float* c = static_cast<float*>(backend.allocate(bytes));
    backend.fill(a, 0.5f, n * n);
    backend.fill(b, 0.25f, n * n);
    int count = 0;
    float probe = 0.0f;
    while (std::chrono::steady_clock::now() < deadline) {
        for (int i = 0; i < 8; ++i) {
            backend.gemm(a, b, c, n, n, n);
            ++count;
        }
        // Copying one value back synchronizes with an asynchronous GPU queue.
        backend.copy(&probe, c, sizeof(float), backend.device() == pulsatrix::DeviceType::Cpu
                                                    ? pulsatrix::CopyDirection::HostToHost
                                                    : pulsatrix::CopyDirection::DeviceToHost);
    }
    backend.free(a);
    backend.free(b);
    backend.free(c);
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    const int seconds = argc > 1 ? std::max(3, std::atoi(argv[1])) : 10;

    pulsatrix::SystemMonitor::Options options;
    options.interval = 1000ms;
    options.log_path = "system_monitor.jsonl";
    pulsatrix::SystemMonitor monitor(options);

    std::printf("SystemMonitor capabilities on this machine:\n");
    for (const pulsatrix::MetricCapability& cap : monitor.capabilities()) {
        std::printf("  %s %-32s %s\n", cap.available ? "[yes]" : "[no] ", cap.metric.c_str(), cap.detail.c_str());
    }

    std::unique_ptr<pulsatrix::DeviceBackend> gpu;
#if defined(PULSATRIX_DEMO_WITH_HIP)
    try {
        gpu = std::make_unique<pulsatrix::HIPBackend>();
    } catch (const std::exception& e) {
        std::printf("HIP backend unavailable (%s); GPU phase runs on the CPU\n", e.what());
    }
#elif defined(PULSATRIX_DEMO_WITH_CUDA)
    try {
        gpu = std::make_unique<pulsatrix::CUDABackend>();
    } catch (const std::exception& e) {
        std::printf("CUDA backend unavailable (%s); GPU phase runs on the CPU\n", e.what());
    }
#endif
    pulsatrix::CPUBackend cpu;

    std::printf("\nMonitoring for %d s, logging to %s (one JSON object per line, flushed live)\n\n", seconds,
                options.log_path.c_str());
    monitor.on_sample(print_status);
    monitor.start();

    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();
    const auto phase = std::chrono::milliseconds(seconds * 1000 / 3);

    monitor.mark("idle");
    std::this_thread::sleep_for(phase);

    monitor.mark("cpu gemm 256");
    const int cpu_gemms = gemm_until(cpu, 256, begin + 2 * phase);

    int gpu_gemms = 0;
    if (gpu) {
        monitor.mark("gpu gemm 2048");
        gpu_gemms = gemm_until(*gpu, 2048, begin + std::chrono::seconds(seconds));
    } else {
        monitor.mark("cpu gemm 256 (no GPU backend in this build)");
        gemm_until(cpu, 256, begin + std::chrono::seconds(seconds));
    }
    std::this_thread::sleep_for(100ms);
    monitor.stop();

    std::printf("\nRan %d CPU gemms and %d GPU gemms; %zu samples in history; log: %s\n", cpu_gemms, gpu_gemms,
                monitor.history().size(), options.log_path.c_str());
    return 0;
}
