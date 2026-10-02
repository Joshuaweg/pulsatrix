// Host platform sources for SystemMonitor: Linux (/proc + /sys via LinuxSources), Windows
// (Win32 APIs), a stub for everything else, plus NVIDIA GPUs through NVML loaded at runtime
// with dlopen/LoadLibrary. Nothing here adds a link-time dependency beyond system libraries
// (libdl on older glibc, psapi on Windows).

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/system_monitor_detail.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
// windows.h must precede psapi.h.
#include <psapi.h>
#else
#include <dlfcn.h>
#endif

namespace pulsatrix::detail {

namespace {

// ---------------------------------------------------------------------------------------------
// NVML, declared minimally here (no nvml.h / CUDA toolkit needed to build). Signatures and
// struct layouts match nvml.h; every call returns nvmlReturn_t where 0 == NVML_SUCCESS.
// ---------------------------------------------------------------------------------------------
using nvmlReturn_t = int;
using nvmlDevice_t = struct nvmlDevice_st*;
struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
};
struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};
constexpr nvmlReturn_t kNvmlSuccess = 0;
constexpr int kNvmlTemperatureGpu = 0;   // NVML_TEMPERATURE_GPU
constexpr unsigned int kNvmlNameBufferSize = 96;  // NVML_DEVICE_NAME_V2_BUFFER_SIZE

#ifdef _WIN32
using LibraryHandle = HMODULE;
#else
using LibraryHandle = void*;
#endif

LibraryHandle open_library(const char* name) {
#ifdef _WIN32
    return LoadLibraryA(name);
#else
    return dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
}

void close_library(LibraryHandle lib) {
#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
}

// Copies the symbol address into a function pointer via memcpy: a direct object-pointer ->
// function-pointer cast is only conditionally supported in ISO C++.
template <typename Fn>
bool resolve(LibraryHandle lib, const char* name, Fn& out) {
#ifdef _WIN32
    FARPROC symbol = GetProcAddress(lib, name);
#else
    void* symbol = dlsym(lib, name);
#endif
    if (symbol == nullptr) {
        return false;
    }
    static_assert(sizeof(symbol) == sizeof(out), "function pointer size mismatch");
    std::memcpy(&out, &symbol, sizeof(out));
    return true;
}

class NvmlSource {
public:
    NvmlSource() {
#ifdef _WIN32
        std::vector<std::string> candidates = {"nvml.dll"};
        char program_files[MAX_PATH] = {};
        const DWORD len = GetEnvironmentVariableA("ProgramFiles", program_files, MAX_PATH);
        if (len > 0 && len < MAX_PATH) {
            candidates.push_back(std::string(program_files) + "\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        }
        const char* tried = "nvml.dll / %ProgramFiles%\\NVIDIA Corporation\\NVSMI\\nvml.dll";
#else
        std::vector<std::string> candidates = {"libnvidia-ml.so.1"};
        const char* tried = "libnvidia-ml.so.1";
#endif
        for (const std::string& candidate : candidates) {
            lib_ = open_library(candidate.c_str());
            if (lib_ != nullptr) {
                library_name_ = candidate;
                break;
            }
        }
        if (lib_ == nullptr) {
            status_ = std::string("NVML library not found (") + tried + "); no NVIDIA driver installed";
            return;
        }
        const bool resolved = resolve(lib_, "nvmlInit_v2", init_) && resolve(lib_, "nvmlShutdown", shutdown_) &&
                              resolve(lib_, "nvmlDeviceGetCount_v2", get_count_) &&
                              resolve(lib_, "nvmlDeviceGetHandleByIndex_v2", get_handle_) &&
                              resolve(lib_, "nvmlDeviceGetName", get_name_) &&
                              resolve(lib_, "nvmlDeviceGetUtilizationRates", get_utilization_) &&
                              resolve(lib_, "nvmlDeviceGetMemoryInfo", get_memory_) &&
                              resolve(lib_, "nvmlDeviceGetTemperature", get_temperature_) &&
                              resolve(lib_, "nvmlDeviceGetPowerUsage", get_power_);
        if (!resolved) {
            status_ = library_name_ + " is missing an expected NVML entry point (driver too old?)";
            release();
            return;
        }
        const nvmlReturn_t init_rc = init_();
        if (init_rc != kNvmlSuccess) {
            status_ = "nvmlInit_v2 failed with NVML error " + std::to_string(init_rc);
            release();
            return;
        }
        initialized_ = true;
        unsigned int count = 0;
        if (get_count_(&count) != kNvmlSuccess) {
            status_ = "nvmlDeviceGetCount_v2 failed";
            return;
        }
        for (unsigned int i = 0; i < count; ++i) {
            nvmlDevice_t handle = nullptr;
            if (get_handle_(i, &handle) != kNvmlSuccess) {
                continue;
            }
            char name[kNvmlNameBufferSize] = {};
            std::string device_name = "NVIDIA GPU " + std::to_string(i);
            if (get_name_(handle, name, kNvmlNameBufferSize) == kNvmlSuccess) {
                device_name = name;
            }
            devices_.push_back(Device{handle, device_name});
        }
        status_ = std::to_string(devices_.size()) + " NVIDIA GPU(s) via NVML (" + library_name_ + ")";
    }

    ~NvmlSource() { release(); }

    NvmlSource(const NvmlSource&) = delete;
    NvmlSource& operator=(const NvmlSource&) = delete;

    [[nodiscard]] std::vector<GpuSample> read(int first_index) const {
        std::vector<GpuSample> gpus;
        int index = first_index;
        for (const Device& device : devices_) {
            GpuSample gpu;
            gpu.index = index++;
            gpu.name = device.name;
            gpu.vendor = "NVIDIA";
            nvmlUtilization_t utilization{};
            if (get_utilization_(device.handle, &utilization) == kNvmlSuccess) {
                gpu.utilization_percent = static_cast<double>(utilization.gpu);
            }
            nvmlMemory_t memory{};
            if (get_memory_(device.handle, &memory) == kNvmlSuccess) {
                gpu.memory_used_bytes = static_cast<double>(memory.used);
                gpu.memory_total_bytes = static_cast<double>(memory.total);
            }
            unsigned int temperature = 0;
            if (get_temperature_(device.handle, kNvmlTemperatureGpu, &temperature) == kNvmlSuccess) {
                gpu.temperature_c = static_cast<double>(temperature);
            }
            unsigned int milliwatts = 0;
            if (get_power_(device.handle, &milliwatts) == kNvmlSuccess) {
                gpu.power_watts = static_cast<double>(milliwatts) / 1000.0;
            }
            gpus.push_back(std::move(gpu));
        }
        return gpus;
    }

    [[nodiscard]] std::vector<MetricCapability> capabilities(int first_index) const {
        std::vector<MetricCapability> caps;
        for (const GpuSample& gpu : read(first_index)) {
            const std::string prefix = "gpu" + std::to_string(gpu.index) + ".";
            auto add = [&](const char* metric, bool ok, const char* call) {
                caps.push_back(MetricCapability{prefix + metric, ok,
                                                ok ? std::string("NVML ") + call
                                                   : gpu.name + ": NVML " + call + " not supported"});
            };
            add("utilization_percent", gpu.utilization_percent.has_value(), "nvmlDeviceGetUtilizationRates");
            add("memory_used_bytes", gpu.memory_used_bytes.has_value(), "nvmlDeviceGetMemoryInfo");
            add("memory_total_bytes", gpu.memory_total_bytes.has_value(), "nvmlDeviceGetMemoryInfo");
            caps.push_back(MetricCapability{prefix + "memory_gtt_used_bytes", false,
                                            "NVIDIA has no GTT counter (AMD sysfs only)"});
            add("temperature_c", gpu.temperature_c.has_value(), "nvmlDeviceGetTemperature");
            add("power_watts", gpu.power_watts.has_value(), "nvmlDeviceGetPowerUsage");
        }
        caps.push_back(MetricCapability{"nvidia_gpus", !devices_.empty(), status_});
        return caps;
    }

private:
    struct Device {
        nvmlDevice_t handle;
        std::string name;
    };

    void release() {
        if (initialized_ && shutdown_ != nullptr) {
            shutdown_();
        }
        initialized_ = false;
        if (lib_ != nullptr) {
            close_library(lib_);
            lib_ = nullptr;
        }
    }

    LibraryHandle lib_ = nullptr;
    std::string library_name_;
    std::string status_;
    bool initialized_ = false;
    std::vector<Device> devices_;
    nvmlReturn_t (*init_)() = nullptr;
    nvmlReturn_t (*shutdown_)() = nullptr;
    nvmlReturn_t (*get_count_)(unsigned int*) = nullptr;
    nvmlReturn_t (*get_handle_)(unsigned int, nvmlDevice_t*) = nullptr;
    nvmlReturn_t (*get_name_)(nvmlDevice_t, char*, unsigned int) = nullptr;
    nvmlReturn_t (*get_utilization_)(nvmlDevice_t, nvmlUtilization_t*) = nullptr;
    nvmlReturn_t (*get_memory_)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
    nvmlReturn_t (*get_temperature_)(nvmlDevice_t, int, unsigned int*) = nullptr;
    nvmlReturn_t (*get_power_)(nvmlDevice_t, unsigned int*) = nullptr;
};

// ---------------------------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------------------------
#ifdef _WIN32
double filetime_seconds(const FILETIME& ft) {
    const unsigned long long ticks =
        (static_cast<unsigned long long>(ft.dwHighDateTime) << 32) | static_cast<unsigned long long>(ft.dwLowDateTime);
    return static_cast<double>(ticks) * 1e-7;  // 100 ns units
}

class WindowsSources : public PlatformSources {
public:
    RawReading read() override {
        RawReading raw;
        FILETIME idle{}, kernel{}, user{};
        if (GetSystemTimes(&idle, &kernel, &user)) {
            // Kernel time includes idle time.
            const double total = filetime_seconds(kernel) + filetime_seconds(user);
            raw.system_cpu = CpuTimes{total - filetime_seconds(idle), total};
        }
        FILETIME created{}, exited{}, proc_kernel{}, proc_user{};
        if (GetProcessTimes(GetCurrentProcess(), &created, &exited, &proc_kernel, &proc_user)) {
            raw.process_cpu_seconds = filetime_seconds(proc_kernel) + filetime_seconds(proc_user);
        }
        MEMORYSTATUSEX memory{};
        memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory)) {
            raw.memory_total_bytes = static_cast<double>(memory.ullTotalPhys);
            raw.memory_used_bytes = static_cast<double>(memory.ullTotalPhys - memory.ullAvailPhys);
        }
        PROCESS_MEMORY_COUNTERS counters{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
            raw.process_rss_bytes = static_cast<double>(counters.WorkingSetSize);
        }
        return raw;
    }

    std::vector<MetricCapability> capabilities() override {
        const RawReading raw = read();
        std::vector<MetricCapability> caps;
        auto add = [&caps](const char* metric, bool ok, const char* source) {
            caps.push_back(MetricCapability{metric, ok, ok ? std::string(source) : std::string(source) + " failed"});
        };
        add("cpu_utilization_percent", raw.system_cpu.has_value(), "GetSystemTimes");
        add("process_cpu_percent", raw.process_cpu_seconds.has_value(), "GetProcessTimes");
        add("memory_total_bytes", raw.memory_total_bytes.has_value(), "GlobalMemoryStatusEx");
        add("memory_used_bytes", raw.memory_used_bytes.has_value(), "GlobalMemoryStatusEx");
        add("process_rss_bytes", raw.process_rss_bytes.has_value(), "GetProcessMemoryInfo (WorkingSetSize)");
        caps.push_back(MetricCapability{"cpu_temperature_c", false,
                                        "Windows: CPU temperature not available without WMI/admin"});
        caps.push_back(MetricCapability{"amd_gpus", false,
                                        "Windows: AMD GPU metrics need ADLX, which is not integrated"});
        return caps;
    }
};
#endif

// ---------------------------------------------------------------------------------------------
// Anything that is neither Linux nor Windows: compiles, reads nothing, says so.
// ---------------------------------------------------------------------------------------------
class UnsupportedSources : public PlatformSources {
public:
    RawReading read() override { return {}; }

    std::vector<MetricCapability> capabilities() override {
        std::vector<MetricCapability> caps;
        for (const char* metric : {"cpu_utilization_percent", "process_cpu_percent", "memory_total_bytes",
                                   "memory_used_bytes", "process_rss_bytes", "cpu_temperature_c", "amd_gpus"}) {
            caps.push_back(MetricCapability{metric, false,
                                            "no source implemented for this platform (Linux and Windows only)"});
        }
        return caps;
    }
};

// Base OS sources + NVML. NVML GPUs are numbered after the base sources' GPUs.
class HostSources : public PlatformSources {
public:
    explicit HostSources(std::unique_ptr<PlatformSources> base) : base_(std::move(base)) {}

    RawReading read() override {
        RawReading raw = base_->read();
        std::vector<GpuSample> nvidia = nvml_.read(static_cast<int>(raw.gpus.size()));
        for (GpuSample& gpu : nvidia) {
            raw.gpus.push_back(std::move(gpu));
        }
        return raw;
    }

    std::vector<MetricCapability> capabilities() override {
        std::vector<MetricCapability> caps = base_->capabilities();
        const std::size_t base_gpus = base_->read().gpus.size();
        for (MetricCapability& cap : nvml_.capabilities(static_cast<int>(base_gpus))) {
            caps.push_back(std::move(cap));
        }
        return caps;
    }

private:
    std::unique_ptr<PlatformSources> base_;
    NvmlSource nvml_;
};

}  // namespace

std::unique_ptr<PlatformSources> make_host_sources() {
#if defined(_WIN32)
    return std::make_unique<HostSources>(std::make_unique<WindowsSources>());
#elif defined(__linux__)
    return std::make_unique<HostSources>(std::make_unique<LinuxSources>());
#else
    return std::make_unique<HostSources>(std::make_unique<UnsupportedSources>());
#endif
}

}  // namespace pulsatrix::detail
