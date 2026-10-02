// LinuxSources: /proc and /sys parsers behind a configurable filesystem root. Pure
// std::filesystem/ifstream code with no POSIX headers beyond sysconf (guarded), so it compiles
// on every platform and its unit tests run against fake trees everywhere.

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "pulsatrix/system_monitor_detail.hpp"

#ifndef _WIN32
#include <unistd.h>
#endif

namespace pulsatrix::detail {

namespace fs = std::filesystem;

namespace {

std::optional<std::string> read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return std::nullopt;
    }
    std::ostringstream out;
    out << in.rdbuf();
    if (in.bad()) {
        return std::nullopt;
    }
    return out.str();
}

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// Whole-file numeric value, as sysfs attributes are. Anything unparseable is missing data.
std::optional<double> read_number(const std::string& path) {
    const auto text = read_text(path);
    if (!text) {
        return std::nullopt;
    }
    std::istringstream in(trim(*text));
    in.imbue(std::locale::classic());
    double value = 0.0;
    if (!(in >> value)) {
        return std::nullopt;
    }
    return value;
}

bool exists(const std::string& path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

// Directory entries' names, sorted so selection is deterministic.
std::vector<std::string> list_dir(const std::string& path) {
    std::vector<std::string> names;
    std::error_code ec;
    fs::directory_iterator it(path, ec);
    if (ec) {
        return names;
    }
    for (const fs::directory_entry& entry : it) {
        names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    return names;
}

// "meminfo"-style "Key:   12345 kB" line value in bytes.
std::optional<double> read_kb_field(const std::string& path, const std::string& key) {
    const auto text = read_text(path);
    if (!text) {
        return std::nullopt;
    }
    std::istringstream lines(*text);
    std::string line;
    const std::string prefix = key + ":";
    while (std::getline(lines, line)) {
        if (line.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        std::istringstream fields(line.substr(prefix.size()));
        fields.imbue(std::locale::classic());
        double value = 0.0;
        std::string unit;
        if (!(fields >> value)) {
            return std::nullopt;
        }
        fields >> unit;
        return unit == "kB" ? value * 1024.0 : value;
    }
    return std::nullopt;
}

bool is_card_entry(const std::string& name) {
    if (name.size() <= 4 || name.compare(0, 4, "card") != 0) {
        return false;
    }
    return std::all_of(name.begin() + 4, name.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// Prefers the labelled channel in @p preferred (in order, prefix match), else temp1_input.
std::optional<std::string> pick_temperature_input(const std::string& hwmon_dir,
                                                  const std::vector<std::string>& preferred) {
    std::vector<std::pair<std::string, std::string>> labelled;  // (label, input path)
    for (const std::string& name : list_dir(hwmon_dir)) {
        const std::string suffix = "_label";
        if (name.compare(0, 4, "temp") != 0 || name.size() <= suffix.size() ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            continue;
        }
        const std::string input = hwmon_dir + "/" + name.substr(0, name.size() - suffix.size()) + "_input";
        const auto label = read_text(hwmon_dir + "/" + name);
        if (label && exists(input)) {
            labelled.emplace_back(trim(*label), input);
        }
    }
    for (const std::string& want : preferred) {
        for (const auto& [label, input] : labelled) {
            if (label.compare(0, want.size(), want) == 0) {
                return input;
            }
        }
    }
    const std::string first = hwmon_dir + "/temp1_input";
    if (exists(first)) {
        return first;
    }
    if (!labelled.empty()) {
        return labelled.front().second;
    }
    return std::nullopt;
}

}  // namespace

LinuxSources::LinuxSources(std::string proc_root, std::string sys_root, long clock_ticks_per_second)
    : proc_root_(std::move(proc_root)), sys_root_(std::move(sys_root)), clock_ticks_per_second_(100.0) {
    if (clock_ticks_per_second > 0) {
        clock_ticks_per_second_ = static_cast<double>(clock_ticks_per_second);
    } else {
#ifndef _WIN32
        const long ticks = sysconf(_SC_CLK_TCK);
        if (ticks > 0) {
            clock_ticks_per_second_ = static_cast<double>(ticks);
        }
#endif
    }
}

std::optional<CpuTimes> LinuxSources::read_system_cpu_times() const {
    const auto text = read_text(proc_root_ + "/stat");
    if (!text) {
        return std::nullopt;
    }
    std::istringstream lines(*text);
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream fields(line);
        fields.imbue(std::locale::classic());
        std::string tag;
        fields >> tag;
        if (tag != "cpu") {
            continue;
        }
        // user nice system idle iowait irq softirq steal [guest guest_nice]. guest time is
        // already included in user/nice, so it is not added again.
        std::array<double, 8> v{};
        std::size_t n = 0;
        while (n < v.size() && (fields >> v[n])) {
            ++n;
        }
        if (n < 4) {
            return std::nullopt;
        }
        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            total += v[i];
        }
        const double idle = v[3] + (n > 4 ? v[4] : 0.0);
        return CpuTimes{total - idle, total};
    }
    return std::nullopt;
}

std::optional<double> LinuxSources::read_process_cpu_seconds() const {
    const auto text = read_text(proc_root_ + "/self/stat");
    if (!text) {
        return std::nullopt;
    }
    // Field 2 (comm) is parenthesised and may contain spaces/parens: parse after the LAST ')'.
    const auto close = text->rfind(')');
    if (close == std::string::npos) {
        return std::nullopt;
    }
    std::istringstream fields(text->substr(close + 1));
    fields.imbue(std::locale::classic());
    std::vector<std::string> tokens;
    std::string token;
    while (tokens.size() < 13 && fields >> token) {
        tokens.push_back(token);
    }
    if (tokens.size() < 13) {
        return std::nullopt;
    }
    // tokens[0] is field 3 (state); utime is field 14, stime field 15.
    try {
        const double utime = std::stod(tokens[11]);
        const double stime = std::stod(tokens[12]);
        return (utime + stime) / clock_ticks_per_second_;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<double> LinuxSources::read_memory_total_bytes() const {
    return read_kb_field(proc_root_ + "/meminfo", "MemTotal");
}

std::optional<double> LinuxSources::read_memory_used_bytes() const {
    const auto total = read_kb_field(proc_root_ + "/meminfo", "MemTotal");
    const auto available = read_kb_field(proc_root_ + "/meminfo", "MemAvailable");
    if (!total || !available) {
        return std::nullopt;
    }
    return *total - *available;
}

std::optional<double> LinuxSources::read_process_rss_bytes() const {
    return read_kb_field(proc_root_ + "/self/status", "VmRSS");
}

std::optional<std::string> LinuxSources::find_cpu_temperature_input() const {
    static const std::array<const char*, 5> kPriority = {"k10temp", "zenpower", "coretemp", "cpu_thermal", "acpitz"};
    const std::string hwmon_root = sys_root_ + "/class/hwmon";
    const std::vector<std::string> entries = list_dir(hwmon_root);
    for (const char* wanted : kPriority) {
        for (const std::string& entry : entries) {
            const std::string dir = hwmon_root + "/" + entry;
            const auto name = read_text(dir + "/name");
            if (!name || trim(*name) != wanted) {
                continue;
            }
            if (auto input = pick_temperature_input(dir, {"Tctl", "Tdie", "Package id"})) {
                return input;
            }
        }
    }
    return std::nullopt;
}

std::optional<double> LinuxSources::read_cpu_temperature_c() const {
    const auto input = find_cpu_temperature_input();
    if (!input) {
        return std::nullopt;
    }
    const auto millidegrees = read_number(*input);
    if (!millidegrees) {
        return std::nullopt;
    }
    return *millidegrees / 1000.0;
}

std::vector<AmdGpuDevice> LinuxSources::find_amd_gpus() const {
    std::vector<AmdGpuDevice> devices;
    const std::string drm = sys_root_ + "/class/drm";
    std::vector<std::string> cards;
    for (const std::string& name : list_dir(drm)) {
        if (is_card_entry(name)) {
            cards.push_back(name);
        }
    }
    // Numeric order (card2 before card10).
    std::sort(cards.begin(), cards.end(), [](const std::string& a, const std::string& b) {
        return a.size() != b.size() ? a.size() < b.size() : a < b;
    });
    for (const std::string& card : cards) {
        const std::string device_dir = drm + "/" + card + "/device";
        const auto vendor = read_text(device_dir + "/vendor");
        if (!vendor || trim(*vendor) != "0x1002") {
            continue;
        }
        const auto pci_device = read_text(device_dir + "/device");
        devices.push_back(AmdGpuDevice{card, device_dir, pci_device ? trim(*pci_device) : std::string()});
    }
    return devices;
}

GpuSample LinuxSources::read_amd_gpu(const AmdGpuDevice& device, int index) const {
    GpuSample gpu;
    gpu.index = index;
    gpu.vendor = "AMD";
    const auto product = read_text(device.device_dir + "/product_name");
    if (product && !trim(*product).empty()) {
        gpu.name = trim(*product);
    } else {
        gpu.name = "AMD GPU" + (device.pci_device.empty() ? std::string() : " " + device.pci_device) + " (" +
                   device.card + ")";
    }
    gpu.utilization_percent = read_number(device.device_dir + "/gpu_busy_percent");
    gpu.memory_used_bytes = read_number(device.device_dir + "/mem_info_vram_used");
    gpu.memory_total_bytes = read_number(device.device_dir + "/mem_info_vram_total");
    gpu.memory_gtt_used_bytes = read_number(device.device_dir + "/mem_info_gtt_used");

    const std::string hwmon_root = device.device_dir + "/hwmon";
    for (const std::string& entry : list_dir(hwmon_root)) {
        const std::string dir = hwmon_root + "/" + entry;
        if (!gpu.temperature_c) {
            if (const auto input = pick_temperature_input(dir, {"edge"})) {
                if (const auto milli = read_number(*input)) {
                    gpu.temperature_c = *milli / 1000.0;
                }
            }
        }
        if (!gpu.power_watts) {
            // power1_average is the documented amdgpu attribute but some APUs only expose (or
            // only successfully read) power1_input. Both are microwatts.
            auto micro = read_number(dir + "/power1_average");
            if (!micro) {
                micro = read_number(dir + "/power1_input");
            }
            if (micro) {
                gpu.power_watts = *micro / 1e6;
            }
        }
    }
    return gpu;
}

RawReading LinuxSources::read() {
    RawReading raw;
    raw.system_cpu = read_system_cpu_times();
    raw.process_cpu_seconds = read_process_cpu_seconds();
    raw.memory_total_bytes = read_memory_total_bytes();
    raw.memory_used_bytes = read_memory_used_bytes();
    raw.process_rss_bytes = read_process_rss_bytes();
    raw.cpu_temperature_c = read_cpu_temperature_c();
    int index = 0;
    for (const AmdGpuDevice& device : find_amd_gpus()) {
        raw.gpus.push_back(read_amd_gpu(device, index++));
    }
    return raw;
}

std::vector<MetricCapability> LinuxSources::capabilities() {
    std::vector<MetricCapability> caps;
    const std::string stat = proc_root_ + "/stat";
    const std::string self_stat = proc_root_ + "/self/stat";
    const std::string meminfo = proc_root_ + "/meminfo";
    const std::string status = proc_root_ + "/self/status";

    auto add = [&caps](const std::string& metric, bool ok, const std::string& source, const std::string& reason) {
        caps.push_back(MetricCapability{metric, ok, ok ? source : reason});
    };
    add("cpu_utilization_percent", read_system_cpu_times().has_value(), stat,
        "no parseable aggregate 'cpu' line in " + stat);
    add("process_cpu_percent", read_process_cpu_seconds().has_value(), self_stat,
        "cannot read utime/stime from " + self_stat);
    add("memory_total_bytes", read_memory_total_bytes().has_value(), meminfo + " (MemTotal)",
        "no MemTotal in " + meminfo);
    add("memory_used_bytes", read_memory_used_bytes().has_value(), meminfo + " (MemTotal - MemAvailable)",
        "no MemTotal/MemAvailable in " + meminfo + " (MemAvailable needs Linux >= 3.14)");
    add("process_rss_bytes", read_process_rss_bytes().has_value(), status + " (VmRSS)",
        "no VmRSS in " + status);
    const auto temp_input = find_cpu_temperature_input();
    const bool temp_ok = temp_input && read_cpu_temperature_c().has_value();
    add("cpu_temperature_c", temp_ok, temp_input ? *temp_input : std::string(),
        temp_input ? "cannot read " + *temp_input
                   : "no k10temp/zenpower/coretemp/cpu_thermal/acpitz hwmon sensor found under " + sys_root_ +
                         "/class/hwmon");

    const std::vector<AmdGpuDevice> devices = find_amd_gpus();
    int index = 0;
    for (const AmdGpuDevice& device : devices) {
        const GpuSample gpu = read_amd_gpu(device, index);
        const std::string prefix = "gpu" + std::to_string(index) + ".";
        const std::string where = device.device_dir;
        add(prefix + "utilization_percent", gpu.utilization_percent.has_value(), where + "/gpu_busy_percent",
            device.card + ": gpu_busy_percent not readable");
        add(prefix + "memory_used_bytes", gpu.memory_used_bytes.has_value(), where + "/mem_info_vram_used",
            device.card + ": mem_info_vram_used not readable");
        add(prefix + "memory_total_bytes", gpu.memory_total_bytes.has_value(), where + "/mem_info_vram_total",
            device.card + ": mem_info_vram_total not readable");
        add(prefix + "memory_gtt_used_bytes", gpu.memory_gtt_used_bytes.has_value(), where + "/mem_info_gtt_used",
            device.card + ": mem_info_gtt_used not readable");
        add(prefix + "temperature_c", gpu.temperature_c.has_value(), where + "/hwmon/*/temp*_input",
            device.card + ": no readable hwmon temperature");
        add(prefix + "power_watts", gpu.power_watts.has_value(), where + "/hwmon/*/power1_{average,input}",
            device.card + ": no readable hwmon power1_average/power1_input");
        ++index;
    }
    add("amd_gpus", !devices.empty(),
        std::to_string(devices.size()) + " AMD GPU(s) via " + sys_root_ + "/class/drm (amdgpu sysfs)",
        "no AMD (vendor 0x1002) card under " + sys_root_ + "/class/drm");
    return caps;
}

}  // namespace pulsatrix::detail
