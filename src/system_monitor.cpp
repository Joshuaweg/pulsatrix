// SystemMonitor: background sampling thread, ring-buffer history, and the live JSON Lines /
// CSV log. Platform reads live in system_monitor_linux.cpp / system_monitor_host.cpp.

#include "pulsatrix/system_monitor.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

#include "pulsatrix/system_monitor_detail.hpp"

namespace pulsatrix {

namespace detail {

std::optional<double> cpu_percent_from_deltas(const CpuTimes& previous, const CpuTimes& current) {
    const double total = current.total - previous.total;
    if (!(total > 0.0)) {
        return std::nullopt;
    }
    // Linux's iowait counter can step backwards, so a busy delta can marginally exceed the
    // total delta; clamp to the physically meaningful range.
    const double percent = 100.0 * (current.busy - previous.busy) / total;
    return std::clamp(percent, 0.0, 100.0);
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 2);
    for (const char ch : text) {
        const auto c = static_cast<unsigned char>(ch);
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += ch;  // UTF-8 passes through unchanged
                }
        }
    }
    return out;
}

std::string format_number(double value) {
    if (!std::isfinite(value)) {
        return "";
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());  // never a locale-dependent decimal comma
    if (value == std::floor(value) && std::fabs(value) < 1e15) {
        out << std::fixed << std::setprecision(0) << (value == 0.0 ? 0.0 : value);
    } else {
        out << std::fixed << std::setprecision(3) << value;
    }
    return out.str();
}

namespace {

std::string json_number(const std::optional<double>& value) {
    if (!value) {
        return "null";
    }
    const std::string text = format_number(*value);
    return text.empty() ? "null" : text;
}

std::string csv_number(const std::optional<double>& value) { return value ? format_number(*value) : ""; }

std::string csv_field(const std::string& text) {
    if (text.find_first_of(",\"\r\n") == std::string::npos) {
        return text;
    }
    std::string out = "\"";
    for (const char c : text) {
        out += c;
        if (c == '"') {
            out += '"';
        }
    }
    return out + "\"";
}

const char* const kCsvGpuColumns[] = {"util", "mem_used", "mem_total", "gtt_used", "temp", "power"};
constexpr std::size_t kCsvGpuColumnCount = sizeof(kCsvGpuColumns) / sizeof(kCsvGpuColumns[0]);

}  // namespace

std::string mark_to_json(std::chrono::system_clock::time_point timestamp, double elapsed_seconds,
                         const std::string& label) {
    return "{\"event\":\"mark\",\"timestamp\":\"" + format_iso8601_utc(timestamp) +
           "\",\"elapsed_seconds\":" + json_number(elapsed_seconds) + ",\"label\":\"" + json_escape(label) + "\"}";
}

std::string csv_header(std::size_t gpu_count) {
    std::string header =
        "event,timestamp,elapsed_seconds,label,cpu_utilization_percent,process_cpu_percent,"
        "memory_used_bytes,memory_total_bytes,process_rss_bytes,cpu_temperature_c";
    for (std::size_t i = 0; i < gpu_count; ++i) {
        for (const char* column : kCsvGpuColumns) {
            header += ",gpu" + std::to_string(i) + "_" + column;
        }
    }
    return header;
}

std::string sample_to_csv(const SystemSample& s, std::size_t gpu_count) {
    std::string row = "sample," + format_iso8601_utc(s.timestamp) + "," + format_number(s.elapsed_seconds) + "," +
                      csv_field(s.label) + "," + csv_number(s.cpu_utilization_percent) + "," +
                      csv_number(s.process_cpu_percent) + "," + csv_number(s.memory_used_bytes) + "," +
                      csv_number(s.memory_total_bytes) + "," + csv_number(s.process_rss_bytes) + "," +
                      csv_number(s.cpu_temperature_c);
    for (std::size_t i = 0; i < gpu_count; ++i) {
        if (i < s.gpus.size()) {
            const GpuSample& g = s.gpus[i];
            row += "," + csv_number(g.utilization_percent) + "," + csv_number(g.memory_used_bytes) + "," +
                   csv_number(g.memory_total_bytes) + "," + csv_number(g.memory_gtt_used_bytes) + "," +
                   csv_number(g.temperature_c) + "," + csv_number(g.power_watts);
        } else {
            row += std::string(kCsvGpuColumnCount, ',');
        }
    }
    return row;
}

std::string mark_to_csv(std::chrono::system_clock::time_point timestamp, double elapsed_seconds,
                        const std::string& label, std::size_t gpu_count) {
    // event,timestamp,elapsed,label then 6 empty system columns and the empty GPU columns.
    return "mark," + format_iso8601_utc(timestamp) + "," + format_number(elapsed_seconds) + "," + csv_field(label) +
           std::string(6 + gpu_count * kCsvGpuColumnCount, ',');
}

}  // namespace detail

std::string format_iso8601_utc(std::chrono::system_clock::time_point tp) {
    const long long total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(tp.time_since_epoch()).count();
    long long seconds = total_ms / 1000;
    long long millis = total_ms % 1000;
    if (millis < 0) {  // floor for pre-1970 times
        millis += 1000;
        seconds -= 1;
    }
    const std::time_t t = static_cast<std::time_t>(seconds);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ", tm.tm_year + 1900, tm.tm_mon + 1,
                  tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<int>(millis));
    return buf;
}

std::string sample_to_json(const SystemSample& s) {
    using detail::json_escape;
    using detail::json_number;
    std::string out;
    out.reserve(512);
    out += "{\"event\":\"sample\",\"timestamp\":\"" + format_iso8601_utc(s.timestamp) + "\"";
    out += ",\"elapsed_seconds\":" + json_number(s.elapsed_seconds);
    out += ",\"label\":\"" + json_escape(s.label) + "\"";
    out += ",\"cpu_utilization_percent\":" + json_number(s.cpu_utilization_percent);
    out += ",\"process_cpu_percent\":" + json_number(s.process_cpu_percent);
    out += ",\"memory_used_bytes\":" + json_number(s.memory_used_bytes);
    out += ",\"memory_total_bytes\":" + json_number(s.memory_total_bytes);
    out += ",\"process_rss_bytes\":" + json_number(s.process_rss_bytes);
    out += ",\"cpu_temperature_c\":" + json_number(s.cpu_temperature_c);
    out += ",\"gpus\":[";
    for (std::size_t i = 0; i < s.gpus.size(); ++i) {
        const GpuSample& g = s.gpus[i];
        out += i == 0 ? "{" : ",{";
        out += "\"index\":" + std::to_string(g.index);
        out += ",\"name\":\"" + json_escape(g.name) + "\"";
        out += ",\"vendor\":\"" + json_escape(g.vendor) + "\"";
        out += ",\"utilization_percent\":" + json_number(g.utilization_percent);
        out += ",\"memory_used_bytes\":" + json_number(g.memory_used_bytes);
        out += ",\"memory_total_bytes\":" + json_number(g.memory_total_bytes);
        out += ",\"memory_gtt_used_bytes\":" + json_number(g.memory_gtt_used_bytes);
        out += ",\"temperature_c\":" + json_number(g.temperature_c);
        out += ",\"power_watts\":" + json_number(g.power_watts);
        out += "}";
    }
    out += "]}";
    return out;
}

namespace {

using SteadyClock = std::chrono::steady_clock;

// The previous raw reading a sampling path derives its rates from.
struct Baseline {
    detail::RawReading raw;
    SteadyClock::time_point at;
};

// Set on the monitor thread so stop() can refuse to join itself.
thread_local const void* tls_current_monitor = nullptr;

}  // namespace

struct SystemMonitor::Impl {
    Options options;
    std::unique_ptr<detail::PlatformSources> sources;

    // Lifecycle: start()/stop() serialize on this.
    std::mutex lifecycle_mutex;
    std::atomic<bool> running{false};
    std::thread thread;

    // Stop signalling for the monitor thread.
    std::mutex control_mutex;
    std::condition_variable cv;
    bool stop_requested = false;

    // Shared sample state.
    mutable std::mutex data_mutex;
    std::optional<SystemSample> latest;
    std::deque<SystemSample> history;
    std::string label;
    std::function<void(const SystemSample&)> callback;
    SteadyClock::time_point epoch = SteadyClock::now();

    // Log.
    std::mutex log_mutex;
    std::ofstream log;
    std::size_t csv_gpu_count = 0;
    bool log_error_reported = false;
    bool started_once = false;

    // Rate baselines: one per sampling path, so sample_now() never shortens the background
    // thread's measurement window (or vice versa).
    std::mutex now_mutex;
    Baseline now_baseline;
    Baseline thread_baseline;  // touched only by start() (before the thread exists) and the thread

    std::atomic<bool> callback_error_reported{false};
    std::atomic<bool> sample_error_reported{false};

    SystemSample build(const detail::RawReading& raw, Baseline& baseline) {
        const SteadyClock::time_point now = SteadyClock::now();
        SystemSample s;
        s.timestamp = std::chrono::system_clock::now();
        if (raw.system_cpu && baseline.raw.system_cpu) {
            s.cpu_utilization_percent = detail::cpu_percent_from_deltas(*baseline.raw.system_cpu, *raw.system_cpu);
        }
        const double wall = std::chrono::duration<double>(now - baseline.at).count();
        if (raw.process_cpu_seconds && baseline.raw.process_cpu_seconds && wall > 0.0) {
            const double cpu = *raw.process_cpu_seconds - *baseline.raw.process_cpu_seconds;
            if (cpu >= 0.0) {
                s.process_cpu_percent = 100.0 * cpu / wall;
            }
        }
        s.memory_used_bytes = raw.memory_used_bytes;
        s.memory_total_bytes = raw.memory_total_bytes;
        s.process_rss_bytes = raw.process_rss_bytes;
        s.cpu_temperature_c = raw.cpu_temperature_c;
        s.gpus = raw.gpus;
        // Keep the old counter if this read lost it, so a transient failure costs one rate,
        // not two.
        if (raw.system_cpu) {
            baseline.raw.system_cpu = raw.system_cpu;
        }
        if (raw.process_cpu_seconds) {
            baseline.raw.process_cpu_seconds = raw.process_cpu_seconds;
            baseline.at = now;
        }
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            s.label = label;
            s.elapsed_seconds = std::chrono::duration<double>(now - epoch).count();
        }
        return s;
    }

    void write_log_line(const std::string& line) {
        std::lock_guard<std::mutex> lock(log_mutex);
        if (!log.is_open()) {
            return;
        }
        log << line << '\n';
        log.flush();  // live: every record is on disk before the next one is taken
        if (!log) {
            if (!log_error_reported) {
                std::cerr << "pulsatrix::SystemMonitor: failed to write log '" << options.log_path
                          << "'; further write errors are not reported\n";
                log_error_reported = true;
            }
            log.clear();
        }
    }

    void take_background_sample() {
        SystemSample s = build(sources->read(), thread_baseline);
        std::function<void(const SystemSample&)> cb;
        {
            std::lock_guard<std::mutex> lock(data_mutex);
            latest = s;
            if (options.history_capacity > 0) {
                history.push_back(s);
                while (history.size() > options.history_capacity) {
                    history.pop_front();
                }
            }
            cb = callback;
        }
        if (!options.log_path.empty()) {
            std::size_t gpu_columns = 0;
            {
                std::lock_guard<std::mutex> lock(log_mutex);
                gpu_columns = csv_gpu_count;
            }
            write_log_line(options.format == LogFormat::Csv ? detail::sample_to_csv(s, gpu_columns)
                                                            : sample_to_json(s));
        }
        if (cb) {
            try {
                cb(s);
            } catch (const std::exception& e) {
                if (!callback_error_reported.exchange(true)) {
                    std::cerr << "pulsatrix::SystemMonitor: on_sample callback threw: " << e.what()
                              << " (further callback errors are not reported)\n";
                }
            } catch (...) {
                if (!callback_error_reported.exchange(true)) {
                    std::cerr << "pulsatrix::SystemMonitor: on_sample callback threw a non-std exception"
                              << " (further callback errors are not reported)\n";
                }
            }
        }
    }

    void run() {
        tls_current_monitor = this;
        std::unique_lock<std::mutex> lock(control_mutex);
        SteadyClock::time_point next = SteadyClock::now() + options.interval;
        for (;;) {
            if (cv.wait_until(lock, next, [this] { return stop_requested; })) {
                break;
            }
            lock.unlock();
            try {
                take_background_sample();
            } catch (const std::exception& e) {
                if (!sample_error_reported.exchange(true)) {
                    std::cerr << "pulsatrix::SystemMonitor: sampling failed: " << e.what()
                              << " (further sampling errors are not reported)\n";
                }
            }
            lock.lock();
            next += options.interval;
            const SteadyClock::time_point now = SteadyClock::now();
            if (next <= now) {
                next = now + options.interval;  // fell behind: skip, don't burst
            }
        }
        tls_current_monitor = nullptr;
    }

    void open_log() {
        if (options.log_path.empty()) {
            return;
        }
        const bool append = options.append || started_once;
        std::lock_guard<std::mutex> lock(log_mutex);
        log.open(options.log_path, append ? std::ios::out | std::ios::app : std::ios::out | std::ios::trunc);
        if (!log.is_open()) {
            throw std::runtime_error("pulsatrix::SystemMonitor: cannot open log file '" + options.log_path +
                                     "' for writing");
        }
        log_error_reported = false;
        csv_gpu_count = thread_baseline.raw.gpus.size();
        if (options.format == LogFormat::Csv) {
            log.seekp(0, std::ios::end);
            if (log.tellp() == std::streampos(0)) {
                log << detail::csv_header(csv_gpu_count) << '\n';
                log.flush();
            }
        }
    }

    void close_log() {
        std::lock_guard<std::mutex> lock(log_mutex);
        if (log.is_open()) {
            log.close();
        }
    }
};

SystemMonitor::SystemMonitor() : SystemMonitor(Options{}) {}

SystemMonitor::SystemMonitor(Options options) : SystemMonitor(std::move(options), detail::make_host_sources()) {}

SystemMonitor::SystemMonitor(Options options, std::unique_ptr<detail::PlatformSources> sources)
    : impl_(std::make_unique<Impl>()) {
    if (options.interval.count() <= 0) {
        throw std::invalid_argument("pulsatrix::SystemMonitor: interval must be positive");
    }
    if (!sources) {
        throw std::invalid_argument("pulsatrix::SystemMonitor: sources must not be null");
    }
    impl_->options = std::move(options);
    impl_->sources = std::move(sources);
    impl_->now_baseline = Baseline{impl_->sources->read(), SteadyClock::now()};
}

SystemMonitor::~SystemMonitor() { stop(); }

void SystemMonitor::start() {
    std::lock_guard<std::mutex> lifecycle(impl_->lifecycle_mutex);
    if (impl_->running) {
        return;
    }
    impl_->thread_baseline = Baseline{impl_->sources->read(), SteadyClock::now()};
    impl_->open_log();  // throws before any thread exists
    impl_->started_once = true;
    {
        std::lock_guard<std::mutex> lock(impl_->data_mutex);
        impl_->epoch = impl_->thread_baseline.at;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->control_mutex);
        impl_->stop_requested = false;
    }
    impl_->thread = std::thread([impl = impl_.get()] { impl->run(); });
    impl_->running = true;
}

void SystemMonitor::stop() {
    if (tls_current_monitor == impl_.get()) {
        throw std::logic_error("pulsatrix::SystemMonitor::stop called from the monitor thread (on_sample callback)");
    }
    std::lock_guard<std::mutex> lifecycle(impl_->lifecycle_mutex);
    if (!impl_->running) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->control_mutex);
        impl_->stop_requested = true;
    }
    impl_->cv.notify_all();
    impl_->thread.join();
    impl_->running = false;
    impl_->close_log();
}

bool SystemMonitor::running() const { return impl_->running; }

SystemSample SystemMonitor::sample_now() {
    std::lock_guard<std::mutex> lock(impl_->now_mutex);
    return impl_->build(impl_->sources->read(), impl_->now_baseline);
}

std::optional<SystemSample> SystemMonitor::latest() const {
    std::lock_guard<std::mutex> lock(impl_->data_mutex);
    return impl_->latest;
}

std::vector<SystemSample> SystemMonitor::history() const {
    std::lock_guard<std::mutex> lock(impl_->data_mutex);
    return std::vector<SystemSample>(impl_->history.begin(), impl_->history.end());
}

void SystemMonitor::on_sample(std::function<void(const SystemSample&)> callback) {
    std::lock_guard<std::mutex> lock(impl_->data_mutex);
    impl_->callback = std::move(callback);
}

void SystemMonitor::mark(std::string label) {
    const auto timestamp = std::chrono::system_clock::now();
    double elapsed = 0.0;
    {
        std::lock_guard<std::mutex> lock(impl_->data_mutex);
        impl_->label = label;
        elapsed = std::chrono::duration<double>(SteadyClock::now() - impl_->epoch).count();
    }
    if (impl_->options.log_path.empty()) {
        return;
    }
    std::size_t gpu_columns = 0;
    {
        std::lock_guard<std::mutex> lock(impl_->log_mutex);
        gpu_columns = impl_->csv_gpu_count;
    }
    impl_->write_log_line(impl_->options.format == LogFormat::Csv
                              ? detail::mark_to_csv(timestamp, elapsed, label, gpu_columns)
                              : detail::mark_to_json(timestamp, elapsed, label));
}

std::vector<MetricCapability> SystemMonitor::capabilities() const { return impl_->sources->capabilities(); }

const SystemMonitor::Options& SystemMonitor::options() const noexcept { return impl_->options; }

}  // namespace pulsatrix
