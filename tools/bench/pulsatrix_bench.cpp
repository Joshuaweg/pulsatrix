// pulsatrix_bench: the benchmark suite (roadmap KS-2). Measures training step time, explanation
// time and LRP conservation error on every backend this build has, and compares two builds.
//
//   pulsatrix_bench run [--device cpu|hip|cuda|all] [--quick] [--filter TEXT] [--repeats N]
//                       [--warmup N] [--label TEXT] [--check] [-o report.json]
//   pulsatrix_bench compare --baseline A.json... --candidate B.json...
//                           [--time-tolerance 0.10] [--metric-tolerance 1e-5] [--significance 0.05]
//
// run prints a table and, with -o, writes a pulsatrix.benchmark.v1 report. --check exits 1 if an
// LRP conservation error is above 1e-3, a correctness gate that doesn't depend on timing noise.
// compare exits 1 if the candidate regressed. scripts/bench_ab.sh runs two builds interleaved
// and compares them.
//
// Exit status: 0 on success, 1 on a failed check or a regression, 2 on a usage error.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/benchmark.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/integrated_gradients.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/lrp_conservation.hpp"
#include "pulsatrix/tagger_finetune_example.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"
#include "workloads.hpp"
#ifdef PULSATRIX_BENCH_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif
#ifdef PULSATRIX_BENCH_WITH_CUDA
#include "pulsatrix/cuda_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

// The relative conservation error the Conv2D LRP tests accept per layer
// (conv2d_stride_padding_test.cpp). In float32 these models land near 2e-4: rounding over
// thousands of inputs, plus what the epsilon stabilizer absorbs.
constexpr double kConservationLimit = 1e-3;

constexpr const char* kUsage =
    "usage: pulsatrix_bench run [--device cpu|hip|cuda|all] [--quick] [--filter TEXT] [--repeats N]\n"
    "                           [--warmup N] [--label TEXT] [--check] [-o report.json]\n"
    "       pulsatrix_bench compare --baseline A.json... --candidate B.json...\n"
    "                               [--time-tolerance 0.10] [--metric-tolerance 1e-5] [--significance 0.05]\n";

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_bench: " << problem << "\n" << kUsage;
    std::exit(2);
}

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("can't open " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct RunOptions {
    std::string device = "all";
    std::string filter;
    std::string label;
    std::string output;
    int warmup = 2;
    int repeats = 10;
    bool check = false;
};

// Collects records for one device, skipping benchmarks the filter excludes.
class Suite {
public:
    Suite(std::string device, const RunOptions& o, BenchmarkReport& report) : device_(std::move(device)), o_(o), report_(report) {}

    bool wanted(const std::string& name) const { return o_.filter.empty() || name.find(o_.filter) != std::string::npos; }

    // @p per divides every timing, for a closure that runs @p per units of work.
    void time(const std::string& name, const std::string& unit, const std::function<void()>& fn, int per = 1) {
        if (!wanted(name)) return;
        BenchmarkRecord r;
        r.name = name;
        r.device = device_;
        r.kind = BenchmarkRecord::Kind::Time;
        r.unit = unit;
        r.stats = TimeIt(fn, o_.warmup, o_.repeats);
        for (double* v : {&r.stats.min_ms, &r.stats.p10_ms, &r.stats.median_ms, &r.stats.p90_ms, &r.stats.mean_ms}) {
            *v /= per;
        }
        report_.records.push_back(r);
        std::printf("  %-38s %-5s %10.3f ms  (p10 %.3f, p90 %.3f, n=%lld)\n", name.c_str(), device_.c_str(), r.stats.median_ms,
                    r.stats.p10_ms, r.stats.p90_ms, static_cast<long long>(r.stats.repeats));
        std::fflush(stdout);
    }

    void metric(const std::string& name, const std::string& unit, const std::function<double()>& fn) {
        if (!wanted(name)) return;
        BenchmarkRecord r;
        r.name = name;
        r.device = device_;
        r.kind = BenchmarkRecord::Kind::Metric;
        r.unit = unit;
        r.value = fn();
        report_.records.push_back(r);
        std::printf("  %-38s %-5s %13.3e  %s\n", name.c_str(), device_.c_str(), r.value, unit.c_str());
        std::fflush(stdout);
    }

private:
    std::string device_;
    const RunOptions& o_;
    BenchmarkReport& report_;
};

// One full training step: forward, loss, backward, AdamW. Reading the loss waits for the device.
std::function<void()> TrainStep(bench::Classifier& c, DeviceBackend* backend) {
    auto opt = std::make_shared<AdamWOptimizer>(1e-3f, backend);
    auto loss = std::make_shared<TokenCrossEntropyLoss>(backend);
    return [&c, opt, loss] {
        opt->zero_grad(*c.model);
        volatile float l = loss->forward(c.model->forward(c.x), c.y);
        (void)l;
        (void)c.model->backward(loss->backward());
        opt->step(*c.model);
    };
}

// The predicted class of c.single, and its logit.
std::pair<int64_t, float> Predict(ExplainerContext& ctx, const bench::Classifier& c) {
    std::vector<float> logits = ctx.forward_pass(c.single).to_host_vector();
    int64_t best = 0;
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > logits[static_cast<size_t>(best)]) best = static_cast<int64_t>(i);
    }
    return {best, logits[static_cast<size_t>(best)]};
}

// |sum of input relevance - explained logit| / |explained logit|, LRPSeed::OutputValue.
double ConservationError(const LRP& lrp, ExplainerContext& ctx, const bench::Classifier& c, DeviceBackend* backend) {
    auto [target, logit] = Predict(ctx, c);
    Attribution a = lrp.explain(ctx, c.single, target, backend);
    std::vector<float> seed(10, 0.0f);
    seed[static_cast<size_t>(target)] = logit;
    CPUBackend host;
    ConservationResult r = ComputeConservation(Tensor(a.values.shape(), &host, a.values.to_host_vector()),
                                               Tensor(Shape({1, 10}), &host, seed));
    return std::abs(static_cast<double>(r.delta())) / std::max(std::abs(static_cast<double>(logit)), 1e-30);
}

void RunDevice(const std::string& device, DeviceBackend* backend, const RunOptions& o, BenchmarkReport& report) {
    std::printf("%s\n", device.c_str());
    Suite s(device, o, report);

    // Training step time.
    if (s.wanted("train.mlp")) {
        bench::Classifier mlp = bench::MakeMlp(backend);
        s.time("train.mlp", "ms per step", TrainStep(mlp, backend));
    }
    if (s.wanted("train.cnn")) {
        bench::Classifier cnn = bench::MakeCnn(backend);
        s.time("train.cnn", "ms per step", TrainStep(cnn, backend));
    }
    // TrainTagger runs a whole schedule, so each sample is 5 steps (plus model setup), divided by 5.
    s.time(
        "train.tagger", "ms per step",
        [backend] {
            TinyTagger tagger(backend);
            InitTagger(tagger, 1);
            FineTuneConfig config;
            config.steps = 5;
            config.warmup = 1;
            volatile float l = TrainTagger(tagger, TaggingRule::SumWithPrevious, config, backend).back();
            (void)l;
        },
        5);

    // Explanation time and correctness, on one input each.
    bench::Classifier mlp = bench::MakeMlp(backend);
    bench::Classifier conv = bench::MakeConvNet(backend, false);
    ExplainerContext mlp_ctx(mlp.layers);
    ExplainerContext conv_ctx(conv.layers);
    const LRP epsilon;
    const LRP epsilon_plus = LRP::epsilon_plus();
    const int64_t mlp_target = Predict(mlp_ctx, mlp).first;
    const int64_t conv_target = Predict(conv_ctx, conv).first;
    const IntegratedGradients ig;
    const Tensor mlp_baseline(mlp.single.shape(), backend, std::vector<float>(static_cast<size_t>(mlp.single.numel()), 0.0f));

    s.time("explain.lrp_epsilon.mlp", "ms per explanation", [&] {
        volatile float v = epsilon.explain(mlp_ctx, mlp.single, mlp_target, backend).values.to_host_vector()[0];
        (void)v;
    });
    s.time("explain.lrp_epsilon_plus.convnet", "ms per explanation", [&] {
        volatile float v = epsilon_plus.explain(conv_ctx, conv.single, conv_target, backend).values.to_host_vector()[0];
        (void)v;
    });
    s.time("explain.integrated_gradients.mlp", "ms per explanation (32 steps)", [&] {
        volatile float v = ig.explain(mlp_ctx, mlp.single, mlp_baseline, mlp_target, 32, backend).values.to_host_vector()[0];
        (void)v;
    });

    s.metric("conservation.lrp_epsilon.mlp", "relative error", [&] { return ConservationError(epsilon, mlp_ctx, mlp, backend); });
    s.metric("conservation.lrp_epsilon.convnet", "relative error",
             [&] { return ConservationError(epsilon, conv_ctx, conv, backend); });
    s.metric("conservation.lrp_epsilon_plus.convnet", "relative error",
             [&] { return ConservationError(epsilon_plus, conv_ctx, conv, backend); });
    // Completeness: IG's attributions sum to f(x) - f(baseline), up to its Riemann-sum error. With
    // zero biases this ReLU MLP is positively homogeneous, so the gradient is constant along the
    // path from the zero baseline and the sum is exact up to rounding.
    s.metric("completeness.integrated_gradients.mlp", "relative error", [&] {
        std::vector<float> a = ig.explain(mlp_ctx, mlp.single, mlp_baseline, mlp_target, 32, backend).values.to_host_vector();
        double sum = 0.0;
        for (float v : a) sum += v;
        const double fx = mlp_ctx.forward_pass(mlp.single).to_host_vector()[static_cast<size_t>(mlp_target)];
        const double fb = mlp_ctx.forward_pass(mlp_baseline).to_host_vector()[static_cast<size_t>(mlp_target)];
        return std::abs(sum - (fx - fb)) / std::max(std::abs(fx - fb), 1e-30);
    });
}

std::string Compiler() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_FULL_VER);
#else
    return "unknown";
#endif
}

int Run(int argc, char** argv) {
    RunOptions o;
    bool quick = false;
    bool repeats_set = false, warmup_set = false;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--device") o.device = value();
        else if (a == "--filter") o.filter = value();
        else if (a == "--label") o.label = value();
        else if (a == "-o" || a == "--output") o.output = value();
        else if (a == "--repeats") { o.repeats = std::atoi(value().c_str()); repeats_set = true; }
        else if (a == "--warmup") { o.warmup = std::atoi(value().c_str()); warmup_set = true; }
        else if (a == "--quick") quick = true;
        else if (a == "--check") o.check = true;
        else Usage("unknown option " + a);
    }
    if (quick) {
        if (!repeats_set) o.repeats = 3;
        if (!warmup_set) o.warmup = 1;
    }
    if (o.repeats < 1 || o.warmup < 0) Usage("--repeats must be at least 1 and --warmup at least 0");

    BenchmarkReport report;
    report.label = o.label;
    report.environment["version"] = PULSATRIX_BENCH_VERSION;
    report.environment["build_type"] = PULSATRIX_BENCH_BUILD_TYPE;
    report.environment["compiler"] = Compiler();
    report.environment["repeats"] = std::to_string(o.repeats);
    report.environment["warmup"] = std::to_string(o.warmup);

    bool ran = false;
    auto want = [&](const char* d) { return o.device == "all" || o.device == d; };
    if (want("cpu")) {
        CPUBackend cpu;
        RunDevice("cpu", &cpu, o, report);
        ran = true;
    }
#ifdef PULSATRIX_BENCH_WITH_HIP
    if (want("hip")) {
        HIPBackend hip;
        RunDevice("hip", &hip, o, report);
        ran = true;
    }
#endif
#ifdef PULSATRIX_BENCH_WITH_CUDA
    if (want("cuda")) {
        CUDABackend cuda;
        RunDevice("cuda", &cuda, o, report);
        ran = true;
    }
#endif
    if (!ran) Usage("device \"" + o.device + "\" is not in this build");

    if (!o.output.empty()) {
        std::ofstream out(o.output, std::ios::binary);
        out << ToJson(report);
        if (!out) throw std::runtime_error("can't write " + o.output);
    }
    if (o.check) {
        bool ok = true;
        for (const BenchmarkRecord& r : report.records) {
            if (r.name.rfind("conservation.", 0) == 0 && !(r.value <= kConservationLimit)) {
                std::fprintf(stderr, "check failed: %s on %s is %.3e, above %.0e\n", r.name.c_str(), r.device.c_str(), r.value,
                             kConservationLimit);
                ok = false;
            }
        }
        if (!ok) return 1;
        std::printf("check passed: every LRP conservation error is at most %.0e\n", kConservationLimit);
    }
    return 0;
}

int Compare(int argc, char** argv) {
    std::vector<std::string> base_paths, cand_paths;
    CompareOptions options;
    std::vector<std::string>* list = nullptr;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--baseline") list = &base_paths;
        else if (a == "--candidate") list = &cand_paths;
        else if (a == "--time-tolerance" || a == "--metric-tolerance" || a == "--significance") {
            if (i + 1 >= argc) Usage(a + " needs a value");
            char* end = nullptr;
            const double v = std::strtod(argv[++i], &end);
            if (*end != '\0') Usage(a + " needs a number");
            (a == "--time-tolerance" ? options.time_tolerance : a == "--significance" ? options.significance : options.metric_tolerance) = v;
            list = nullptr;
        } else if (!a.empty() && a[0] == '-') Usage("unknown option " + a);
        else if (list == nullptr) Usage("a report path must follow --baseline or --candidate");
        else list->push_back(a);
    }
    if (base_paths.empty() || cand_paths.empty()) Usage("compare needs --baseline and --candidate reports");
    std::vector<BenchmarkReport> base, cand;
    for (const auto& p : base_paths) base.push_back(ParseBenchmarkReport(ReadFile(p)));
    for (const auto& p : cand_paths) cand.push_back(ParseBenchmarkReport(ReadFile(p)));

    int regressions = 0;
    std::printf("%-38s %-5s %13s %13s %9s %7s\n", "benchmark", "dev", "baseline", "candidate", "change", "p");
    for (const BenchmarkComparison& c : CompareBenchmarks(base, cand, options)) {
        const bool time = c.kind == BenchmarkRecord::Kind::Time;
        if (!c.missing_from.empty()) {
            std::printf("%-38s %-5s  (missing from %s)\n", c.name.c_str(), c.device.c_str(), c.missing_from.c_str());
            continue;
        }
        char change[32];
        if (time) std::snprintf(change, sizeof change, "%+.1f%%", c.change * 100.0);
        else std::snprintf(change, sizeof change, "%+.1e", c.change);
        char p[16] = "";
        if (time && !std::isnan(c.p_value)) std::snprintf(p, sizeof p, "%.3f", c.p_value);
        std::printf(time ? "%-38s %-5s %10.3f ms %10.3f ms %9s %7s%s\n" : "%-38s %-5s %13.3e %13.3e %9s %7s%s\n", c.name.c_str(),
                    c.device.c_str(), c.baseline, c.candidate, change, p, c.regression ? "  REGRESSION" : "");
        regressions += c.regression ? 1 : 0;
    }
    std::printf("%d regression%s (time tolerance %.0f%% and p < %.2f with 3+ rounds per side, metric tolerance %.0e)\n",
                regressions, regressions == 1 ? "" : "s", options.time_tolerance * 100.0, options.significance,
                options.metric_tolerance);
    return regressions > 0 ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string command = argc > 1 ? argv[1] : "";
    try {
        if (command == "run") return Run(argc, argv);
        if (command == "compare") return Compare(argc, argv);
        if (command == "-h" || command == "--help") {
            std::cout << kUsage;
            return 0;
        }
        Usage(command.empty() ? "no command" : "unknown command " + command);
    } catch (const std::exception& e) {
        std::cerr << "pulsatrix_bench: " << e.what() << "\n";
        return 1;
    }
}
