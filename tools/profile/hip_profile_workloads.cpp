// Fixed GPU training workloads for scripts/profile_hip.sh (roadmap HIP-1): the baselines that
// HIP-2, HIP-4 and HIP-5 are measured against. Built with PULSATRIX_ENABLE_HIP=ON.
//
//   hip_profile_workloads <tagger|cnn|mlp> [steps]
//
//   tagger  the TRN-6 recipe's transformer tagger: embedding, attention, norms, small GEMMs
//   cnn     Conv2D(3->16) -> BatchNorm -> ReLU -> Conv2D(16->32, stride 2) -> BatchNorm -> ReLU
//           -> Flatten -> Linear, on 32 images of 3x32x32: im2col, col2im, per-channel norms
//   mlp     Linear(256->512) -> ReLU -> Linear(512->512) -> ReLU -> Linear(512->10), batch 64
//   reduce  DeviceBackend::dot and sum over 16M floats, the HIP-5 microbenchmark
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/hip_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/tagger_finetune_example.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"
#include "workloads.hpp"  // tools/bench: the same models pulsatrix_bench measures

using namespace pulsatrix;

namespace {

// One full training step per iteration: forward, loss, backward, AdamW.
void run_classifier(Module& model, DeviceBackend* backend, const Tensor& x, const Tensor& y, int steps) {
    AdamWOptimizer opt(1e-3f, backend);
    TokenCrossEntropyLoss loss(backend);
    for (int step = 0; step < steps; ++step) {
        const auto t0 = std::chrono::steady_clock::now();
        opt.zero_grad(model);
        const float l = loss.forward(model.forward(x), y);
        (void)model.backward(loss.backward());
        opt.step(model);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("step %d  loss %.4f  %.2f ms\n", step, l, ms);
    }
}

}  // namespace

// Allocator counters (HIP-3): unaffected by other programs sharing the GPU, unlike timings.
void print_memory(const HIPBackend& backend) {
    const CachingAllocator::Stats s = backend.memory_stats();
    std::printf("allocator: %zu hipMalloc calls, %zu cache hits, peak %.1f MB in use\n", s.raw_allocs, s.cache_hits,
                static_cast<double>(s.peak_in_use_bytes) / (1 << 20));
}

int run(const std::string& workload, int steps, HIPBackend& backend);

int main(int argc, char** argv) {
    const std::string workload = argc > 1 ? argv[1] : "";
    const int steps = argc > 2 ? std::atoi(argv[2]) : 20;
    HIPBackend backend;
    const int status = run(workload, steps, backend);
    if (status == 0) {
        print_memory(backend);
    } else {
        std::fprintf(stderr, "usage: %s <tagger|cnn|mlp|reduce> [steps]\n", argv[0]);
    }
    return status;
}

int run(const std::string& workload, int steps, HIPBackend& backend) {

    if (workload == "tagger") {
        TinyTagger tagger(&backend);
        InitTagger(tagger, 1);
        FineTuneConfig config;
        config.steps = steps;
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> losses = TrainTagger(tagger, TaggingRule::SumWithPrevious, config, &backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        std::printf("tagger: %d steps, final loss %.4f, %.2f ms per step\n", steps, losses.back(), ms / steps);
        return 0;
    }
    if (workload == "cnn" || workload == "mlp") {
        bench::Classifier c = workload == "cnn" ? bench::MakeCnn(&backend) : bench::MakeMlp(&backend);
        run_classifier(*c.model, &backend, c.x, c.y, steps);
        return 0;
    }
    if (workload == "reduce") {
        constexpr int64_t n = int64_t{1} << 24;
        Tensor a(Shape({n}), &backend, bench::Pattern(static_cast<size_t>(n), 0.01f));
        Tensor b(Shape({n}), &backend, bench::Pattern(static_cast<size_t>(n), 0.02f));
        for (int step = 0; step < steps; ++step) {
            const auto t0 = std::chrono::steady_clock::now();
            const float d = backend.dot(a.data(), b.data(), static_cast<size_t>(n));
            const float s = backend.sum(a.data(), static_cast<size_t>(n));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("step %d  dot %.3f  sum %.3f  %.3f ms\n", step, d, s, ms);
        }
        return 0;
    }
    return 2;
}
