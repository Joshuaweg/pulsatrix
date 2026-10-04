// Fixed GPU training workloads for scripts/profile_hip.sh (roadmap HIP-1): the baselines that
// HIP-2, HIP-4 and HIP-5 are measured against. Built with PULSATRIX_ENABLE_HIP=ON.
//
//   hip_profile_workloads <tagger|cnn|mlp> [steps]
//
//   tagger  the TRN-6 recipe's transformer tagger: embedding, attention, norms, small GEMMs
//   cnn     Conv2D(3->16) -> BatchNorm -> ReLU -> Conv2D(16->32, stride 2) -> BatchNorm -> ReLU
//           -> Flatten -> Linear, on 32 images of 3x32x32: im2col, col2im, per-channel norms
//   mlp     Linear(256->512) -> ReLU -> Linear(512->512) -> ReLU -> Linear(512->10), batch 64
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

using namespace pulsatrix;

namespace {

// Small deterministic values: matrices +-1/sqrt(fan_in), 1-D parameters 1 (norm scales) or 0.
void init(Module& m) {
    uint64_t s = 12345;
    auto uniform = [&s] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(s >> 40) / static_cast<float>(1ULL << 24);
    };
    for (const NamedParamRef& p : m.named_parameters()) {
        Tensor& t = *p.ref.value;
        std::vector<float> v(static_cast<size_t>(t.numel()));
        if (t.rank() >= 2) {
            const float a = 1.0f / std::sqrt(static_cast<float>(t.numel() / t.shape().dim(t.rank() - 1)));
            for (float& x : v) x = (2.0f * uniform() - 1.0f) * a;
        } else {
            const float fill = p.name.find("weight") != std::string::npos ? 1.0f : 0.0f;
            for (float& x : v) x = fill;
        }
        t = Tensor(t.shape(), t.backend(), v, t.device());
    }
}

std::vector<float> pattern(size_t n, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * static_cast<float>(static_cast<int>((i * 7919) % 17) - 8);
    return v;
}

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

int main(int argc, char** argv) {
    const std::string workload = argc > 1 ? argv[1] : "";
    const int steps = argc > 2 ? std::atoi(argv[2]) : 20;
    HIPBackend backend;

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
    if (workload == "cnn") {
        constexpr int64_t n = 32;
        Conv2DModule c1(3, 16, 3, 3, &backend, 1, 1), c2(16, 32, 3, 3, &backend, 2, 1);
        BatchNormModule b1(16, &backend), b2(32, &backend);
        ReluModule r1(&backend), r2(&backend);
        FlattenModule flat(&backend);
        LinearModule fc(32 * 16 * 16, 10, &backend);
        SequentialModule model({&c1, &b1, &r1, &c2, &b2, &r2, &flat, &fc});
        init(model);
        Tensor x(Shape({n, 3, 32, 32}), &backend, pattern(n * 3 * 32 * 32, 0.1f));
        std::vector<float> labels(n);
        for (int64_t i = 0; i < n; ++i) labels[static_cast<size_t>(i)] = static_cast<float>(i % 10);
        run_classifier(model, &backend, x, Tensor(Shape({n}), &backend, labels), steps);
        return 0;
    }
    if (workload == "mlp") {
        constexpr int64_t n = 64;
        LinearModule l1(256, 512, &backend), l2(512, 512, &backend), l3(512, 10, &backend);
        ReluModule r1(&backend), r2(&backend);
        SequentialModule model({&l1, &r1, &l2, &r2, &l3});
        init(model);
        Tensor x(Shape({n, 256}), &backend, pattern(n * 256, 0.05f));
        std::vector<float> labels(n);
        for (int64_t i = 0; i < n; ++i) labels[static_cast<size_t>(i)] = static_cast<float>(i % 10);
        run_classifier(model, &backend, x, Tensor(Shape({n}), &backend, labels), steps);
        return 0;
    }
    std::fprintf(stderr, "usage: %s <tagger|cnn|mlp> [steps]\n", argv[0]);
    return 2;
}
