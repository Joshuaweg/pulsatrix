// KD-1 on the GPU: DistillationLoss on the HIP backend gives the CPU's loss and gradient, with and
// without dropped rows.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/distillation_loss.hpp"
#include "pulsatrix/feature_distillation.hpp"
#include "pulsatrix/hip_backend.hpp"

namespace pulsatrix {
namespace {

struct Outcome {
    float loss;
    std::vector<float> grad;
};

Outcome Distill(DeviceBackend* backend, DistillationDivergence divergence, bool with_targets) {
    const Shape shape({3, 5, 40});
    std::vector<float> s(static_cast<size_t>(shape.numel())), t(s.size());
    for (size_t k = 0; k < s.size(); ++k) {
        s[k] = static_cast<float>(2 * std::sin(0.7 * k + 0.3));
        t[k] = static_cast<float>(3 * std::cos(0.45 * k + 0.1));
    }
    std::vector<float> y(15);
    for (size_t k = 0; k < y.size(); ++k) y[k] = k % 4 == 1 ? -100.0f : static_cast<float>((7 * k) % 40);
    DistillationOptions o;
    o.temperature = 2.5f;
    o.alpha = 0.6f;
    o.divergence = divergence;
    DistillationLoss loss(backend, o);
    const Tensor st(shape, backend, s, backend->device()), tt(shape, backend, t, backend->device());
    const float value = with_targets ? loss.forward(st, tt, Tensor(Shape({3, 5}), backend, y, backend->device()))
                                     : loss.forward(st, tt);
    const Tensor grad = loss.backward();
    EXPECT_EQ(grad.device(), backend->device());
    return {value, grad.to_host_vector()};
}

TEST(DistillationLossHip, TheGpuGivesTheCpusLossAndGradient) {
    HIPBackend hip;
    CPUBackend cpu;
    for (DistillationDivergence divergence : {DistillationDivergence::Forward, DistillationDivergence::Reverse}) {
        for (bool with_targets : {false, true}) {
            const Outcome g = Distill(&hip, divergence, with_targets), c = Distill(&cpu, divergence, with_targets);
            EXPECT_NEAR(g.loss, c.loss, 1e-5 * std::abs(c.loss) + 1e-6);
            ASSERT_EQ(g.grad.size(), c.grad.size());
            for (size_t k = 0; k < g.grad.size(); ++k) EXPECT_NEAR(g.grad[k], c.grad[k], 1e-6) << k;
        }
    }
}

// KD-3: the hint loss (through its regressor) and attention transfer on the GPU give the CPU's.
std::vector<float> Wave(size_t n, double a, double b) {
    std::vector<float> v(n);
    for (size_t k = 0; k < n; ++k) v[k] = static_cast<float>(std::sin(a * static_cast<double>(k) + b));
    return v;
}

std::vector<std::vector<float>> FeatureLosses(DeviceBackend* backend) {
    const DeviceType d = backend->device();
    HintLoss hint(6, 10, backend, 4);
    const float h = hint.forward(Tensor(Shape({3, 4, 6}), backend, Wave(72, 0.7, 0.3), d), Tensor(Shape({3, 4, 10}), backend, Wave(120, 0.45, 0.1), d));
    AttentionTransferLoss at(backend);
    const float a = at.forward(Tensor(Shape({2, 3, 4, 5}), backend, Wave(120, 0.9, 0.2), d), Tensor(Shape({2, 6, 4, 5}), backend, Wave(240, 0.35, 0.6), d));
    const Tensor hg = hint.backward(), ag = at.backward();
    EXPECT_EQ(hg.device(), d);
    EXPECT_EQ(ag.device(), d);
    return {{h, a}, hg.to_host_vector(), hint.regressor()->weight_grad().to_host_vector(), ag.to_host_vector()};
}

TEST(FeatureDistillationHip, TheGpuGivesTheCpusLossesAndGradients) {
    HIPBackend hip;
    CPUBackend cpu;
    const std::vector<std::vector<float>> g = FeatureLosses(&hip), c = FeatureLosses(&cpu);
    for (size_t part = 0; part < g.size(); ++part) {
        ASSERT_EQ(g[part].size(), c[part].size());
        for (size_t k = 0; k < g[part].size(); ++k) EXPECT_NEAR(g[part][k], c[part][k], 1e-5 * std::abs(c[part][k]) + 1e-6) << part << " " << k;
    }
}

}  // namespace
}  // namespace pulsatrix
