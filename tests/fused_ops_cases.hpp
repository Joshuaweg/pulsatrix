// HIP-6: the fused training ops give what the unfused ones did. Shared by fused_ops_test.cpp (CPU)
// and the HIP and CUDA suites.
#pragma once

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix::fused_cases {

inline std::vector<float> Random(size_t n, unsigned seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

inline std::vector<float> Host(const Tensor& t) {
    if (t.device() == DeviceType::Cpu) return t.to_host_vector();
    CPUBackend cpu;
    Tensor h = t;
    h.to(DeviceType::Cpu, &cpu);
    return h.to_host_vector();
}

inline Tensor On(DeviceBackend* b, const std::vector<float>& v, Shape shape) { return Tensor(shape, b, v, b->device()); }

// 40 tensors (more than one launch's 32), some empty, in each decay mode, three steps: the one
// call equals axpby() then adam_step() per tensor, bit for bit.
inline void AdamMultiMatchesPerTensor(DeviceBackend* b) {
    for (int mode = 0; mode < 3; ++mode) {
        SCOPED_TRACE("mode " + std::to_string(mode));
        std::vector<Tensor> p1, p2, g, m1, v1, m2, v2;
        std::vector<size_t> sizes;
        for (unsigned t = 0; t < 40; ++t) {
            const size_t n = t % 7 == 3 ? 0 : 1 + (t * 37) % 300;
            sizes.push_back(n);
            const Shape s({static_cast<int64_t>(n == 0 ? 1 : n)});
            p1.push_back(On(b, Random(n == 0 ? 1 : n, 10 + t), s));
            p2.push_back(p1.back());
            g.push_back(On(b, Random(n == 0 ? 1 : n, 100 + t), s));
            for (auto* vec : {&m1, &v1, &m2, &v2}) {
                vec->push_back(Tensor(s, b, b->device()));
                vec->back().fill(0.0f);
            }
        }
        for (int step = 1; step <= 3; ++step) {
            std::vector<AdamTensorStep> steps;
            for (size_t t = 0; t < 40; ++t) {
                AdamTensorStep s;
                s.param = p1[t].data();
                s.grad = g[t].data();
                s.m = m1[t].data();
                s.v = v1[t].data();
                s.n = sizes[t];
                s.lr = 1e-3f * static_cast<float>(1 + t % 3);
                s.bias_correction1 = 1.0f - std::pow(0.9f, static_cast<float>(step));
                s.bias_correction2 = 1.0f - std::pow(0.999f, static_cast<float>(step));
                if (mode == 1) s.decay = 1.0f - s.lr * 0.01f;
                if (mode == 2) s.l2 = 0.01f;
                steps.push_back(s);
                // The unfused reference on the second copy.
                if (s.n == 0) continue;
                if (mode == 1) b->axpby(s.decay, p2[t].data(), 0.0f, p2[t].data(), p2[t].data(), s.n);
                Tensor decayed(Shape({static_cast<int64_t>(s.n)}), b, b->device());
                const float* grad = g[t].data();
                if (mode == 2) {
                    b->axpby(s.l2, p2[t].data(), 1.0f, g[t].data(), decayed.data(), s.n);
                    grad = decayed.data();
                }
                b->adam_step(p2[t].data(), grad, m2[t].data(), v2[t].data(), s.n, s.lr, 0.9f, 0.999f, 1e-8f,
                             s.bias_correction1, s.bias_correction2);
            }
            b->adam_step_multi(steps.data(), steps.size(), 0.9f, 0.999f, 1e-8f);
        }
        for (size_t t = 0; t < 40; ++t) {
            EXPECT_EQ(Host(p1[t]), Host(p2[t])) << "tensor " << t;
            EXPECT_EQ(Host(m1[t]), Host(m2[t])) << "tensor " << t;
            EXPECT_EQ(Host(v1[t]), Host(v2[t])) << "tensor " << t;
        }
    }
}

inline void DotIntoMatchesDot(DeviceBackend* b) {
    Tensor out(Shape({4}), b, b->device());
    out.fill(-1.0f);
    std::vector<float> expected;
    const size_t sizes[] = {0, 1, 1000, 300000};
    for (size_t k = 0; k < 4; ++k) {
        const size_t n = sizes[k];
        const Tensor x = On(b, Random(n == 0 ? 1 : n, 7 + static_cast<unsigned>(k)), Shape({static_cast<int64_t>(n == 0 ? 1 : n)}));
        b->dot_into(x.data(), x.data(), n, out.data() + k);
        expected.push_back(b->dot(x.data(), x.data(), n));
    }
    EXPECT_EQ(Host(out), expected);
}

// The device-validated loss against the CPU's, its gradient, and its errors.
inline void TokenCrossEntropyOnDevice(DeviceBackend* b) {
    constexpr int64_t kRows = 37, kClasses = 11;
    CPUBackend cpu;
    const std::vector<float> logits = Random(kRows * kClasses, 3, -4.0f, 4.0f);
    std::vector<float> targets(kRows);
    for (int64_t r = 0; r < kRows; ++r) targets[r] = r % 5 == 0 ? -100.0f : static_cast<float>((r * 7) % kClasses);
    TokenCrossEntropyLoss on_cpu(&cpu), on_device(b);
    const float l_cpu = on_cpu.forward(Tensor(Shape({kRows, kClasses}), &cpu, logits), Tensor(Shape({kRows}), &cpu, targets));
    const float l_dev = on_device.forward(On(b, logits, Shape({kRows, kClasses})), On(b, targets, Shape({kRows})));
    EXPECT_NEAR(l_dev, l_cpu, 1e-5f * std::fabs(l_cpu));
    const std::vector<float> g_cpu = on_cpu.backward().to_host_vector(), g_dev = Host(on_device.backward());
    ASSERT_EQ(g_cpu.size(), g_dev.size());
    for (size_t i = 0; i < g_cpu.size(); ++i) EXPECT_NEAR(g_dev[i], g_cpu[i], 1e-6f) << i;

    // Bad targets throw the host-validated messages, and leave the last forward() in place.
    std::vector<float> fractional = targets, outside = targets;
    fractional[9] = 2.5f;
    outside[12] = 11.0f;
    try {
        (void)on_device.forward(On(b, logits, Shape({kRows, kClasses})), On(b, fractional, Shape({kRows})));
        ADD_FAILURE() << "no throw";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("whole numbers"), std::string::npos) << e.what();
    }
    try {
        (void)on_device.forward(On(b, logits, Shape({kRows, kClasses})), On(b, outside, Shape({kRows})));
        ADD_FAILURE() << "no throw";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("target 11 is outside [0, 11)"), std::string::npos) << e.what();
    }
    EXPECT_EQ(Host(on_device.backward()), g_dev);

    // Every target ignored: loss 0, gradient 0.
    const std::vector<float> ignored(kRows, -100.0f);
    EXPECT_EQ(on_device.forward(On(b, logits, Shape({kRows, kClasses})), On(b, ignored, Shape({kRows}))), 0.0f);
    for (float x : Host(on_device.backward())) EXPECT_EQ(x, 0.0f);
}

// SequentialModule runs Linear -> ReLU pairs as one fused kernel. The research notes' risk is a
// fused training path drifting from the unfused one LRP relies on, so: the same layers called
// one by one must give identical outputs, gradients and relevance, bit for bit.
inline void FusedLinearReluMatchesLayerByLayer(DeviceBackend* b) {
    const DeviceType dev = b->device();
    struct Net {
        LinearModule l1, l2, l3;
        ReluModule r1, r2;
        Net(DeviceBackend* b, DeviceType dev)
            : l1(6, 9, b, dev), l2(9, 7, b, dev), l3(7, 4, b, dev), r1(b, dev), r2(b, dev) {}
        std::vector<Module*> layers() { return {&l1, &r1, &l2, &r2, &l3}; }
    };
    Net fused(b, dev), plain(b, dev);
    for (Net* n : {&fused, &plain}) {
        unsigned seed = 50;
        for (LinearModule* l : {&n->l1, &n->l2, &n->l3}) {
            l->set_weight(Random(static_cast<size_t>(l->weight().numel()), seed++));
            l->set_bias(Random(static_cast<size_t>(l->bias().numel()), seed++));
        }
    }
    SequentialModule seq(fused.layers());
    const Tensor x = On(b, Random(5 * 6, 9), Shape({5, 6}));
    const Tensor g = On(b, Random(5 * 4, 10), Shape({5, 4}));

    Tensor h = x;
    for (Module* m : plain.layers()) h = m->forward(h);
    EXPECT_EQ(Host(seq.forward(x)), Host(h));

    Tensor gp = g;
    std::vector<Module*> rev = plain.layers();
    for (auto it = rev.rbegin(); it != rev.rend(); ++it) gp = (*it)->backward(gp);
    EXPECT_EQ(Host(seq.backward(g)), Host(gp));
    EXPECT_EQ(Host(fused.l1.weight_grad()), Host(plain.l1.weight_grad()));
    EXPECT_EQ(Host(fused.l2.bias_grad()), Host(plain.l2.bias_grad()));

    LRPRuleConfig zennit;
    zennit.epsilon_bias_in_denominator = true;
    for (const LRPRuleConfig& rule : {LRPRuleConfig{}, zennit}) {
        Tensor rp = g;
        for (auto it = rev.rbegin(); it != rev.rend(); ++it) rp = (*it)->propagate_relevance(rp, rule);
        EXPECT_EQ(Host(seq.propagate_relevance(g, rule)), Host(rp));
    }
}

}  // namespace pulsatrix::fused_cases
