// HIP-7: Conv2DModule in batch chunks. Whatever the workspace budget, forward, backward (input,
// kernel and bias gradients) and every LRP rule give bit-identical results to the unchunked
// layer. Shared by conv2d_chunking_test.cpp (CPU) and the HIP and CUDA suites.
#pragma once

#include <gtest/gtest.h>

#include <random>
#include <string>
#include <vector>

#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix::conv_chunking_cases {

inline std::vector<float> Random(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
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

struct Run {
    std::vector<float> out, grad_in, kernel_grad, bias_grad;
    std::vector<std::vector<float>> relevance;  // one per rule
};

inline std::vector<LRPRuleConfig> Rules() {
    std::vector<LRPRuleConfig> rules(5);
    rules[1].epsilon_bias_in_denominator = true;
    rules[2].rule = LRPRule::AlphaBeta;
    rules[2].alpha = 2.0f;
    rules[2].beta = 1.0f;
    rules[3].rule = LRPRule::Gamma;
    rules[4].rule = LRPRule::ZBox;
    rules[4].low = -1.0f;
    return rules;
}

// Conv2D(3 -> 4, 3x3, stride 2, padding 1) on (5, 3, 9, 7): P = 27, Q = 5 * 4 = 20, so one
// example's patches are 27 * 20 * 4 = 2160 bytes.
inline Run Once(DeviceBackend* backend, size_t budget) {
    constexpr int64_t N = 5, C = 3, OC = 4, H = 9, W = 7;
    const DeviceType dev = backend->device();
    Conv2DModule conv(C, OC, 3, 3, backend, 2, 1);
    conv.set_kernel(Random(OC * C * 9, 1));
    conv.set_bias(Random(OC, 2));
    conv.set_max_workspace_bytes(budget);
    const Tensor x(Shape({N, C, H, W}), backend, Random(N * C * H * W, 3), dev);
    Run r;
    const Tensor y = conv.forward(x);
    r.out = Host(y);
    const Tensor g(y.shape(), backend, Random(static_cast<size_t>(y.numel()), 4), dev);
    r.grad_in = Host(conv.backward(g));
    r.kernel_grad = Host(conv.kernel_grad());
    r.bias_grad = Host(conv.bias_grad());
    for (const LRPRuleConfig& rule : Rules()) r.relevance.push_back(Host(conv.propagate_relevance(g, rule)));
    return r;
}

inline void ChunkedMatchesWholeBatch(DeviceBackend* backend) {
    const Run whole = Once(backend, Conv2DModule::kDefaultMaxWorkspaceBytes);
    // Budgets for 0 (still one example), 1, 2 and 3 examples per chunk; 5 is not a multiple of 2 or 3.
    for (size_t examples : {0u, 1u, 2u, 3u}) {
        SCOPED_TRACE("examples per chunk " + std::to_string(examples));
        const Run c = Once(backend, examples * 2160);
        EXPECT_EQ(c.out, whole.out);
        EXPECT_EQ(c.grad_in, whole.grad_in);
        EXPECT_EQ(c.kernel_grad, whole.kernel_grad);
        EXPECT_EQ(c.bias_grad, whole.bias_grad);
        ASSERT_EQ(c.relevance.size(), whole.relevance.size());
        for (size_t i = 0; i < c.relevance.size(); ++i) EXPECT_EQ(c.relevance[i], whole.relevance[i]) << "rule " << i;
    }
}

// A frozen kernel skips the patches in backward(); the input and bias gradients don't change.
inline void FrozenKernelInChunks(DeviceBackend* backend) {
    const DeviceType dev = backend->device();
    auto run = [&](size_t budget) {
        Conv2DModule conv(2, 3, 3, 3, backend, 1, 1);
        conv.set_kernel(Random(54, 5));
        conv.set_bias(Random(3, 6));
        conv.set_requires_grad(false, "weight");
        conv.set_max_workspace_bytes(budget);
        const Tensor x(Shape({3, 2, 4, 4}), backend, Random(96, 7), dev);
        const Tensor y = conv.forward(x);
        const Tensor g(y.shape(), backend, Random(static_cast<size_t>(y.numel()), 8), dev);
        return std::make_pair(Host(conv.backward(g)), Host(conv.bias_grad()));
    };
    EXPECT_EQ(run(1), run(Conv2DModule::kDefaultMaxWorkspaceBytes));
}

}  // namespace pulsatrix::conv_chunking_cases
