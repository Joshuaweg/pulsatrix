#include <gtest/gtest.h>

#include <vector>

#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/flatten_module.hpp"
#include "exai/grad_cam.hpp"
#include "exai/integrated_gradients.hpp"
#include "exai/kernel_shap.hpp"
#include "exai/lime.hpp"
#include "exai/linear_module.hpp"
#include "exai/pdp.hpp"
#include "exai/relu_module.hpp"
#include "exai/saliency.hpp"

// Phase 4 Mission 1: charter's "repeated-run variance measured and documented" audit
// category, run for every explainer this project has shipped, not just LIME. Five of six
// are deterministic by construction (no RNG anywhere in their implementation, confirmed by
// inspection) -- variance must be exactly 0.0f. LIME is the one genuinely stochastic
// method; its stability test must vary the seed across repeated runs or it trivially shows
// zero variance and fails to demonstrate the real, expected instability the charter wants
// measured (see mission_stability_suite.md's Recon).
namespace exai {
namespace {

float ComputeVariance(const std::vector<float>& values) {
    float mean = 0.0f;
    for (float v : values) {
        mean += v;
    }
    mean /= static_cast<float>(values.size());

    float variance = 0.0f;
    for (float v : values) {
        variance += (v - mean) * (v - mean);
    }
    return variance / static_cast<float>(values.size());
}

// For a genuinely deterministic explainer, every repeated-run result should be
// bit-for-bit identical to the first -- checked directly (exact equality against
// results[0]) rather than via ComputeVariance's mean/sum, which can accumulate tiny
// (~1e-15) floating-point summation-order noise even when every underlying value is
// exactly identical (e.g. summing N copies of a value and dividing by N doesn't always
// exactly round-trip). ComputeVariance is reserved for the one case that needs a "clearly
// nonzero" signal (LIME across independent seeds), not a "should be exactly zero" one.
bool AllValuesExactlyEqual(const std::vector<float>& values) {
    for (float v : values) {
        if (v != values[0]) {
            return false;
        }
    }
    return true;
}

TEST(StabilityTest, SaliencyIsDeterministic) {
    CPUBackend backend;
    LinearModule linear(3, 2, &backend);
    linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    linear.set_bias({0.1f, -0.1f});
    ExplainerContext ctx({&linear});
    Tensor input(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});

    Saliency saliency;
    std::vector<float> results;
    for (int i = 0; i < 10; ++i) {
        Attribution attr = saliency.explain(ctx, input, /*target_index=*/0, &backend);
        results.push_back(attr.values.data()[0]);
    }
    EXPECT_TRUE(AllValuesExactlyEqual(results));
}

TEST(StabilityTest, IntegratedGradientsIsDeterministic) {
    CPUBackend backend;
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});
    ExplainerContext ctx({&linear1, &relu, &linear2});

    Tensor input(Shape({3}), &backend, {0.5f, -0.3f, 1.2f});
    Tensor baseline(Shape({3}), &backend, {0.0f, 0.0f, 0.0f});

    IntegratedGradients ig;
    std::vector<float> results;
    for (int i = 0; i < 10; ++i) {
        Attribution attr = ig.explain(ctx, input, baseline, /*target_index=*/0, /*steps=*/50, &backend);
        results.push_back(attr.values.data()[0]);
    }
    EXPECT_TRUE(AllValuesExactlyEqual(results));
}

TEST(StabilityTest, GradCAMIsDeterministic) {
    CPUBackend backend;
    Conv2DModule conv(1, 2, 2, 2, &backend);
    conv.set_kernel({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f});
    conv.set_bias({0.0f, 0.0f});
    ReluModule relu(&backend);
    FlattenModule flatten(&backend);
    LinearModule linear(8, 2, &backend);
    linear.set_weight({1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f});
    linear.set_bias({0.0f, 0.0f});
    ExplainerContext ctx({&conv, &relu, &flatten, &linear});
    Tensor input(Shape({1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    GradCAM gradcam;
    std::vector<float> results;
    for (int i = 0; i < 10; ++i) {
        Attribution attr = gradcam.explain(ctx, input, /*target_index=*/0, &backend);
        results.push_back(attr.values.data()[0]);
    }
    EXPECT_TRUE(AllValuesExactlyEqual(results));
}

TEST(StabilityTest, KernelSHAPIsDeterministic) {
    CPUBackend backend;
    LinearModule linear(3, 1, &backend);
    linear.set_weight({2.0f, -3.0f, 5.0f});
    linear.set_bias({100.0f});
    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor baseline(Shape({3}), &backend, {0.0f, 0.0f, 0.0f});

    KernelSHAP shap;
    std::vector<float> results;
    for (int i = 0; i < 10; ++i) {
        Attribution attr = shap.explain(predict, input, baseline, /*target_index=*/0, &backend);
        results.push_back(attr.values.data()[0]);
    }
    EXPECT_TRUE(AllValuesExactlyEqual(results));
}

TEST(StabilityTest, PDPIsDeterministic) {
    CPUBackend backend;
    LinearModule linear(3, 1, &backend);
    linear.set_weight({2.0f, -3.0f, 5.0f});
    linear.set_bias({100.0f});
    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    std::vector<Tensor> background;
    background.emplace_back(Shape({3}), &backend, std::initializer_list<float>{1.0f, 1.0f, 1.0f});
    background.emplace_back(Shape({3}), &backend, std::initializer_list<float>{5.0f, -2.0f, 0.0f});

    PDP pdp;
    std::vector<float> results;
    for (int i = 0; i < 10; ++i) {
        Attribution attr = pdp.explain(predict, background, /*feature_index=*/1, /*target_index=*/0, -2.0f, 4.0f, 7,
                                        &backend);
        results.push_back(attr.values.data()[0]);
    }
    EXPECT_TRUE(AllValuesExactlyEqual(results));
}

// LIME is the one genuinely stochastic explainer. A network with real nonlinearity
// (ReLU) is deliberately used here, not a linear-only network like every other test in
// this project -- LIME's weighted least squares fit on an exactly-linear function is
// forced to the same exact answer regardless of which points get sampled (Phase 3 Mission
// 0's own correctness proof), so a linear network would show zero variance even with
// varying seeds and fail to exercise what this test exists to check. The ReLU kink means
// different sampled neighborhoods genuinely produce different local linear
// approximations.
TEST(StabilityTest, LIMEVariesAcrossIndependentSeeds) {
    CPUBackend backend;
    LinearModule linear1(2, 4, &backend);
    linear1.set_weight({0.6f, -0.3f, 0.4f, -0.7f, 0.2f, 0.5f, -0.6f, 0.1f});
    linear1.set_bias({0.0f, 0.0f, 0.0f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 1, &backend);
    linear2.set_weight({0.5f, -0.4f, 0.3f, 0.6f});
    linear2.set_bias({0.0f});
    ExplainerContext ctx({&linear1, &relu, &linear2});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({2}), &backend, {0.5f, 0.5f});

    LIME lime;
    std::vector<float> results;
    for (unsigned seed = 0; seed < 10; ++seed) {
        Attribution attr =
            lime.explain(predict, input, /*target_index=*/0, /*num_samples=*/50, /*sigma=*/0.5f, 0.0f, seed, &backend);
        results.push_back(attr.values.data()[0]);
    }
    EXPECT_GT(ComputeVariance(results), 0.0f)
        << "LIME showed zero variance across independently-seeded runs -- either the "
           "network isn't exercising real nonlinearity, or LIME has silently become "
           "deterministic regardless of seed";
}

TEST(StabilityTest, LIMEIsDeterministicForAFixedSeed) {
    CPUBackend backend;
    LinearModule linear1(2, 4, &backend);
    linear1.set_weight({0.6f, -0.3f, 0.4f, -0.7f, 0.2f, 0.5f, -0.6f, 0.1f});
    linear1.set_bias({0.0f, 0.0f, 0.0f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 1, &backend);
    linear2.set_weight({0.5f, -0.4f, 0.3f, 0.6f});
    linear2.set_bias({0.0f});
    ExplainerContext ctx({&linear1, &relu, &linear2});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({2}), &backend, {0.5f, 0.5f});

    LIME lime;
    std::vector<float> results;
    for (int i = 0; i < 10; ++i) {
        Attribution attr =
            lime.explain(predict, input, /*target_index=*/0, /*num_samples=*/50, /*sigma=*/0.5f, 0.0f, /*seed=*/42,
                         &backend);
        results.push_back(attr.values.data()[0]);
    }
    EXPECT_TRUE(AllValuesExactlyEqual(results));
}

}  // namespace
}  // namespace exai
