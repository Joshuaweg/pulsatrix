#include <gtest/gtest.h>

#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/grad_cam.hpp"
#include "pulsatrix/integrated_gradients.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/saliency.hpp"

// Phase 4 Mission 2: charter's "Explanation fidelity" audit category checked against an
// independent Python implementation (Captum), not just this project's own closed-form
// derivations. Reference values generated offline via
// tools/generate_captum_reference_values.py (`py -3.11 tools/generate_captum_reference_values.py`)
// and hand-transcribed below -- zero live Python dependency at test/build time.
//
// Scoped to Saliency/IntegratedGradients/GradCAM only. Captum's Lime/KernelShap both
// transitively require sklearn, which fails to import on this machine (missing
// vcomp140.dll); not worth a second shared-environment repair for methods this project
// already benchmarks against a closed-form Shapley/exact-recovery oracle in Phase 3
// (kernel_shap_test.cpp, lime_test.cpp). PDP has no Captum counterpart at all (global,
// not per-instance) and keeps its own Phase 3 closed-form oracle. See
// mission_benchmark_suite.md's Recon for the full story.
namespace pulsatrix {
namespace {

// Model A: single Linear(3,1), weight {2,-3,5}, bias {100} -- same network as
// KernelSHAPIsDeterministic/PDPIsDeterministic in stability_test.cpp.
TEST(CaptumBenchmarkTest, SaliencyMatchesCaptumOnLinearModel) {
    CPUBackend backend;
    LinearModule linear(3, 1, &backend);
    linear.set_weight({2.0f, -3.0f, 5.0f});
    linear.set_bias({100.0f});
    ExplainerContext ctx({&linear});
    Tensor input(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});

    Saliency saliency;
    Attribution attr = saliency.explain(ctx, input, /*target_index=*/0, &backend);

    // Captum reference (abs=False -- raw signed gradient, matching this library's
    // Saliency semantics; Captum's abs=True default takes |gradient|): [[2.0, -3.0, 5.0]]
    EXPECT_NEAR(attr.values.data()[0], 2.0f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[1], -3.0f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[2], 5.0f, 1e-3f);
}

// Model B: Linear(3,4)->ReLU->Linear(4,2) -- same network as
// IntegratedGradientsIsDeterministic in stability_test.cpp.
TEST(CaptumBenchmarkTest, SaliencyMatchesCaptumOnNonlinearModel) {
    CPUBackend backend;
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});
    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});

    Saliency saliency;
    Attribution attr = saliency.explain(ctx, input, /*target_index=*/0, &backend);

    // Captum reference (abs=False): [[0.05000000074505806, -0.11000000685453415, 0.2200000137090683]]
    EXPECT_NEAR(attr.values.data()[0], 0.05f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[1], -0.11f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[2], 0.22f, 1e-3f);
}

TEST(CaptumBenchmarkTest, IntegratedGradientsMatchesCaptumOnNonlinearModel) {
    CPUBackend backend;
    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});
    ExplainerContext ctx({&linear1, &relu, &linear2});
    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    Tensor baseline(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});

    IntegratedGradients ig;
    Attribution attr = ig.explain(ctx, input, baseline, /*target_index=*/0, /*steps=*/50, &backend);

    // Captum reference: IntegratedGradients: [[0.04609340445676667, 0.0626379946747575, 0.2133758545451633]]
    EXPECT_NEAR(attr.values.data()[0], 0.04609f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[1], 0.06264f, 1e-3f);
    EXPECT_NEAR(attr.values.data()[2], 0.21338f, 1e-3f);
}

// Model C: Conv2D(1,2,2,2)->ReLU->Flatten->Linear(8,2) -- same network as
// GradCAMIsDeterministic in stability_test.cpp.
TEST(CaptumBenchmarkTest, GradCAMMatchesCaptumLayerGradCam) {
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
    Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 5.0f, 6.0f, 7.0f, 8.0f});

    GradCAM gradcam;
    Attribution attr = gradcam.explain(ctx, input, /*target_index=*/0, &backend);

    // Captum reference: LayerGradCam (pre-upsample): [[[[13.0, 13.0], [23.0, 32.0]]]]
    // (row-major over the 2x2 spatial map: (h0,w0), (h0,w1), (h1,w0), (h1,w1))
    EXPECT_NEAR(attr.values.at({0, 0, 0}), 13.0f, 1e-3f);
    EXPECT_NEAR(attr.values.at({0, 0, 1}), 13.0f, 1e-3f);
    EXPECT_NEAR(attr.values.at({0, 1, 0}), 23.0f, 1e-3f);
    EXPECT_NEAR(attr.values.at({0, 1, 1}), 32.0f, 1e-3f);
}

}  // namespace
}  // namespace pulsatrix
