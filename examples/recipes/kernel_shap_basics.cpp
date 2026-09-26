/** @file kernel_shap_basics.cpp
 *  @brief Recipe: KernelSHAP against a linear network, where the true Shapley values have a
 *         known closed form (phi_i = w_i * (x_i - baseline_i)), so the explainer's output can
 *         be checked exactly. Paired with docs/recipes/interpretability/kernel_shap_basics.md.
 */
#include <cstdio>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/kernel_shap.hpp"
#include "pulsatrix/linear_module.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    LinearModule linear(3, 1, &backend);
    const std::vector<float> weight = {2.0f, -3.0f, 5.0f};
    linear.set_weight(weight);
    linear.set_bias({100.0f});  // deliberately large/irrelevant -- cancels in every f(S)-f(baseline)

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor baseline(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});

    KernelSHAP shap;
    Attribution attr = shap.explain(predict, input, baseline, /*target_index=*/0, &backend);

    std::printf("KernelSHAP recipe -- linear network f(x) = 2*x0 - 3*x1 + 5*x2 + 100\n\n");
    std::printf("%-8s %10s %10s\n", "feature", "phi (SHAP)", "w*(x-b)");
    for (int64_t i = 0; i < 3; ++i) {
        const float closed_form = weight[static_cast<size_t>(i)] * (input.data()[i] - baseline.data()[i]);
        std::printf("x%-7lld %10.4f %10.4f\n", static_cast<long long>(i), attr.values.data()[i], closed_form);
    }

    std::printf(
        "\nFor a linear model, Shapley's efficiency axiom collapses to the exact per-feature\n"
        "contribution w_i*(x_i - baseline_i) -- KernelSHAP's coalition-weighted regression\n"
        "recovers it exactly (bias cancels out, since it appears in every coalition's f(S)).\n");

    return 0;
}
