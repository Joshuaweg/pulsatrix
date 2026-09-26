/** @file lime_basics.cpp
 *  @brief Recipe: LIME's local linear surrogate against an exactly-linear network, where
 *         weighted least squares on noise-free linear data recovers the true weights.
 *         Paired with docs/recipes/interpretability/lime_basics.md.
 */
#include <cstdio>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/lime.hpp"
#include "pulsatrix/linear_module.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    LinearModule linear(3, 2, &backend);
    const std::vector<float> weight = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    linear.set_weight(weight);
    linear.set_bias({100.0f, 100.0f});  // deliberately large/irrelevant

    ExplainerContext ctx({&linear});
    auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

    Tensor input(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});

    LIME lime;
    Attribution attr = lime.explain(predict, input, /*target_index=*/1, /*num_samples=*/300, /*sigma=*/1.0f,
                                     /*l2_lambda=*/0.0f, /*seed=*/42, &backend);

    std::printf("LIME recipe -- output 1 of a linear network, true weight column [2, 4, 6]\n\n");
    std::printf("%-8s %12s %12s\n", "feature", "LIME coeff", "true weight");
    const float true_weights[3] = {2.0f, 4.0f, 6.0f};
    for (int64_t i = 0; i < 3; ++i) {
        std::printf("x%-7lld %12.4f %12.1f\n", static_cast<long long>(i), attr.values.data()[i],
                    true_weights[static_cast<size_t>(i)]);
    }

    std::printf(
        "\nLIME perturbs the input with Gaussian noise (sigma=1.0), weights each sample by an\n"
        "exponential locality kernel of the same width, and fits a local linear surrogate\n"
        "g(z) = f(x) + w^T(z - x). Against a network that is already exactly linear, the\n"
        "surrogate recovers the true weight column almost exactly, regardless of the (large,\n"
        "irrelevant) bias -- the surrogate only ever sees output differences.\n");

    return 0;
}
