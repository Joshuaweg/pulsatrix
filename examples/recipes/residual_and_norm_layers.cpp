/** @file residual_and_norm_layers.cpp
 *  @brief Recipe: ResidualModule's skip connection and BatchNormModule's per-channel
 *         normalization, each demonstrated forward-only on a small hand-traceable input.
 *         Paired with docs/recipes/deep-learning/residual_and_norm_layers.md.
 */
#include <cstdio>

#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/residual_module.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    std::printf("Residual connections and normalization layers recipe\n\n");

    // --- ResidualModule: y = x + inner->forward(x) ---
    std::printf("=== ResidualModule ===\n");
    LinearModule inner(2, 2, &backend);
    inner.set_weight({1.0f, 0.0f, 0.0f, 1.0f});  // identity
    inner.set_bias({0.5f, -0.5f});

    ResidualModule residual(&inner, &backend);
    Tensor x(Shape({1, 2}), &backend, {1.0f, 2.0f});
    Tensor y = residual.forward(x);

    std::printf("inner = Linear(identity weight, bias=[0.5, -0.5])\n");
    std::printf("x = [%.1f, %.1f]\n", x.data()[0], x.data()[1]);
    std::printf("inner->forward(x) = [%.1f, %.1f]  (identity + bias)\n", x.data()[0] + 0.5f, x.data()[1] - 0.5f);
    std::printf("y = x + inner->forward(x) = [%.1f, %.1f]\n\n", y.data()[0], y.data()[1]);

    // --- BatchNormModule: per-channel standardization over (N, H, W) jointly ---
    std::printf("=== BatchNormModule ===\n");
    constexpr int64_t kChannels = 2;
    BatchNormModule bn(kChannels, &backend);
    bn.set_gamma({1.0f, 1.0f});  // identity scale -- constructor zero-initializes gamma/beta
    bn.set_beta({0.0f, 0.0f});

    // 4 batch rows, 2 channels, 1x1 spatial -- channel 0 centered at 10, channel 1 at -5.
    Tensor batch(Shape({4, kChannels, 1, 1}), &backend,
                 {10.0f, -5.0f, 12.0f, -3.0f, 8.0f, -7.0f, 14.0f, -1.0f});
    Tensor normalized = bn.forward(batch);

    for (int64_t c = 0; c < kChannels; ++c) {
        float mean = 0.0f;
        for (int64_t n = 0; n < 4; ++n) {
            mean += normalized.at({n, c, 0, 0});
        }
        mean /= 4.0f;
        std::printf("channel %lld: input mean %.2f -> normalized mean %.4f (expect ~0.0)\n",
                    static_cast<long long>(c), (c == 0 ? 11.0f : -4.0f), mean);
    }

    std::printf(
        "\nBatchNorm computes mu_c/std_c over every (batch, spatial) element for each channel\n"
        "jointly, so with gamma=1/beta=0 each channel's normalized output is mean-zero\n"
        "regardless of that channel's original scale.\n");

    return 0;
}
