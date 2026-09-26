/** @file sparse_autoencoder_probe.cpp
 *  @brief Recipe: trains a SparseAutoencoder to reconstruct synthetic activations, then trains
 *         a LinearProbe on the same activations to test whether a concept baked into their
 *         construction is linearly decodable. Paired with
 *         docs/recipes/mechanistic-interpretability/sparse_autoencoder_probe.md.
 */
#include <cstdio>
#include <random>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_probe.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    constexpr int64_t kDim = 4;
    constexpr int64_t kHiddenDim = 12;  // overcomplete: hidden_dim > dim
    constexpr int64_t kBatchSize = 32;

    // Synthetic "activations": each row's sign on dimension 0 is the concept a probe should
    // be able to recover -- a positive control, by construction linearly separable.
    std::mt19937 rng(7);
    std::normal_distribution<float> noise(0.0f, 0.5f);
    std::vector<float> activation_data(static_cast<size_t>(kBatchSize * kDim));
    std::vector<float> label_data(static_cast<size_t>(kBatchSize));
    for (int64_t n = 0; n < kBatchSize; ++n) {
        const bool concept = (n % 2 == 0);
        activation_data[static_cast<size_t>(n * kDim + 0)] = (concept ? 1.0f : -1.0f) + noise(rng);
        for (int64_t d = 1; d < kDim; ++d) {
            activation_data[static_cast<size_t>(n * kDim + d)] = noise(rng);
        }
        label_data[static_cast<size_t>(n)] = concept ? 1.0f : 0.0f;
    }
    Tensor activations(Shape({kBatchSize, kDim}), &backend, activation_data);
    Tensor labels(Shape({kBatchSize, 1}), &backend, label_data);

    // --- Sparse autoencoder: reconstruct the activations through an L1-penalized hidden layer ---
    std::printf("Sparse autoencoder + linear probe recipe\n\n");
    std::printf("=== SparseAutoencoder (dim=%lld, hidden_dim=%lld, l1_lambda=0.01) ===\n",
                static_cast<long long>(kDim), static_cast<long long>(kHiddenDim));

    SparseAutoencoder sae(kDim, kHiddenDim, /*l1_lambda=*/0.01f, &backend);
    AdamOptimizer sae_optimizer(0.01f, &backend);

    for (int epoch = 0; epoch <= 300; ++epoch) {
        float loss = sae.train_step(activations, sae_optimizer);
        if (epoch % 100 == 0) {
            std::printf("epoch %3d | reconstruction loss %.6f | mean hidden activation %.4f\n", epoch, loss,
                        sae.mean_hidden_activation(activations));
        }
    }
    std::printf("final reconstruction error: %.6f\n\n", sae.reconstruction_error(activations));

    // --- Linear probe: is the concept (sign of dimension 0) linearly decodable? ---
    std::printf("=== LinearProbe (activation_dim=%lld) ===\n", static_cast<long long>(kDim));

    LinearProbe probe(kDim, &backend);
    AdamOptimizer probe_optimizer(0.1f, &backend);

    for (int epoch = 0; epoch <= 200; ++epoch) {
        probe.train_step(activations, labels, probe_optimizer);
    }
    float accuracy = probe.accuracy(activations, labels);
    std::printf("probe accuracy: %.1f%% (chance level is 50%%)\n", 100.0f * accuracy);

    std::printf(
        "\nThe probe recovers the concept because it was constructed to be linearly\n"
        "separable (dimension 0's sign). A negative control -- labels independent of every\n"
        "feature -- would instead plateau near 50%% accuracy; see tests/linear_probe_test.cpp\n"
        "for that paired control.\n");

    return 0;
}
