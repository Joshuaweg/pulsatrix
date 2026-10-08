// FEAT-1: the Featurizer interface on SparseAutoencoder (unit-norm decoders, encode/decode, the
// shared training step), L0, and dead- and dense-latent tracking.
#include "pulsatrix/featurizer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Data(int64_t n, int64_t d, unsigned seed) {
    std::vector<float> v(static_cast<size_t>(n * d));
    uint64_t s = seed;
    for (float& x : v) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        x = static_cast<float>(static_cast<uint32_t>(s >> 32)) / 4294967296.0f * 2.0f - 1.0f;
    }
    return v;
}

TEST(Featurizer, NormalizingTheDecoderKeepsReconstructionsAndMakesDirectionsUnit) {
    CPUBackend cpu;
    SparseAutoencoder sae(6, 12, 0.01f, &cpu, 3u);
    // Break the unit norms the constructor set, then restore them.
    std::vector<float> dec = sae.decoder().weight().to_host_vector();
    for (size_t i = 0; i < dec.size(); ++i) dec[i] *= 0.3f + 0.1f * static_cast<float>(i % 7);
    sae.decoder().set_weight(dec);
    const Tensor x(Shape({5, 6}), &cpu, Data(5, 6, 1));
    const std::vector<float> before = sae.reconstruct(x).to_host_vector();
    sae.normalize_decoder();
    const std::vector<float> after = sae.reconstruct(x).to_host_vector();
    for (size_t i = 0; i < before.size(); ++i) EXPECT_NEAR(after[i], before[i], 1e-5) << i;
    for (int64_t f = 0; f < sae.num_features(); ++f) {
        double sq = 0;
        for (float v : sae.decoder_direction(f)) sq += static_cast<double>(v) * v;
        EXPECT_NEAR(std::sqrt(sq), 1.0, 1e-5) << f;
    }
    // decode(encode(x)) is the reconstruction.
    const std::vector<float> round = sae.decode(sae.encode(x)).to_host_vector();
    for (size_t i = 0; i < before.size(); ++i) EXPECT_NEAR(round[i], after[i], 1e-6);
    EXPECT_THROW((void)sae.decode(Tensor(Shape({2, 6}), &cpu, std::vector<float>(12, 0.0f))), std::invalid_argument);
    EXPECT_THROW((void)sae.decoder_direction(12), std::invalid_argument);
}

TEST(Featurizer, TrainingThroughTheInterfaceLowersTheLossAndKeepsUnitNorms) {
    CPUBackend cpu;
    SparseAutoencoder sae(6, 24, 0.005f, &cpu, 5u);
    Featurizer& f = sae;
    AdamOptimizer opt(0.01f, &cpu);
    FeatureActivityTracker tracker(f.num_features());
    const Tensor x(Shape({64, 6}), &cpu, Data(64, 6, 2));
    const FeaturizerLoss first = TrainFeaturizer(f, x, opt, true, &tracker);
    FeaturizerLoss last = first;
    for (int step = 0; step < 200; ++step) last = TrainFeaturizer(f, x, opt, true, &tracker);
    EXPECT_LT(last.total, first.total * 0.5f);
    EXPECT_NEAR(last.total, last.reconstruction + last.sparsity, 1e-6);
    EXPECT_GT(last.sparsity, 0.0f);
    EXPECT_EQ(tracker.inputs_seen(), 64 * 201);
    for (int64_t i = 0; i < f.num_features(); ++i) {
        double sq = 0;
        for (float v : f.decoder_direction(i)) sq += static_cast<double>(v) * v;
        EXPECT_NEAR(std::sqrt(sq), 1.0, 1e-5);
    }
    const double l0 = MeanL0(f, x);
    EXPECT_GT(l0, 0.0);
    EXPECT_LE(l0, 24.0);
}

TEST(Featurizer, TheAutoencoderIsAModuleWithNamedParametersAndCheckpoints) {
    CPUBackend cpu;
    SparseAutoencoder sae(4, 8, 0.01f, &cpu, 9u);
    const std::vector<NamedParamRef> params = sae.named_parameters();
    ASSERT_EQ(params.size(), 4u);
    EXPECT_EQ(params[0].name, "encoder.weight");
    EXPECT_EQ(params[3].name, "decoder.bias");
    const Tensor x(Shape({3, 4}), &cpu, Data(3, 4, 4));
    const std::vector<float> y = sae.forward(x).to_host_vector(), r = sae.reconstruct(x).to_host_vector();
    for (size_t i = 0; i < y.size(); ++i) EXPECT_FLOAT_EQ(y[i], r[i]);
    EXPECT_EQ(sae.propagate_relevance(Tensor(Shape({3, 4}), &cpu, y), LRPRuleConfig{}).shape(), Shape({3, 4}));

    const std::string path = (std::filesystem::temp_directory_path() / "pulsatrix_sae_checkpoint.safetensors").string();
    SaveCheckpoint(path, sae);
    SparseAutoencoder other(4, 8, 0.01f, &cpu, 10u);
    LoadCheckpoint(path, other);
    const std::vector<float> z = other.reconstruct(x).to_host_vector();
    for (size_t i = 0; i < y.size(); ++i) EXPECT_FLOAT_EQ(z[i], y[i]);
    std::filesystem::remove(path);
}

TEST(Featurizer, BackwardMatchesFiniteDifferences) {
    CPUBackend cpu;
    SparseAutoencoder sae(3, 5, 0.0f, &cpu, 2u);
    const std::vector<float> xv = Data(2, 3, 7);
    auto loss = [&](const std::vector<float>& in) {
        const std::vector<float> y = sae.forward(Tensor(Shape({2, 3}), &cpu, in)).to_host_vector();
        double s = 0;
        for (float v : y) s += 0.5 * v * v;
        return s;
    };
    const std::vector<float> y = sae.forward(Tensor(Shape({2, 3}), &cpu, xv)).to_host_vector();
    const std::vector<float> g = sae.backward(Tensor(Shape({2, 3}), &cpu, y)).to_host_vector();
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> up = xv, down = xv;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(g[i], (loss(up) - loss(down)) / 2e-3, 2e-3) << i;
    }
}

TEST(FeatureActivityTracker, FindsDeadAndDenseFeatures) {
    FeatureActivityTracker t(3);
    // Feature 0 fires on every input, feature 1 only on the first, feature 2 never.
    t.observe({1, 1, 0,  //
               1, 0, 0,  //
               2, 0, 0,  //
               5, 0, 0});
    EXPECT_EQ(t.inputs_seen(), 4);
    EXPECT_EQ(t.firing_rates(), (std::vector<double>{1.0, 0.25, 0.0}));
    EXPECT_EQ(t.dead(4), (std::vector<int64_t>{2}));     // feature 1 fired 4 inputs ago
    EXPECT_EQ(t.dead(3), (std::vector<int64_t>{1, 2}));  // but not in the last 3
    EXPECT_NEAR(t.dead_fraction(3), 2.0 / 3.0, 1e-12);
    EXPECT_EQ(t.dense(0.5), (std::vector<int64_t>{0}));
    // A threshold counts only codes above it.
    t.observe({1.5f, 0, 0}, 2.0f);
    EXPECT_EQ(t.firing_rates()[0], 0.8);
    t.reset();
    EXPECT_EQ(t.inputs_seen(), 0);
    EXPECT_EQ(t.dead(1).size(), 3u);
    EXPECT_THROW(t.observe({1, 2}), std::invalid_argument);
    EXPECT_THROW(FeatureActivityTracker(0), std::invalid_argument);
    EXPECT_THROW((void)t.dead(0), std::invalid_argument);
}

TEST(MeanL0, CountsActiveCodesPerInput) {
    EXPECT_DOUBLE_EQ(MeanL0({1, 0, 2, 0, 0, 0}, 3), 1.0);  // 2 then 0 active
    EXPECT_DOUBLE_EQ(MeanL0({1, 0.5f, 2, 0.1f}, 2, 0.2f), 1.5);
    EXPECT_THROW((void)MeanL0({1, 2, 3}, 2), std::invalid_argument);
    EXPECT_THROW((void)MeanL0({}, 2), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
