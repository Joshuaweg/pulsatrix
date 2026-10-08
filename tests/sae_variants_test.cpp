// FEAT-4: BatchTopK, Matryoshka and JumpReLU sparse autoencoders, against PyTorch renderings of
// their papers (tools/golden/make_sae_variants_golden.py), and on synthetic data.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/featurizer_metrics.hpp"
#include "pulsatrix/jumprelu_sparse_autoencoder.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/topk_sparse_autoencoder.hpp"

namespace pulsatrix {
namespace {

std::string Golden() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/sae/sae_variants_golden.safetensors"; }

std::vector<float> Floats(const SafetensorsFile& f, const std::string& name) {
    CPUBackend cpu;
    return f.tensor(name, &cpu).to_host_vector();
}

double MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    double d = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::abs(static_cast<double>(a[i]) - b[i]));
    return d;
}

TopKSparseAutoencoder GoldenBatchSae(const SafetensorsFile& g, CPUBackend* cpu) {
    TopKSaeOptions o;
    o.k = 3;
    o.k_aux = 4;
    o.dead_after = 100;
    o.batch_topk = true;
    o.matryoshka_prefixes = {5, 12};
    TopKSparseAutoencoder sae(6, 20, cpu, o);
    sae.encoder().set_weight(Floats(g, "batch.w_enc"));
    sae.encoder().set_bias(Floats(g, "batch.b_enc"));
    sae.decoder().set_weight(Floats(g, "batch.w_dec"));
    sae.decoder().set_bias(Floats(g, "batch.b_dec"));
    return sae;
}

void MarkDead(TopKSparseAutoencoder& sae, const SafetensorsFile& g) {
    auto [ptr, size] = g.bytes("batch.dead");
    std::vector<int64_t> dead(size / sizeof(int64_t));
    std::memcpy(dead.data(), ptr, size);
    std::vector<int64_t> since(20, 0);
    for (int64_t d : dead) since[static_cast<size_t>(d)] = 100;
    sae.set_inputs_since_fired(since);
}

TEST(SaeVariants, MatryoshkaBatchTopKMatchesPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = SafetensorsFile::Map(Golden());
    TopKSparseAutoencoder sae = GoldenBatchSae(g, &cpu);
    EXPECT_EQ(sae.options().matryoshka_prefixes, (std::vector<int64_t>{5, 12, 20}));
    MarkDead(sae, g);
    const Tensor x(Shape({9, 6}), &cpu, Floats(g, "batch.x"));
    const FeaturizerLoss loss = sae.loss_and_backward(x);
    EXPECT_NEAR(loss.total, Floats(g, "batch.loss")[0], 1e-5);
    EXPECT_NEAR(loss.reconstruction, Floats(g, "batch.reconstruction")[0], 1e-5);
    EXPECT_NEAR(loss.sparsity, Floats(g, "batch.aux")[0], 1e-6);
    const char* names[] = {"batch.grad_w_enc", "batch.grad_b_enc", "batch.grad_w_dec", "batch.grad_b_dec"};
    const std::vector<NamedParamRef> p = sae.named_parameters();
    for (size_t i = 0; i < 4; ++i) EXPECT_LT(MaxDiff(p[i].ref.grad->to_host_vector(), Floats(g, names[i])), 1e-5) << p[i].name;

    TopKSparseAutoencoder trained = GoldenBatchSae(g, &cpu);
    AdamOptimizer opt(1e-2f, &cpu);
    for (int step = 0; step < 2; ++step) {
        MarkDead(trained, g);
        opt.zero_grad(trained);
        (void)trained.loss_and_backward(x);
        opt.step(trained);
        trained.normalize_decoder();
    }
    const char* after[] = {"batch.after_w_enc", "batch.after_b_enc", "batch.after_w_dec", "batch.after_b_dec"};
    const std::vector<NamedParamRef> q = trained.named_parameters();
    for (size_t i = 0; i < 4; ++i) EXPECT_LT(MaxDiff(q[i].ref.value->to_host_vector(), Floats(g, after[i])), 1e-5) << q[i].name;
    EXPECT_NEAR(trained.threshold(), Floats(g, "batch.threshold")[0], 1e-5);
}

TEST(SaeVariants, JumpReLUMatchesPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = SafetensorsFile::Map(Golden());
    auto make = [&] {
        JumpReLUSaeOptions o;
        o.l0_coefficient = 0.05f;
        o.bandwidth = 0.5f;
        JumpReLUSparseAutoencoder sae(6, 20, &cpu, o);
        sae.encoder().set_weight(Floats(g, "jump.w_enc"));
        sae.encoder().set_bias(Floats(g, "jump.b_enc"));
        sae.decoder().set_weight(Floats(g, "jump.w_dec"));
        sae.decoder().set_bias(Floats(g, "jump.b_dec"));
        for (const NamedParamRef& p : sae.named_parameters()) {
            if (p.name == "log_threshold") *p.ref.value = Tensor(Shape({20}), &cpu, Floats(g, "jump.log_threshold"));
        }
        return sae;
    };
    JumpReLUSparseAutoencoder sae = make();
    const Tensor x(Shape({9, 6}), &cpu, Floats(g, "jump.x"));
    const FeaturizerLoss loss = sae.loss_and_backward(x);
    EXPECT_NEAR(loss.total, Floats(g, "jump.loss")[0], 1e-5);
    EXPECT_NEAR(loss.reconstruction, Floats(g, "jump.reconstruction")[0], 1e-5);
    EXPECT_NEAR(loss.sparsity, Floats(g, "jump.sparsity")[0], 1e-6);
    const std::vector<NamedParamRef> p = sae.named_parameters();
    ASSERT_EQ(p.size(), 5u);
    EXPECT_EQ(p[4].name, "log_threshold");
    const char* names[] = {"jump.grad_w_enc", "jump.grad_b_enc", "jump.grad_w_dec", "jump.grad_b_dec", "jump.grad_log_threshold"};
    for (size_t i = 0; i < 5; ++i) EXPECT_LT(MaxDiff(p[i].ref.grad->to_host_vector(), Floats(g, names[i])), 1e-5) << p[i].name;

    JumpReLUSparseAutoencoder trained = make();
    AdamOptimizer opt(1e-2f, &cpu);
    for (int step = 0; step < 2; ++step) (void)TrainFeaturizer(trained, x, opt);
    const char* after[] = {"jump.after_w_enc", "jump.after_b_enc", "jump.after_w_dec", "jump.after_b_dec", "jump.after_log_threshold"};
    const std::vector<NamedParamRef> q = trained.named_parameters();
    for (size_t i = 0; i < 5; ++i) EXPECT_LT(MaxDiff(q[i].ref.value->to_host_vector(), Floats(g, after[i])), 1e-5) << q[i].name;
}

/** @brief Inputs that are sums of 1 to 5 of 12 known unit directions, so L0 varies per input. */
struct SparseData {
    std::vector<float> x;
    std::vector<float> directions;  ///< (12, dim)
    std::vector<int> active;        ///< per input
};

SparseData MakeData(int64_t n, int64_t dim, uint64_t seed) {
    SparseData d;
    uint64_t s = seed;
    auto u = [&] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(static_cast<uint32_t>(s >> 32)) / 4294967296.0;
    };
    d.directions.resize(static_cast<size_t>(12 * dim));
    for (int64_t i = 0; i < 12; ++i) {
        double sq = 0;
        for (int64_t j = 0; j < dim; ++j) {
            const double v = u() * 2 - 1;
            d.directions[static_cast<size_t>(i * dim + j)] = static_cast<float>(v);
            sq += v * v;
        }
        for (int64_t j = 0; j < dim; ++j) d.directions[static_cast<size_t>(i * dim + j)] /= static_cast<float>(std::sqrt(sq));
    }
    d.x.assign(static_cast<size_t>(n * dim), 0.0f);
    for (int64_t r = 0; r < n; ++r) {
        const int count = 1 + static_cast<int>(u() * 5);
        d.active.push_back(count);
        for (int t = 0; t < count; ++t) {
            const auto i = static_cast<int64_t>(u() * 12) % 12;
            const auto w = static_cast<float>(1.0 + 2.0 * u());
            for (int64_t j = 0; j < dim; ++j) d.x[static_cast<size_t>(r * dim + j)] += w * d.directions[static_cast<size_t>(i * dim + j)];
        }
    }
    return d;
}

int Recovered(Featurizer& f, const SparseData& data, int64_t dim) {
    int recovered = 0;
    for (int64_t t = 0; t < 12; ++t) {
        double best = 0;
        for (int64_t i = 0; i < f.num_features(); ++i) {
            const std::vector<float> d = f.decoder_direction(i);
            double dot = 0;
            for (int64_t j = 0; j < dim; ++j) dot += static_cast<double>(d[static_cast<size_t>(j)]) * data.directions[static_cast<size_t>(t * dim + j)];
            best = std::max(best, dot);
        }
        recovered += best > 0.9 ? 1 : 0;
    }
    return recovered;
}

TEST(SaeVariants, BatchTopKSpendsLatentsWhereInputsNeedThem) {
    CPUBackend cpu;
    const int64_t dim = 16, n = 2048;
    const SparseData data = MakeData(n, dim, 3);
    const Tensor x(Shape({n, dim}), &cpu, data.x);
    TopKSaeOptions o;
    o.k = 3;
    o.batch_topk = true;
    o.dead_after = 4096;
    TopKSparseAutoencoder sae(dim, 24, &cpu, o);
    sae.initialize_bias(x);
    EXPECT_LT(sae.threshold(), 0.0f);
    AdamOptimizer opt(5e-3f, &cpu);
    std::vector<float> codes;
    for (int step = 0; step < 600; ++step) (void)TrainFeaturizer(sae, x, opt);
    // A training batch keeps exactly N * k activations (all positive here).
    (void)sae.loss_and_backward(x, &codes);
    EXPECT_NEAR(MeanL0(codes, sae.num_features()), 3.0, 1e-9);
    EXPECT_GT(sae.threshold(), 0.0f);
    EXPECT_GE(Recovered(sae, data, dim), 11);
    // At inference the threshold decides: about k on average, more where more directions are present.
    const std::vector<float> c = sae.encode(x).to_host_vector();
    double few = 0, many = 0;
    int64_t n_few = 0, n_many = 0;
    for (int64_t r = 0; r < n; ++r) {
        int64_t active = 0;
        for (int64_t i = 0; i < 24; ++i) active += c[static_cast<size_t>(r * 24 + i)] > 0 ? 1 : 0;
        if (data.active[static_cast<size_t>(r)] == 1) few += static_cast<double>(active), ++n_few;
        if (data.active[static_cast<size_t>(r)] == 5) many += static_cast<double>(active), ++n_many;
    }
    const double l0 = MeanL0(sae, x);
    std::printf("[BatchTopK] inference L0 %.2f; %.2f on one-direction inputs, %.2f on five\n", l0, few / n_few, many / n_many);
    EXPECT_NEAR(l0, 3.0, 0.5);
    EXPECT_GT(many / static_cast<double>(n_many), few / static_cast<double>(n_few) + 1.5);

    // The threshold is a buffer: checkpoints keep it.
    const std::string path = (std::filesystem::temp_directory_path() / "pulsatrix_batchtopk.safetensors").string();
    SaveCheckpoint(path, sae);
    TopKSparseAutoencoder other(dim, 24, &cpu, o);
    LoadCheckpoint(path, other);
    EXPECT_EQ(other.threshold(), sae.threshold());
    std::filesystem::remove(path);
    EXPECT_THROW(TopKSparseAutoencoder(dim, 24, &cpu, TopKSaeOptions{3, 0, 0.03f, 10, 0, true, 0.99f, {8, 4}}), std::invalid_argument);
}

TEST(SaeVariants, JumpReLUTradesL0ForReconstructionThroughLambda) {
    CPUBackend cpu;
    const int64_t dim = 16, n = 2048;
    const SparseData data = MakeData(n, dim, 5);
    const Tensor x(Shape({n, dim}), &cpu, data.x);
    auto train = [&](float lambda) {
        JumpReLUSaeOptions o;
        o.l0_coefficient = lambda;
        o.bandwidth = 0.05f;
        o.initial_threshold = 0.1f;
        auto sae = std::make_unique<JumpReLUSparseAutoencoder>(dim, 24, &cpu, o);
        sae->initialize_bias(x);
        AdamOptimizer opt(5e-3f, &cpu);
        for (int step = 0; step < 800; ++step) (void)TrainFeaturizer(*sae, x, opt);
        return sae;
    };
    auto low = train(0.005f), high = train(0.05f);
    const ReconstructionMetrics rl = EvaluateReconstruction(*low, x), rh = EvaluateReconstruction(*high, x);
    std::printf("[JumpReLU] lambda 0.005: L0 %.2f, EV %.3f, %d/12 directions; lambda 0.05: L0 %.2f, EV %.3f, %d/12\n", rl.l0,
                rl.explained_variance, Recovered(*low, data, dim), rh.l0, rh.explained_variance, Recovered(*high, data, dim));
    EXPECT_LT(rh.l0, rl.l0);
    EXPECT_GT(rl.explained_variance, rh.explained_variance);
    EXPECT_GT(rh.explained_variance, 0.8);
    EXPECT_GE(Recovered(*high, data, dim), 11);
    const std::vector<float> theta = high->thresholds();
    EXPECT_GT(*std::max_element(theta.begin(), theta.end()), 0.1f);  // thresholds learned, not fixed
}

TEST(SaeVariants, JumpReLUNormalizesDecodersWithoutChangingCodesAndIsAModule) {
    CPUBackend cpu;
    JumpReLUSaeOptions o;
    o.initial_threshold = 0.05f;
    JumpReLUSparseAutoencoder sae(5, 12, &cpu, o);
    std::vector<float> dec = sae.decoder().weight().to_host_vector();
    for (size_t i = 0; i < dec.size(); ++i) dec[i] *= 0.4f + 0.15f * static_cast<float>(i % 5);
    sae.decoder().set_weight(dec);
    std::vector<float> xv(4 * 5);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = std::sin(0.9f * static_cast<float>(i)) + 0.3f;
    const Tensor x(Shape({4, 5}), &cpu, xv);
    const std::vector<float> before = sae.decode(sae.encode(x)).to_host_vector();
    const double l0 = MeanL0(sae, x);
    sae.normalize_decoder();
    const std::vector<float> after = sae.decode(sae.encode(x)).to_host_vector();
    for (size_t i = 0; i < before.size(); ++i) EXPECT_NEAR(after[i], before[i], 1e-5);
    EXPECT_DOUBLE_EQ(MeanL0(sae, x), l0);
    for (int64_t f = 0; f < 12; ++f) {
        double sq = 0;
        for (float v : sae.decoder_direction(f)) sq += static_cast<double>(v) * v;
        EXPECT_NEAR(std::sqrt(sq), 1.0, 1e-5);
    }
    // Backward against finite differences of 0.5 |x̂|², away from threshold crossings.
    auto f = [&](const std::vector<float>& in) {
        double s = 0;
        for (float v : sae.forward(Tensor(Shape({4, 5}), &cpu, in)).to_host_vector()) s += 0.5 * v * v;
        return s;
    };
    const std::vector<float> y = sae.forward(x).to_host_vector();
    const std::vector<float> gx = sae.backward(Tensor(Shape({4, 5}), &cpu, y)).to_host_vector();
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> up = xv, down = xv;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(gx[i], (f(up) - f(down)) / 2e-3, 3e-3) << i;
    }
    EXPECT_EQ(sae.propagate_relevance(Tensor(Shape({4, 5}), &cpu, y), LRPRuleConfig{}).shape(), Shape({4, 5}));
    // Checkpoints keep the thresholds.
    const std::string path = (std::filesystem::temp_directory_path() / "pulsatrix_jumprelu.safetensors").string();
    SaveCheckpoint(path, sae);
    JumpReLUSparseAutoencoder other(5, 12, &cpu);
    LoadCheckpoint(path, other);
    EXPECT_EQ(other.thresholds(), sae.thresholds());
    std::filesystem::remove(path);
    EXPECT_THROW(JumpReLUSparseAutoencoder(5, 12, &cpu, JumpReLUSaeOptions{-1.0f}), std::invalid_argument);
    EXPECT_THROW(sae.set_l0_coefficient(-1.0f), std::invalid_argument);
}

/**
 * @brief Chanin et al.'s toy hierarchy: a parent feature (e0) present on 40% of inputs; when it
 *        is, one of ten children (e1..e10) is present half the time, so each child is rare.
 *        Twelve unrelated features fire independently. A sparse autoencoder saves a latent per
 *        input by merging the parent into each child's latent, and then its parent latent
 *        skips those inputs: absorption.
 */
struct Toy {
    std::vector<float> x;
    std::vector<int> parent;
};

constexpr int64_t kToyDim = 23;

Toy MakeToy(int64_t n, uint64_t seed) {
    Toy t;
    uint64_t s = seed;
    auto u = [&] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(static_cast<uint32_t>(s >> 32)) / 4294967296.0;
    };
    for (int64_t r = 0; r < n; ++r) {
        std::vector<float> row(kToyDim, 0.0f);
        const bool parent = u() < 0.4;
        if (parent) {
            row[0] = 1.0f;
            if (u() < 0.5) row[static_cast<size_t>(1 + static_cast<int>(u() * 10) % 10)] = 1.0f;
        }
        for (int64_t j = 11; j < kToyDim; ++j) {
            if (u() < 0.08) row[static_cast<size_t>(j)] = 1.0f;
        }
        t.x.insert(t.x.end(), row.begin(), row.end());
        t.parent.push_back(parent ? 1 : 0);
    }
    return t;
}

TEST(SaeVariants, MatryoshkaAbsorbsLessThanPlainBatchTopK) {
    CPUBackend cpu;
    const int64_t n = 2048;
    const Toy toy = MakeToy(n, 17);
    const Tensor x(Shape({n, kToyDim}), &cpu, toy.x);
    auto absorption = [&](std::vector<int64_t> prefixes) {
        TopKSaeOptions o;
        o.k = 2;  // tight enough that merging parent and child saves a latent
        o.batch_topk = true;
        o.dead_after = 2 * n;
        o.matryoshka_prefixes = std::move(prefixes);
        o.seed = 2;
        TopKSparseAutoencoder sae(kToyDim, 48, &cpu, o);
        sae.initialize_bias(x);
        AdamOptimizer opt(5e-3f, &cpu);
        for (int step = 0; step < 500; ++step) (void)TrainFeaturizer(sae, x, opt);
        return FeatureAbsorption(sae, x, toy.parent);
    };
    const AbsorptionResult plain = absorption({}), nested = absorption({4, 16});
    std::printf("[Matryoshka] parent absorbed: plain %.1f%% (main F1 %.3f), Matryoshka %.1f%% (main F1 %.3f)\n", 100 * plain.absorption_rate,
                plain.main_f1, 100 * nested.absorption_rate, nested.main_f1);
    EXPECT_GT(plain.probe_f1, 0.99);
    // Data seeds 3, 4, 5 and 17 gave plain 6.6% to 12.6%, and Matryoshka 0% every time.
    EXPECT_GT(plain.absorption_rate, 0.02);
    EXPECT_LT(nested.absorption_rate, 0.01);
    EXPECT_GE(nested.main_f1, plain.main_f1);
}

}  // namespace
}  // namespace pulsatrix
