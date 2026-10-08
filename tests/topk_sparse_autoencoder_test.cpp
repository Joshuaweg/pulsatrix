// FEAT-2: the TopK sparse autoencoder against a PyTorch rendering of Gao et al.'s
// (tools/golden/make_topk_sae_golden.py), and on synthetic sparse data: it recovers a known
// dictionary, doesn't shrink codes the way an L1 penalty does, and AuxK revives dead latents.
#include "pulsatrix/topk_sparse_autoencoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Floats(const SafetensorsFile& f, const std::string& name) {
    CPUBackend cpu;
    return f.tensor(name, &cpu).to_host_vector();
}

double MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d = std::max(d, std::abs(static_cast<double>(a[i]) - b[i]));
    return d;
}

/** @brief A TopK SAE set to the golden's weights, with its dead latents marked dead. */
TopKSparseAutoencoder GoldenSae(const SafetensorsFile& g, CPUBackend* cpu) {
    TopKSaeOptions o;
    o.k = 3;
    o.k_aux = 4;
    o.dead_after = 100;
    TopKSparseAutoencoder sae(6, 20, cpu, o);
    sae.encoder().set_weight(Floats(g, "w_enc"));
    sae.encoder().set_bias(Floats(g, "b_enc"));
    sae.decoder().set_weight(Floats(g, "w_dec"));
    sae.decoder().set_bias(Floats(g, "b_dec"));
    return sae;
}

void MarkDead(TopKSparseAutoencoder& sae, const SafetensorsFile& g) {
    auto [ptr, size] = g.bytes("dead");
    std::vector<int64_t> dead(size / sizeof(int64_t));
    std::memcpy(dead.data(), ptr, size);
    std::vector<int64_t> since(20, 0);
    for (int64_t d : dead) since[static_cast<size_t>(d)] = 100;
    sae.set_inputs_since_fired(since);
}

TEST(TopKSparseAutoencoder, LossAndGradientsMatchPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = SafetensorsFile::Map(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/sae/topk_sae_golden.safetensors");
    TopKSparseAutoencoder sae = GoldenSae(g, &cpu);
    MarkDead(sae, g);
    EXPECT_EQ(sae.dead_latents(), (std::vector<int64_t>{1, 4, 7, 11, 13, 17}));
    const Tensor x(Shape({9, 6}), &cpu, Floats(g, "x"));
    const FeaturizerLoss loss = sae.loss_and_backward(x);
    EXPECT_NEAR(loss.reconstruction, Floats(g, "reconstruction")[0], 1e-5);
    EXPECT_NEAR(loss.sparsity, Floats(g, "aux")[0], 1e-6);
    EXPECT_NEAR(loss.total, Floats(g, "loss")[0], 1e-5);
    const std::vector<NamedParamRef> p = sae.named_parameters();
    ASSERT_EQ(p.size(), 4u);
    const char* names[] = {"grad_w_enc", "grad_b_enc", "grad_w_dec", "grad_b_dec"};
    for (size_t i = 0; i < 4; ++i) EXPECT_LT(MaxDiff(p[i].ref.grad->to_host_vector(), Floats(g, names[i])), 1e-5) << p[i].name;

    // Two Adam steps, each followed by unit-norm decoder rows, the dead set held as the reference holds it.
    TopKSparseAutoencoder trained = GoldenSae(g, &cpu);
    AdamOptimizer opt(1e-2f, &cpu);
    for (int step = 0; step < 2; ++step) {
        MarkDead(trained, g);
        opt.zero_grad(trained);
        (void)trained.loss_and_backward(x);
        opt.step(trained);
        trained.normalize_decoder();
    }
    const char* after[] = {"after_w_enc", "after_b_enc", "after_w_dec", "after_b_dec"};
    const std::vector<NamedParamRef> q = trained.named_parameters();
    for (size_t i = 0; i < 4; ++i) EXPECT_LT(MaxDiff(q[i].ref.value->to_host_vector(), Floats(g, after[i])), 1e-5) << q[i].name;
}

TEST(TopKSparseAutoencoder, KeepsExactlyKLatentsAndBackpropagatesThroughThem) {
    CPUBackend cpu;
    TopKSaeOptions o;
    o.k = 4;
    TopKSparseAutoencoder sae(5, 16, &cpu, o);
    std::vector<float> xv(3 * 5);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = std::sin(1.3f * static_cast<float>(i)) + 0.5f;
    const Tensor x(Shape({3, 5}), &cpu, xv);
    const std::vector<float> codes = sae.encode(x).to_host_vector();
    for (int r = 0; r < 3; ++r) {
        int active = 0;
        for (int i = 0; i < 16; ++i) active += codes[static_cast<size_t>(r * 16 + i)] > 0 ? 1 : 0;
        EXPECT_LE(active, 4);
        EXPECT_GE(active, 1);
    }
    EXPECT_LE(MeanL0(sae, x), 4.0);
    // The module's backward against finite differences of 0.5 |x̂|², away from selection changes.
    auto f = [&](const std::vector<float>& in) {
        double s = 0;
        for (float v : sae.forward(Tensor(Shape({3, 5}), &cpu, in)).to_host_vector()) s += 0.5 * v * v;
        return s;
    };
    const std::vector<float> y = sae.forward(x).to_host_vector();
    const std::vector<float> gx = sae.backward(Tensor(Shape({3, 5}), &cpu, y)).to_host_vector();
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> up = xv, down = xv;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(gx[i], (f(up) - f(down)) / 2e-3, 3e-3) << i;
    }
    EXPECT_THROW(TopKSparseAutoencoder(5, 16, &cpu, TopKSaeOptions{17}), std::invalid_argument);
    EXPECT_THROW((void)sae.encode(Tensor(Shape({2, 4}), &cpu, std::vector<float>(8, 0.0f))), std::invalid_argument);
}

/** @brief Inputs that are sums of exactly 3 of 12 known unit directions, with positive weights. */
struct SparseData {
    std::vector<float> x;
    std::vector<float> directions;  ///< (12, dim)
    std::vector<float> weights;     ///< (N, 12), mostly zero
};

SparseData MakeSparseData(int64_t n, int64_t dim, uint64_t seed) {
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
    d.weights.assign(static_cast<size_t>(n * 12), 0.0f);
    for (int64_t r = 0; r < n; ++r) {
        for (int t = 0; t < 3; ++t) {
            const auto i = static_cast<int64_t>(u() * 12) % 12;
            const float w = static_cast<float>(1.0 + 2.0 * u());
            d.weights[static_cast<size_t>(r * 12 + i)] += w;
            for (int64_t j = 0; j < dim; ++j) d.x[static_cast<size_t>(r * dim + j)] += w * d.directions[static_cast<size_t>(i * dim + j)];
        }
    }
    return d;
}

TEST(TopKSparseAutoencoder, RecoversASparseDictionaryWithoutShrinkingCodes) {
    CPUBackend cpu;
    const int64_t dim = 16, n = 2048;
    const SparseData data = MakeSparseData(n, dim, 9);
    const Tensor x(Shape({n, dim}), &cpu, data.x);
    TopKSaeOptions o;
    o.k = 3;
    o.dead_after = 4096;
    TopKSparseAutoencoder sae(dim, 24, &cpu, o);
    sae.initialize_bias(x);
    AdamOptimizer opt(5e-3f, &cpu);
    for (int step = 0; step < 600; ++step) (void)TrainFeaturizer(sae, x, opt);
    // Every true direction has a learned decoder direction close to it.
    int recovered = 0;
    for (int64_t t = 0; t < 12; ++t) {
        double best = 0;
        for (int64_t f = 0; f < sae.num_features(); ++f) {
            const std::vector<float> d = sae.decoder_direction(f);
            double dot = 0;
            for (int64_t j = 0; j < dim; ++j) dot += static_cast<double>(d[static_cast<size_t>(j)]) * data.directions[static_cast<size_t>(t * dim + j)];
            best = std::max(best, dot);
        }
        recovered += best > 0.9 ? 1 : 0;
    }
    EXPECT_GE(recovered, 11) << "of 12 true directions";

    // Shrinkage: an L1 SAE at a comparable L0 reconstructs with smaller codes than the truth's;
    // TopK's codes keep the true size. Measured as |x̂| / |x| over the batch.
    SparseAutoencoder l1(dim, 24, 0.05f, &cpu, 3u);
    AdamOptimizer opt_l1(5e-3f, &cpu);
    for (int step = 0; step < 600; ++step) (void)TrainFeaturizer(l1, x, opt_l1);
    auto norm_ratio = [&](Featurizer& f) {
        const std::vector<float> xh = f.decode(f.encode(x)).to_host_vector();
        double a = 0, b = 0;
        for (size_t i = 0; i < xh.size(); ++i) {
            a += static_cast<double>(xh[i]) * xh[i];
            b += static_cast<double>(data.x[i]) * data.x[i];
        }
        return std::sqrt(a / b);
    };
    const double topk_ratio = norm_ratio(sae), l1_ratio = norm_ratio(l1);
    std::printf("[TopKSparseAutoencoder] |x̂|/|x|: TopK %.3f, L1 %.3f; L0: TopK %.2f, L1 %.2f\n", topk_ratio, l1_ratio, MeanL0(sae, x),
                MeanL0(l1, x));
    EXPECT_NEAR(topk_ratio, 1.0, 0.05);
    EXPECT_LT(l1_ratio, topk_ratio - 0.05);
}

TEST(TopKSparseAutoencoder, TheAuxiliaryLossRevivesDeadLatents) {
    CPUBackend cpu;
    const int64_t dim = 16, n = 1024;
    const SparseData data = MakeSparseData(n, dim, 4);
    const Tensor x(Shape({n, dim}), &cpu, data.x);
    auto dead_after_training = [&](float aux) {
        TopKSaeOptions o;
        o.k = 2;
        o.aux_coefficient = aux;
        o.dead_after = 2 * n;  // dead after two steps without firing
        TopKSparseAutoencoder sae(dim, 64, &cpu, o);
        // Start most latents dead the way they die in practice: they still respond to the data,
        // but too weakly to win a top-k slot against the 8 strong ones.
        std::vector<float> enc = sae.encoder().weight().to_host_vector();
        for (size_t i = 0; i < enc.size(); ++i) {
            if (i % 64 >= 8) enc[i] *= 0.02f;
        }
        sae.encoder().set_weight(enc);
        sae.initialize_bias(x);
        AdamOptimizer opt(1e-2f, &cpu);
        for (int step = 0; step < 300; ++step) (void)TrainFeaturizer(sae, x, opt);
        return sae.dead_latents().size();
    };
    const size_t with_aux = dead_after_training(1.0f / 32), without = dead_after_training(0.0f);
    std::printf("[TopKSparseAutoencoder] dead latents of 64 after training: %zu with AuxK, %zu without\n", with_aux, without);
    // The data has 12 directions for 64 latents, so many stay dead either way (measured: 30 with,
    // 48 without); AuxK must still bring back a clear share.
    EXPECT_LE(with_aux + 12, without);
}

}  // namespace
}  // namespace pulsatrix
