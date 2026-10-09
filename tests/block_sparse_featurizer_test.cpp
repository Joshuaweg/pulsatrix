// FEAT-6: block-sparse featurizers, against a PyTorch rendering of the paper and its reference
// code (tools/golden/make_bsf_golden.py), and on data that is a sparse sum of planted
// two-dimensional concepts.
#include "pulsatrix/block_sparse_featurizer.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/featurizer_metrics.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/topk_sparse_autoencoder.hpp"

namespace pulsatrix {
namespace {

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

SafetensorsFile Golden() { return SafetensorsFile::Map(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/sae/bsf_golden.safetensors"); }

void SetParam(Module& m, const std::string& name, const std::vector<float>& v) {
    for (const NamedParamRef& p : m.named_parameters()) {
        if (p.name == name) {
            *p.ref.value = Tensor(p.ref.value->shape(), p.ref.value->backend(), v);
            return;
        }
    }
    FAIL() << "no parameter " << name;
}

/** @brief Checks the loss, every gradient, and two Adam steps against the golden. */
void ExpectMatches(const std::string& prefix, const std::function<std::unique_ptr<BlockSparseFeaturizer>()>& make,
                   const std::vector<std::pair<std::string, std::string>>& params) {
    CPUBackend cpu;
    const SafetensorsFile g = Golden();
    const Tensor x(Shape({9, 6}), &cpu, Floats(g, "x"));
    auto f = make();
    const FeaturizerLoss loss = f->loss_and_backward(x);
    EXPECT_NEAR(loss.total, Floats(g, prefix + ".loss")[0], 1e-5) << prefix;
    const std::vector<NamedParamRef> p = f->named_parameters();
    ASSERT_EQ(p.size(), params.size()) << prefix;
    for (size_t i = 0; i < params.size(); ++i) {
        EXPECT_EQ(p[i].name, params[i].first);
        EXPECT_LT(MaxDiff(p[i].ref.grad->to_host_vector(), Floats(g, prefix + ".grad_" + params[i].second)), 1e-5) << prefix << " " << p[i].name;
    }
    auto trained = make();
    AdamOptimizer opt(1e-2f, &cpu);
    for (int step = 0; step < 2; ++step) (void)TrainFeaturizer(*trained, x, opt);
    const std::vector<NamedParamRef> q = trained->named_parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        EXPECT_LT(MaxDiff(q[i].ref.value->to_host_vector(), Floats(g, prefix + ".after_" + params[i].second)), 1e-5) << prefix << " " << q[i].name;
    }
}

TEST(BlockSparseFeaturizer, VanillaMatchesPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = Golden();
    ExpectMatches("vanilla", [&] {
        BlockSparseOptions o;
        o.block_size = 2;
        o.k = 2;
        auto f = std::make_unique<BlockSparseFeaturizer>(6, 5, &cpu, o);
        SetParam(*f, "encoder.weight", Floats(g, "vanilla.w_enc"));
        SetParam(*f, "encoder.bias", Floats(g, "vanilla.b_enc"));
        SetParam(*f, "decoder.weight", Floats(g, "vanilla.w_dec"));
        return f;
    }, {{"encoder.weight", "w_enc"}, {"encoder.bias", "b_enc"}, {"decoder.weight", "w_dec"}});
}

TEST(BlockSparseFeaturizer, GrassmannianMatchesPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = Golden();
    ExpectMatches("grass", [&] {
        BlockSparseOptions o;
        o.variant = BsfVariant::Grassmannian;
        o.block_size = 2;
        o.k = 2;
        auto f = std::make_unique<BlockSparseFeaturizer>(6, 5, &cpu, o);
        SetParam(*f, "decoder.weight", Floats(g, "grass.frames"));
        SetParam(*f, "gamma", Floats(g, "grass.gamma"));
        return f;
    }, {{"decoder.weight", "frames"}, {"gamma", "gamma"}});
}

TEST(BlockSparseFeaturizer, GroupLassoMatchesPyTorchFromItsColdStart) {
    CPUBackend cpu;
    const SafetensorsFile g = Golden();
    auto make = [&] {
        BlockSparseOptions o;
        o.variant = BsfVariant::GroupLasso;
        o.block_size = 2;
        o.target_l0 = 2;
        auto f = std::make_unique<BlockSparseFeaturizer>(6, 5, &cpu, o);
        SetParam(*f, "encoder.weight", Floats(g, "lasso.w_enc"));
        SetParam(*f, "encoder.bias", Floats(g, "lasso.b_enc"));
        SetParam(*f, "decoder.weight", Floats(g, "lasso.w_dec"));
        return f;
    };
    ExpectMatches("lasso", make,
                  {{"encoder.weight", "w_enc"}, {"encoder.bias", "b_enc"}, {"decoder.weight", "w_dec"}, {"threshold_raw", "threshold_raw"}});
    auto f = make();
    const Tensor x(Shape({9, 6}), &cpu, Floats(g, "x"));
    const FeaturizerLoss loss = f->loss_and_backward(x);
    EXPECT_NEAR(loss.reconstruction, Floats(g, "lasso.reconstruction")[0], 1e-5);
    EXPECT_NEAR(loss.sparsity, Floats(g, "lasso.sparsity")[0], 1e-6);
    EXPECT_NEAR(std::log(f->l0_multiplier()), Floats(g, "lasso.log_lambda_after_one")[0], 1e-5);
    auto trained = make();
    AdamOptimizer opt(1e-2f, &cpu);
    for (int step = 0; step < 2; ++step) (void)TrainFeaturizer(*trained, x, opt);
    EXPECT_NEAR(std::log(trained->l0_multiplier()), Floats(g, "lasso.after_log_lambda")[0], 1e-5);
    for (const NamedBufferRef& b : trained->named_buffers()) {
        if (b.name == "bandwidth") EXPECT_NEAR(b.value->to_host_vector()[0], Floats(g, "lasso.after_bandwidth")[0], 1e-6);
    }
}

TEST(BlockSparseFeaturizer, TournamentSkipsAnOverlappingBlockAndRecordsTheVerdict) {
    CPUBackend cpu;
    BlockSparseOptions o;
    o.block_size = 1;
    o.k = 2;
    o.selection = BlockSelection::Tournament;
    BlockSparseFeaturizer f(4, 3, &cpu, o);
    // Block 1 nearly repeats block 0 (overlap 0.81); block 2 is orthogonal to both.
    const std::vector<float> dec = {1, 0, 0, 0, 0.9f, 0.43589f, 0, 0, 0, 0, 1, 0};
    f.decoder().set_weight(dec);
    std::vector<float> enc(12);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) enc[static_cast<size_t>(j * 3 + i)] = dec[static_cast<size_t>(i * 4 + j)];
    }
    f.encoder().set_weight(enc);
    const Tensor x(Shape({1, 4}), &cpu, std::vector<float>{1, 0, 0.5f, 0});  // norms 1, 0.9, 0.5
    auto active = [&] {
        const std::vector<float> c = f.encode(x).to_host_vector();
        return std::vector<bool>{c[0] != 0, c[1] != 0, c[2] != 0};
    };
    // Plain top-k would keep blocks 0 and 1; the tournament keeps 0 and 2.
    EXPECT_EQ(active(), (std::vector<bool>{true, false, true}));
    BlockSparseOptions plain = o;
    plain.selection = BlockSelection::TopK;
    BlockSparseFeaturizer p(4, 3, &cpu, plain);
    p.decoder().set_weight(dec);
    p.encoder().set_weight(enc);
    const std::vector<float> c = p.encode(x).to_host_vector();
    EXPECT_TRUE(c[0] != 0 && c[1] != 0 && c[2] == 0);
    // In training the duel is won by block 0; above the verdict overlap block 1 is out for good,
    // even for an input that only it would explain.
    (void)f.loss_and_backward(x);
    const Tensor only_one(Shape({1, 4}), &cpu, std::vector<float>{0, 1, 0, 0});
    EXPECT_EQ(f.encode(only_one).to_host_vector()[1], 0.0f);
}

/** @brief A sparse sum of planted 2D concepts: each input has 2 of 6, each at a random angle on a
 *         circle of radius 1 to 2 in its own random plane of R^12. */
struct Planted {
    std::vector<float> x;
    std::vector<std::vector<float>> planes;  ///< per concept, 2 x 12 orthonormal rows
};

Planted MakePlanted(int64_t n, uint64_t seed, int64_t d = 12, int concepts = 6) {
    Planted p;
    uint64_t s = seed;
    auto u = [&] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(static_cast<uint32_t>(s >> 32)) / 4294967296.0;
    };
    for (int c = 0; c < concepts; ++c) {
        std::vector<float> q(static_cast<size_t>(2 * d));
        for (float& v : q) v = static_cast<float>(u() * 2 - 1);
        for (int i = 0; i < 2; ++i) {
            float* r = q.data() + i * d;
            for (int j = 0; j < i; ++j) {
                double dot = 0;
                for (int64_t t = 0; t < d; ++t) dot += static_cast<double>(r[t]) * q[static_cast<size_t>(j * d + t)];
                for (int64_t t = 0; t < d; ++t) r[t] = static_cast<float>(r[t] - dot * q[static_cast<size_t>(j * d + t)]);
            }
            double n2 = 0;
            for (int64_t t = 0; t < d; ++t) n2 += static_cast<double>(r[t]) * r[t];
            for (int64_t t = 0; t < d; ++t) r[t] = static_cast<float>(r[t] / std::sqrt(n2));
        }
        p.planes.push_back(q);
    }
    for (int64_t r = 0; r < n; ++r) {
        std::vector<double> row(static_cast<size_t>(d), 0.0);
        const int a = static_cast<int>(u() * concepts) % concepts;
        int b = static_cast<int>(u() * (concepts - 1)) % (concepts - 1);
        if (b >= a) ++b;
        for (int c : {a, b}) {
            const double radius = 1 + u(), angle = 2 * 3.14159265358979 * u();
            for (int64_t t = 0; t < d; ++t) {
                row[static_cast<size_t>(t)] += radius * (std::cos(angle) * p.planes[static_cast<size_t>(c)][static_cast<size_t>(t)] +
                                                         std::sin(angle) * p.planes[static_cast<size_t>(c)][static_cast<size_t>(d + t)]);
            }
        }
        for (double v : row) p.x.push_back(static_cast<float>(v + 0.01 * (u() - 0.5)));
    }
    return p;
}

/** @brief How many planted planes some block spans: overlap `|Q_true Q_gᵀ|²_F / 2` above 0.9. */
int Recovered(BlockSparseFeaturizer& f, const Planted& p, int64_t d = 12) {
    int found = 0;
    for (const std::vector<float>& plane : p.planes) {
        double best = 0;
        for (int64_t g = 0; g < f.num_blocks(); ++g) {
            std::vector<float> q = f.block_frame(g);
            const int64_t b = f.block_size();
            for (int64_t i = 0; i < b; ++i) {  // orthonormalize the block's rows
                float* r = q.data() + i * d;
                for (int64_t j = 0; j < i; ++j) {
                    double dot = 0;
                    for (int64_t t = 0; t < d; ++t) dot += static_cast<double>(r[t]) * q[static_cast<size_t>(j * d + t)];
                    for (int64_t t = 0; t < d; ++t) r[t] = static_cast<float>(r[t] - dot * q[static_cast<size_t>(j * d + t)]);
                }
                double n2 = 0;
                for (int64_t t = 0; t < d; ++t) n2 += static_cast<double>(r[t]) * r[t];
                for (int64_t t = 0; t < d; ++t) r[t] = static_cast<float>(r[t] / std::max(1e-12, std::sqrt(n2)));
            }
            double s = 0;
            for (int i = 0; i < 2; ++i) {
                for (int64_t j = 0; j < b; ++j) {
                    double dot = 0;
                    for (int64_t t = 0; t < d; ++t) dot += static_cast<double>(plane[static_cast<size_t>(i * d + t)]) * q[static_cast<size_t>(j * d + t)];
                    s += dot * dot;
                }
            }
            best = std::max(best, s / 2);
        }
        found += best > 0.9 ? 1 : 0;
    }
    return found;
}

std::unique_ptr<BlockSparseFeaturizer> TrainBsf(const Tensor& x, BlockSparseOptions o, int64_t blocks, int steps, float lr, CPUBackend* cpu) {
    auto f = std::make_unique<BlockSparseFeaturizer>(x.shape().dim(1), blocks, cpu, o);
    AdamOptimizer opt(lr, cpu);
    for (int step = 0; step < steps; ++step) (void)TrainFeaturizer(*f, x, opt);
    return f;
}

TEST(BlockSparseFeaturizer, BlocksRecoverPlantedPlanesAndDescribeThemMoreCompactlyThanAnSae) {
    CPUBackend cpu;
    const int64_t n = 1024, d = 32;
    const Planted p = MakePlanted(n, 21, d);
    const Tensor x(Shape({n, d}), &cpu, p.x);
    BlockSparseOptions o;
    o.block_size = 2;
    o.k = 2;
    auto vanilla = TrainBsf(x, o, 8, 1000, 2e-2f, &cpu);
    o.variant = BsfVariant::Grassmannian;
    auto grass = TrainBsf(x, o, 8, 1000, 2e-2f, &cpu);
    TopKSaeOptions t;
    t.k = 4;  // the same active dimensions as 2 blocks of 2
    t.dead_after = 2 * n;
    TopKSparseAutoencoder sae(d, 16, &cpu, t);
    sae.initialize_bias(x);
    AdamOptimizer opt(2e-2f, &cpu);
    for (int step = 0; step < 1000; ++step) (void)TrainFeaturizer(sae, x, opt);
    const ReconstructionMetrics ev = EvaluateReconstruction(*vanilla, x), eg = EvaluateReconstruction(*grass, x), es = EvaluateReconstruction(sae, x);
    const DescriptionLength lv = MeasureDescriptionLength(*vanilla, x), ls = MeasureDescriptionLength(sae, x);
    std::printf("[BSF] planes recovered (of 6): vanilla %d, Grassmannian %d; EV vanilla %.3f, Grassmannian %.3f, TopK SAE %.3f\n",
                Recovered(*vanilla, p, d), Recovered(*grass, p, d), ev.explained_variance, eg.explained_variance, es.explained_variance);
    std::printf("[BSF] bits per input at 10%% distortion: vanilla BSF %.1f (support %.1f, code %.1f, residual %.1f), "
                "TopK SAE %.1f (support %.1f, code %.1f, residual %.1f)\n",
                lv.total, lv.support, lv.code, lv.residual, ls.total, ls.support, ls.code, ls.residual);
    // Seeds 5, 8 and 21 all gave 6 of 6, EV 1.000 and 19.9 bits against the SAE's 32.4 to 32.7.
    EXPECT_GE(Recovered(*vanilla, p, d), 5);
    EXPECT_NEAR(ev.l0, 2.0, 1e-9);  // blocks, not codes
    EXPECT_GT(ev.explained_variance, 0.95);
    EXPECT_GT(ev.explained_variance, es.explained_variance);
    EXPECT_LT(lv.total, 0.8 * ls.total);
    // Tied frames get stuck here (1 to 4 planes, EV about 0.9, at any learning rate tried).
    EXPECT_GT(eg.explained_variance, 0.8);
}

TEST(BlockSparseFeaturizer, ABlocksRankSaturatesAtItsConceptsDimension) {
    CPUBackend cpu;
    const int64_t n = 1024, d = 32;
    const Planted p = MakePlanted(n, 21, d);
    const Tensor x(Shape({n, d}), &cpu, p.x);
    BlockSparseOptions o;
    o.block_size = 4;  // twice the concepts' dimension
    o.k = 2;
    auto f = TrainBsf(x, o, 8, 1000, 2e-2f, &cpu);
    const BlockGeometry geo = MeasureBlockGeometry(*f, x);
    std::printf("[BSF] b = 4 on 2D concepts: stable rank %.2f, participation ratio %.2f, effective rank %.2f\n", geo.mean_stable_rank,
                geo.mean_participation_ratio, geo.mean_effective_rank);
    // Seeds 5, 8 and 21: stable rank 1.4 to 1.8, participation ratio 2.4 to 3.0, of 4.
    EXPECT_LT(geo.mean_stable_rank, 2.2);
    EXPECT_LT(geo.mean_participation_ratio, 3.2);
}

TEST(BlockSparseFeaturizer, GroupLassoHoldsItsTargetL0) {
    CPUBackend cpu;
    const int64_t n = 2048;
    const Planted p = MakePlanted(n, 33);
    const Tensor x(Shape({n, 12}), &cpu, p.x);
    BlockSparseOptions o;
    o.variant = BsfVariant::GroupLasso;
    o.block_size = 2;
    o.target_l0 = 2;
    auto f = TrainBsf(x, o, 8, 600, 5e-3f, &cpu);
    const ReconstructionMetrics r = EvaluateReconstruction(*f, x);
    std::printf("[BSF] group lasso: L0 %.2f (target 2), EV %.3f, lambda %.4f\n", r.l0, r.explained_variance, f->l0_multiplier());
    EXPECT_NEAR(r.l0, 2.0, 0.6);
    EXPECT_GT(r.explained_variance, 0.8);
    // Checkpoints keep λ, the bandwidth and the thresholds.
    const std::string path = (std::filesystem::temp_directory_path() / "pulsatrix_bsf.safetensors").string();
    SaveCheckpoint(path, *f);
    BlockSparseFeaturizer other(12, 8, &cpu, o);
    LoadCheckpoint(path, other);
    EXPECT_EQ(other.thresholds(), f->thresholds());
    EXPECT_EQ(other.l0_multiplier(), f->l0_multiplier());
    EXPECT_EQ(other.encode(x).to_host_vector(), f->encode(x).to_host_vector());
    std::filesystem::remove(path);
}

TEST(BlockSparseFeaturizer, RelevanceThroughBlockSelectionIsConservedAndBackwardIsExact) {
    CPUBackend cpu;
    BlockSparseOptions o;
    o.variant = BsfVariant::Grassmannian;
    o.block_size = 3;
    o.k = 2;
    BlockSparseFeaturizer f(5, 4, &cpu, o);
    std::vector<float> xv(3 * 5);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = std::sin(0.8f * static_cast<float>(i)) + 0.2f;
    const Tensor x(Shape({3, 5}), &cpu, xv);
    const Tensor y = f.forward(x);
    LRPRuleConfig config;
    config.epsilon = 1e-9f;
    const std::vector<float> r = f.propagate_relevance(y, config).to_host_vector(), out = y.to_host_vector();
    double in_sum = 0, out_sum = 0;
    for (float v : r) in_sum += v;
    for (float v : out) out_sum += v;
    EXPECT_NEAR(in_sum, out_sum, 1e-3 * std::abs(out_sum) + 1e-4);  // no biases, so relevance is conserved
    // Backward, with the selection held fixed, against finite differences of 0.5 |x̂|².
    BlockSparseOptions v = o;
    v.variant = BsfVariant::Vanilla;
    BlockSparseFeaturizer g(5, 4, &cpu, v);
    auto loss = [&](const std::vector<float>& in) {
        double s = 0;
        for (float t : g.forward(Tensor(Shape({3, 5}), &cpu, in)).to_host_vector()) s += 0.5 * t * t;
        return s;
    };
    const std::vector<float> gy = g.forward(x).to_host_vector();
    const std::vector<float> gx = g.backward(Tensor(Shape({3, 5}), &cpu, gy)).to_host_vector();
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> up = xv, down = xv;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(gx[i], (loss(up) - loss(down)) / 2e-3, 3e-3) << i;
    }
    // Metrics count blocks, and absorption refuses them.
    EXPECT_NEAR(MeanL0(f, x), 2.0, 1e-12);
    EXPECT_THROW((void)FeatureAbsorption(f, x, {0, 1, 1}), std::invalid_argument);
    EXPECT_THROW(BlockSparseFeaturizer(5, 4, &cpu, BlockSparseOptions{BsfVariant::Vanilla, 3, 5}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
