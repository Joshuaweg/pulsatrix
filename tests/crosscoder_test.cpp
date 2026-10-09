// FEAT-8: crosscoders, against a PyTorch rendering (tools/golden/make_crosscoder_golden.py), and
// model diffing on planted features: shared, base-only and fine-tune-only directions.
#include "pulsatrix/crosscoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/safetensors.hpp"

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

struct GoldenCase {
    const char* name;
    int64_t S, d, m, N;
    CrosscoderOptions options;
};

GoldenCase Case(const std::string& name) {
    CrosscoderOptions o;
    if (name == "l1") {
        o.sparsity = CrosscoderSparsity::L1;
        o.l1_coefficient = 0.1f;
        return {"l1", 3, 4, 10, 7, o};
    }
    o.k_aux = 4;
    o.dead_after = 100;
    if (name == "btk") {
        o.k = 3;
        return {"btk", 2, 5, 12, 9, o};
    }
    o.delta = true;
    o.k = 2;
    o.k_shared = 2;
    o.shared_fraction = 0.2;  // 3 of 15
    o.delta_coefficient = 0.5f;
    return {"delta", 2, 5, 15, 9, o};
}

std::vector<int64_t> DeadSince(const SafetensorsFile& g, const std::string& name, int64_t m) {
    std::vector<int64_t> since(static_cast<size_t>(m), 0);
    if (!g.contains(name + ".dead")) return since;
    auto [ptr, size] = g.bytes(name + ".dead");
    std::vector<int64_t> dead(size / sizeof(int64_t));
    std::memcpy(dead.data(), ptr, size);
    for (int64_t d : dead) since[static_cast<size_t>(d)] = 100;
    return since;
}

void Load(Crosscoder& c, const SafetensorsFile& g, const std::string& n) {
    c.encoder().set_weight(Floats(g, n + ".w_enc"));
    c.encoder().set_bias(Floats(g, n + ".b_enc"));
    c.decoder().set_weight(Floats(g, n + ".w_dec"));
    c.decoder().set_bias(Floats(g, n + ".b_dec"));
    c.set_inputs_since_fired(DeadSince(g, n, c.num_features()));
}

class CrosscoderGolden : public ::testing::TestWithParam<std::string> {};

TEST_P(CrosscoderGolden, MatchesPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = SafetensorsFile::Map(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/sae/crosscoder_golden.safetensors");
    const GoldenCase gc = Case(GetParam());
    const std::string n = gc.name;
    Crosscoder c(gc.S, gc.d, gc.m, &cpu, gc.options);
    Load(c, g, n);
    const Tensor x(Shape({gc.N, gc.S * gc.d}), &cpu, Floats(g, n + ".x"));
    const FeaturizerLoss loss = c.loss_and_backward(x);
    EXPECT_NEAR(loss.total, Floats(g, n + ".loss")[0], 1e-5);
    EXPECT_NEAR(loss.reconstruction, Floats(g, n + ".reconstruction")[0], 1e-5);
    EXPECT_NEAR(loss.sparsity, Floats(g, n + ".sparsity")[0], 1e-6);
    EXPECT_NEAR(c.delta_loss(), Floats(g, n + ".delta")[0], 1e-6);
    if (n != "l1") EXPECT_GT(loss.sparsity, 0.0f);  // AuxK ran
    if (n == "delta") EXPECT_GT(c.delta_loss(), 0.0f);
    const std::vector<NamedParamRef> p = c.named_parameters();
    ASSERT_EQ(p.size(), 4u);
    const char* names[] = {"grad_w_enc", "grad_b_enc", "grad_w_dec", "grad_b_dec"};
    for (size_t i = 0; i < 4; ++i) EXPECT_LT(MaxDiff(p[i].ref.grad->to_host_vector(), Floats(g, n + "." + names[i])), 1e-5) << p[i].name;

    // Two Adam steps without decoder normalization, the dead set held as the reference holds it.
    Crosscoder trained(gc.S, gc.d, gc.m, &cpu, gc.options);
    Load(trained, g, n);
    AdamOptimizer opt(1e-2f, &cpu);
    for (int step = 0; step < 2; ++step) {
        trained.set_inputs_since_fired(DeadSince(g, n, gc.m));
        (void)TrainFeaturizer(trained, x, opt, /*unit_norm_decoder=*/false);
    }
    const char* after[] = {"after_w_enc", "after_b_enc", "after_w_dec", "after_b_dec"};
    const std::vector<NamedParamRef> q = trained.named_parameters();
    for (size_t i = 0; i < 4; ++i) EXPECT_LT(MaxDiff(q[i].ref.value->to_host_vector(), Floats(g, n + "." + after[i])), 1e-5) << q[i].name;
}

INSTANTIATE_TEST_SUITE_P(Variants, CrosscoderGolden, ::testing::Values("l1", "btk", "delta"));

CrosscoderOptions Opts(CrosscoderSparsity sparsity, int64_t k, bool delta = false, double shared_fraction = 0.2) {
    CrosscoderOptions o;
    o.sparsity = sparsity;
    o.k = k;
    o.delta = delta;
    o.shared_fraction = shared_fraction;
    return o;
}

constexpr auto kL1 = CrosscoderSparsity::L1;
constexpr auto kBtk = CrosscoderSparsity::BatchTopK;

std::vector<float> Wave(int64_t n, float a, float b) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = std::sin(a * static_cast<float>(i) + b) + 0.3f;
    return v;
}

TEST(Crosscoder, StartsWithTheSameDecoderInEverySourceAndTheEncoderItsTranspose) {
    CPUBackend cpu;
    Crosscoder btk(3, 4, 6, &cpu, Opts(kBtk, 2)), l1(2, 4, 6, &cpu, Opts(kL1, 1));
    for (int64_t i = 0; i < 6; ++i) {
        const std::vector<float> a = btk.decoder_direction(i, 0);
        EXPECT_EQ(btk.decoder_direction(i, 1), a);
        EXPECT_EQ(btk.decoder_direction(i, 2), a);
        double sq = 0;
        for (float v : a) sq += static_cast<double>(v) * v;
        EXPECT_NEAR(std::sqrt(sq), 1.0, 1e-6);
        double sq1 = 0;
        for (float v : l1.decoder_direction(i, 1)) sq1 += static_cast<double>(v) * v;
        EXPECT_NEAR(std::sqrt(sq1), 0.05, 1e-7);
    }
    const std::vector<float> enc = btk.encoder().weight().to_host_vector(), dec = btk.decoder().weight().to_host_vector();
    for (int64_t i = 0; i < 6; ++i) {
        for (int64_t j = 0; j < 12; ++j) EXPECT_EQ(enc[static_cast<size_t>(j * 6 + i)], dec[static_cast<size_t>(i * 12 + j)]);
    }
    for (const CrosscoderLatentStats& s : CrosscoderLatents(btk, 0, 2)) {
        EXPECT_DOUBLE_EQ(s.relative_norm, 0.5);
        EXPECT_NEAR(s.cosine, 1.0, 1e-6);
        EXPECT_EQ(ClassifyLatent(s), LatentClass::Shared);
    }
}

TEST(Crosscoder, NormalizingTheDecoderChangesNoReconstructionPenaltyOrSelection) {
    CPUBackend cpu;
    for (bool l1 : {true, false}) {
        CrosscoderOptions o;
        o.sparsity = l1 ? CrosscoderSparsity::L1 : CrosscoderSparsity::BatchTopK;
        o.k = 3;
        Crosscoder c(2, 5, 12, &cpu, o);
        std::vector<float> w = c.decoder().weight().to_host_vector();
        for (size_t i = 0; i < w.size(); ++i) w[i] *= 0.3f + 0.2f * static_cast<float>(i % 7);  // uneven norms
        c.decoder().set_weight(w);
        c.encoder().set_bias(Wave(12, 0.9f, 0.1f));
        const Tensor x(Shape({6, 10}), &cpu, Wave(60, 0.37f, 0.2f));
        const std::vector<float> codes = c.encode(x).to_host_vector(), xh = c.decode(c.encode(x)).to_host_vector();
        const FeaturizerLoss l0 = c.loss_and_backward(x);
        c.set_thresholds({-1.0f});  // select across the batch again, as before
        c.normalize_decoder();
        for (int64_t i = 0; i < 12; ++i) {
            double sq = 0;
            for (float v : c.decoder_direction(i)) sq += static_cast<double>(v) * v;
            EXPECT_NEAR(std::sqrt(sq), 1.0, 1e-5);
        }
        const std::vector<float> codes2 = c.encode(x).to_host_vector();
        for (size_t i = 0; i < codes.size(); ++i) EXPECT_EQ(codes[i] > 0.0f, codes2[i] > 0.0f) << i;
        EXPECT_LT(MaxDiff(c.decode(c.encode(x)).to_host_vector(), xh), 1e-5);
        const FeaturizerLoss l1loss = c.loss_and_backward(x);
        EXPECT_NEAR(l1loss.total, l0.total, 1e-5);
        EXPECT_NEAR(l1loss.sparsity, l0.sparsity, 1e-6);
    }
}

TEST(Crosscoder, RejectsBadOptions) {
    CPUBackend cpu;
    EXPECT_THROW(Crosscoder(0, 4, 8, &cpu), std::invalid_argument);
    EXPECT_THROW(Crosscoder(2, 4, 8, &cpu, Opts(kBtk, 9)), std::invalid_argument);
    CrosscoderOptions bad;
    bad.l1_coefficient = -1;
    EXPECT_THROW(Crosscoder(2, 4, 8, &cpu, bad), std::invalid_argument);
    bad = CrosscoderOptions{};
    bad.threshold_decay = 1;
    EXPECT_THROW(Crosscoder(2, 4, 8, &cpu, bad), std::invalid_argument);
    // The Delta-Crosscoder needs two sources, BatchTopK, and both partitions.
    EXPECT_THROW(Crosscoder(3, 4, 20, &cpu, Opts(kBtk, 2, true)), std::invalid_argument);
    EXPECT_THROW(Crosscoder(2, 4, 20, &cpu, Opts(kL1, 2, true)), std::invalid_argument);
    EXPECT_THROW(Crosscoder(2, 4, 20, &cpu, Opts(kBtk, 2, true, 0.0)), std::invalid_argument);
    EXPECT_THROW(Crosscoder(2, 4, 20, &cpu, Opts(kBtk, 3, true)), std::invalid_argument);  // k_shared 6 > 4 shared
    const Crosscoder ok(2, 4, 20, &cpu, Opts(kBtk, 2, true));
    EXPECT_EQ(ok.num_shared(), 4);
    EXPECT_EQ(ok.options().k_shared, 4);
    Crosscoder c(2, 4, 8, &cpu, Opts(kBtk, 2));
    EXPECT_THROW((void)c.encode(Tensor(Shape({3, 4}), &cpu, std::vector<float>(12, 1))), std::invalid_argument);
    EXPECT_THROW(c.set_thresholds({1, 2}), std::invalid_argument);
    EXPECT_THROW((void)CrosscoderLatents(c, 0, 0), std::invalid_argument);
    EXPECT_THROW((void)MeasureLatentScaling(c, Tensor(Shape({1, 8}), &cpu, std::vector<float>(8, 1)), {8}), std::invalid_argument);
}

TEST(Crosscoder, IsAModuleWithAThresholdForInferenceAndCheckpoints) {
    CPUBackend cpu;
    CrosscoderOptions o;
    o.k = 2;
    o.delta = true;
    o.shared_fraction = 0.25;  // 3 of 12, k_shared 4 > 3
    o.k_shared = 2;
    Crosscoder c(2, 4, 12, &cpu, o);
    const Tensor x(Shape({5, 8}), &cpu, Wave(40, 0.53f, 0.4f));
    EXPECT_EQ(c.thresholds(), (std::vector<float>{-1, -1}));
    AdamOptimizer opt(1e-3f, &cpu);
    (void)TrainFeaturizer(c, x, opt, false);
    const std::vector<float> theta = c.thresholds();
    EXPECT_GT(theta[0], 0.0f);
    EXPECT_GT(theta[1], 0.0f);
    // The first batch sets each threshold to its own smallest kept value, which this batch sits
    // on; lower them so no value is at a threshold.
    c.set_thresholds({0.8f * theta[0], 0.8f * theta[1]});
    // Backward against finite differences of 0.5 |x̂|², with the thresholds fixed.
    const std::vector<float> xv = x.to_host_vector();
    auto f = [&](const std::vector<float>& in) {
        double s = 0;
        for (float v : c.forward(Tensor(Shape({5, 8}), &cpu, in)).to_host_vector()) s += 0.5 * v * v;
        return s;
    };
    const std::vector<float> y = c.forward(x).to_host_vector();
    const std::vector<float> gx = c.backward(Tensor(Shape({5, 8}), &cpu, y)).to_host_vector();
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> up = xv, down = xv;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(gx[i], (f(up) - f(down)) / 2e-3, 3e-3) << i;
    }
    (void)c.forward(x);
    EXPECT_EQ(c.propagate_relevance(Tensor(Shape({5, 8}), &cpu, y), LRPRuleConfig{}).shape(), Shape({5, 8}));
    // Checkpoints keep the thresholds, so inference codes match.
    const std::string path = (std::filesystem::temp_directory_path() / "pulsatrix_crosscoder.safetensors").string();
    SaveCheckpoint(path, c);
    Crosscoder other(2, 4, 12, &cpu, o);
    LoadCheckpoint(path, other);
    EXPECT_EQ(other.thresholds(), c.thresholds());
    EXPECT_EQ(other.encode(x).to_host_vector(), c.encode(x).to_host_vector());
    std::filesystem::remove(path);
}

TEST(Crosscoder, LatentStatsAndClasses) {
    CPUBackend cpu;
    Crosscoder c(2, 2, 4, &cpu, Opts(kBtk, 1));
    // Latent 0 only in source 0, 1 only in source 1, 2 even and aligned, 3 even and opposite.
    c.decoder().set_weight({3, 4, 0, 0, /**/ 0, 0, 1, 0, /**/ 1, 0, 1, 0, /**/ 0, 2, 0, -2});
    const std::vector<CrosscoderLatentStats> s = CrosscoderLatents(c);
    EXPECT_DOUBLE_EQ(s[0].norm_a, 5);
    EXPECT_DOUBLE_EQ(s[0].relative_norm, 0);
    EXPECT_DOUBLE_EQ(s[0].delta_norm, 0);
    EXPECT_EQ(ClassifyLatent(s[0]), LatentClass::AOnly);
    EXPECT_DOUBLE_EQ(s[1].relative_norm, 1);
    EXPECT_EQ(ClassifyLatent(s[1]), LatentClass::BOnly);
    EXPECT_DOUBLE_EQ(s[2].cosine, 1);
    EXPECT_EQ(ClassifyLatent(s[2]), LatentClass::Shared);
    EXPECT_DOUBLE_EQ(s[3].cosine, -1);
    // Δnorm and the relative norm agree at the ends and the middle, not between: |a| = 1, |b| = 3.
    c.decoder().set_weight({1, 0, 3, 0, /**/ 0, 0, 1, 0, /**/ 1, 0, 1, 0, /**/ 0, 2, 0, -2});
    const CrosscoderLatentStats t = CrosscoderLatents(c)[0];
    EXPECT_DOUBLE_EQ(t.relative_norm, 0.75);
    EXPECT_DOUBLE_EQ(t.delta_norm, 0.5 * (1 + 2.0 / 3.0));
    EXPECT_EQ(ClassifyLatent(t), LatentClass::Other);
}

TEST(Crosscoder, LatentScalingMeasuresWhatEachSourceNeeds) {
    CPUBackend cpu;
    // One latent reading and writing u in source b only. Source a holds c · t · u: the decoder
    // would need c · u there, so ν_error is c (complete shrinkage when c > 0). Nothing else
    // writes u to a, so ν_reconstruction is 0.
    const int64_t d = 3;
    const std::vector<float> u = {0.6f, 0.0f, 0.8f};
    for (double c_true : {0.0, 0.7}) {
        Crosscoder c(2, d, 1, &cpu, Opts(kL1, 1));
        c.decoder().set_weight({0, 0, 0, u[0], u[1], u[2]});
        c.encoder().set_weight({0, 0, 0, u[0], u[1], u[2]});
        std::vector<float> x;
        for (int r = 0; r < 20; ++r) {
            const float t = r % 4 == 0 ? 0.0f : 0.2f + 0.1f * static_cast<float>(r);
            for (int64_t j = 0; j < d; ++j) x.push_back(static_cast<float>(c_true * t * u[static_cast<size_t>(j)]));
            for (int64_t j = 0; j < d; ++j) x.push_back(t * u[static_cast<size_t>(j)]);
        }
        const Tensor xt(Shape({20, 2 * d}), &cpu, x);
        const LatentScaling ls = MeasureLatentScaling(c, xt, {0}, 0, 1, 7)[0];
        EXPECT_EQ(ls.active, 15);
        EXPECT_NEAR(ls.beta_error_b, 1.0, 1e-5);
        EXPECT_NEAR(ls.beta_error_a, c_true, 1e-5);
        EXPECT_NEAR(ls.nu_error, c_true, 1e-5);
        EXPECT_NEAR(ls.beta_reconstruction_b, 1.0, 1e-5);
        EXPECT_NEAR(ls.nu_reconstruction, 0.0, 1e-6);
        const std::vector<double> ev = ExplainedVarianceBySource(c, xt, 6);
        EXPECT_NEAR(ev[1], 1.0, 1e-5);
        if (c_true > 0) EXPECT_LT(ev[0], 0.0);  // source a: nothing reconstructed, worse than its mean
    }
}


/**
 * @brief Two "models" over the same inputs, d = 24: 16 shared features (the same direction in
 *        both), 4 only in the base (source 0) and 4 only in the fine-tune (source 1). Each fires
 *        independently with its own rate, at a magnitude in [0.5, 1.5], plus a little noise.
 */
struct PlantedDiff {
    int64_t d = 24;
    std::vector<std::vector<double>> shared, base_only, ft_only;
    std::vector<float> x;  ///< (n, 2d)
};

PlantedDiff MakePlantedDiff(int64_t n, double shared_rate, double base_rate, double ft_rate, uint64_t seed) {
    uint64_t s = seed;
    auto u = [&] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(static_cast<uint32_t>(s >> 32)) / 4294967296.0;
    };
    auto gauss = [&] { return std::sqrt(-2.0 * std::log(u() + 1e-12)) * std::cos(6.283185307179586 * u()); };
    PlantedDiff p;
    auto direction = [&] {
        std::vector<double> v(static_cast<size_t>(p.d));
        double sq = 0;
        for (double& e : v) {
            e = gauss();
            sq += e * e;
        }
        for (double& e : v) e /= std::sqrt(sq);
        return v;
    };
    for (int i = 0; i < 16; ++i) p.shared.push_back(direction());
    for (int i = 0; i < 4; ++i) p.base_only.push_back(direction());
    for (int i = 0; i < 4; ++i) p.ft_only.push_back(direction());
    const auto d = static_cast<size_t>(p.d);
    for (int64_t r = 0; r < n; ++r) {
        std::vector<double> a(d), b(d);
        for (size_t j = 0; j < d; ++j) {
            a[j] = 0.01 * gauss();
            b[j] = 0.01 * gauss();
        }
        auto fire = [&](const std::vector<std::vector<double>>& dirs, double rate, bool to_a, bool to_b) {
            for (const std::vector<double>& v : dirs) {
                if (u() >= rate) continue;
                const double c = 0.5 + u();
                for (size_t j = 0; j < d; ++j) {
                    if (to_a) a[j] += c * v[j];
                    if (to_b) b[j] += c * v[j];
                }
            }
        };
        fire(p.shared, shared_rate, true, true);
        fire(p.base_only, base_rate, true, false);
        fire(p.ft_only, ft_rate, false, true);
        for (double v : a) p.x.push_back(static_cast<float>(v));
        for (double v : b) p.x.push_back(static_cast<float>(v));
    }
    return p;
}

/** @brief The latent whose decoder in @p source is closest in cosine to @p v, and that cosine. */
std::pair<int64_t, double> BestLatent(Crosscoder& c, const std::vector<double>& v, int64_t source) {
    double best = -2;
    int64_t arg = -1;
    for (int64_t i = 0; i < c.num_features(); ++i) {
        const std::vector<float> w = c.decoder_direction(i, source);
        double dot = 0, sq = 0;
        for (size_t j = 0; j < v.size(); ++j) {
            dot += w[j] * v[j];
            sq += static_cast<double>(w[j]) * w[j];
        }
        const double cos = sq > 0 ? dot / std::sqrt(sq) : 0;
        if (cos > best) {
            best = cos;
            arg = i;
        }
    }
    return {arg, best};
}

/** @brief How many planted features a latent recovers: decoder cosine above 0.9 in the source
 *         the feature lives in, and the right class by Δnorm. */
int Recovered(Crosscoder& c, const std::vector<std::vector<double>>& dirs, int64_t source, LatentClass want) {
    const std::vector<CrosscoderLatentStats> stats = CrosscoderLatents(c);
    int found = 0;
    for (const std::vector<double>& v : dirs) {
        const auto [arg, cos] = BestLatent(c, v, source);
        if (cos > 0.9 && ClassifyLatent(stats[static_cast<size_t>(arg)]) == want) ++found;
    }
    return found;
}

std::unique_ptr<Crosscoder> TrainOnPlanted(CPUBackend& cpu, const PlantedDiff& p, int64_t n, CrosscoderOptions o, int64_t m, int steps,
                                           int64_t batch) {
    auto c = std::make_unique<Crosscoder>(2, p.d, m, &cpu, o);
    c->initialize_bias(Tensor(Shape({n, 2 * p.d}), &cpu, p.x));
    AdamOptimizer opt(3e-3f, &cpu);
    for (int step = 0; step < steps; ++step) {
        const int64_t r0 = (step * batch) % n;
        const Tensor xb(Shape({batch, 2 * p.d}), &cpu, std::vector<float>(p.x.begin() + r0 * 2 * p.d, p.x.begin() + (r0 + batch) * 2 * p.d));
        (void)TrainFeaturizer(*c, xb, opt, false);
    }
    return c;
}

TEST(Crosscoder, ModelDiffingRecoversSharedAndExclusiveFeatures) {
    CPUBackend cpu;
    const int64_t n = 8192;
    const PlantedDiff p = MakePlantedDiff(n, 0.08, 0.08, 0.08, 21);
    const Tensor x(Shape({n, 2 * p.d}), &cpu, p.x);
    for (CrosscoderSparsity sp : {kBtk, kL1}) {
        CrosscoderOptions o;
        o.sparsity = sp;
        o.k = 3;
        o.l1_coefficient = 1e-3f;
        o.dead_after = 4096;
        o.seed = 1;
        std::unique_ptr<Crosscoder> c = TrainOnPlanted(cpu, p, n, o, 48, 1500, 512);
        const int shared = Recovered(*c, p.shared, 0, LatentClass::Shared), base = Recovered(*c, p.base_only, 0, LatentClass::AOnly),
                  ft = Recovered(*c, p.ft_only, 1, LatentClass::BOnly);
        const std::vector<double> ev = ExplainedVarianceBySource(*c, x);
        std::printf("[Crosscoder] %s: recovered shared %d/16, base-only %d/4, fine-tune-only %d/4; EV base %.3f, fine-tune %.3f\n",
                    sp == kL1 ? "L1" : "BatchTopK", shared, base, ft, ev[0], ev[1]);
        // BatchTopK 16/16, 4/4, 4/4; L1 13/16, 4/4, 4/4.
        EXPECT_GE(shared, sp == kL1 ? 12 : 15);
        EXPECT_EQ(base, 4);
        EXPECT_EQ(ft, 4);
        EXPECT_GT(ev[0], 0.99);
        EXPECT_GT(ev[1], 0.99);
    }
}

TEST(Crosscoder, OnANarrowFineTuneLatentScalingFindsWhatDecoderNormsMiss) {
    CPUBackend cpu;
    const int64_t n = 16384;
    // The fine-tune adds 4 rare features (0.5% of inputs each) and keeps the 16 shared ones.
    const PlantedDiff p = MakePlantedDiff(n, 0.08, 0.0, 0.005, 33);
    const Tensor x(Shape({n, 2 * p.d}), &cpu, p.x);
    for (bool delta : {false, true}) {
        CrosscoderOptions o;
        o.dead_after = 4096;
        o.seed = 1;
        o.delta = delta;
        o.k = delta ? 1 : 3;
        o.k_shared = 3;
        o.delta_coefficient = 1.0f;
        std::unique_ptr<Crosscoder> c = TrainOnPlanted(cpu, p, n, o, 64, 2000, 256);
        const std::vector<CrosscoderLatentStats> stats = CrosscoderLatents(*c);
        // One Latent Scaling pass: the fine-tune-only features' latents, then the shared ones'.
        std::vector<int64_t> latents;
        int learned = 0, by_norm = 0, by_scaling = 0, in_delta = 0;
        double mean_delta_norm = 0;
        for (const std::vector<double>& v : p.ft_only) {
            const auto [arg, cos] = BestLatent(*c, v, 1);
            latents.push_back(arg);
            learned += cos > 0.85 ? 1 : 0;
            by_norm += ClassifyLatent(stats[static_cast<size_t>(arg)]) == LatentClass::BOnly ? 1 : 0;
            in_delta += arg >= c->num_shared() ? 1 : 0;
            mean_delta_norm += stats[static_cast<size_t>(arg)].delta_norm / 4;
        }
        for (const std::vector<double>& v : p.shared) latents.push_back(BestLatent(*c, v, 0).first);
        const std::vector<LatentScaling> ls = MeasureLatentScaling(*c, x, latents);
        int shared_kept = 0;
        for (size_t i = 0; i < ls.size(); ++i) {
            if (i < 4) {
                by_scaling += ls[i].nu_reconstruction < 0.5 && ls[i].nu_error < 0.2 ? 1 : 0;  // Minder et al.'s criterion
            } else {
                shared_kept += ls[i].nu_reconstruction > 0.5 ? 1 : 0;  // for contrast: both sources need these
            }
        }
        std::printf("[Crosscoder] narrow fine-tune, %s: fine-tune-only features learned %d/4, called fine-tune-only by Δnorm %d/4 "
                    "(mean Δnorm %.2f), by Latent Scaling %d/4; shared latents both sources need %d/16\n",
                    delta ? "Delta-Crosscoder" : "BatchTopK", learned, by_norm, mean_delta_norm, by_scaling, shared_kept);
        EXPECT_GE(learned, 3);  // above cosine 0.85: BatchTopK 3, Delta 4
        EXPECT_EQ(by_scaling, 4);
        EXPECT_LE(by_norm, 1);  // their base-side decoders keep about half the norm, unused
        EXPECT_GE(shared_kept, 12);
        if (delta) EXPECT_EQ(in_delta, 4);
    }
}

}  // namespace
}  // namespace pulsatrix
