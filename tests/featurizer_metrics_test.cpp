// FEAT-3: featurizer metrics on featurizers with hand-set weights, so every expected value is
// known: reconstruction metrics, splicing into EncoderLM and CausalLM through the hidden-state
// hook (loss recovered), and feature absorption on a planted parent/child hierarchy.
#include "pulsatrix/featurizer_metrics.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/hf_model.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {
namespace {

/** @brief codes = ReLU(x W + b_enc), x̂ = codes D + b_dec, with W `(d, m)` and D `(m, d)` set by hand. */
class FixedFeaturizer : public Featurizer {
public:
    FixedFeaturizer(int64_t d, int64_t m, DeviceBackend* backend)
        : d_(d), m_(m), backend_(backend), w_(static_cast<size_t>(d * m), 0.0f), b_enc_(static_cast<size_t>(m), 0.0f),
          dec_(static_cast<size_t>(m * d), 0.0f), b_dec_(static_cast<size_t>(d), 0.0f), params_(1, 1, backend) {}

    float& W(int64_t in, int64_t f) { return w_[static_cast<size_t>(in * m_ + f)]; }
    float& D(int64_t f, int64_t out) { return dec_[static_cast<size_t>(f * d_ + out)]; }
    float& BEnc(int64_t f) { return b_enc_[static_cast<size_t>(f)]; }

    [[nodiscard]] int64_t input_dim() const override { return d_; }
    [[nodiscard]] int64_t num_features() const override { return m_; }
    [[nodiscard]] Tensor encode(const Tensor& x) override {
        const std::vector<float> v = x.to_host_vector();
        const int64_t n = x.shape().dim(0);
        std::vector<float> c(static_cast<size_t>(n * m_));
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t f = 0; f < m_; ++f) {
                double s = b_enc_[static_cast<size_t>(f)];
                for (int64_t j = 0; j < d_; ++j) s += static_cast<double>(v[static_cast<size_t>(r * d_ + j)]) * w_[static_cast<size_t>(j * m_ + f)];
                c[static_cast<size_t>(r * m_ + f)] = static_cast<float>(std::max(0.0, s));
            }
        }
        return Tensor(Shape({n, m_}), backend_, c);
    }
    [[nodiscard]] Tensor decode(const Tensor& codes) override {
        const std::vector<float> c = codes.to_host_vector();
        const int64_t n = codes.shape().dim(0);
        std::vector<float> x(static_cast<size_t>(n * d_));
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t j = 0; j < d_; ++j) {
                double s = b_dec_[static_cast<size_t>(j)];
                for (int64_t f = 0; f < m_; ++f) s += static_cast<double>(c[static_cast<size_t>(r * m_ + f)]) * dec_[static_cast<size_t>(f * d_ + j)];
                x[static_cast<size_t>(r * d_ + j)] = static_cast<float>(s);
            }
        }
        return Tensor(Shape({n, d_}), backend_, x);
    }
    FeaturizerLoss loss_and_backward(const Tensor&, std::vector<float>*) override { throw std::logic_error("not trainable"); }
    void normalize_decoder() override {}
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i) override {
        return {dec_.begin() + i * d_, dec_.begin() + (i + 1) * d_};
    }
    [[nodiscard]] Module& parameters_module() override { return params_; }

private:
    int64_t d_, m_;
    DeviceBackend* backend_;
    std::vector<float> w_, b_enc_, dec_, b_dec_;
    LinearModule params_;
};

/** @brief Features 2j and 2j + 1 carry dimension j's positive and negative parts: exact
 *         reconstruction, scaled by @p scale. */
FixedFeaturizer SignSplit(int64_t d, DeviceBackend* backend, float scale = 1.0f) {
    FixedFeaturizer f(d, 2 * d, backend);
    for (int64_t j = 0; j < d; ++j) {
        f.W(j, 2 * j) = 1.0f;
        f.W(j, 2 * j + 1) = -1.0f;
        f.D(2 * j, j) = scale;
        f.D(2 * j + 1, j) = -scale;
    }
    return f;
}

std::vector<float> Wave(int64_t n, float offset) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = std::sin(0.7f * static_cast<float>(i)) + offset;
    return v;
}

TEST(FeaturizerMetrics, ReconstructionMetricsHaveTheirDefinitions) {
    CPUBackend cpu;
    const int64_t n = 40, d = 3;
    const std::vector<float> xv = Wave(n * d, 2.0f);  // every value positive
    const Tensor x(Shape({n, d}), &cpu, xv);
    FixedFeaturizer exact = SignSplit(d, &cpu);
    ReconstructionMetrics r = EvaluateReconstruction(exact, x, 0.1, /*batch=*/7);
    EXPECT_EQ(r.inputs, n);
    EXPECT_NEAR(r.explained_variance, 1.0, 1e-9);
    EXPECT_NEAR(r.cosine, 1.0, 1e-9);
    EXPECT_NEAR(r.norm_ratio, 1.0, 1e-9);
    EXPECT_NEAR(r.l0, 3.0, 1e-12);
    // The negative halves never fire; the positive ones fire on everything.
    EXPECT_NEAR(r.dead_fraction, 0.5, 1e-12);
    EXPECT_NEAR(r.dense_fraction, 0.5, 1e-12);
    EXPECT_EQ(r.firing_rates[0], 1.0);
    EXPECT_EQ(r.firing_rates[1], 0.0);

    // Half-size reconstructions: error 0.25 |x|², variance about the mean.
    FixedFeaturizer half = SignSplit(d, &cpu, 0.5f);
    r = EvaluateReconstruction(half, x);
    double err = 0, var = 0;
    for (int64_t j = 0; j < d; ++j) {
        double mean = 0;
        for (int64_t i = 0; i < n; ++i) mean += xv[static_cast<size_t>(i * d + j)] / static_cast<double>(n);
        for (int64_t i = 0; i < n; ++i) {
            const double v = xv[static_cast<size_t>(i * d + j)];
            err += 0.25 * v * v;
            var += (v - mean) * (v - mean);
        }
    }
    EXPECT_NEAR(r.explained_variance, 1.0 - err / var, 1e-6);
    EXPECT_NEAR(r.mse, err / static_cast<double>(n * d), 1e-6);
    EXPECT_NEAR(r.cosine, 1.0, 1e-9);
    EXPECT_NEAR(r.norm_ratio, 0.5, 1e-6);
    EXPECT_THROW((void)EvaluateReconstruction(half, Tensor(Shape({2, 4}), &cpu, std::vector<float>(8, 1.0f))), std::invalid_argument);
}

/** @brief A loss that any change to the logits raises: the squared distance from @p clean. */
double DistanceFrom(const std::vector<float>& clean, const Tensor& logits) {
    const std::vector<float> v = logits.to_host_vector();
    double s = 0;
    for (size_t i = 0; i < v.size(); ++i) s += (static_cast<double>(v[i]) - clean[i]) * (static_cast<double>(v[i]) - clean[i]);
    return s;
}

TEST(FeaturizerMetrics, LossRecoveredSplicesIntoAnEncoder) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm", &cpu);
    const int64_t h = model->config().hidden_size, position = 1;
    const Tensor ids(Shape({1, 7}), &cpu, std::vector<float>{0, 5, 8, 11, 6, 13, 2});
    const std::vector<float> clean = model->forward(ids).to_host_vector();
    auto loss = [&](const HiddenStateHook& hook) {
        model->set_hidden_state_hook(hook);
        const double l = DistanceFrom(clean, model->forward(ids));
        model->set_hidden_state_hook({});
        return l;
    };
    // The hook sees every position, and its replacement is what the model continues with.
    std::vector<int64_t> seen;
    model->set_hidden_state_hook([&](int64_t at, const Tensor& hidden) {
        seen.push_back(at);
        EXPECT_EQ(hidden.shape(), Shape({1, 7, h}));
        return hidden;
    });
    (void)model->forward(ids);
    EXPECT_EQ(static_cast<int64_t>(seen.size()), model->num_layers() + 1);
    EXPECT_EQ(seen.front(), 0);
    model->set_hidden_state_hook([](int64_t, const Tensor& hidden) { return Tensor(Shape({1, 1}), hidden.backend(), std::vector<float>{0}); });
    EXPECT_THROW((void)model->forward(ids), std::invalid_argument);
    model->set_hidden_state_hook({});

    // A perfect featurizer recovers everything; one that reconstructs nothing is the ablation.
    FixedFeaturizer exact = SignSplit(h, &cpu);
    LossRecovered r = MeasureLossRecovered(exact, position, loss);
    EXPECT_DOUBLE_EQ(r.spliced, r.clean);
    EXPECT_GT(r.ablated, r.clean);
    EXPECT_DOUBLE_EQ(r.recovered, 1.0);
    FixedFeaturizer nothing(h, 4, &cpu);
    r = MeasureLossRecovered(nothing, position, loss);
    EXPECT_DOUBLE_EQ(r.spliced, r.ablated);
    EXPECT_DOUBLE_EQ(r.recovered, 0.0);
    // Halfway: rows left out are untouched, so splicing nowhere changes nothing.
    r = MeasureLossRecovered(nothing, position, loss, std::vector<bool>(7, false));
    EXPECT_DOUBLE_EQ(r.spliced, r.clean);
    EXPECT_TRUE(std::isnan(r.recovered));
    EXPECT_THROW((void)MeasureLossRecovered(nothing, position, loss, std::vector<bool>(3, true)), std::invalid_argument);
    FixedFeaturizer wrong = SignSplit(h + 1, &cpu);
    EXPECT_THROW((void)MeasureLossRecovered(wrong, position, loss), std::invalid_argument);
}

TEST(FeaturizerMetrics, LossRecoveredSplicesIntoACausalModel) {
    CPUBackend cpu;
    const HfModelConfig config = ReadHfConfig(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/qwen2/config.json");
    CausalLM model(config, &cpu);
    uint64_t seed = 3;
    for (const NamedParamRef& p : model.named_parameters()) {
        std::vector<float> v(static_cast<size_t>(p.ref.value->numel()));
        for (float& w : v) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            w = 0.2f * (static_cast<float>(static_cast<uint32_t>(seed >> 32)) / 4294967296.0f - 0.5f);
        }
        *p.ref.value = Tensor(p.ref.value->shape(), &cpu, v);
    }
    const Tensor ids(Shape({2, 4}), &cpu, std::vector<float>{1, 4, 9, 2, 7, 3, 3, 8});
    const std::vector<float> clean = model.forward(ids).to_host_vector();
    auto loss = [&](const HiddenStateHook& hook) {
        model.set_hidden_state_hook(hook);
        const double l = DistanceFrom(clean, model.forward(ids));
        model.set_hidden_state_hook({});
        return l;
    };
    FixedFeaturizer exact = SignSplit(config.hidden_size, &cpu), half = SignSplit(config.hidden_size, &cpu, 0.5f);
    for (int64_t position = 0; position <= model.num_layers(); ++position) {
        const LossRecovered r = MeasureLossRecovered(exact, position, loss);
        EXPECT_DOUBLE_EQ(r.spliced, r.clean) << position;
        EXPECT_DOUBLE_EQ(r.recovered, 1.0) << position;
    }
    const LossRecovered r = MeasureLossRecovered(half, 1, loss);
    EXPECT_GT(r.spliced, r.clean);
    EXPECT_GT(r.recovered, 0.0);
    EXPECT_LT(r.recovered, 1.0);
    // A loss that ablation doesn't raise leaves nothing to recover.
    EXPECT_TRUE(std::isnan(MeasureLossRecovered(half, 1, [&](const HiddenStateHook& hook) { return -loss(hook); }).recovered));
}

/**
 * @brief A planted hierarchy in 16 dimensions: concept A along e0; ten rare children c, each
 *        A plus e(1 + c); and distractors along e11..e15. Feature 0 stands for A but, in the
 *        absorbing featurizer, is silenced by any child; features 1..10 fire on their child and
 *        write along (A + child), absorbing A there.
 */
struct Hierarchy {
    std::vector<float> x;
    std::vector<int> labels;
    std::vector<int> child;  ///< -1 unless the input is a child
};

Hierarchy MakeHierarchy() {
    Hierarchy h;
    uint64_t s = 11;
    auto noise = [&] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return 0.01f * (static_cast<float>(static_cast<uint32_t>(s >> 32)) / 4294967296.0f - 0.5f);
    };
    auto add = [&](int label, int child, int distractor) {
        std::vector<float> row(16);
        for (float& v : row) v = noise();
        if (label == 1) row[0] += 2.0f;
        if (child >= 0) row[static_cast<size_t>(1 + child)] += 2.0f;
        if (distractor >= 0) row[static_cast<size_t>(11 + distractor)] += 2.0f;
        h.x.insert(h.x.end(), row.begin(), row.end());
        h.labels.push_back(label);
        h.child.push_back(child);
    };
    for (int i = 0; i < 1000; ++i) {
        if (i % 2 == 0) {
            add(0, -1, (i / 2) % 5);  // negatives
        } else if (i % 10 == 1) {
            add(1, (i / 10) % 10, -1);  // 100 children, 10 of each
        } else {
            add(1, -1, -1);  // 400 plain A
        }
    }
    return h;
}

FixedFeaturizer HierarchyFeaturizer(DeviceBackend* backend, bool absorbing) {
    FixedFeaturizer f(16, 16, backend);
    f.W(0, 0) = 1.0f;
    f.BEnc(0) = -0.5f;  // above the noise
    f.D(0, 0) = 1.0f;
    for (int c = 0; c < 10; ++c) {
        if (absorbing) f.W(1 + c, 0) = -3.0f;  // a child silences A's feature
        f.W(1 + c, 1 + c) = 2.0f;
        f.BEnc(1 + c) = -1.0f;
        f.D(1 + c, 0) = absorbing ? 1.0f / std::sqrt(2.0f) : 0.0f;
        f.D(1 + c, 1 + c) = absorbing ? 1.0f / std::sqrt(2.0f) : 1.0f;
    }
    for (int k = 0; k < 5; ++k) {
        f.W(11 + k, 11 + k) = 1.0f;
        f.BEnc(11 + k) = -1.0f;
        f.D(11 + k, 11 + k) = 1.0f;
    }
    return f;
}

TEST(FeaturizerMetrics, FeatureAbsorptionFindsAPlantedHierarchy) {
    CPUBackend cpu;
    const Hierarchy h = MakeHierarchy();
    const Tensor x(Shape({1000, 16}), &cpu, h.x);
    FixedFeaturizer absorbing = HierarchyFeaturizer(&cpu, true);
    const AbsorptionResult r = FeatureAbsorption(absorbing, x, h.labels);
    std::printf("[FeatureAbsorption] probe F1 %.3f, main F1 %.3f, %lld of %lld hits missed, %lld absorbed\n", r.probe_f1, r.main_f1,
                static_cast<long long>(r.missed), static_cast<long long>(r.probe_hits), static_cast<long long>(r.absorbed));
    EXPECT_GT(r.probe_f1, 0.99);
    EXPECT_EQ(r.main_features, (std::vector<int64_t>{0}));  // each child adds too little F1 to join
    EXPECT_EQ(r.probe_hits, r.positives);
    // Every held-out child is missed by feature 0 and absorbed by its own child feature.
    EXPECT_GT(r.missed, 20);
    EXPECT_EQ(r.absorbed, r.missed);
    EXPECT_NEAR(r.absorption_rate, static_cast<double>(r.absorbed) / static_cast<double>(r.probe_hits), 1e-12);
    EXPECT_NEAR(r.absorption_rate, 0.2, 0.06);
    EXPECT_NEAR(r.main_f1, 2 * 0.8 / 1.8, 0.03);  // recall 0.8, precision 1
    ASSERT_FALSE(r.absorbing_features.empty());
    for (const auto& [feature, count] : r.absorbing_features) {
        EXPECT_GE(feature, 1);
        EXPECT_LE(feature, 10);
        EXPECT_GT(count, 0);
    }

    // Without absorption, feature 0 covers every positive.
    FixedFeaturizer clean = HierarchyFeaturizer(&cpu, false);
    const AbsorptionResult c = FeatureAbsorption(clean, x, h.labels);
    EXPECT_EQ(c.missed, 0);
    EXPECT_EQ(c.absorbed, 0);
    EXPECT_GT(c.main_f1, 0.99);

    std::vector<int> bad = h.labels;
    bad[3] = 2;
    EXPECT_THROW((void)FeatureAbsorption(clean, x, bad), std::invalid_argument);
    EXPECT_THROW((void)FeatureAbsorption(clean, x, std::vector<int>(1000, 1)), std::invalid_argument);
    EXPECT_THROW((void)FeatureAbsorption(clean, x, std::vector<int>(10, 1)), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
