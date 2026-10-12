// TRN-9: LoReFT interventions, checked against pyreft's LoreftIntervention on the tiny Qwen2
// (tools/golden/make_loreft_golden.py), and LRP through them.
#include "pulsatrix/loreft.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/generation.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }
const std::vector<float> kTokens = {3, 17, 42, 5, 60, 8};

std::vector<float> Random(std::size_t n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> v(n);
    for (float& x : v) x = scale * normal(rng);
    return v;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    float m = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

class LoReFTLayerTest : public ::testing::Test {
protected:
    CPUBackend backend;
    const int64_t n = 2, len = 5, d = 8, r = 3;
    Tensor Hidden(uint32_t seed) { return Tensor(Shape({n, len, d}), &backend, Random(static_cast<std::size_t>(n * len * d), seed)); }
};

TEST_F(LoReFTLayerTest, EditsOnlyTheChosenPositionsAndOnlyInsideTheSubspace) {
    LoReFTIntervention edit(d, r, /*first=*/1, /*last=*/1, &backend, 4);
    EXPECT_LT(edit.orthonormality_error(), 1e-6);
    const Tensor h = Hidden(5);
    const std::vector<float> in = h.to_host_vector(), out = edit.forward(h).to_host_vector();
    EXPECT_EQ(edit.last_positions(), (std::vector<int64_t>{0, 4, 5, 9}));
    const std::vector<float> p = edit.rotation().to_host_vector(), q = edit.source_weight().to_host_vector(),
                             b = edit.source_bias().to_host_vector();
    for (int64_t s = 0; s < n; ++s) {
        for (int64_t t = 0; t < len; ++t) {
            const float* x = &in[static_cast<std::size_t>((s * len + t) * d)];
            const float* y = &out[static_cast<std::size_t>((s * len + t) * d)];
            if (t != 0 && t != len - 1) {
                for (int64_t i = 0; i < d; ++i) EXPECT_EQ(y[i], x[i]);
                continue;
            }
            // Inside the subspace the coordinates become W h + b; outside it nothing moves.
            std::vector<double> moved(static_cast<std::size_t>(d));
            for (int64_t k = 0; k < r; ++k) {
                double ry = 0, target = b[static_cast<std::size_t>(k)];
                for (int64_t i = 0; i < d; ++i) {
                    ry += static_cast<double>(p[static_cast<std::size_t>(i * r + k)]) * y[i];
                    target += static_cast<double>(q[static_cast<std::size_t>(i * r + k)]) * x[i];
                }
                EXPECT_NEAR(ry, target, 1e-5);
            }
            for (int64_t i = 0; i < d; ++i) moved[static_cast<std::size_t>(i)] = static_cast<double>(y[i]) - x[i];
            for (int64_t i = 0; i < d; ++i) {  // y - x lies in the span of R's rows: (I - P P^T)(y - x) = 0
                double outside = moved[static_cast<std::size_t>(i)];
                for (int64_t k = 0; k < r; ++k) {
                    double c = 0;
                    for (int64_t j = 0; j < d; ++j) c += p[static_cast<std::size_t>(j * r + k)] * moved[static_cast<std::size_t>(j)];
                    outside -= p[static_cast<std::size_t>(i * r + k)] * c;
                }
                EXPECT_NEAR(outside, 0.0, 1e-5);
            }
        }
    }
}

// sum(y * u) is quadratic in R, so central differences carry an O(h^2) error; it's linear in W, b and h.
TEST_F(LoReFTLayerTest, GradientsMatchFiniteDifferences) {
    LoReFTIntervention edit(d, r, 2, 1, &backend, 6);
    const Tensor h = Hidden(7);
    const std::vector<float> u = Random(static_cast<std::size_t>(n * len * d), 8);
    auto loss = [&](const Tensor& input) {
        const std::vector<float> y = edit.forward(input).to_host_vector();
        double s = 0;
        for (std::size_t i = 0; i < y.size(); ++i) s += static_cast<double>(y[i]) * u[i];
        return s;
    };
    (void)edit.forward(h);
    const std::vector<float> grad_h = edit.backward(Tensor(h.shape(), &backend, u)).to_host_vector();
    std::vector<std::vector<float>> grads;
    for (const NamedParamRef& p : edit.named_parameters()) grads.push_back(p.ref.grad->to_host_vector());
    const double step = 1e-3;
    for (std::size_t which = 0; which < 3; ++which) {
        for (std::size_t i = 0; i < grads[which].size(); ++i) {
            std::vector<float> values[3] = {edit.rotation().to_host_vector(), edit.source_weight().to_host_vector(),
                                            edit.source_bias().to_host_vector()};
            const float saved = values[which][i];
            values[which][i] = saved + static_cast<float>(step);
            edit.set_parameters(values[0], values[1], values[2]);
            const double up = loss(h);
            values[which][i] = saved - static_cast<float>(step);
            edit.set_parameters(values[0], values[1], values[2]);
            const double down = loss(h);
            values[which][i] = saved;
            edit.set_parameters(values[0], values[1], values[2]);
            EXPECT_NEAR(grads[which][i], (up - down) / (2 * step), 5e-3) << which << " " << i;
        }
    }
    std::vector<float> xv = h.to_host_vector();
    for (std::size_t i = 0; i < xv.size(); ++i) {
        const float saved = xv[i];
        xv[i] = saved + static_cast<float>(step);
        const double up = loss(Tensor(h.shape(), &backend, xv));
        xv[i] = saved - static_cast<float>(step);
        const double down = loss(Tensor(h.shape(), &backend, xv));
        xv[i] = saved;
        EXPECT_NEAR(grad_h[i], (up - down) / (2 * step), 5e-3) << "h " << i;
    }
}

// Epsilon on the whole affine map h M + c (M = I + (W^T - R^T) R): x_i M_ij R_j / stab(z_j), z = h M.
TEST_F(LoReFTLayerTest, EpsilonRelevanceIsThatOfTheAffineEdit) {
    LoReFTIntervention edit(d, r, 1, 0, &backend, 9);
    const Tensor h = Hidden(10);
    (void)edit.forward(h);
    const Tensor rel(h.shape(), &backend, Random(static_cast<std::size_t>(n * len * d), 11));
    const std::vector<float> got = edit.propagate_relevance(rel, LRPRuleConfig{}).to_host_vector();
    const std::vector<float> edit_part = edit.last_edit_relevance().to_host_vector();
    const std::vector<float> hv = h.to_host_vector(), rv = rel.to_host_vector(), p = edit.rotation().to_host_vector(),
                             q = edit.source_weight().to_host_vector();
    std::vector<double> m(static_cast<std::size_t>(d * d));  // (Q - P) P^T
    for (int64_t i = 0; i < d; ++i)
        for (int64_t j = 0; j < d; ++j)
            for (int64_t k = 0; k < r; ++k)
                m[static_cast<std::size_t>(i * d + j)] += (static_cast<double>(q[static_cast<std::size_t>(i * r + k)]) - p[static_cast<std::size_t>(i * r + k)]) *
                                                          p[static_cast<std::size_t>(j * r + k)];
    const float eps = LRPRuleConfig{}.epsilon;
    for (int64_t s = 0; s < n; ++s) {
        for (int64_t t = 0; t < len; ++t) {
            const std::size_t row = static_cast<std::size_t>((s * len + t) * d);
            if (t != 0) {  // passed through
                for (int64_t i = 0; i < d; ++i) {
                    EXPECT_EQ(got[row + static_cast<std::size_t>(i)], rv[row + static_cast<std::size_t>(i)]);
                    EXPECT_EQ(edit_part[row + static_cast<std::size_t>(i)], 0.0f);
                }
                continue;
            }
            std::vector<double> z(static_cast<std::size_t>(d));
            for (int64_t j = 0; j < d; ++j) {
                z[static_cast<std::size_t>(j)] = hv[row + static_cast<std::size_t>(j)];
                for (int64_t i = 0; i < d; ++i) z[static_cast<std::size_t>(j)] += hv[row + static_cast<std::size_t>(i)] * m[static_cast<std::size_t>(i * d + j)];
            }
            for (int64_t i = 0; i < d; ++i) {
                double kept = 0, edited = 0;
                for (int64_t j = 0; j < d; ++j) {
                    const double zj = z[static_cast<std::size_t>(j)];
                    const double ratio = rv[row + static_cast<std::size_t>(j)] / (zj + (zj >= 0 ? eps : -eps));
                    kept += (i == j ? 1.0 : 0.0) * hv[row + static_cast<std::size_t>(i)] * ratio;
                    edited += hv[row + static_cast<std::size_t>(i)] * m[static_cast<std::size_t>(i * d + j)] * ratio;
                }
                EXPECT_NEAR(edit_part[row + static_cast<std::size_t>(i)], edited, 1e-3 * (1 + std::fabs(edited)));
                EXPECT_NEAR(got[row + static_cast<std::size_t>(i)], kept + edited, 1e-3 * (1 + std::fabs(kept + edited)));
            }
        }
    }
}

TEST_F(LoReFTLayerTest, AnEditThatChangesNothingCarriesNoRelevance) {
    LoReFTIntervention edit(d, r, 1, 1, &backend, 12);
    // W = R and b = 0 write each coordinate back as it was.
    edit.set_parameters(edit.rotation().to_host_vector(), edit.rotation().to_host_vector(), std::vector<float>(static_cast<std::size_t>(r), 0.0f));
    const Tensor h = Hidden(13);
    EXPECT_EQ(edit.forward(h).to_host_vector(), h.to_host_vector());
    const Tensor rel(h.shape(), &backend, Random(static_cast<std::size_t>(n * len * d), 14));
    EXPECT_LT(MaxAbsDiff(edit.propagate_relevance(rel, LRPRuleConfig{}).to_host_vector(), rel.to_host_vector()), 1e-4f);
    for (float v : edit.last_edit_relevance().to_host_vector()) EXPECT_EQ(v, 0.0f);
}

TEST_F(LoReFTLayerTest, OrthonormalizeRestoresTheRotation) {
    LoReFTIntervention edit(d, r, 1, 1, &backend, 15);
    std::vector<float> p = edit.rotation().to_host_vector();
    for (float& v : p) v *= 1.1f;
    p[0] += 0.2f;
    edit.set_parameters(p, edit.source_weight().to_host_vector(), edit.source_bias().to_host_vector());
    EXPECT_GT(edit.orthonormality_error(), 0.1);
    edit.orthonormalize();
    EXPECT_LT(edit.orthonormality_error(), 1e-6);
    EXPECT_THROW(LoReFTIntervention(d, d + 1, 1, 1, &backend, 0), std::invalid_argument);
    EXPECT_THROW(LoReFTIntervention(d, 2, -1, 1, &backend, 0), std::invalid_argument);
}

class LoReFTModelTest : public ::testing::Test {
protected:
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Fixture("qwen2"), &backend);
    SafetensorsFile reference = SafetensorsFile::Read(Fixture("qwen2_loreft") + "/reference.safetensors");
    std::vector<float> Logits() { return model->forward(Tensor(Shape({1, 6}), &backend, kTokens)).to_host_vector(); }
};

TEST_F(LoReFTModelTest, LogitsGradientsAndGenerationMatchPyreft) {
    const std::vector<int64_t> prompt(kTokens.begin(), kTokens.end());
    GenerationConfig greedy;
    greedy.max_new_tokens = 8;
    const std::vector<int64_t> base_tokens = Generate(model->next_token_logits(32), prompt, greedy).new_tokens;

    const std::vector<LoReFTIntervention*> edits = LoadLoReFT(*model, Fixture("qwen2_loreft") + "/loreft.safetensors");
    ASSERT_EQ(edits.size(), 2u);
    EXPECT_EQ(edits[0]->first(), 1);
    EXPECT_EQ(edits[0]->last(), 2);
    EXPECT_LT(edits[0]->orthonormality_error(), 1e-5);
    EXPECT_LT(MaxAbsDiff(Logits(), reference.tensor("logits", &backend).to_host_vector()), 1e-4f);

    (void)model->backward(reference.tensor("upstream", &backend));
    for (int layer = 0; layer < 2; ++layer) {
        const std::string key = "layer." + std::to_string(layer) + ".";
        LoReFTIntervention& e = *edits[static_cast<std::size_t>(layer)];
        std::vector<std::vector<float>> grads;
        for (const NamedParamRef& p : e.named_parameters()) grads.push_back(p.ref.grad->to_host_vector());
        const int64_t d = 32, r = 2;
        const std::vector<float> want_r = reference.tensor("grad." + key + "rotate_layer", &backend).to_host_vector();
        const std::vector<float> want_w = reference.tensor("grad." + key + "learned_source.weight", &backend).to_host_vector();  // (r, d)
        const std::vector<float> want_b = reference.tensor("grad." + key + "learned_source.bias", &backend).to_host_vector();
        EXPECT_LT(MaxAbsDiff(grads[0], want_r), 1e-3f) << key;
        std::vector<float> w_ours(want_w.size());
        for (int64_t k = 0; k < r; ++k)
            for (int64_t i = 0; i < d; ++i) w_ours[static_cast<std::size_t>(i * r + k)] = want_w[static_cast<std::size_t>(k * d + i)];
        EXPECT_LT(MaxAbsDiff(grads[1], w_ours), 1e-3f) << key;
        EXPECT_LT(MaxAbsDiff(grads[2], want_b), 1e-3f) << key;
    }

    const std::vector<float> want = reference.tensor("generated", &backend).to_host_vector();
    const std::vector<int64_t> generated = Generate(model->next_token_logits(32), prompt, greedy).new_tokens;
    EXPECT_EQ(generated, std::vector<int64_t>(want.begin(), want.end()));
    EXPECT_NE(generated, base_tokens);  // the interventions change what the model says
}

TEST_F(LoReFTModelTest, SavingAndLoadingKeepPyreftsLayout) {
    (void)LoadLoReFT(*model, Fixture("qwen2_loreft") + "/loreft.safetensors");
    const std::string path = (std::filesystem::temp_directory_path() / "pulsatrix_loreft.safetensors").string();
    SaveLoReFT(*model, path);
    const SafetensorsFile ours = SafetensorsFile::Read(path), theirs = SafetensorsFile::Read(Fixture("qwen2_loreft") + "/loreft.safetensors");
    EXPECT_EQ(ours.metadata().at("first"), "1");
    EXPECT_EQ(ours.metadata().at("last"), "2");
    for (const std::string& key : theirs.names()) {
        ASSERT_TRUE(ours.contains(key)) << key;
        EXPECT_EQ(ours.info(key).shape, theirs.info(key).shape) << key;
        EXPECT_EQ(ours.tensor(key, &backend).to_host_vector(), theirs.tensor(key, &backend).to_host_vector()) << key;
    }
    EXPECT_THROW((void)LoadLoReFT(*model, path), std::invalid_argument);  // layers already intervened on
    std::filesystem::remove(path);
}

TEST_F(LoReFTModelTest, TrainingMovesOnlyTheInterventionsAndExplainsThroughThem) {
    LoReFTConfig config;
    config.rank = 2;
    config.layers = {1};
    config.first = 0;
    config.last = 2;
    const std::vector<LoReFTIntervention*> edits = ApplyLoReFT(*model, config);
    ASSERT_EQ(edits.size(), 1u);
    EXPECT_EQ(model->intervention(0), nullptr);
    int trainable = 0;
    for (const NamedParamRef& p : model->named_parameters()) {
        if (p.ref.value->requires_grad()) {
            ++trainable;
            EXPECT_EQ(p.name.rfind("interventions.1.", 0), 0u) << p.name;
        }
    }
    EXPECT_EQ(trainable, 3);
    const std::vector<float> embed = model->embed_tokens().weight().to_host_vector();

    // Teach the last position to predict token 9.
    const Tensor ids(Shape({1, 6}), &backend, kTokens);
    std::vector<float> targets(6, static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex));
    targets[5] = 9;
    const Tensor target(Shape({1, 6}), &backend, targets);
    AdamOptimizer opt(5e-2f, &backend);
    TokenCrossEntropyLoss loss(&backend);
    float first = 0, last = 0;
    for (int step = 0; step < 40; ++step) {
        opt.zero_grad(*model);
        last = loss.forward(model->forward(ids), target);
        if (step == 0) first = last;
        (void)model->backward(loss.backward());
        opt.step(*model);
        OrthonormalizeLoReFT(*model);
    }
    EXPECT_LT(last, 0.5f * first) << first << " -> " << last;
    EXPECT_LT(edits[0]->orthonormality_error(), 1e-5);
    EXPECT_EQ(model->embed_tokens().weight().to_host_vector(), embed);

    // The explanation runs through the intervention, which carries part of the relevance.
    const Tensor logits = model->forward(ids);
    std::vector<float> r(static_cast<std::size_t>(logits.numel()), 0.0f);
    r[static_cast<std::size_t>(5 * 64 + 9)] = logits.to_host_vector()[static_cast<std::size_t>(5 * 64 + 9)];
    (void)model->propagate_relevance(Tensor(logits.shape(), &backend, r), LRPRuleConfig{});
    double edit_total = 0;
    for (float v : edits[0]->last_edit_relevance().to_host_vector()) edit_total += std::fabs(v);
    EXPECT_GT(edit_total, 0.0);
}

}  // namespace
}  // namespace pulsatrix
