// LLM-2: TiedLMHeadModule, the language-model head that shares the token embedding table.

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/tied_lm_head_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Pattern(int64_t n, float scale, float shift) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = scale * static_cast<float>((i * 7 + 3) % 11 - 5) + shift;
    }
    return v;
}

// E^T, the (d_model, vocab) weight a bias-free LinearModule needs to compute the same logits.
std::vector<float> Transposed(const std::vector<float>& e, int64_t vocab, int64_t d) {
    std::vector<float> t(e.size());
    for (int64_t r = 0; r < vocab; ++r) {
        for (int64_t c = 0; c < d; ++c) {
            t[static_cast<size_t>(c * vocab + r)] = e[static_cast<size_t>(r * d + c)];
        }
    }
    return t;
}

class TiedLMHeadModuleTest : public ::testing::Test {
protected:
    static constexpr int64_t kVocab = 7;
    static constexpr int64_t kD = 4;
    CPUBackend backend;
    EmbeddingModule embedding{kVocab, kD, &backend};
    std::vector<float> table = Pattern(kVocab * kD, 0.2f, 0.05f);

    void SetUp() override { embedding.set_weight(table); }
};

TEST_F(TiedLMHeadModuleTest, ForwardIsInputTimesTransposedTable) {
    TiedLMHeadModule head(embedding, &backend);
    EXPECT_EQ(head.vocab_size(), kVocab);
    EXPECT_EQ(head.d_model(), kD);
    const std::vector<float> x = Pattern(2 * 3 * kD, 0.3f, -0.1f);
    Tensor y = head.forward(Tensor(Shape({2, 3, kD}), &backend, x));
    ASSERT_EQ(y.shape(), Shape({2, 3, kVocab}));
    for (int64_t t = 0; t < 6; ++t) {
        for (int64_t v = 0; v < kVocab; ++v) {
            float expected = 0.0f;
            for (int64_t c = 0; c < kD; ++c) {
                expected += x[static_cast<size_t>(t * kD + c)] * table[static_cast<size_t>(v * kD + c)];
            }
            EXPECT_NEAR(y.data()[t * kVocab + v], expected, 1e-5f);
        }
    }
}

// The head is a bias-free LinearModule whose weight is E^T: same logits, same input gradient,
// same weight gradient (transposed), and the same relevance under every rule.
TEST_F(TiedLMHeadModuleTest, MatchesBiasFreeLinearWithTransposedTable) {
    TiedLMHeadModule head(embedding, &backend);
    LinearModule linear(kD, kVocab, &backend, /*use_bias=*/false);
    linear.set_weight(Transposed(table, kVocab, kD));

    const Shape in_shape({5, kD});
    const std::vector<float> x = Pattern(5 * kD, 0.3f, -0.1f);
    const std::vector<float> g = Pattern(5 * kVocab, 0.25f, 0.02f);
    Tensor y_head = head.forward(Tensor(in_shape, &backend, x));
    Tensor y_lin = linear.forward(Tensor(in_shape, &backend, x));
    for (int64_t i = 0; i < y_head.numel(); ++i) {
        EXPECT_NEAR(y_head.data()[i], y_lin.data()[i], 1e-5f);
    }
    Tensor dx_head = head.backward(Tensor(Shape({5, kVocab}), &backend, g));
    Tensor dx_lin = linear.backward(Tensor(Shape({5, kVocab}), &backend, g));
    for (int64_t i = 0; i < dx_head.numel(); ++i) {
        EXPECT_NEAR(dx_head.data()[i], dx_lin.data()[i], 1e-5f);
    }
    const std::vector<float> gw_lin = Transposed(linear.weight_grad().to_host_vector(), kD, kVocab);
    for (int64_t i = 0; i < kVocab * kD; ++i) {
        EXPECT_NEAR(embedding.weight_grad().data()[i], gw_lin[static_cast<size_t>(i)], 1e-5f);
    }

    std::vector<LRPRuleConfig> rules(6);
    rules[1].epsilon_bias_in_denominator = true;
    rules[2].rule = LRPRule::Gamma;
    rules[3].rule = LRPRule::AlphaBeta;
    rules[3].alpha = 2.0f;
    rules[3].beta = 1.0f;
    rules[4].rule = LRPRule::ZBox;
    rules[4].low = -1.0f;
    rules[4].high = 1.0f;
    rules[5].epsilon = 0.25f;
    const std::vector<float> r = Pattern(5 * kVocab, 0.4f, 0.1f);
    for (size_t k = 0; k < rules.size(); ++k) {
        Tensor r_head = head.propagate_relevance(Tensor(Shape({5, kVocab}), &backend, r), rules[k]);
        Tensor r_lin = linear.propagate_relevance(Tensor(Shape({5, kVocab}), &backend, r), rules[k]);
        for (int64_t i = 0; i < r_head.numel(); ++i) {
            EXPECT_NEAR(r_head.data()[i], r_lin.data()[i], 1e-4f) << "rule " << k << ", element " << i;
        }
    }
}

// Embedding -> head: the shared table's gradient is the sum of both uses. Checked by central
// differences on the table itself, which perturbs the lookup and the head at once.
TEST_F(TiedLMHeadModuleTest, SharedTableGradientSumsBothUses) {
    TiedLMHeadModule head(embedding, &backend);
    const std::vector<float> ids = {3, 0, 3, 6, 1, 3};
    const std::vector<float> g = Pattern(6 * kVocab, 0.3f, -0.05f);
    auto objective = [&]() {
        Tensor y = head.forward(embedding.forward(Tensor(Shape({2, 3}), &backend, ids)));
        double s = 0.0;
        for (int64_t i = 0; i < y.numel(); ++i) {
            s += static_cast<double>(y.data()[i]) * g[static_cast<size_t>(i)];
        }
        return static_cast<float>(s);
    };
    (void)objective();
    (void)embedding.backward(head.backward(Tensor(Shape({2, 3, kVocab}), &backend, g)));
    const std::vector<float> analytic = embedding.weight_grad().to_host_vector();

    const float h = 1e-3f;
    for (int64_t i = 0; i < kVocab * kD; ++i) {
        std::vector<float> plus = table, minus = table;
        plus[static_cast<size_t>(i)] += h;
        minus[static_cast<size_t>(i)] -= h;
        embedding.set_weight(plus);
        const float fp = objective();
        embedding.set_weight(minus);
        const float fm = objective();
        EXPECT_NEAR(analytic[static_cast<size_t>(i)], (fp - fm) / (2 * h), 2e-3f) << "table element " << i;
    }
}

TEST_F(TiedLMHeadModuleTest, OwnsNoParametersAndFollowsTheEmbedding) {
    TiedLMHeadModule head(embedding, &backend);
    EXPECT_TRUE(head.named_parameters().empty());

    // A table set after construction is the one the head uses.
    std::vector<float> doubled = table;
    for (float& v : doubled) {
        v *= 2.0f;
    }
    Tensor x(Shape({1, kD}), &backend, Pattern(kD, 0.3f, 0.0f));
    Tensor before = head.forward(x);
    embedding.set_weight(doubled);
    Tensor after = head.forward(x);
    for (int64_t i = 0; i < kVocab; ++i) {
        EXPECT_NEAR(after.data()[i], 2.0f * before.data()[i], 1e-5f);
    }

    // Freezing the embedding freezes the head's contribution too.
    for (NamedParamRef& p : embedding.named_parameters()) {
        p.ref.value->set_requires_grad(false);
    }
    (void)head.backward(Tensor(Shape({1, kVocab}), &backend, Pattern(kVocab, 0.5f, 0.1f)));
    for (int64_t i = 0; i < kVocab * kD; ++i) {
        EXPECT_EQ(embedding.weight_grad().data()[i], 0.0f);
    }
}

TEST_F(TiedLMHeadModuleTest, EpsilonRuleConservesRelevance) {
    TiedLMHeadModule head(embedding, &backend);
    Tensor y = head.forward(Tensor(Shape({3, kD}), &backend, Pattern(3 * kD, 0.3f, 0.2f)));
    Tensor r = head.propagate_relevance(y, LRPRuleConfig{1e-9f});
    double in = 0.0, out = 0.0;
    for (int64_t i = 0; i < r.numel(); ++i) {
        in += r.data()[i];
    }
    for (int64_t i = 0; i < y.numel(); ++i) {
        out += y.data()[i];
    }
    EXPECT_NEAR(in, out, 1e-4);
}

TEST_F(TiedLMHeadModuleTest, RejectsBadShapesAndOutOfOrderCalls) {
    TiedLMHeadModule head(embedding, &backend);
    EXPECT_THROW((void)head.backward(Tensor(Shape({1, kVocab}), &backend)), std::logic_error);
    EXPECT_THROW((void)head.propagate_relevance(Tensor(Shape({1, kVocab}), &backend), LRPRuleConfig{}),
                 std::logic_error);
    EXPECT_THROW((void)head.forward(Tensor(Shape({kD}), &backend)), std::invalid_argument);
    EXPECT_THROW((void)head.forward(Tensor(Shape({2, kD + 1}), &backend)), std::invalid_argument);
    (void)head.forward(Tensor(Shape({2, kD}), &backend));
    EXPECT_THROW((void)head.backward(Tensor(Shape({3, kVocab}), &backend)), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
