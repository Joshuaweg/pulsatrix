// LRP-rules campaign Mission 2: Zennit-compatible Epsilon (bias in z), Gamma, AlphaBeta and ZBox
// rules on LinearModule / Conv2DModule, the no-silent-fallback contract, and LRP composites.
//
// Every expected value comes either from a hand derivation (in the comments) or from
// `reference_dense`, a direct per-element transcription of the Zennit 1.0.0 formulas in double
// precision (checked against zennit.rules on torch.nn.Linear to ~1e-15 while writing this test).

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/max_pool2d_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/softmax_module.hpp"

namespace pulsatrix {
namespace {

LRPRuleConfig make_rule(LRPRule rule) {
    LRPRuleConfig c;
    c.rule = rule;
    return c;
}
LRPRuleConfig epsilon_with_bias(float eps = 1e-6f) {
    LRPRuleConfig c{eps};
    c.epsilon_bias_in_denominator = true;
    return c;
}
LRPRuleConfig alpha_beta(float alpha, float beta) {
    LRPRuleConfig c = make_rule(LRPRule::AlphaBeta);
    c.alpha = alpha;
    c.beta = beta;
    return c;
}
LRPRuleConfig gamma_rule(float gamma) {
    LRPRuleConfig c = make_rule(LRPRule::Gamma);
    c.gamma = gamma;
    return c;
}
LRPRuleConfig zbox(float low, float high) {
    LRPRuleConfig c = make_rule(LRPRule::ZBox);
    c.low = low;
    c.high = high;
    return c;
}

std::vector<LRPRuleConfig> all_rules() {
    return {epsilon_with_bias(), alpha_beta(1.0f, 0.0f), alpha_beta(2.0f, 1.0f), gamma_rule(0.25f),
            zbox(-1.5f, 2.0f)};
}

double stab(double z, double eps) { return z + eps * (z >= 0.0 ? 1.0 : -1.0); }
double pos(double v) { return v > 0.0 ? v : 0.0; }
double neg(double v) { return v < 0.0 ? v : 0.0; }

// Zennit 1.0.0 rule for a dense layer, per row: x (N, in), w (in, out), b (out), r (N, out).
std::vector<double> reference_dense(const std::vector<float>& x, const std::vector<float>& w,
                                    const std::vector<float>& b, const std::vector<float>& r, int64_t N, int64_t in,
                                    int64_t out, const LRPRuleConfig& c) {
    const double eps = c.epsilon;
    std::vector<double> r_in(static_cast<size_t>(N * in), 0.0);
    for (int64_t n = 0; n < N; ++n) {
        auto X = [&](int64_t i) { return static_cast<double>(x[n * in + i]); };
        for (int64_t j = 0; j < out; ++j) {
            auto Wt = [&](int64_t i) { return static_cast<double>(w[i * out + j]); };
            const double R = r[n * out + j];
            const double bj = b[j];
            double z = bj, zp = pos(bj), zn = neg(bj);
            const double g = c.gamma;
            double dp = bj + g * pos(bj), dn = bj + g * neg(bj), den_box = 0.0;
            for (int64_t i = 0; i < in; ++i) {
                const double wa = Wt(i) + g * pos(Wt(i)), wb = Wt(i) + g * neg(Wt(i));
                z += X(i) * Wt(i);
                zp += pos(X(i)) * pos(Wt(i)) + neg(X(i)) * neg(Wt(i));
                zn += pos(X(i)) * neg(Wt(i)) + neg(X(i)) * pos(Wt(i));
                dp += pos(X(i)) * wa + neg(X(i)) * wb;
                dn += pos(X(i)) * wb + neg(X(i)) * wa;
                den_box += X(i) * Wt(i) - c.low * pos(Wt(i)) - c.high * neg(Wt(i));
            }
            for (int64_t i = 0; i < in; ++i) {
                const double wa = Wt(i) + g * pos(Wt(i)), wb = Wt(i) + g * neg(Wt(i));
                double contribution = 0.0;
                switch (c.rule) {
                    case LRPRule::Epsilon: {
                        const double zz = c.epsilon_bias_in_denominator ? z : z - bj;
                        contribution = X(i) * Wt(i) * R / stab(zz, eps);
                        break;
                    }
                    case LRPRule::AlphaBeta:
                        contribution = c.alpha * (pos(X(i)) * pos(Wt(i)) + neg(X(i)) * neg(Wt(i))) * R / stab(zp, eps) -
                                       c.beta * (pos(X(i)) * neg(Wt(i)) + neg(X(i)) * pos(Wt(i))) * R / stab(zn, eps);
                        break;
                    case LRPRule::Gamma:
                        contribution = (z > 0.0 ? (pos(X(i)) * wa + neg(X(i)) * wb) * R / stab(dp, eps) : 0.0) +
                                       (z < 0.0 ? (pos(X(i)) * wb + neg(X(i)) * wa) * R / stab(dn, eps) : 0.0);
                        break;
                    case LRPRule::ZBox:
                        contribution =
                            (X(i) * Wt(i) - c.low * pos(Wt(i)) - c.high * neg(Wt(i))) * R / stab(den_box, eps);
                        break;
                }
                r_in[static_cast<size_t>(n * in + i)] += contribution;
            }
        }
    }
    return r_in;
}

void expect_close(const Tensor& actual, const std::vector<double>& expected, double tol = 1e-4) {
    ASSERT_EQ(static_cast<size_t>(actual.numel()), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(actual.data()[i], expected[i], tol * std::max(1.0, std::fabs(expected[i]))) << "index " << i;
    }
}

class LRPRulesTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // One output: W = [1, -2, -1]^T, b = 0.5, x = [2, -1, 1]  ->  xW = 3, z = 3.5, R = 1.
    // x+ = [2, 0, 1], x- = [0, -1, 0], W+ = [1, 0, 0], W- = [0, -2, -1].
    Tensor single_output(const LRPRuleConfig& config) {
        LinearModule lin(3, 1, &backend);
        lin.set_weight({1.0f, -2.0f, -1.0f});
        lin.set_bias({0.5f});
        Tensor x(Shape({1, 3}), &backend, {2.0f, -1.0f, 1.0f});
        (void)lin.forward(x);
        return lin.propagate_relevance(Tensor(Shape({1, 1}), &backend, {1.0f}), config);
    }

    // A batch-of-2, 4 -> 3 layer with mixed signs everywhere and no near-zero outputs.
    const std::vector<float> x_ = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f, 2.0f, -0.75f};
    const std::vector<float> w_ = {0.5f, -1.0f, 0.75f, -0.25f, 2.0f, -1.5f, 1.25f, 0.5f, -0.5f, -0.75f, 0.25f, 1.0f};
    const std::vector<float> b_ = {0.3f, -0.2f, 0.1f};
    const std::vector<float> r_ = {1.0f, -0.5f, 2.0f, 0.25f, 1.5f, -1.0f};

    Tensor batch_linear(const LRPRuleConfig& config, const std::vector<float>& bias) {
        LinearModule lin(4, 3, &backend);
        lin.set_weight(w_);
        lin.set_bias(bias);
        (void)lin.forward(Tensor(Shape({2, 4}), &backend, x_));
        return lin.propagate_relevance(Tensor(Shape({2, 3}), &backend, r_), config);
    }
};

// ---- Hand-computed single-output values ----------------------------------------------------------

TEST_F(LRPRulesTest, EpsilonWithBiasDividesByTheBiasedOutput) {
    // x * W / z = [2, 2, -1] / 3.5
    expect_close(single_output(epsilon_with_bias()), {2.0 / 3.5, 2.0 / 3.5, -1.0 / 3.5});
    // The default (pre-bias) rule is unchanged: [2, 2, -1] / 3.
    expect_close(single_output(LRPRuleConfig{}), {2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0});
}

TEST_F(LRPRulesTest, AlphaBetaMatchesHandComputedValues) {
    // z+ = x+W+ (2) + b+ (0.5) + x-W- (2) = 4.5; positive terms x+W+ + x-W- = [2, 2, 0].
    // z- = x+W- (-1) + b- (0) + x-W+ (0) = -1; negative terms x+W- + x-W+ = [0, 0, -1].
    expect_close(single_output(alpha_beta(1.0f, 0.0f)), {2.0 / 4.5, 2.0 / 4.5, 0.0});
    // alpha 2, beta 1: 2 * [2, 2, 0] / 4.5 - [0, 0, -1] / -1.
    expect_close(single_output(alpha_beta(2.0f, 1.0f)), {4.0 / 4.5, 4.0 / 4.5, -1.0});
}

TEST_F(LRPRulesTest, GammaMatchesHandComputedValues) {
    // gamma 0.5: Wa = W + 0.5 W+ = [1.5, -2, -1], Wb = W + 0.5 W- = [1, -3, -1.5], ba = 0.75.
    // z = 3.5 > 0, so only the positive pair: o0 = x+Wa + ba = 3 - 1 + 0.75 = 2.75,
    // o1 = x-Wb = 3; R_in = (x+ * Wa + x- * Wb) / 5.75 = [3, 3, -1] / 5.75.
    expect_close(single_output(gamma_rule(0.5f)), {3.0 / 5.75, 3.0 / 5.75, -1.0 / 5.75});
}

TEST_F(LRPRulesTest, GammaUsesTheNegativePairForNegativeOutputs) {
    // W = [-1, 1, 0.5]^T, b = -0.5, x = [2, -1, 1]: z = -2 - 1 + 0.5 - 0.5 = -3 < 0.
    // gamma 0.5: Wa = [-1, 1.5, 0.75], Wb = [-1.5, 1, 0.5], bb = -0.5 + 0.5 * -0.5 = -0.75.
    // o2 = x+Wb + bb = -3 + 0.5 - 0.75 = -3.25; o3 = x-Wa = -1.5; den = -4.75.
    // R_in = (x+ * Wb + x- * Wa) * R / den = [-3, -1.5, 0.5] * 2 / -4.75.
    LinearModule lin(3, 1, &backend);
    lin.set_weight({-1.0f, 1.0f, 0.5f});
    lin.set_bias({-0.5f});
    (void)lin.forward(Tensor(Shape({1, 3}), &backend, {2.0f, -1.0f, 1.0f}));
    Tensor r = lin.propagate_relevance(Tensor(Shape({1, 1}), &backend, {2.0f}), gamma_rule(0.5f));
    expect_close(r, {-3.0 * 2.0 / -4.75, -1.5 * 2.0 / -4.75, 0.5 * 2.0 / -4.75});
}

TEST_F(LRPRulesTest, ZBoxMatchesHandComputedValues) {
    // low -1, high 2: den = xW - L W+ - H W- = 3 + 1 + 6 = 10 (the bias cancels);
    // R_in = (x W - L W+ - H W-) / 10 = ([2, 2, -1] - [-1, 0, 0] - [0, -4, -2]) / 10 = [0.3, 0.6, 0.1].
    expect_close(single_output(zbox(-1.0f, 2.0f)), {0.3, 0.6, 0.1});
}

// ---- Batched layer against the reference transcription -------------------------------------------

TEST_F(LRPRulesTest, EveryRuleMatchesTheReferenceOnABatch) {
    for (const LRPRuleConfig& config : all_rules()) {
        SCOPED_TRACE(lrp_rule_name(config.rule));
        expect_close(batch_linear(config, b_), reference_dense(x_, w_, b_, r_, 2, 4, 3, config));
    }
}

TEST_F(LRPRulesTest, AlphaBetaOneZeroIsZPlusOnNonNegativeInputs) {
    // ZPlus (Zennit) on x >= 0: R_i = sum_j x_i W+_ij / (x W+ + b+)_j * R_j.
    LinearModule lin(3, 2, &backend);
    lin.set_weight({1.0f, -2.0f, -0.5f, 3.0f, 2.0f, 1.0f});
    lin.set_bias({0.5f, -1.0f});
    (void)lin.forward(Tensor(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f}));
    Tensor r = lin.propagate_relevance(Tensor(Shape({1, 2}), &backend, {1.0f, 3.0f}), alpha_beta(1.0f, 0.0f));
    // Column 0: W+ = [1, 0, 2], b+ = 0.5 -> den = 1 + 0 + 1 + 0.5 = 2.5, shares [1, 0, 1] / 2.5.
    // Column 1: W+ = [0, 3, 1], b+ = 0 -> den = 6 + 0.5 = 6.5, shares [0, 6, 0.5] / 6.5 * 3.
    expect_close(r, {1.0 / 2.5, 18.0 / 6.5, 1.0 / 2.5 + 1.5 / 6.5});
}

TEST_F(LRPRulesTest, AlphaBetaConservesWithoutBias) {
    // Without bias, and with every unit having both positive and negative contributions
    // (z+ != 0, z- != 0), sum_i positive terms = R and sum_i negative terms = R (up to eps), so
    // sum R_in = (alpha - beta) R = R. Positive inputs and mixed-sign weight columns ensure that.
    const std::vector<float> positive_x = {1.0f, 2.0f, 0.5f, 3.0f, 1.5f, 0.25f, 2.0f, 0.75f};
    for (const LRPRuleConfig& config : {alpha_beta(1.0f, 0.0f), alpha_beta(2.0f, 1.0f), alpha_beta(3.0f, 2.0f)}) {
        LinearModule lin(4, 3, &backend);
        lin.set_weight(w_);
        lin.set_bias({0.0f, 0.0f, 0.0f});
        (void)lin.forward(Tensor(Shape({2, 4}), &backend, positive_x));
        Tensor r = lin.propagate_relevance(Tensor(Shape({2, 3}), &backend, r_), config);
        for (int64_t n = 0; n < 2; ++n) {
            double in_sum = 0.0, out_sum = 0.0;
            for (int64_t i = 0; i < 4; ++i) in_sum += r.data()[n * 4 + i];
            for (int64_t j = 0; j < 3; ++j) out_sum += r_[static_cast<size_t>(n * 3 + j)];
            EXPECT_NEAR(in_sum, out_sum, 1e-4) << "alpha " << config.alpha << " row " << n;
        }
    }
}

TEST_F(LRPRulesTest, GammaZeroIsEpsilonWithBiasWhereOutputsAreNonzero) {
    Tensor gamma0 = batch_linear(gamma_rule(0.0f), b_);
    Tensor epsilon = batch_linear(epsilon_with_bias(), b_);
    for (int64_t i = 0; i < gamma0.numel(); ++i) {
        EXPECT_NEAR(gamma0.data()[i], epsilon.data()[i], 1e-5f) << i;
    }
}

TEST_F(LRPRulesTest, ZBoxWithZeroBoxIsThePreBiasEpsilonRule) {
    // low = high = 0: den = xW and R_in = x (g W^T) -- the bias has cancelled out, which is
    // exactly the default pre-bias epsilon rule.
    Tensor box = batch_linear(zbox(0.0f, 0.0f), b_);
    Tensor epsilon = batch_linear(LRPRuleConfig{}, b_);
    for (int64_t i = 0; i < box.numel(); ++i) {
        EXPECT_NEAR(box.data()[i], epsilon.data()[i], 1e-5f) << i;
    }
}

TEST_F(LRPRulesTest, InvalidAlphaBetaThrows) {
    EXPECT_THROW((void)single_output(alpha_beta(2.0f, 0.0f)), std::invalid_argument);   // alpha - beta != 1
    EXPECT_THROW((void)single_output(alpha_beta(-1.0f, -2.0f)), std::invalid_argument);  // negative alpha/beta
    EXPECT_THROW((void)single_output(alpha_beta(0.5f, -0.5f)), std::invalid_argument);   // negative beta
    Conv2DModule conv(1, 1, 2, 2, &backend);
    (void)conv.forward(Tensor(Shape({1, 1, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f}));
    try {
        (void)conv.propagate_relevance(Tensor(Shape({1, 1, 1, 1}), &backend, {1.0f}), alpha_beta(1.0f, 1.0f));
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("Conv2DModule"), std::string::npos);
    }
}

TEST_F(LRPRulesTest, AggregateInitStillMeansEpsilon) {
    const LRPRuleConfig config{0.01f};
    EXPECT_EQ(config.rule, LRPRule::Epsilon);
    EXPECT_FLOAT_EQ(config.epsilon, 0.01f);
    EXPECT_FALSE(config.epsilon_bias_in_denominator);
}

// ---- Conv2D -------------------------------------------------------------------------------------

// A conv whose kernel covers the whole input has one patch, equal to the flattened input
// (im2col order c, kh, kw == row-major (C, H, W)): every rule must equal the Linear rule with
// W_lin[p][oc] = K[oc][p].
TEST_F(LRPRulesTest, SinglePatchConvEqualsLinear) {
    const std::vector<float> input = {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f, 2.0f, -0.75f};  // (1, 2, 2, 2)
    std::vector<float> kernel(16), w_lin(16);
    for (int i = 0; i < 16; ++i) kernel[static_cast<size_t>(i)] = 0.25f * static_cast<float>((i * 7) % 11) - 1.2f;
    for (int oc = 0; oc < 2; ++oc) {
        for (int p = 0; p < 8; ++p) w_lin[static_cast<size_t>(p * 2 + oc)] = kernel[static_cast<size_t>(oc * 8 + p)];
    }
    const std::vector<float> bias = {0.4f, -0.3f};
    const std::vector<float> r = {1.5f, -0.5f};
    for (const LRPRuleConfig& config : all_rules()) {
        SCOPED_TRACE(lrp_rule_name(config.rule));
        Conv2DModule conv(2, 2, 2, 2, &backend);
        conv.set_kernel(kernel);
        conv.set_bias(bias);
        (void)conv.forward(Tensor(Shape({1, 2, 2, 2}), &backend, input));
        Tensor rc = conv.propagate_relevance(Tensor(Shape({1, 2, 1, 1}), &backend, r), config);
        LinearModule lin(8, 2, &backend);
        lin.set_weight(w_lin);
        lin.set_bias(bias);
        (void)lin.forward(Tensor(Shape({1, 8}), &backend, input));
        Tensor rl = lin.propagate_relevance(Tensor(Shape({1, 2}), &backend, r), config);
        for (int i = 0; i < 8; ++i) {
            EXPECT_NEAR(rc.data()[i], rl.data()[i], 1e-5f) << i;
        }
    }
}

// Overlapping windows, a batch of 2, 2 -> 3 channels: the rule is applied per patch (each output
// position is an independent dense layer over its patch) and the patch relevances are summed back
// into the input pixels.
TEST_F(LRPRulesTest, MultiPositionConvMatchesPerPatchReference) {
    const int64_t N = 2, C = 2, H = 4, W = 4, OC = 3, KH = 2, KW = 2, OH = 3, OW = 3, P = C * KH * KW, Q = OH * OW;
    std::vector<float> input(static_cast<size_t>(N * C * H * W)), kernel(static_cast<size_t>(OC * P));
    std::vector<float> r(static_cast<size_t>(N * OC * Q));
    for (size_t i = 0; i < input.size(); ++i) input[i] = 0.3f * static_cast<float>((i * 5) % 13) - 1.7f;
    for (size_t i = 0; i < kernel.size(); ++i) kernel[i] = 0.2f * static_cast<float>((i * 3) % 7) - 0.55f;
    for (size_t i = 0; i < r.size(); ++i) r[i] = 0.1f * static_cast<float>((i * 11) % 9) - 0.35f;
    const std::vector<float> bias = {0.15f, -0.25f, 0.05f};
    std::vector<float> w_patch(static_cast<size_t>(P * OC));
    for (int64_t oc = 0; oc < OC; ++oc) {
        for (int64_t p = 0; p < P; ++p) {
            w_patch[static_cast<size_t>(p * OC + oc)] = kernel[static_cast<size_t>(oc * P + p)];
        }
    }

    for (const LRPRuleConfig& config : all_rules()) {
        SCOPED_TRACE(lrp_rule_name(config.rule));
        std::vector<double> expected(input.size(), 0.0);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t q = 0; q < Q; ++q) {
                const int64_t oy = q / OW, ox = q % OW;
                std::vector<float> patch(static_cast<size_t>(P)), r_patch(static_cast<size_t>(OC));
                auto pixel = [&](int64_t p) {
                    const int64_t c = p / (KH * KW), ky = (p / KW) % KH, kx = p % KW;
                    return ((n * C + c) * H + oy + ky) * W + ox + kx;
                };
                for (int64_t p = 0; p < P; ++p) patch[static_cast<size_t>(p)] = input[static_cast<size_t>(pixel(p))];
                for (int64_t oc = 0; oc < OC; ++oc) {
                    r_patch[static_cast<size_t>(oc)] = r[static_cast<size_t>((n * OC + oc) * Q + q)];
                }
                const std::vector<double> rp = reference_dense(patch, w_patch, bias, r_patch, 1, P, OC, config);
                for (int64_t p = 0; p < P; ++p) expected[static_cast<size_t>(pixel(p))] += rp[static_cast<size_t>(p)];
            }
        }
        Conv2DModule conv(C, OC, KH, KW, &backend);
        conv.set_kernel(kernel);
        conv.set_bias(bias);
        (void)conv.forward(Tensor(Shape({N, C, H, W}), &backend, input));
        Tensor actual = conv.propagate_relevance(Tensor(Shape({N, OC, OH, OW}), &backend, r), config);
        expect_close(actual, expected, 2e-4);
    }
}

// ---- No silent fallback ---------------------------------------------------------------------------

TEST_F(LRPRulesTest, EpsilonOnlyModuleRejectsOtherRules) {
    LinearModule lin(3, 2, &backend);
    lin.set_weight({1.0f, 0.5f, -1.0f, 2.0f, 0.25f, -0.5f});
    SoftmaxModule softmax(&backend);
    ExplainerContext ctx({&lin, &softmax});
    Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor y = ctx.forward_pass(x);
    LRPRuleConfig gamma = gamma_rule(0.25f);
    try {
        (void)ctx.relevance_pass(y, gamma);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("SoftmaxModule"), std::string::npos) << what;
        EXPECT_NE(what.find("gamma"), std::string::npos) << what;
    }
    EXPECT_THROW((void)LRP(gamma).explain(ctx, x, 0, &backend), std::invalid_argument);
    EXPECT_NO_THROW((void)ctx.relevance_pass(y, LRPRuleConfig{}));  // epsilon still works
    EXPECT_FALSE(softmax.supports_lrp_rule(LRPRule::ZBox));
    EXPECT_TRUE(softmax.supports_lrp_rule(LRPRule::Epsilon));
}

TEST_F(LRPRulesTest, PassThroughModulesWorkWithEveryRule) {
    Conv2DModule conv(1, 2, 2, 2, &backend);
    conv.set_kernel({0.5f, -1.0f, 0.75f, 0.25f, -0.5f, 1.0f, 0.5f, -0.25f});
    conv.set_bias({0.1f, -0.1f});
    ReluModule relu(&backend);
    DropoutModule dropout(0.5f, &backend);
    dropout.set_training(false);
    MaxPool2DModule pool(2, 2, &backend);
    FlattenModule flatten(&backend);
    LinearModule lin(2, 2, &backend);
    lin.set_weight({1.0f, -0.5f, 0.25f, 2.0f});
    lin.set_bias({0.05f, -0.05f});
    ExplainerContext ctx({&conv, &relu, &dropout, &pool, &flatten, &lin});
    Tensor x(Shape({1, 1, 3, 3}), &backend, {1.0f, -2.0f, 0.5f, 3.0f, -1.5f, 0.25f, 2.0f, -0.75f, 1.25f});
    for (const LRPRuleConfig& config : all_rules()) {
        SCOPED_TRACE(lrp_rule_name(config.rule));
        Tensor y = ctx.forward_pass(x);
        Tensor seed(y.shape(), &backend, {1.0f, 0.0f});
        Tensor via_ctx = ctx.relevance_pass(seed, config);
        // Manual chain: the pass-through modules hand relevance on unchanged (reshaped/unpooled).
        Tensor r = lin.propagate_relevance(seed, config);
        r = flatten.propagate_relevance(r, config);
        r = pool.propagate_relevance(r, config);
        r = dropout.propagate_relevance(r, config);
        r = relu.propagate_relevance(r, config);
        r = conv.propagate_relevance(r, config);
        ASSERT_EQ(via_ctx.shape(), x.shape());
        for (int64_t i = 0; i < r.numel(); ++i) EXPECT_FLOAT_EQ(via_ctx.data()[i], r.data()[i]);
    }
}

TEST_F(LRPRulesTest, SequentialSupportsARuleOnlyIfEveryLayerDoes) {
    LinearModule lin(2, 2, &backend);
    ReluModule relu(&backend);
    SoftmaxModule softmax(&backend);
    SequentialModule affine({&lin, &relu});
    SequentialModule with_softmax({&lin, &softmax});
    EXPECT_TRUE(affine.supports_lrp_rule(LRPRule::Gamma));
    EXPECT_FALSE(with_softmax.supports_lrp_rule(LRPRule::Gamma));
    EXPECT_TRUE(with_softmax.supports_lrp_rule(LRPRule::Epsilon));
}

// ---- Composites ---------------------------------------------------------------------------------

class LRPCompositeTest : public LRPRulesTest {
protected:
    // conv1 -> relu -> conv2 -> relu -> flatten -> linear
    Conv2DModule conv1{1, 2, 2, 2, &backend};
    ReluModule relu1{&backend};
    Conv2DModule conv2{2, 2, 2, 2, &backend};
    ReluModule relu2{&backend};
    FlattenModule flatten{&backend};
    LinearModule lin{2, 3, &backend};
    Tensor x{Shape({1, 1, 3, 3}), &backend, {0.9f, 0.1f, 0.5f, 0.3f, 0.7f, 0.2f, 0.8f, 0.4f, 0.6f}};

    void SetUp() override {
        conv1.set_kernel({0.5f, -0.3f, 0.8f, 0.2f, -0.4f, 0.6f, 0.3f, 0.7f});
        conv1.set_bias({0.05f, 0.1f});
        conv2.set_kernel({0.4f, -0.2f, 0.3f, 0.5f, -0.6f, 0.2f, 0.1f, 0.4f,
                          0.3f, 0.6f, -0.5f, 0.2f, 0.4f, -0.1f, 0.7f, 0.2f});
        conv2.set_bias({0.02f, -0.03f});
        lin.set_weight({1.0f, -0.5f, 0.3f, 0.6f, 0.8f, -0.4f});
        lin.set_bias({0.1f, 0.0f, -0.1f});
    }
    std::vector<Module*> chain() { return {&conv1, &relu1, &conv2, &relu2, &flatten, &lin}; }

    // Forward, then each module's propagate_relevance with its own config, in reverse.
    Tensor manual(const std::vector<LRPRuleConfig>& configs, int64_t target) {
        ExplainerContext ctx(chain());
        Tensor y = ctx.forward_pass(x);
        std::vector<float> seed(static_cast<size_t>(y.numel()), 0.0f);
        seed[static_cast<size_t>(target)] = y.data()[target];
        Tensor r(y.shape(), &backend, seed);
        std::vector<Module*> modules = chain();
        for (size_t i = modules.size(); i-- > 0;) r = modules[i]->propagate_relevance(r, configs[i]);
        return r;
    }
};

TEST_F(LRPCompositeTest, PresetsAssignTheDocumentedRules) {
    ExplainerContext ctx(chain());
    struct Case {
        LRP lrp;
        std::string name;
        std::string rules;
    };
    std::vector<Case> cases = {
        {LRP::epsilon_plus(), "composite:epsilon_plus", "alpha_beta,epsilon,alpha_beta,epsilon,epsilon,epsilon"},
        {LRP::epsilon_alpha2_beta1(), "composite:epsilon_alpha2_beta1",
         "alpha_beta,epsilon,alpha_beta,epsilon,epsilon,epsilon"},
        {LRP::epsilon_gamma_box(0.0f, 1.0f), "composite:epsilon_gamma_box",
         "zbox,epsilon,gamma,epsilon,epsilon,epsilon"},
    };
    for (Case& c : cases) {
        Attribution a = c.lrp.explain(ctx, x, 1, &backend);
        EXPECT_EQ(a.metadata.at("rule"), c.name);
        EXPECT_EQ(a.metadata.at("rules"), c.rules);
        // Running twice gives the same rules (epsilon_gamma_box's first-layer state resets).
        EXPECT_EQ(c.lrp.explain(ctx, x, 1, &backend).metadata.at("rules"), c.rules);
    }
    // A Linear-first network: ZBox lands on the first Linear, Epsilon on the others.
    LinearModule a(3, 2, &backend), b(2, 2, &backend);
    ReluModule relu(&backend);
    ExplainerContext dense({&a, &relu, &b});
    Attribution d = LRP::epsilon_gamma_box(-1.0f, 1.0f).explain(dense, Tensor(Shape({1, 3}), &backend,
                                                                             {0.5f, -0.5f, 0.25f}), 0, &backend);
    EXPECT_EQ(d.metadata.at("rules"), "zbox,epsilon,epsilon");
    // The uniform LRP reports its own rule.
    EXPECT_EQ(LRP(gamma_rule(0.25f)).explain(ctx, x, 1, &backend).metadata.at("rules"),
              "gamma,gamma,gamma,gamma,gamma,gamma");
}

TEST_F(LRPCompositeTest, PresetConfigsCarryZennitParameters) {
    const LRPComposite box = lrp_composite::epsilon_gamma_box(-2.0f, 3.0f, 0.5f, 1e-5f);
    const LRPRuleConfig c0 = box(0, conv1), c2 = box(2, conv2), c5 = box(5, lin), c1 = box(1, relu1);
    EXPECT_EQ(c0.rule, LRPRule::ZBox);
    EXPECT_FLOAT_EQ(c0.low, -2.0f);
    EXPECT_FLOAT_EQ(c0.high, 3.0f);
    EXPECT_EQ(c2.rule, LRPRule::Gamma);
    EXPECT_FLOAT_EQ(c2.gamma, 0.5f);
    EXPECT_FLOAT_EQ(c2.epsilon, 1e-5f);
    EXPECT_EQ(c5.rule, LRPRule::Epsilon);
    EXPECT_TRUE(c5.epsilon_bias_in_denominator);
    EXPECT_EQ(c1.rule, LRPRule::Epsilon);
    const LRPRuleConfig ab = lrp_composite::epsilon_alpha2_beta1()(0, conv1);
    EXPECT_EQ(ab.rule, LRPRule::AlphaBeta);
    EXPECT_FLOAT_EQ(ab.alpha, 2.0f);
    EXPECT_FLOAT_EQ(ab.beta, 1.0f);
    const LRPRuleConfig zplus = lrp_composite::epsilon_plus()(0, conv1);
    EXPECT_FLOAT_EQ(zplus.alpha, 1.0f);
    EXPECT_FLOAT_EQ(zplus.beta, 0.0f);
    EXPECT_TRUE(lrp_composite::epsilon_plus()(5, lin).epsilon_bias_in_denominator);
}

TEST_F(LRPCompositeTest, CompositeExplainEqualsManualPerModuleCalls) {
    std::vector<std::pair<LRP, LRPComposite>> cases = {
        {LRP::epsilon_plus(), lrp_composite::epsilon_plus()},
        {LRP::epsilon_alpha2_beta1(), lrp_composite::epsilon_alpha2_beta1()},
        {LRP::epsilon_gamma_box(0.0f, 1.0f), lrp_composite::epsilon_gamma_box(0.0f, 1.0f)},
    };
    for (auto& [lrp, composite] : cases) {
        std::vector<Module*> modules = chain();
        std::vector<LRPRuleConfig> configs;
        for (size_t i = 0; i < modules.size(); ++i) configs.push_back(composite(i, *modules[i]));
        Tensor expected = manual(configs, 2);
        ExplainerContext ctx(chain());
        Tensor actual = lrp.explain(ctx, x, 2, &backend).values;
        ASSERT_EQ(actual.shape(), expected.shape());
        for (int64_t i = 0; i < actual.numel(); ++i) EXPECT_FLOAT_EQ(actual.data()[i], expected.data()[i]) << i;
    }
}

TEST_F(LRPCompositeTest, CustomLambdaCompositeChoosesPerLayer) {
    LRPComposite custom = [](size_t index, const Module&) {
        if (index == 0) return zbox(0.0f, 1.0f);
        if (index == 2) return gamma_rule(0.1f);
        return epsilon_with_bias(1e-4f);
    };
    ExplainerContext ctx(chain());
    Attribution a = LRP(custom, "mine").explain(ctx, x, 0, &backend);
    EXPECT_EQ(a.metadata.at("rule"), "composite:mine");
    EXPECT_EQ(a.metadata.at("rules"), "zbox,epsilon,gamma,epsilon,epsilon,epsilon");
    std::vector<LRPRuleConfig> configs;
    std::vector<Module*> modules = chain();
    for (size_t i = 0; i < modules.size(); ++i) configs.push_back(custom(i, *modules[i]));
    Tensor expected = manual(configs, 0);
    for (int64_t i = 0; i < expected.numel(); ++i) EXPECT_FLOAT_EQ(a.values.data()[i], expected.data()[i]);

    // A composite that asks an epsilon-only module for Gamma is rejected, not silently run.
    SoftmaxModule softmax(&backend);
    ExplainerContext with_softmax({&lin, &softmax});
    LRPComposite all_gamma = [](size_t, const Module&) { return gamma_rule(0.25f); };
    EXPECT_THROW((void)LRP(all_gamma).explain(with_softmax, Tensor(Shape({1, 2}), &backend, {0.5f, 0.25f}), 0,
                                               &backend),
                 std::invalid_argument);
    EXPECT_THROW(LRP(LRPComposite{}), std::invalid_argument);
}

TEST_F(LRPCompositeTest, ConfigVectorSizeMustMatchTheModuleCount) {
    ExplainerContext ctx(chain());
    Tensor y = ctx.forward_pass(x);
    EXPECT_THROW((void)ctx.relevance_pass(y, std::vector<LRPRuleConfig>(2)), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
