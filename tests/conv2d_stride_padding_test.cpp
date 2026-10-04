#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

std::vector<float> random_values(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

struct Geometry {
    int64_t n, in_c, out_c, h, w, k, stride, pad;
    int64_t out_h() const { return (h + 2 * pad - k) / stride + 1; }
    int64_t out_w() const { return (w + 2 * pad - k) / stride + 1; }
};

// Naive direct convolution, independent of im2col: out[e,o,oh,ow] = b[o] + sum over (c, i, j)
// of w[o,c,i,j] * x[e,c,oh*s-p+i, ow*s-p+j], with out-of-range input reading as zero.
std::vector<double> reference_forward(const Geometry& g, const std::vector<float>& x, const std::vector<float>& kernel,
                                      const std::vector<float>& bias) {
    std::vector<double> out(static_cast<size_t>(g.n * g.out_c * g.out_h() * g.out_w()));
    for (int64_t e = 0; e < g.n; ++e)
        for (int64_t o = 0; o < g.out_c; ++o)
            for (int64_t oh = 0; oh < g.out_h(); ++oh)
                for (int64_t ow = 0; ow < g.out_w(); ++ow) {
                    double acc = bias[o];
                    for (int64_t c = 0; c < g.in_c; ++c)
                        for (int64_t i = 0; i < g.k; ++i)
                            for (int64_t j = 0; j < g.k; ++j) {
                                const int64_t ih = oh * g.stride - g.pad + i, iw = ow * g.stride - g.pad + j;
                                if (ih < 0 || ih >= g.h || iw < 0 || iw >= g.w) continue;
                                acc += static_cast<double>(kernel[((o * g.in_c + c) * g.k + i) * g.k + j]) *
                                       x[((e * g.in_c + c) * g.h + ih) * g.w + iw];
                            }
                    out[((e * g.out_c + o) * g.out_h() + oh) * g.out_w() + ow] = acc;
                }
    return out;
}

// The adjoint of reference_forward: input and kernel gradients for upstream gradient dy.
void reference_backward(const Geometry& g, const std::vector<float>& x, const std::vector<float>& kernel,
                        const std::vector<float>& dy, std::vector<double>& dx, std::vector<double>& dk) {
    dx.assign(x.size(), 0.0);
    dk.assign(kernel.size(), 0.0);
    for (int64_t e = 0; e < g.n; ++e)
        for (int64_t o = 0; o < g.out_c; ++o)
            for (int64_t oh = 0; oh < g.out_h(); ++oh)
                for (int64_t ow = 0; ow < g.out_w(); ++ow) {
                    const double d = dy[((e * g.out_c + o) * g.out_h() + oh) * g.out_w() + ow];
                    for (int64_t c = 0; c < g.in_c; ++c)
                        for (int64_t i = 0; i < g.k; ++i)
                            for (int64_t j = 0; j < g.k; ++j) {
                                const int64_t ih = oh * g.stride - g.pad + i, iw = ow * g.stride - g.pad + j;
                                if (ih < 0 || ih >= g.h || iw < 0 || iw >= g.w) continue;
                                const size_t xi = static_cast<size_t>(((e * g.in_c + c) * g.h + ih) * g.w + iw);
                                const size_t ki = static_cast<size_t>(((o * g.in_c + c) * g.k + i) * g.k + j);
                                dx[xi] += d * kernel[ki];
                                dk[ki] += d * x[xi];
                            }
                }
}

const Geometry kGeometries[] = {
    {2, 3, 4, 7, 6, 3, 2, 1},  // ResNet-style 3x3 stride 2 pad 1, odd sizes
    {1, 2, 3, 5, 5, 3, 1, 1},  // "same" padding
    {2, 2, 3, 6, 7, 1, 2, 0},  // ResNet downsample 1x1 stride 2: some inputs are skipped
    {1, 1, 2, 9, 8, 7, 2, 3},  // ResNet stem 7x7 stride 2 pad 3
    {1, 2, 2, 5, 4, 2, 3, 2},  // stride larger than the kernel, padding larger than half the kernel
};

class Conv2DStridePaddingTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Conv2DModule make(const Geometry& g, unsigned seed) {
        Conv2DModule conv(g.in_c, g.out_c, g.k, g.k, &backend, g.stride, g.pad);
        conv.set_kernel(random_values(static_cast<size_t>(g.out_c * g.in_c * g.k * g.k), seed));
        conv.set_bias(random_values(static_cast<size_t>(g.out_c), seed + 1));
        return conv;
    }
};

TEST_F(Conv2DStridePaddingTest, OutputShapeFollowsTheStandardFormula) {
    Conv2DModule stem(1, 2, 7, 7, &backend, 2, 3);
    EXPECT_EQ(stem.forward(Tensor(Shape({1, 1, 8, 8}), &backend)).shape(), Shape({1, 2, 4, 4}));
    Conv2DModule same(1, 2, 3, 3, &backend, 1, 1);
    EXPECT_EQ(same.forward(Tensor(Shape({1, 1, 5, 5}), &backend)).shape(), Shape({1, 2, 5, 5}));
    Conv2DModule down(1, 2, 1, 1, &backend, 2, 0);
    EXPECT_EQ(down.forward(Tensor(Shape({1, 1, 5, 5}), &backend)).shape(), Shape({1, 2, 3, 3}));
}

TEST_F(Conv2DStridePaddingTest, ForwardMatchesDirectConvolution) {
    unsigned seed = 10;
    for (const Geometry& g : kGeometries) {
        Conv2DModule conv = make(g, seed);
        std::vector<float> x = random_values(static_cast<size_t>(g.n * g.in_c * g.h * g.w), seed + 2);
        Tensor y = conv.forward(Tensor(Shape({g.n, g.in_c, g.h, g.w}), &backend, x));
        ASSERT_EQ(y.shape(), Shape({g.n, g.out_c, g.out_h(), g.out_w()}));
        std::vector<double> expected = reference_forward(g, x, values_of(conv.kernel()), values_of(conv.bias()));
        for (size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(y.data()[i], expected[i], 1e-5) << "seed " << seed;
        seed += 10;
    }
}

TEST_F(Conv2DStridePaddingTest, BackwardMatchesTheAdjointOfDirectConvolution) {
    unsigned seed = 100;
    for (const Geometry& g : kGeometries) {
        Conv2DModule conv = make(g, seed);
        std::vector<float> x = random_values(static_cast<size_t>(g.n * g.in_c * g.h * g.w), seed + 2);
        std::vector<float> dy = random_values(static_cast<size_t>(g.n * g.out_c * g.out_h() * g.out_w()), seed + 3);
        (void)conv.forward(Tensor(Shape({g.n, g.in_c, g.h, g.w}), &backend, x));
        Tensor dx = conv.backward(Tensor(Shape({g.n, g.out_c, g.out_h(), g.out_w()}), &backend, dy));
        std::vector<double> ref_dx, ref_dk;
        reference_backward(g, x, values_of(conv.kernel()), dy, ref_dx, ref_dk);
        for (size_t i = 0; i < ref_dx.size(); ++i) EXPECT_NEAR(dx.data()[i], ref_dx[i], 1e-5) << "seed " << seed;
        const Tensor& dk = *conv.parameters()[0].grad;
        for (size_t i = 0; i < ref_dk.size(); ++i) EXPECT_NEAR(dk.data()[i], ref_dk[i], 1e-5) << "seed " << seed;
        seed += 10;
    }
}

TEST_F(Conv2DStridePaddingTest, StridedOutputIsTheSubsampledStrideOneOutput) {
    Conv2DModule strided(2, 3, 3, 3, &backend, 2, 1), dense(2, 3, 3, 3, &backend, 1, 1);
    std::vector<float> kernel = random_values(3 * 2 * 3 * 3, 7), bias = random_values(3, 8);
    strided.set_kernel(kernel);
    strided.set_bias(bias);
    dense.set_kernel(kernel);
    dense.set_bias(bias);
    Tensor x(Shape({1, 2, 7, 7}), &backend, random_values(2 * 7 * 7, 9));
    Tensor ys = strided.forward(x), yd = dense.forward(x);
    ASSERT_EQ(ys.shape(), Shape({1, 3, 4, 4}));
    for (int64_t o = 0; o < 3; ++o)
        for (int64_t oh = 0; oh < 4; ++oh)
            for (int64_t ow = 0; ow < 4; ++ow)
                EXPECT_EQ(ys.data()[(o * 4 + oh) * 4 + ow], yd.data()[(o * 7 + 2 * oh) * 7 + 2 * ow]);
}

std::vector<LRPRuleConfig> zero_input_safe_rules() {
    std::vector<LRPRuleConfig> rules(4);
    rules[0].rule = LRPRule::Epsilon;  // the original pre-bias rule
    rules[1].rule = LRPRule::Epsilon;
    rules[1].epsilon_bias_in_denominator = true;
    rules[2].rule = LRPRule::AlphaBeta;
    rules[2].alpha = 2.0f;
    rules[2].beta = 1.0f;
    rules[3].rule = LRPRule::Gamma;
    return rules;
}

// Padding is zeros, and these rules give a zero input zero relevance: a padded conv must
// explain exactly like an unpadded conv run on an explicitly zero-padded input, cropped.
TEST_F(Conv2DStridePaddingTest, LRPOnPaddedConvEqualsLRPOnExplicitlyPaddedInput) {
    const int64_t pad = 1, h = 5, w = 6, ph = h + 2 * pad, pw = w + 2 * pad;
    Conv2DModule padded(2, 3, 3, 3, &backend, 2, pad), explicit_pad(2, 3, 3, 3, &backend, 2, 0);
    std::vector<float> kernel = random_values(3 * 2 * 3 * 3, 21), bias = random_values(3, 22);
    for (Conv2DModule* m : {&padded, &explicit_pad}) {
        m->set_kernel(kernel);
        m->set_bias(bias);
    }
    std::vector<float> x = random_values(2 * h * w, 23);
    std::vector<float> xp(2 * ph * pw, 0.0f);
    for (int64_t c = 0; c < 2; ++c)
        for (int64_t i = 0; i < h; ++i)
            for (int64_t j = 0; j < w; ++j) xp[(c * ph + i + pad) * pw + j + pad] = x[(c * h + i) * w + j];
    Tensor y = padded.forward(Tensor(Shape({1, 2, h, w}), &backend, x));
    Tensor yp = explicit_pad.forward(Tensor(Shape({1, 2, ph, pw}), &backend, xp));
    ASSERT_EQ(values_of(y), values_of(yp));
    Tensor r(y.shape(), &backend, random_values(static_cast<size_t>(y.numel()), 24));
    for (const LRPRuleConfig& config : zero_input_safe_rules()) {
        SCOPED_TRACE(lrp_rule_name(config.rule));
        Tensor ri = padded.propagate_relevance(r, config);
        Tensor rp = explicit_pad.propagate_relevance(r, config);
        ASSERT_EQ(ri.shape(), Shape({1, 2, h, w}));
        for (int64_t c = 0; c < 2; ++c)
            for (int64_t i = 0; i < h; ++i)
                for (int64_t j = 0; j < w; ++j)
                    EXPECT_NEAR(ri.data()[(c * h + i) * w + j], rp.data()[(c * ph + i + pad) * pw + j + pad], 1e-6f);
    }
}

// With zero bias, every rule here conserves relevance. ZBox is the sharp case: Zennit pads the
// bound tensors with zeros, so a padded position gets no relevance. Filling the patches with
// the scalar bounds instead would hand relevance to padding, which col2im then drops.
TEST_F(Conv2DStridePaddingTest, EveryRuleConservesRelevanceThroughStrideAndPadding) {
    std::vector<LRPRuleConfig> rules = zero_input_safe_rules();
    LRPRuleConfig zbox;
    zbox.rule = LRPRule::ZBox;
    zbox.low = -1.0f;
    zbox.high = 1.0f;
    rules.push_back(zbox);
    Conv2DModule conv(2, 3, 3, 3, &backend, 2, 1);
    conv.set_kernel(random_values(3 * 2 * 3 * 3, 31));
    conv.set_bias({0.0f, 0.0f, 0.0f});
    std::vector<float> x = random_values(2 * 7 * 6, 32);
    for (float& v : x) v = 0.5f * (v + 1.0f) * 0.9f;  // in [0, 0.9]: inside ZBox's [-1, 1] box
    Tensor y = conv.forward(Tensor(Shape({1, 2, 7, 6}), &backend, x));
    for (const LRPRuleConfig& config : rules) {
        SCOPED_TRACE(lrp_rule_name(config.rule));
        Tensor ri = conv.propagate_relevance(y, config);
        double in = 0.0, out = 0.0;
        for (float v : values_of(ri)) in += v;
        for (float v : values_of(y)) out += v;
        EXPECT_NEAR(in, out, 1e-3 * std::max(1.0, std::fabs(out)));
    }
}

TEST_F(Conv2DStridePaddingTest, RejectsInvalidStridePaddingAndTooSmallInput) {
    EXPECT_THROW(Conv2DModule(1, 1, 3, 3, &backend, 0, 0), std::invalid_argument);
    EXPECT_THROW(Conv2DModule(1, 1, 3, 3, &backend, 1, -1), std::invalid_argument);
    Conv2DModule conv(1, 1, 5, 5, &backend, 1, 1);
    EXPECT_THROW((void)conv.forward(Tensor(Shape({1, 1, 2, 2}), &backend)), std::invalid_argument);
    EXPECT_NO_THROW((void)conv.forward(Tensor(Shape({1, 1, 3, 3}), &backend)));  // 3 + 2*1 == 5
}

TEST_F(Conv2DStridePaddingTest, ExposesItsGeometry) {
    Conv2DModule conv(1, 1, 3, 3, &backend, 2, 1);
    EXPECT_EQ(conv.stride(), 2);
    EXPECT_EQ(conv.padding(), 1);
    Conv2DModule plain(1, 1, 3, 3, &backend);
    EXPECT_EQ(plain.stride(), 1);
    EXPECT_EQ(plain.padding(), 0);
}

}  // namespace
}  // namespace pulsatrix
