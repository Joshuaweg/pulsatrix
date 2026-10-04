#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/grad_clipping.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

class GradClippingTest : public ::testing::Test {
protected:
    CPUBackend backend;
    // Linear(2 -> 2): weight grads 3, 0, 0, 4 and bias grads 0, 0 -> global norm 5.
    LinearModule m{2, 2, &backend};
    void SetUp() override {
        std::vector<ParamRef> p = m.parameters();
        *p[0].grad = Tensor(p[0].grad->shape(), &backend, {3.0f, 0.0f, 0.0f, 4.0f});
        p[1].grad->fill(0.0f);
    }
    std::vector<float> weight_grad() { return values_of(*m.parameters()[0].grad); }
};

TEST_F(GradClippingTest, ReturnsTheGlobalNormBeforeClipping) {
    EXPECT_FLOAT_EQ(ClipGradNorm(m, 10.0f), 5.0f);
}

TEST_F(GradClippingTest, LeavesGradientsAloneBelowTheLimit) {
    (void)ClipGradNorm(m, 5.5f);
    EXPECT_EQ(weight_grad(), (std::vector<float>{3.0f, 0.0f, 0.0f, 4.0f}));
}

// torch.nn.utils.clip_grad_norm_: every gradient is scaled by max_norm / (norm + 1e-6), the
// same factor for all parameters, so the direction is kept and the norm becomes max_norm.
TEST_F(GradClippingTest, ScalesEveryGradientByOneFactorAboveTheLimit) {
    (void)ClipGradNorm(m, 1.0f);
    const float coef = 1.0f / (5.0f + 1e-6f);
    const std::vector<float> g = weight_grad();
    EXPECT_FLOAT_EQ(g[0], 3.0f * coef);
    EXPECT_FLOAT_EQ(g[3], 4.0f * coef);
    EXPECT_EQ(g[1], 0.0f);
    EXPECT_NEAR(std::sqrt(g[0] * g[0] + g[3] * g[3]), 1.0f, 1e-5f);
}

TEST_F(GradClippingTest, FrozenParametersAreNeitherCountedNorScaled) {
    std::vector<ParamRef> p = m.parameters();
    *p[1].grad = Tensor(p[1].grad->shape(), &backend, {100.0f, 100.0f});
    m.set_requires_grad(false, "bias");
    EXPECT_FLOAT_EQ(ClipGradNorm(m, 10.0f), 5.0f);
    (void)ClipGradNorm(m, 1.0f);
    EXPECT_EQ(values_of(*p[1].grad), (std::vector<float>{100.0f, 100.0f}));
}

TEST_F(GradClippingTest, NonFiniteNormLeavesGradientsAloneOrThrows) {
    std::vector<ParamRef> p = m.parameters();
    p[1].grad->fill(std::numeric_limits<float>::infinity());
    const float norm = ClipGradNorm(m, 1.0f);
    EXPECT_TRUE(std::isinf(norm));
    EXPECT_EQ(weight_grad(), (std::vector<float>{3.0f, 0.0f, 0.0f, 4.0f}));  // the caller can skip the step
    EXPECT_THROW((void)ClipGradNorm(m, 1.0f, /*error_if_nonfinite=*/true), std::runtime_error);
    p[1].grad->fill(std::numeric_limits<float>::quiet_NaN());
    EXPECT_TRUE(std::isnan(ClipGradNorm(m, 1.0f)));
}

TEST_F(GradClippingTest, RejectsANonPositiveOrNonFiniteLimit) {
    EXPECT_THROW((void)ClipGradNorm(m, 0.0f), std::invalid_argument);
    EXPECT_THROW((void)ClipGradNorm(m, -1.0f), std::invalid_argument);
    EXPECT_THROW((void)ClipGradNorm(m, std::numeric_limits<float>::infinity()), std::invalid_argument);
}

TEST_F(GradClippingTest, AModuleWithoutParametersHasNormZero) {
    LinearModule unused(1, 1, &backend);
    unused.set_requires_grad(false);
    EXPECT_EQ(ClipGradNorm(unused, 1.0f), 0.0f);
}

}  // namespace
}  // namespace pulsatrix
