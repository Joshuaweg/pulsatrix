#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/rms_norm_module.hpp"

namespace exai {
namespace {

class RMSNormModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Scalar loss L = dot(grad_output, forward(x)) -- lets a single Tensor perturbation
    // (of x or of gamma) drive a central-difference gradient estimate, per
    // cpp_tdd/context_tdd_fundamentals.md's guidance on finite-difference checks for
    // numerically-checkable invariants. Reuses one module instance across calls --
    // forward() has no cross-call state dependency other than what it itself overwrites.
    static float loss(RMSNormModule& norm, const Tensor& x, const Tensor& grad_output) {
        Tensor y = norm.forward(x);
        float total = 0.0f;
        for (int64_t i = 0; i < y.numel(); ++i) {
            total += grad_output.data()[i] * y.data()[i];
        }
        return total;
    }
};

TEST_F(RMSNormModuleTest, ConstructionThrowsOnZeroNumFeatures) {
    EXPECT_THROW({ RMSNormModule norm(0, &backend); }, std::invalid_argument);
}

TEST_F(RMSNormModuleTest, ConstructionThrowsOnNegativeNumFeatures) {
    EXPECT_THROW({ RMSNormModule norm(-3, &backend); }, std::invalid_argument);
}

TEST_F(RMSNormModuleTest, GammaGradientStartsAtZero) {
    RMSNormModule norm(3, &backend);
    EXPECT_FLOAT_EQ(norm.gamma_grad().data()[0], 0.0f);
}

// D=4, x uniform, gamma uniform -> rms = sqrt(mean(x_i^2)) = |x_i|, y_i = gamma_i (eps=0
// isolates the rule's core correctness, same convention linear_module_test.cpp uses).
TEST_F(RMSNormModuleTest, ForwardComputesHandVerifiedOutputForUniformInput) {
    RMSNormModule norm(4, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({2.0f, 2.0f, 2.0f, 2.0f});

    Tensor x(Shape({4}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor y = norm.forward(x);

    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(y.data()[i], 2.0f);
    }
}

// Same x, per-element gamma -- proves gamma scales each output element independently,
// not just uniformly (the uniform-gamma case above can't distinguish a correct
// per-element multiply from an accidental scalar one).
TEST_F(RMSNormModuleTest, ForwardScalesEachOutputByItsOwnGamma) {
    RMSNormModule norm(4, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({1.0f, 2.0f, 3.0f, 4.0f});

    Tensor x(Shape({4}), &backend, {1.0f, 1.0f, 1.0f, 1.0f});
    Tensor y = norm.forward(x);

    EXPECT_FLOAT_EQ(y.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(y.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(y.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(y.data()[3], 4.0f);
}

TEST_F(RMSNormModuleTest, ForwardThrowsOnWrongRank) {
    RMSNormModule norm(4, &backend);
    Tensor wrong_rank(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_THROW({ (void)norm.forward(wrong_rank); }, std::invalid_argument);
}

TEST_F(RMSNormModuleTest, ForwardThrowsOnWrongNumel) {
    RMSNormModule norm(4, &backend);
    Tensor wrong_size(Shape({2}), &backend, {1.0f, 2.0f});
    EXPECT_THROW({ (void)norm.forward(wrong_size); }, std::invalid_argument);
}

TEST_F(RMSNormModuleTest, BackwardThrowsIfCalledBeforeForward) {
    RMSNormModule norm(3, &backend);
    Tensor grad(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(grad); }, std::logic_error);
}

TEST_F(RMSNormModuleTest, BackwardThrowsOnWrongGradOutputSize) {
    RMSNormModule norm(3, &backend);
    Tensor x(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);
    Tensor wrong_size_grad(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.backward(wrong_size_grad); }, std::invalid_argument);
}

// The real acceptance criterion for backward()'s correctness -- central finite
// differences against the analytic input gradient, avoiding hand-arithmetic risk on a
// deliberately asymmetric (non-uniform-x) case, per this project's own finite-difference
// discipline for numerically-checkable invariants.
TEST_F(RMSNormModuleTest, BackwardInputGradientMatchesFiniteDifference) {
    RMSNormModule norm(3, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({1.0f, 1.0f, 1.0f});

    Tensor x(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor grad_output(Shape({3}), &backend, {1.0f, 0.5f, -0.25f});

    const float h = 1e-3f;
    std::vector<float> fd_grad(3);
    for (int64_t i = 0; i < 3; ++i) {
        std::vector<float> x_plus = {1.0f, 2.0f, 3.0f};
        std::vector<float> x_minus = {1.0f, 2.0f, 3.0f};
        x_plus[i] += h;
        x_minus[i] -= h;
        Tensor xp(Shape({3}), &backend, x_plus);
        Tensor xm(Shape({3}), &backend, x_minus);
        float lp = loss(norm, xp, grad_output);
        float lm = loss(norm, xm, grad_output);
        fd_grad[i] = (lp - lm) / (2.0f * h);
    }

    (void)norm.forward(x);  // restore cached state to the unperturbed input before backward()
    Tensor grad_input = norm.backward(grad_output);

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(grad_input.data()[i], fd_grad[i], 1e-2f);
    }
}

TEST_F(RMSNormModuleTest, BackwardGammaGradientMatchesFiniteDifference) {
    RMSNormModule norm(3, &backend, DeviceType::Cpu, 0.0f);
    norm.set_gamma({1.0f, 1.5f, 2.0f});

    Tensor x(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor grad_output(Shape({3}), &backend, {1.0f, 0.5f, -0.25f});

    const float h = 1e-3f;
    std::vector<float> fd_grad(3);
    for (int64_t i = 0; i < 3; ++i) {
        std::vector<float> g_plus = {1.0f, 1.5f, 2.0f};
        std::vector<float> g_minus = {1.0f, 1.5f, 2.0f};
        g_plus[i] += h;
        g_minus[i] -= h;

        norm.set_gamma(g_plus);
        float lp = loss(norm, x, grad_output);
        norm.set_gamma(g_minus);
        float lm = loss(norm, x, grad_output);
        fd_grad[i] = (lp - lm) / (2.0f * h);
    }

    norm.set_gamma({1.0f, 1.5f, 2.0f});
    (void)norm.forward(x);
    (void)norm.backward(grad_output);

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(norm.gamma_grad().data()[i], fd_grad[i], 1e-2f);
    }
}

TEST_F(RMSNormModuleTest, PropagateRelevanceThrowsIfCalledBeforeForward) {
    RMSNormModule norm(3, &backend);
    Tensor relevance(Shape({3}), &backend, {1.0f, 1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.propagate_relevance(relevance, LRPRuleConfig{}); }, std::logic_error);
}

TEST_F(RMSNormModuleTest, PropagateRelevanceThrowsOnWrongSize) {
    RMSNormModule norm(3, &backend);
    Tensor x(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);
    Tensor wrong_size_relevance(Shape({2}), &backend, {1.0f, 1.0f});
    EXPECT_THROW({ (void)norm.propagate_relevance(wrong_size_relevance, LRPRuleConfig{}); }, std::invalid_argument);
}

// The mission's real acceptance criterion for the LRP half -- identity rule (AttnLRP,
// Achtibat et al. 2024), verified explicitly rather than assumed from the implementation.
TEST_F(RMSNormModuleTest, PropagateRelevanceIsIdentityPassThrough) {
    RMSNormModule norm(3, &backend);
    Tensor x(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);

    Tensor relevance_out(Shape({3}), &backend, {4.0f, -1.0f, 2.5f});
    Tensor relevance_in = norm.propagate_relevance(relevance_out, LRPRuleConfig{});

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_FLOAT_EQ(relevance_in.data()[i], relevance_out.data()[i]);
    }
}

TEST_F(RMSNormModuleTest, PropagateRelevanceConservesTotalRelevance) {
    RMSNormModule norm(3, &backend);
    Tensor x(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    (void)norm.forward(x);

    Tensor relevance_out(Shape({3}), &backend, {4.0f, -1.0f, 2.5f});
    Tensor relevance_in = norm.propagate_relevance(relevance_out, LRPRuleConfig{});

    float sum_in = relevance_in.data()[0] + relevance_in.data()[1] + relevance_in.data()[2];
    float sum_out = relevance_out.data()[0] + relevance_out.data()[1] + relevance_out.data()[2];
    EXPECT_FLOAT_EQ(sum_in, sum_out);
}

TEST_F(RMSNormModuleTest, ParametersExposesGammaByPointer) {
    RMSNormModule norm(2, &backend);
    auto params = norm.parameters();

    ASSERT_EQ(params.size(), 1u);
    EXPECT_EQ(params[0].value, &norm.gamma());
    EXPECT_EQ(params[0].grad, &norm.gamma_grad());
}

using RMSNormModuleDeathTest = RMSNormModuleTest;

// Not yet backend-generic -- raw host loop in backward(), matching every existing Module
// subclass's Phase 1.5 scope decision. No real GPU needed: Tensor::device() is metadata
// decoupled from which DeviceBackend* actually allocated its buffer.
TEST_F(RMSNormModuleDeathTest, BackwardAbortsOnNonCpuGradOutput) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    RMSNormModule norm(2, &backend);
    Tensor x(Shape({2}), &backend, {1.0f, 1.0f});
    (void)norm.forward(x);

    Tensor grad_output(Shape({2}), &backend, {1.0f, 1.0f}, DeviceType::Cuda);
    EXPECT_DEATH({ (void)norm.backward(grad_output); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
