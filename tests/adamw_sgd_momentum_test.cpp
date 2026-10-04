#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/param_groups.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

class OptimizerVariantsTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Linear(1 -> 1): weight w, bias b, with gradients set per step by set_grads().
    LinearModule make(float w, float b) {
        LinearModule m(1, 1, &backend);
        m.set_weight({w});
        m.set_bias({b});
        return m;
    }
    static void set_grads(LinearModule& m, float gw, float gb) {
        std::vector<ParamRef> p = m.parameters();
        p[0].grad->fill(gw);
        p[1].grad->fill(gb);
    }
    static float w(LinearModule& m) { return m.parameters()[0].value->data()[0]; }
    static float b(LinearModule& m) { return m.parameters()[1].value->data()[0]; }
};

// --- AdamW -------------------------------------------------------------------------------

// The research notes' test: decoupled decay is w -= lr * wd * w, not part of the gradient. With a
// zero gradient that moves w by lr*wd*w; L2 decay would go through Adam's normalization and move
// it by about lr.
TEST_F(OptimizerVariantsTest, AdamWDecayIsDecoupledFromTheGradient) {
    LinearModule decoupled = make(1.0f, 0.0f), coupled = make(1.0f, 0.0f);
    AdamWOptimizer adamw(0.1f, &backend, /*weight_decay=*/0.01f);
    AdamOptimizer adam_l2(0.1f, &backend);
    adam_l2.set_weight_decay(0.01f);
    set_grads(decoupled, 0.0f, 0.0f);
    set_grads(coupled, 0.0f, 0.0f);
    adamw.step(decoupled);
    adam_l2.step(coupled);
    EXPECT_NEAR(w(decoupled), 1.0f - 0.1f * 0.01f, 1e-7f);
    EXPECT_NEAR(w(coupled), 1.0f - 0.1f, 1e-4f);
}

TEST_F(OptimizerVariantsTest, AdamWFirstStepMatchesTheClosedForm) {
    // Step 1: w <- w * (1 - lr*wd), then w -= lr * m_hat / (sqrt(v_hat) + eps) with m_hat = g and
    // v_hat = g^2 after bias correction.
    LinearModule m = make(2.0f, -1.0f);
    set_grads(m, 0.5f, -0.25f);
    AdamWOptimizer adamw(0.01f, &backend, 0.1f);
    adamw.step(m);
    const double lr = 0.01, wd = 0.1, eps = 1e-8;
    EXPECT_NEAR(w(m), 2.0 * (1 - lr * wd) - lr * 0.5 / (0.5 + eps), 1e-6);
    EXPECT_NEAR(b(m), -1.0 * (1 - lr * wd) - lr * -0.25 / (0.25 + eps), 1e-6);
}

TEST_F(OptimizerVariantsTest, AdamWWithoutDecayIsAdam) {
    LinearModule a = make(2.0f, -1.0f), c = make(2.0f, -1.0f);
    AdamWOptimizer adamw(0.01f, &backend, 0.0f);
    AdamOptimizer adam(0.01f, &backend);
    for (int i = 0; i < 3; ++i) {
        set_grads(a, 0.5f - 0.1f * i, 0.2f);
        set_grads(c, 0.5f - 0.1f * i, 0.2f);
        adamw.step(a);
        adam.step(c);
    }
    EXPECT_EQ(w(a), w(c));
    EXPECT_EQ(b(a), b(c));
}

TEST_F(OptimizerVariantsTest, AdamWDefaultsMatchPyTorch) {
    AdamWOptimizer adamw(1e-3f, &backend);
    EXPECT_FLOAT_EQ(adamw.weight_decay(), 0.01f);
}

TEST_F(OptimizerVariantsTest, AdamWHonorsPerGroupDecay) {
    LinearModule m = make(1.0f, 1.0f);
    set_grads(m, 0.0f, 0.0f);
    AdamWOptimizer adamw(0.1f, &backend, 0.5f);
    adamw.set_param_groups({{"no decay", param_select::one_dimensional(), 0.1f, 0.0f}});
    adamw.step(m);
    // Linear's weight is (1, 1), rank 2: decayed. Its bias is rank 1: in the no-decay group.
    EXPECT_NEAR(w(m), 1.0f - 0.1f * 0.5f, 1e-7f);
    EXPECT_EQ(b(m), 1.0f);
}

TEST_F(OptimizerVariantsTest, AdamWCheckpointsLikeAdam) {
    const std::string path = ::testing::TempDir() + "pulsatrix_adamw_" +
                             ::testing::UnitTest::GetInstance()->current_test_info()->name() + ".safetensors";
    LinearModule a = make(2.0f, -1.0f);
    AdamWOptimizer adamw(0.01f, &backend, 0.1f);
    set_grads(a, 0.5f, 0.2f);
    adamw.step(a);
    SaveCheckpoint(path, a, adamw);
    LinearModule c = make(0.0f, 0.0f);
    AdamWOptimizer resumed(0.01f, &backend, 0.1f);
    LoadCheckpoint(path, c, resumed);
    set_grads(a, 0.3f, -0.1f);
    set_grads(c, 0.3f, -0.1f);
    adamw.step(a);
    resumed.step(c);
    EXPECT_EQ(w(a), w(c));
    std::remove(path.c_str());
    std::remove(OptimizerStatePath(path).c_str());
}

// --- SGD with momentum and Nesterov ------------------------------------------------------

// PyTorch's rule: buf = g on the first step, then buf = momentum * buf + g; the step is
// -lr * buf, or -lr * (g + momentum * buf) with Nesterov.
TEST_F(OptimizerVariantsTest, SGDMomentumFollowsPyTorchsBufferRule) {
    LinearModule m = make(1.0f, 0.0f);
    SGDOptimizer sgd(0.1f, /*momentum=*/0.9f);
    const float grads[3] = {1.0f, 0.5f, -2.0f};
    double weight = 1.0, buf = 0.0;
    for (int i = 0; i < 3; ++i) {
        set_grads(m, grads[i], 0.0f);
        sgd.step(m);
        buf = i == 0 ? grads[i] : 0.9 * buf + grads[i];
        weight -= 0.1 * buf;
        EXPECT_NEAR(w(m), weight, 1e-6) << "step " << i;
    }
}

TEST_F(OptimizerVariantsTest, SGDNesterovLooksAhead) {
    LinearModule m = make(1.0f, 0.0f);
    SGDOptimizer sgd(0.1f, 0.9f, /*nesterov=*/true);
    const float grads[3] = {1.0f, 0.5f, -2.0f};
    double weight = 1.0, buf = 0.0;
    for (int i = 0; i < 3; ++i) {
        set_grads(m, grads[i], 0.0f);
        sgd.step(m);
        buf = i == 0 ? grads[i] : 0.9 * buf + grads[i];
        weight -= 0.1 * (grads[i] + 0.9 * buf);
        EXPECT_NEAR(w(m), weight, 1e-6) << "step " << i;
    }
}

TEST_F(OptimizerVariantsTest, SGDMomentumAppliesWeightDecayToTheGradientFirst) {
    LinearModule m = make(2.0f, 0.0f);
    SGDOptimizer sgd(0.1f, 0.5f);
    sgd.set_weight_decay(0.1f);
    set_grads(m, 1.0f, 0.0f);
    sgd.step(m);  // buf = g + wd*w = 1.2
    EXPECT_NEAR(w(m), 2.0 - 0.1 * 1.2, 1e-6);
    set_grads(m, 1.0f, 0.0f);
    sgd.step(m);  // buf = 0.5*1.2 + (1 + 0.1*1.88)
    EXPECT_NEAR(w(m), 1.88 - 0.1 * (0.5 * 1.2 + 1.0 + 0.1 * 1.88), 1e-6);
}

TEST_F(OptimizerVariantsTest, SGDMomentumSkipsFrozenParameters) {
    LinearModule m = make(1.0f, 1.0f);
    m.set_requires_grad(false, "bias");
    SGDOptimizer sgd(0.1f, 0.9f);
    set_grads(m, 1.0f, 1.0f);
    sgd.step(m);
    EXPECT_EQ(b(m), 1.0f);
    EXPECT_EQ(sgd.momentum_buffer(m.parameters()[1].value), nullptr);
    EXPECT_NE(sgd.momentum_buffer(m.parameters()[0].value), nullptr);
}

TEST_F(OptimizerVariantsTest, SGDRejectsInvalidMomentumSettings) {
    EXPECT_THROW(SGDOptimizer(0.1f, -0.1f), std::invalid_argument);
    EXPECT_THROW(SGDOptimizer(0.1f, 1.0f), std::invalid_argument);
    EXPECT_THROW(SGDOptimizer(0.1f, 0.0f, /*nesterov=*/true), std::invalid_argument);
    EXPECT_THROW(AdamWOptimizer(0.1f, &backend, -0.01f), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
