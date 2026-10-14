// KD-3: SoftDecisionTree against tools/generate_soft_decision_tree_reference_values.py (numpy,
// gradients by central differences), its paths, and a tree distilled from a network.
#include "pulsatrix/soft_decision_tree.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/distillation_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

namespace reference {
const std::vector<double> kOutput = {-1.159966070191, -1.012035080732, -1.130026599040, -1.172910007800, -1.051428558696, -1.075589215190, -1.196116899140, -1.094407575503, -1.013638734335, -1.204859070953, -1.071526138005, -1.027836101394, -1.194705654893, -1.048818858355, -1.058846683837};
const double kPenalty = 0.140860085332;
const std::vector<double> kLeafReach = {0.129148576121, 0.114629536971, 0.511468341137, 0.244753545772, 0.132268559687, 0.123099282336, 0.432719504777, 0.311912653199, 0.245459326849, 0.190509182014, 0.236409841021, 0.327621650116, 0.367818016878, 0.231655766695, 0.151408957508, 0.249117258919, 0.334603711668, 0.209854578053, 0.228120542968, 0.227421167311};
const std::vector<double> kGradInput = {-0.000875079120, 0.001669361982, 0.000056794347, -0.014673918525, 0.025873069021, 0.039933431673, 0.013524430464, -0.008768064053, -0.010710498044, 0.004231729633, 0.001041143838, -0.010431575381, -0.012614266520, 0.013600170767, 0.017312010270};
const std::vector<double> kGradWeight = {0.025191218667, 0.021874016820, 0.000368843622, -0.004348514215, 0.015346291127, -0.096449402243, -0.030799292672, -0.002082624517, -0.124755104247};
const std::vector<double> kGradBias = {0.041923771565, -0.013135988519, -0.099991624314};
const std::vector<double> kGradLeaves = {-0.006599308211, 0.018021540038, -0.011422231827, 0.010993057353, 0.006288788601, -0.017281845732, 0.185023053767, -0.023730250875, -0.161292802003, 0.077995642611, 0.008284942732, -0.086280584677};
}  // namespace reference

void ExpectNear(const std::vector<float>& got, const std::vector<double>& expected, double tolerance) {
    ASSERT_EQ(got.size(), expected.size());
    for (size_t k = 0; k < got.size(); ++k) EXPECT_NEAR(got[k], expected[k], tolerance) << k;
}

class SoftDecisionTreeTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The generator's tree: depth 2, 3 inputs, 3 classes, beta 1.5, penalty 0.1, gates from seed 2.
    SoftDecisionTree Reference() {
        SoftDecisionTreeOptions o;
        o.depth = 2;
        o.beta = 1.5f;
        o.penalty = 0.1f;
        o.seed = 2;
        SoftDecisionTree tree(3, 3, &backend, o);
        std::vector<float> phi;
        for (int k = 0; k < 12; ++k) phi.push_back(static_cast<float>(0.8 * std::sin(1.1 * k + 0.4)));
        tree.leaf_logits() = Tensor(Shape({4, 3}), &backend, phi);
        return tree;
    }

    Tensor Inputs() {
        std::vector<float> x;
        for (int k = 0; k < 15; ++k) x.push_back(static_cast<float>(std::sin(0.37 * k + 0.5 * (k % 3))));
        return Tensor(Shape({5, 3}), &backend, x);
    }
};

TEST_F(SoftDecisionTreeTest, MatchesTheReference) {
    SoftDecisionTree tree = Reference();
    const Tensor y = tree.forward(Inputs());
    EXPECT_EQ(y.shape(), Shape({5, 3}));
    ExpectNear(y.to_host_vector(), reference::kOutput, 1e-6);
    EXPECT_NEAR(tree.penalty(), reference::kPenalty, 1e-6);
    const std::vector<double>& reach = tree.leaf_probabilities();
    ExpectNear(std::vector<float>(reach.begin(), reach.end()), reference::kLeafReach, 1e-6);

    std::vector<float> g;
    for (int k = 0; k < 15; ++k) g.push_back(static_cast<float>(std::sin(0.6 * k + 0.9)));
    const Tensor dx = tree.backward(Tensor(Shape({5, 3}), &backend, g));
    ExpectNear(dx.to_host_vector(), reference::kGradInput, 1e-5);
    ExpectNear(tree.gates().weight_grad().to_host_vector(), reference::kGradWeight, 1e-5);
    ExpectNear(tree.gates().bias_grad().to_host_vector(), reference::kGradBias, 1e-5);
    std::vector<NamedParamRef> params = tree.named_parameters();
    ASSERT_EQ(params.size(), 3u);
    EXPECT_EQ(params[0].name, "gates.weight");
    EXPECT_EQ(params[1].name, "gates.bias");
    EXPECT_EQ(params[2].name, "leaves");
    ExpectNear(params[2].ref.grad->to_host_vector(), reference::kGradLeaves, 1e-5);
}

TEST_F(SoftDecisionTreeTest, PathsFollowTheHeapNumbering) {
    SoftDecisionTree tree = Reference();
    using Step = std::pair<int64_t, bool>;
    EXPECT_EQ(tree.PathTo(0), (std::vector<Step>{{0, false}, {1, false}}));
    EXPECT_EQ(tree.PathTo(2), (std::vector<Step>{{0, true}, {2, false}}));
    EXPECT_EQ(tree.PathTo(3), (std::vector<Step>{{0, true}, {2, true}}));
    EXPECT_THROW((void)tree.PathTo(4), std::invalid_argument);
    EXPECT_THROW((void)tree.MostProbableLeaf(), std::logic_error);
    (void)tree.forward(Inputs());
    const std::vector<int64_t> leaves = tree.MostProbableLeaf();
    ASSERT_EQ(leaves.size(), 5u);
    for (size_t r = 0; r < 5; ++r) {
        // Every step of the greedy path takes the likelier branch.
        for (const Step& step : tree.PathTo(leaves[r])) {
            const double p = tree.right_probabilities()[r * 3 + static_cast<size_t>(step.first)];
            EXPECT_EQ(step.second, p > 0.5) << r;
        }
    }
}

TEST_F(SoftDecisionTreeTest, WithoutThePenaltyItsGradientIsGone) {
    SoftDecisionTreeOptions o;
    o.depth = 2;
    SoftDecisionTree tree(3, 3, &backend, o);
    (void)tree.forward(Inputs());
    EXPECT_EQ(tree.penalty(), 0.0);
    // A zero output gradient leaves every gradient at zero.
    const std::vector<float> dx = tree.backward(Tensor(Shape({5, 3}), &backend, std::vector<float>(15, 0.0f))).to_host_vector();
    for (float v : dx) EXPECT_EQ(v, 0.0f);
}

TEST_F(SoftDecisionTreeTest, RejectsBadSettingsAndInputs) {
    SoftDecisionTreeOptions o;
    o.depth = 0;
    EXPECT_THROW(SoftDecisionTree(3, 3, &backend, o), std::invalid_argument);
    o.depth = 17;
    EXPECT_THROW(SoftDecisionTree(3, 3, &backend, o), std::invalid_argument);
    o.depth = 2;
    EXPECT_THROW(SoftDecisionTree(3, 1, &backend, o), std::invalid_argument);
    o.beta = 0.0f;
    EXPECT_THROW(SoftDecisionTree(3, 3, &backend, o), std::invalid_argument);
    SoftDecisionTree tree = Reference();
    EXPECT_THROW((void)tree.backward(Tensor(Shape({5, 3}), &backend)), std::logic_error);
    EXPECT_THROW((void)tree.forward(Tensor(Shape({5, 4}), &backend)), std::invalid_argument);
    (void)tree.forward(Inputs());
    EXPECT_THROW((void)tree.backward(Tensor(Shape({4, 3}), &backend)), std::invalid_argument);
    EXPECT_THROW((void)tree.propagate_relevance(Tensor(Shape({5, 3}), &backend), LRPRuleConfig{}), std::logic_error);
}

TEST_F(SoftDecisionTreeTest, ATreeDistilledFromANetworkAgreesWithIt) {
    // Teacher: 2 -> 16 -> 3 on points in the square; the classes are regions of the plane.
    LinearModule t1(2, 16, &backend), t2(16, 3, &backend);
    ReluModule relu(&backend);
    SequentialModule teacher({&t1, &relu, &t2});
    std::vector<float> w1, w2;
    for (int k = 0; k < 32; ++k) w1.push_back(static_cast<float>(1.5 * std::sin(1.3 * k + 0.2)));
    for (int k = 0; k < 48; ++k) w2.push_back(static_cast<float>(std::cos(0.9 * k + 0.4)));
    t1.set_weight(w1);
    t2.set_weight(w2);
    auto points = [&](int n, double offset) {
        std::vector<float> x;
        for (int k = 0; k < n; ++k) {
            x.push_back(static_cast<float>(std::sin(1.7 * k + offset)));
            x.push_back(static_cast<float>(std::cos(2.3 * k + 0.5 * offset)));
        }
        return Tensor(Shape({n, 2}), &backend, x);
    };
    const Tensor train = points(256, 0.0), test = points(128, 0.37);
    const Tensor teacher_train = teacher.forward(train), teacher_test = teacher.forward(test);

    SoftDecisionTreeOptions o;
    o.depth = 3;
    o.penalty = 0.01f;
    SoftDecisionTree tree(2, 3, &backend, o);
    AdamOptimizer opt(0.05f, &backend);
    DistillationLoss loss(&backend);
    for (int step = 0; step < 500; ++step) {
        opt.zero_grad(tree);
        (void)loss.forward(tree.forward(train), teacher_train);
        (void)tree.backward(loss.backward());
        opt.step(tree);
    }
    const std::vector<float> a = tree.forward(test).to_host_vector(), b = teacher_test.to_host_vector();
    int agree = 0;
    std::vector<int> counts(3, 0);
    for (size_t r = 0; r < 128; ++r) {
        const auto ta = std::max_element(a.begin() + 3 * r, a.begin() + 3 * r + 3) - (a.begin() + 3 * r);
        const auto tb = std::max_element(b.begin() + 3 * r, b.begin() + 3 * r + 3) - (b.begin() + 3 * r);
        agree += ta == tb;
        ++counts[static_cast<size_t>(tb)];
    }
    // Every class is common enough that agreement isn't the majority class alone.
    for (int c : counts) EXPECT_GE(c, 15);
    std::printf("soft tree: %d of 128 held-out points agree with the teacher (classes %d, %d, %d)\n", agree, counts[0], counts[1], counts[2]);
    EXPECT_GE(agree, 110);
}

}  // namespace
}  // namespace pulsatrix
