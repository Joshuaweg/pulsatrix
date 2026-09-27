#include "pulsatrix/neuro_symbolic_toy_kb.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

namespace pulsatrix {
namespace {

// Mission file Stage 3's own hand-picked toy KB: N=4 groundings, one shared feature per
// grounding, feeding two independent sigmoid-squashed LinearModule(1,1) predicates.
std::vector<float> kXValues{-1.5f, -0.5f, 0.5f, 1.5f};
constexpr int64_t kN = 4;

Tensor MakeX(DeviceBackend* backend) { return Tensor(Shape({kN, 1}), backend, kXValues); }

float Sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }

class ToyKnowledgeBaseTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// ---------------------------------------------------------------------------
// Forward correctness -- De Morgan closed form (mission Stage 3, Decision 1)
// ---------------------------------------------------------------------------

TEST_F(ToyKnowledgeBaseTest, RuleEqualsOneMinusABClosedFormForEveryGrounding) {
    // Composed pipeline: not(A) or not(B) [Product t-conorm] must equal the algebraically
    // simplified closed form 1 - a*b, exactly (up to floating-point rounding) -- this is the
    // dedicated confirmation that De Morgan over NegationModule/DisjunctionModule alone is
    // sufficient to express A(x) -> not(B(x)), with no dedicated Implication Module.
    ToyKnowledgeBase kb(&backend, 2.0f);
    Tensor x = MakeX(&backend);
    (void)kb.forward(x);

    const Tensor& a = kb.last_a();
    const Tensor& b = kb.last_b();
    const Tensor& rule = kb.last_rule();
    ASSERT_EQ(rule.numel(), kN);
    for (int64_t i = 0; i < kN; ++i) {
        const float expected = 1.0f - a.data()[i] * b.data()[i];
        EXPECT_NEAR(rule.data()[i], expected, 1e-5f) << "grounding " << i;
    }
}

TEST_F(ToyKnowledgeBaseTest, ForwardProducesTruthDegreesStrictlyInOpenUnitInterval) {
    // Sigmoid-squashed predicates -- every a_i/b_i must land in (0,1), never at the boundary.
    ToyKnowledgeBase kb(&backend, 2.0f);
    Tensor x = MakeX(&backend);
    (void)kb.forward(x);
    for (int64_t i = 0; i < kN; ++i) {
        EXPECT_GT(kb.last_a().data()[i], 0.0f);
        EXPECT_LT(kb.last_a().data()[i], 1.0f);
        EXPECT_GT(kb.last_b().data()[i], 0.0f);
        EXPECT_LT(kb.last_b().data()[i], 1.0f);
    }
}

TEST_F(ToyKnowledgeBaseTest, ForwardMatchesHandComputedPredicateValuesAtInitialWeights) {
    // W_A=0.6, b_A=0.0, W_B=-0.4, b_B=0.0 (mission file Stage 3's own initial values).
    ToyKnowledgeBase kb(&backend, 2.0f);
    Tensor x = MakeX(&backend);
    (void)kb.forward(x);
    for (int64_t i = 0; i < kN; ++i) {
        const float expected_a = Sigmoid(0.6f * kXValues[static_cast<size_t>(i)]);
        const float expected_b = Sigmoid(-0.4f * kXValues[static_cast<size_t>(i)]);
        EXPECT_NEAR(kb.last_a().data()[i], expected_a, 1e-5f) << "a[" << i << "]";
        EXPECT_NEAR(kb.last_b().data()[i], expected_b, 1e-5f) << "b[" << i << "]";
    }
}

// ---------------------------------------------------------------------------
// Backward -- hand-derived closed form (mission file Stage 3), cross-checked against the
// composed module chain and central finite differences on forward()'s scalar loss output.
// ---------------------------------------------------------------------------

// Reproduces the mission file's own hand-derived closed form independently of any module
// call: dL/dW_A = sum_i[(1/N)*m^(1/p-1)*rule_i^(p-1) * b_i*a_i*(1-a_i)*x_i], etc.
struct HandDerivedGrads {
    float dW_a, db_a, dW_b, db_b;
};

HandDerivedGrads ComputeHandDerivedGrads(float W_a, float b_a, float W_b, float b_b, float p) {
    std::vector<float> a(kN), b(kN), rule(kN);
    for (int64_t i = 0; i < kN; ++i) {
        const float x = kXValues[static_cast<size_t>(i)];
        a[static_cast<size_t>(i)] = Sigmoid(W_a * x + b_a);
        b[static_cast<size_t>(i)] = Sigmoid(W_b * x + b_b);
        rule[static_cast<size_t>(i)] = 1.0f - a[static_cast<size_t>(i)] * b[static_cast<size_t>(i)];
    }
    float m = 0.0f;
    for (int64_t i = 0; i < kN; ++i) {
        m += std::pow(rule[static_cast<size_t>(i)], p);
    }
    m /= static_cast<float>(kN);

    HandDerivedGrads g{0.0f, 0.0f, 0.0f, 0.0f};
    const float m_pow = std::pow(m, 1.0f / p - 1.0f);
    for (int64_t i = 0; i < kN; ++i) {
        const size_t si = static_cast<size_t>(i);
        const float x = kXValues[si];
        const float common = (1.0f / static_cast<float>(kN)) * m_pow * std::pow(rule[si], p - 1.0f);
        const float dL_da = b[si] * common;
        const float dL_db = a[si] * common;
        const float dL_dzA = dL_da * a[si] * (1.0f - a[si]);
        const float dL_dzB = dL_db * b[si] * (1.0f - b[si]);
        g.dW_a += dL_dzA * x;
        g.db_a += dL_dzA;
        g.dW_b += dL_dzB * x;
        g.db_b += dL_dzB;
    }
    return g;
}

TEST_F(ToyKnowledgeBaseTest, BackwardMatchesHandDerivedClosedForm) {
    ToyKnowledgeBase kb(&backend, 2.0f);
    Tensor x = MakeX(&backend);
    (void)kb.forward(x);
    kb.backward();

    const HandDerivedGrads expected = ComputeHandDerivedGrads(0.6f, 0.0f, -0.4f, 0.0f, 2.0f);

    EXPECT_NEAR(kb.predicate_a().weight_grad().data()[0], expected.dW_a, 1e-4f);
    EXPECT_NEAR(kb.predicate_a().bias_grad().data()[0], expected.db_a, 1e-4f);
    EXPECT_NEAR(kb.predicate_b().weight_grad().data()[0], expected.dW_b, 1e-4f);
    EXPECT_NEAR(kb.predicate_b().bias_grad().data()[0], expected.db_b, 1e-4f);
}

TEST_F(ToyKnowledgeBaseTest, BackwardMatchesCentralFiniteDifferencesOnAllFourParameters) {
    ToyKnowledgeBase kb(&backend, 2.0f);
    Tensor x = MakeX(&backend);
    (void)kb.forward(x);
    kb.backward();

    const float grad_Wa = kb.predicate_a().weight_grad().data()[0];
    const float grad_ba = kb.predicate_a().bias_grad().data()[0];
    const float grad_Wb = kb.predicate_b().weight_grad().data()[0];
    const float grad_bb = kb.predicate_b().bias_grad().data()[0];

    auto loss_at = [&](float W_a, float b_a, float W_b, float b_b) {
        ToyKnowledgeBase probe(&backend, 2.0f);
        probe.predicate_a().set_weight({W_a});
        probe.predicate_a().set_bias({b_a});
        probe.predicate_b().set_weight({W_b});
        probe.predicate_b().set_bias({b_b});
        Tensor xp = MakeX(&backend);
        return probe.forward(xp);
    };

    constexpr float h = 1e-3f;
    const float fd_Wa = (loss_at(0.6f + h, 0.0f, -0.4f, 0.0f) - loss_at(0.6f - h, 0.0f, -0.4f, 0.0f)) / (2.0f * h);
    const float fd_ba = (loss_at(0.6f, 0.0f + h, -0.4f, 0.0f) - loss_at(0.6f, 0.0f - h, -0.4f, 0.0f)) / (2.0f * h);
    const float fd_Wb = (loss_at(0.6f, 0.0f, -0.4f + h, 0.0f) - loss_at(0.6f, 0.0f, -0.4f - h, 0.0f)) / (2.0f * h);
    const float fd_bb = (loss_at(0.6f, 0.0f, -0.4f, 0.0f + h) - loss_at(0.6f, 0.0f, -0.4f, 0.0f - h)) / (2.0f * h);

    EXPECT_NEAR(grad_Wa, fd_Wa, 5e-3f) << "W_A";
    EXPECT_NEAR(grad_ba, fd_ba, 5e-3f) << "b_A";
    EXPECT_NEAR(grad_Wb, fd_Wb, 5e-3f) << "W_B";
    EXPECT_NEAR(grad_bb, fd_bb, 5e-3f) << "b_B";
}

TEST_F(ToyKnowledgeBaseTest, BackwardBeforeForwardThrows) {
    ToyKnowledgeBase kb(&backend);
    EXPECT_THROW({ kb.backward(); }, std::logic_error);
}

// ---------------------------------------------------------------------------
// End-to-end LRP conservation -- Phase 2 Mission 0's own genuinely new deliverable (see
// mission_0_phase2_lrp_verification_and_e2e_conservation.md). Every per-operator
// propagate_relevance already has its own isolated conservation test (Mission 0/1's own
// test files, and the shared lrp_conservation_test.cpp TEST_P suite) -- this is the first
// check that relevance stays conserved across the *whole composed chain*:
// LinearModule(A)/LinearModule(B) -> sigmoid(x2, file-local glue) -> NegationModule(x2) ->
// DisjunctionModule (Product) -> the rule's own output. Mirrors
// lrp_conservation_test.cpp's own LRPConservationEndToEndTest pattern (arbitrary relevance
// seed at the final output, chained propagate_relevance calls back to the raw inputs,
// sum(relevance_in) compared to sum(relevance_seed)) but for this campaign's own composed
// pipeline instead of a plain Linear->Relu->Linear network.
//
// **Design decision, resolved here (mission file Stage 2/Task 2's own open question):
// relevance propagation stops at each LinearModule's pre-sigmoid output -- it does NOT
// flow through the sigmoid nonlinearity via its own y*(1-y) gradient.** Rationale: this
// codebase's own existing convention for a monotonic, single-input elementwise
// nonlinearity is a plain, unweighted pass-through of relevance (ReluModule::
// propagate_relevance -- relu_module.cpp -- returns relevance_out unchanged despite
// ReLU's own backward() being gated by sign(x); NegationModule::propagate_relevance --
// negation_module.cpp -- does the same for `1-x`, and its own header doc comment states
// this explicitly: "LRP rules are defined across combinations of multiple
// relevance-bearing inputs... a single-input, monotonic, bijective elementwise
// reparametrization... passes relevance through unchanged"). Sigmoid is exactly this
// shape (single-input, monotonic, bijective on its domain) -- reusing its own
// backward()-style y*(1-y) derivative here would be a *gradient*-shaped rule bolted onto
// an activation, which is precisely the thing ReluModule/NegationModule's own rules
// deliberately do NOT do (their backward() gradients are non-trivial -- ReLU's gate,
// Negation's -1 -- yet their propagate_relevance rules are still plain pass-through).
// Applying the training-gradient's y*(1-y) factor to relevance instead would be a genuinely
// new, uncited departure from this codebase's own established LRP convention for
// activations, not a neutral default -- so the relevance chain here treats the sigmoid
// output's relevance (R_a, R_b) as identically the relevance at LinearModule's cached
// pre-bias/pre-sigmoid output (z_a, z_b), fed directly into LinearModule::
// propagate_relevance(), matching the pass-through precedent exactly.
namespace {

// Splits a DisjunctionModule::propagate_relevance()-shaped stacked relevance tensor
// (leading dim 2, produced by disjunction_module.cpp's own combine_operands) back into its
// two operand-shaped halves -- test-local mirror of neuro_symbolic_toy_kb.cpp's anonymous-
// namespace split_stacked_grad (not accessible from here), same layout convention.
std::pair<Tensor, Tensor> SplitStackedRelevance(const Tensor& stacked, DeviceBackend* backend) {
    std::vector<int64_t> operand_dims;
    for (int64_t i = 1; i < stacked.rank(); ++i) {
        operand_dims.push_back(stacked.shape().dim(static_cast<size_t>(i)));
    }
    Shape operand_shape(operand_dims);
    const int64_t half = operand_shape.numel();
    Tensor a(operand_shape, backend);
    Tensor b(operand_shape, backend);
    for (int64_t i = 0; i < half; ++i) {
        a.data()[i] = stacked.data()[i];
        b.data()[i] = stacked.data()[half + i];
    }
    return {std::move(a), std::move(b)};
}

}  // namespace

// Measured gap (not just asserted under tolerance): sum_in=4.99998 vs. sum_seed=5, a gap
// of ~2.05e-5 -- the epsilon stabilizer's bounded residual compounding across
// DisjunctionModule's Product rule + LinearModule's epsilon rule (x2), comfortably inside
// this test's 1e-2 tolerance (same tolerance class as lrp_conservation_test.cpp's own
// end-to-end test and every epsilon-stabilized rule in this codebase).
TEST_F(ToyKnowledgeBaseTest, EndToEndRelevanceConservationAcrossFullComposedChain) {
    ToyKnowledgeBase kb(&backend, 2.0f);
    Tensor x = MakeX(&backend);
    (void)kb.forward(x);

    // Arbitrary relevance seed at the rule's own output (DisjunctionModule's cached (N,1)
    // output), distinct per grounding -- same "arbitrary seed, not the output's own value"
    // convention as lrp_conservation_test.cpp's LRPConservationEndToEndTest.
    Tensor relevance_seed(Shape({kN, 1}), &backend, {1.0f, 2.0f, 0.5f, 1.5f});
    LRPRuleConfig config;

    // DisjunctionModule -> stacked relevance over (not_a, not_b).
    Tensor relevance_stacked = kb.disjunction().propagate_relevance(relevance_seed, config);
    auto [relevance_not_a, relevance_not_b] = SplitStackedRelevance(relevance_stacked, &backend);

    // NegationModule (x2): pass-through, unchanged -- relevance at each predicate's sigmoid
    // output (a, b).
    Tensor relevance_a = kb.negation_a().propagate_relevance(relevance_not_a, config);
    Tensor relevance_b = kb.negation_b().propagate_relevance(relevance_not_b, config);

    // Sigmoid: pass-through, unchanged (this test's own resolved design decision, above) --
    // relevance at each predicate's pre-sigmoid LinearModule output (z_a, z_b) is identical
    // to relevance_a/relevance_b.

    // LinearModule (x2): epsilon-rule redistribution down to the raw input x.
    Tensor relevance_x_a = kb.predicate_a().propagate_relevance(relevance_a, config);
    Tensor relevance_x_b = kb.predicate_b().propagate_relevance(relevance_b, config);

    float sum_seed = 0.0f;
    for (int64_t i = 0; i < relevance_seed.numel(); ++i) sum_seed += relevance_seed.data()[i];

    float sum_in = 0.0f;
    for (int64_t i = 0; i < relevance_x_a.numel(); ++i) sum_in += relevance_x_a.data()[i];
    for (int64_t i = 0; i < relevance_x_b.numel(); ++i) sum_in += relevance_x_b.data()[i];

    EXPECT_NEAR(sum_in, sum_seed, 1e-2f)
        << "end-to-end LRP conservation violated across the full toy-KB composed chain";
}

// ---------------------------------------------------------------------------
// End-to-end training -- the phase gate's literal "trainable end-to-end via ordinary
// gradient descent" requirement: satisfaction (1 - loss) must increase over epochs.
// ---------------------------------------------------------------------------

TEST_F(ToyKnowledgeBaseTest, TrainingViaOrdinaryGradientDescentIncreasesSatisfactionOverEpochs) {
    ToyKnowledgeBase kb(&backend, 2.0f);
    SGDOptimizer optimizer(0.1f);
    Tensor x = MakeX(&backend);

    const float initial_loss = kb.train_step(x, optimizer);
    float loss = initial_loss;
    for (int epoch = 0; epoch < 199; ++epoch) {
        loss = kb.train_step(x, optimizer);
    }

    EXPECT_LT(loss, initial_loss) << "satisfaction should have increased (loss decreased) after 200 steps";
    EXPECT_LT(loss, 0.5f * initial_loss) << "expected substantial satisfaction improvement on this toy KB";
}

TEST_F(ToyKnowledgeBaseTest, TrainingLossIsMonotonicallyNonIncreasingAcrossEarlyEpochs) {
    // A weaker, structural sanity check independent of the exact convergence rate: full-batch
    // SGD on a smooth loss with a small enough learning rate should not increase the loss
    // step-over-step this early in training.
    ToyKnowledgeBase kb(&backend, 2.0f);
    SGDOptimizer optimizer(0.05f);
    Tensor x = MakeX(&backend);

    float previous = kb.train_step(x, optimizer);
    for (int epoch = 0; epoch < 20; ++epoch) {
        const float current = kb.train_step(x, optimizer);
        EXPECT_LE(current, previous + 1e-4f) << "epoch " << epoch;
        previous = current;
    }
}

// ---------------------------------------------------------------------------
// Adversarial / boundary-condition
// ---------------------------------------------------------------------------

TEST_F(ToyKnowledgeBaseTest, ForwardRejectsRankOneInput) {
    ToyKnowledgeBase kb(&backend);
    Tensor x(Shape({kN}), &backend, kXValues);
    EXPECT_THROW({ (void)kb.forward(x); }, std::invalid_argument);
}

TEST_F(ToyKnowledgeBaseTest, ForwardRejectsWrongTrailingDimension) {
    ToyKnowledgeBase kb(&backend);
    Tensor x(Shape({2, 2}), &backend, {0.1f, 0.2f, 0.3f, 0.4f});
    EXPECT_THROW({ (void)kb.forward(x); }, std::invalid_argument);
}

TEST_F(ToyKnowledgeBaseTest, ForwardRejectsZeroGroundings) {
    ToyKnowledgeBase kb(&backend);
    Tensor x(Shape({0, 1}), &backend);
    EXPECT_THROW({ (void)kb.forward(x); }, std::invalid_argument);
}

TEST_F(ToyKnowledgeBaseTest, ForwardHandlesSingleGroundingWithoutError) {
    ToyKnowledgeBase kb(&backend);
    Tensor x(Shape({1, 1}), &backend, {0.3f});
    EXPECT_NO_THROW({ (void)kb.forward(x); });
    kb.backward();
}

// Note: no device-guard death test here (unlike Mission 0/1's own Module test files).
// ToyKnowledgeBase::forward()'s device tag is set by each predicate's own LinearModule
// construction (defaulted to Cpu), not by the caller-supplied x's device tag (LinearModule's
// own forward_impl -- see linear_module.cpp -- tags its output with weight_'s device, not
// input's), so every internal Tensor this class's sigmoid()/sigmoid_backward() helpers ever
// see is already Cpu by construction -- their own PULSATRIX_ASSERT(device == Cpu) guards are
// defensive-consistent with this codebase's convention but not reachable through any
// external-boundary input this class's own public API accepts. The device-guard behavior of
// every module ToyKnowledgeBase composes (LinearModule/NegationModule/DisjunctionModule/
// AggregatorModule) is already covered by Missions 0-1's own test suites -- not re-tested
// here to avoid a misleading/unreachable adversarial case.

}  // namespace
}  // namespace pulsatrix
