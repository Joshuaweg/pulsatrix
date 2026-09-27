#include "pulsatrix/neuro_symbolic_datalog_bridge.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/datalog_lrp.hpp"
#include "pulsatrix/datalog_semiring.hpp"
#include "pulsatrix/datalog_weighted_engine.hpp"

namespace pulsatrix::datalog {
namespace {

// ---------------------------------------------------------------------------
// Hand-derived closed form (worked on paper before writing this test, per this project's
// standing discipline -- see neuro_symbolic_datalog_bridge.hpp's own class-level doc comment
// for the full toy-program write-up):
//
//   edge(a,b) = s = sigmoid(W*x + b),  W=0.6, b=0.0, x=0.5  (this bridge's neural predicate)
//   edge(a,c) = 0.4,  edge(b,d) = 0.6,  edge(c,d) = 0.3       (constants)
//   ancestor(a,d) = edge(a,b)*edge(b,d) + edge(a,c)*edge(c,d) = 0.6*s + 0.12
//
//   z = W*x + b = 0.3
//   s = sigmoid(0.3) = 0.574442516811659
//   Q = 0.6*s + 0.12 = 0.4646655100869954
//
//   dQ/ds = 0.6                              (the closed form's own coefficient on s)
//   ds/dz = s*(1-s) = 0.24445831169074586    (sigmoid's own closed-form derivative)
//   dQ/dz = dQ/ds * ds/dz = 0.1466749870144475
//   dQ/dW = dQ/dz * x = 0.07333749350722375  (LinearModule: dz/dW = x, N=1/in=1/out=1)
//   dQ/db = dQ/dz * 1 = 0.1466749870144475   (LinearModule: dz/db = 1)
// ---------------------------------------------------------------------------

constexpr float kW = 0.6f;
constexpr float kB = 0.0f;
constexpr float kX = 0.5f;
constexpr double kExpectedS = 0.574442516811659;
constexpr double kExpectedQ = 0.4646655100869954;
constexpr double kExpectedDQdS = 0.6;
constexpr double kExpectedDQdW = 0.07333749350722375;
constexpr double kExpectedDQdB = 0.1466749870144475;

class NeuralPredicateDatalogBridgeTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor MakeX(float value = kX) { return Tensor(Shape({1, 1}), &backend, {value}); }
};

// ---------------------------------------------------------------------------
// Forward correctness -- query weight matches the hand-derived closed form.
// ---------------------------------------------------------------------------

TEST_F(NeuralPredicateDatalogBridgeTest, EvaluateProducesQueryWeightMatchingHandDerivedClosedForm) {
    NeuralPredicateDatalogBridge bridge(&backend);
    NeuralPredicateQueryResult result = bridge.evaluate(MakeX());

    EXPECT_NEAR(result.query_weight, kExpectedQ, 1e-4);
}

TEST_F(NeuralPredicateDatalogBridgeTest, EvaluateProducesGradientWrtPredicateOutputMatchingHandDerivedClosedForm) {
    NeuralPredicateDatalogBridge bridge(&backend);
    NeuralPredicateQueryResult result = bridge.evaluate(MakeX());

    EXPECT_NEAR(result.grad_wrt_predicate_output, kExpectedDQdS, 1e-4);
}

// ---------------------------------------------------------------------------
// Gradient verification against finite differences -- the literal phase-gate wording
// ("matches a hand-derived closed form within a defined tolerance"), checked against BOTH
// the closed form (above) and an independent numerical estimate.
// ---------------------------------------------------------------------------

TEST_F(NeuralPredicateDatalogBridgeTest, GradWrtPredicateOutputMatchesFiniteDifferenceOnTheRawEngine) {
    // Central finite difference of Q(s) = 0.6*s + 0.12 directly against the real weighted
    // engine (not the bridge's own DualSemiring pass) -- an independent computation path.
    const double h = 1e-5;
    auto query_at = [](double s_value) {
        WeightedFactDatabase<double> facts = NeuralPredicateDatalogBridge::constant_edge_facts();
        facts.set(Atom("edge", {Term::make_constant("a"), Term::make_constant("b")}), s_value);
        WeightedFactDatabase<double> result =
            naive_evaluate_weighted<RealSemiring<double>>(NeuralPredicateDatalogBridge::diamond_ancestor_program(), facts);
        return result.weight_of(Atom("ancestor", {Term::make_constant("a"), Term::make_constant("d")}), 0.0);
    };
    const double fd_grad = (query_at(kExpectedS + h) - query_at(kExpectedS - h)) / (2.0 * h);

    NeuralPredicateDatalogBridge bridge(&backend);
    NeuralPredicateQueryResult result = bridge.evaluate(MakeX());

    EXPECT_NEAR(result.grad_wrt_predicate_output, fd_grad, 1e-4);
    EXPECT_NEAR(fd_grad, kExpectedDQdS, 1e-4);
}

TEST_F(NeuralPredicateDatalogBridgeTest, BackwardAccumulatesWeightGradMatchingHandDerivedClosedFormAndFiniteDifference) {
    NeuralPredicateDatalogBridge bridge(&backend);
    (void)bridge.evaluate(MakeX());
    bridge.backward();

    EXPECT_NEAR(bridge.predicate().weight_grad().at({0, 0}), kExpectedDQdW, 1e-3);

    // Finite difference w.r.t. W: perturb the predicate's own weight, re-evaluate end-to-end.
    const float h = 1e-3f;
    NeuralPredicateDatalogBridge plus(&backend);
    plus.predicate().set_weight({kW + h});
    plus.predicate().set_bias({kB});
    double q_plus = plus.evaluate(MakeX()).query_weight;

    NeuralPredicateDatalogBridge minus(&backend);
    minus.predicate().set_weight({kW - h});
    minus.predicate().set_bias({kB});
    double q_minus = minus.evaluate(MakeX()).query_weight;

    double fd_dQdW = (q_plus - q_minus) / (2.0 * h);
    EXPECT_NEAR(bridge.predicate().weight_grad().at({0, 0}), fd_dQdW, 5e-3);
}

TEST_F(NeuralPredicateDatalogBridgeTest, BackwardAccumulatesBiasGradMatchingHandDerivedClosedFormAndFiniteDifference) {
    NeuralPredicateDatalogBridge bridge(&backend);
    (void)bridge.evaluate(MakeX());
    bridge.backward();

    EXPECT_NEAR(bridge.predicate().bias_grad().at({0}), kExpectedDQdB, 1e-3);

    const float h = 1e-3f;
    NeuralPredicateDatalogBridge plus(&backend);
    plus.predicate().set_weight({kW});
    plus.predicate().set_bias({kB + h});
    double q_plus = plus.evaluate(MakeX()).query_weight;

    NeuralPredicateDatalogBridge minus(&backend);
    minus.predicate().set_weight({kW});
    minus.predicate().set_bias({kB - h});
    double q_minus = minus.evaluate(MakeX()).query_weight;

    double fd_dQdB = (q_plus - q_minus) / (2.0 * h);
    EXPECT_NEAR(bridge.predicate().bias_grad().at({0}), fd_dQdB, 5e-3);
}

// ---------------------------------------------------------------------------
// Composition with ordinary constant-weighted facts -- constant_edge_facts() never changes
// across evaluate() calls, only edge(a,b) (the neural predicate) does.
// ---------------------------------------------------------------------------

TEST_F(NeuralPredicateDatalogBridgeTest, QueryWeightChangesWithNeuralPredicateInputButProgramStructureStaysFixed) {
    NeuralPredicateDatalogBridge bridge(&backend);
    double q1 = bridge.evaluate(MakeX(0.5f)).query_weight;
    double q2 = bridge.evaluate(MakeX(-0.5f)).query_weight;
    EXPECT_NE(q1, q2);
}

// ---------------------------------------------------------------------------
// Adversarial/boundary tests -- every new public entry point.
// ---------------------------------------------------------------------------

TEST_F(NeuralPredicateDatalogBridgeTest, EvaluateThrowsOnWrongRank) {
    NeuralPredicateDatalogBridge bridge(&backend);
    Tensor bad(Shape({1}), &backend, {kX});
    EXPECT_THROW(bridge.evaluate(bad), std::invalid_argument);
}

TEST_F(NeuralPredicateDatalogBridgeTest, EvaluateThrowsOnWrongLeadingDimension) {
    NeuralPredicateDatalogBridge bridge(&backend);
    Tensor bad(Shape({2, 1}), &backend, {kX, kX});
    EXPECT_THROW(bridge.evaluate(bad), std::invalid_argument);
}

TEST_F(NeuralPredicateDatalogBridgeTest, EvaluateThrowsOnWrongTrailingDimension) {
    NeuralPredicateDatalogBridge bridge(&backend);
    Tensor bad(Shape({1, 2}), &backend, {kX, kX});
    EXPECT_THROW(bridge.evaluate(bad), std::invalid_argument);
}

TEST_F(NeuralPredicateDatalogBridgeTest, BackwardThrowsIfCalledBeforeEvaluate) {
    NeuralPredicateDatalogBridge bridge(&backend);
    EXPECT_THROW(bridge.backward(), std::logic_error);
}

// ---------------------------------------------------------------------------
// Phase 3 Mission 3 -- LRP for the Datalog/provenance-semiring circuit, extended
// end-to-end through this bridge's own neural predicate. Design question 2's resolution:
// relevance does NOT stop at edge(a,b) -- it continues through sigmoid (pass-through) and
// LinearModule::propagate_relevance() (composition, no new rule) down to the predicate's raw
// input x.
//
// Hand-derived closed form (worked before this test, mirroring the gradient derivation
// above): with Q = 0.6*s + 0.12 and the two-path diamond structure
// (edge(a,b)*edge(b,d) + edge(a,c)*edge(c,d)), the eps=0 weighted-sum/bilinear-split
// composition (see datalog_lrp.hpp's own worked example) gives edge(a,b)'s own
// Datalog-level relevance as:
//   R_s = (0.6*s / Q) / 2 = 0.3*s / Q
// (the bilinear split of a two-factor product is always exactly half-and-half at eps=0,
// independent of the operand values, and the ⊕-split for a two-term sum is that term's own
// share of the total). This is verified below against the implementation's actual output at
// eps=0, not merely asserted to hold generically.
// ---------------------------------------------------------------------------

TEST_F(NeuralPredicateDatalogBridgeTest, PropagateRelevanceMatchesHandDerivedClosedFormAtEpsilonZero) {
    NeuralPredicateDatalogBridge bridge(&backend);
    NeuralPredicateQueryResult query = bridge.evaluate(MakeX());

    LRPRuleConfig zero_eps;
    zero_eps.epsilon = 0.0f;
    NeuralPredicateRelevanceResult relevance = bridge.propagate_relevance(1.0, zero_eps);

    const double s = kExpectedS;
    const double q = query.query_weight;
    const double expected_r_s = (0.6 * s) / q / 2.0;

    const double r_s = relevance.base_fact_relevance.at(Atom("edge", {Term::make_constant("a"), Term::make_constant("b")}));
    EXPECT_NEAR(r_s, expected_r_s, 1e-4);
}

TEST_F(NeuralPredicateDatalogBridgeTest, PropagateRelevanceConservesEndToEndThroughPredicateToRawInput) {
    NeuralPredicateDatalogBridge bridge(&backend);
    (void)bridge.evaluate(MakeX());

    constexpr double kSeed = 5.0;
    NeuralPredicateRelevanceResult relevance = bridge.propagate_relevance(kSeed);

    const double r_s = relevance.base_fact_relevance.at(Atom("edge", {Term::make_constant("a"), Term::make_constant("b")}));

    double other_base_facts_total = 0.0;
    for (const auto& [atom, r] : relevance.base_fact_relevance) {
        if (atom != Atom("edge", {Term::make_constant("a"), Term::make_constant("b")})) {
            other_base_facts_total += r;
        }
    }

    const double r_x = static_cast<double>(relevance.relevance_wrt_x.at({0, 0}));

    // The predicate's own LinearModule::propagate_relevance() conserves r_s into r_x
    // near-exactly (single in_features/out_features, so this is a trivial one-term epsilon
    // split) -- composition, not a new rule.
    EXPECT_NEAR(r_x, r_s, 1e-3);

    // Full end-to-end conservation: every base fact's relevance except edge(a,b)'s own
    // (which continued on into r_x instead of terminating there) sums with r_x back to the
    // seeded relevance -- a genuine trace from the Datalog query to the predicate's raw input.
    EXPECT_NEAR(other_base_facts_total + r_x, kSeed, 1e-2);
}

TEST_F(NeuralPredicateDatalogBridgeTest, PropagateRelevanceThrowsIfCalledBeforeEvaluate) {
    NeuralPredicateDatalogBridge bridge(&backend);
    EXPECT_THROW(bridge.propagate_relevance(1.0), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix::datalog
