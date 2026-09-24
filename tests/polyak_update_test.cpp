#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "exai/cpu_backend.hpp"
#include "exai/dqn_target.hpp"  // SyncTargetNetwork -- the tau == 1 cross-check
#include "exai/linear_module.hpp"
#include "exai/polyak_update.hpp"
#include "exai/relu_module.hpp"
#include "exai/sequential_module.hpp"

namespace exai {
namespace {

// Two structurally identical two-layer networks with *different, known, non-zero* weights on
// both sides. The destination deliberately does not keep LinearModule's zero initialization
// (unlike SyncTargetNetworkTest's fixture): a soft update reads the destination's current value
// as an operand, so a zero destination would make `tau*source + (1-tau)*destination`
// indistinguishable from `tau*source` and hide the entire second half of the formula.
class PolyakUpdateTest : public ::testing::Test {
protected:
    PolyakUpdateTest()
        : source_a(2, 3, &backend),
          source_b(3, 2, &backend),
          destination_a(2, 3, &backend),
          destination_b(3, 2, &backend),
          source({&source_a, &source_b}),
          destination({&destination_a, &destination_b}) {
        source_a.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        source_a.set_bias({7.0f, 8.0f, 9.0f});
        source_b.set_weight({10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f});
        source_b.set_bias({16.0f, 17.0f});

        destination_a.set_weight({-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f});
        destination_a.set_bias({-7.0f, -8.0f, -9.0f});
        destination_b.set_weight({-10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f});
        destination_b.set_bias({-16.0f, -17.0f});
    }

    CPUBackend backend;
    LinearModule source_a;
    LinearModule source_b;
    LinearModule destination_a;
    LinearModule destination_b;
    SequentialModule source;
    SequentialModule destination;
};

// ---------------------------------------------------------------------------------------
// The formula, hand-derived.
// ---------------------------------------------------------------------------------------

// tau = 0.1, so every element becomes 0.1*source + 0.9*destination. With the fixture's
// destination being exactly -source, that is 0.1*s + 0.9*(-s) = -0.8*s -- hand-computed per
// element below rather than expressed as a loop over the formula, so the test cannot agree with
// a wrong implementation by sharing its arithmetic.
TEST_F(PolyakUpdateTest, BlendsEveryParameterPerHandDerivedFormula) {
    PolyakUpdate(source, destination, 0.1f);

    // destination_a.weight was {-1..-6}, source_a.weight {1..6}: 0.1*1 + 0.9*(-1) = -0.8, etc.
    EXPECT_NEAR(destination_a.weight().data()[0], -0.8f, 1e-6f);
    EXPECT_NEAR(destination_a.weight().data()[1], -1.6f, 1e-6f);
    EXPECT_NEAR(destination_a.weight().data()[5], -4.8f, 1e-6f);
    EXPECT_NEAR(destination_a.bias().data()[0], -5.6f, 1e-6f);   // 0.1*7 + 0.9*(-7)
    EXPECT_NEAR(destination_a.bias().data()[2], -7.2f, 1e-6f);   // 0.1*9 + 0.9*(-9)
    EXPECT_NEAR(destination_b.weight().data()[0], -8.0f, 1e-6f); // 0.1*10 + 0.9*(-10)
    EXPECT_NEAR(destination_b.weight().data()[5], -12.0f, 1e-6f);
    EXPECT_NEAR(destination_b.bias().data()[1], -13.6f, 1e-6f);  // 0.1*17 + 0.9*(-17)

    // The source is untouched -- this is a one-directional blend.
    EXPECT_FLOAT_EQ(source_a.weight().data()[0], 1.0f);
    EXPECT_FLOAT_EQ(source_b.bias().data()[1], 17.0f);
}

// The same property with binary-exact arithmetic, so it can be asserted as *equality* rather
// than a tolerance: tau = 0.25 and 0.75 are both exactly representable, and the chosen operands
// make the products exact too. 0.25*4 + 0.75*(-4) = 1 - 3 = -2, exactly.
TEST_F(PolyakUpdateTest, BlendIsExactForBinaryRepresentableTau) {
    PolyakUpdate(source, destination, 0.25f);

    EXPECT_FLOAT_EQ(destination_a.weight().data()[3], -2.0f);   // source 4, destination -4
    EXPECT_FLOAT_EQ(destination_a.bias().data()[1], -4.0f);     // source 8, destination -8
    EXPECT_FLOAT_EQ(destination_b.weight().data()[2], -6.0f);   // source 12, destination -12
}

// The exponential-moving-average property, which is the whole reason the update is "soft": the
// destination approaches the source geometrically, retaining (1-tau)^k of its original offset
// after k calls. tau = 0.5 keeps every step binary-exact, so this is an equality test:
// starting at 0 toward 8 -- 4, then 6, then 7.
TEST_F(PolyakUpdateTest, RepeatedUpdatesMoveTheDestinationGeometricallyTowardTheSource) {
    LinearModule from(1, 1, &backend);
    LinearModule into(1, 1, &backend);
    from.set_weight({8.0f});
    from.set_bias({0.0f});
    into.set_weight({0.0f});
    into.set_bias({0.0f});

    PolyakUpdate(from, into, 0.5f);
    EXPECT_FLOAT_EQ(into.weight().data()[0], 4.0f);
    PolyakUpdate(from, into, 0.5f);
    EXPECT_FLOAT_EQ(into.weight().data()[0], 6.0f);
    PolyakUpdate(from, into, 0.5f);
    EXPECT_FLOAT_EQ(into.weight().data()[0], 7.0f);
    // Never overshoots and never quite arrives -- the defining behavior of the soft update.
    EXPECT_LT(into.weight().data()[0], 8.0f);
}

// ---------------------------------------------------------------------------------------
// The tau == 1 cross-check against the already-verified SyncTargetNetwork.
// ---------------------------------------------------------------------------------------

// tau = 1 degenerates the blend to `1*source + 0*destination`, i.e. exactly Phase 2's hard
// copy. Checked against SyncTargetNetwork's *own output* on an identically-initialized second
// destination rather than against the source's values directly: that makes this a real
// cross-check between two independent implementations of the same limiting case, so if either
// one drifts the disagreement surfaces here.
TEST_F(PolyakUpdateTest, TauOneReproducesSyncTargetNetworkExactly) {
    LinearModule hard_a(2, 3, &backend);
    LinearModule hard_b(3, 2, &backend);
    hard_a.set_weight({-1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f});
    hard_a.set_bias({-7.0f, -8.0f, -9.0f});
    hard_b.set_weight({-10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f});
    hard_b.set_bias({-16.0f, -17.0f});
    SequentialModule hard_destination({&hard_a, &hard_b});

    // Precondition: the two destinations start identical, and differ from the source.
    std::vector<ParamRef> soft_params = destination.parameters();
    std::vector<ParamRef> hard_params = hard_destination.parameters();
    ASSERT_EQ(soft_params.size(), hard_params.size());
    for (size_t p = 0; p < soft_params.size(); ++p) {
        ASSERT_EQ(soft_params[p].value->numel(), hard_params[p].value->numel()) << "parameter " << p;
        for (int64_t e = 0; e < soft_params[p].value->numel(); ++e) {
            ASSERT_FLOAT_EQ(soft_params[p].value->data()[e], hard_params[p].value->data()[e])
                << "parameter " << p << ", element " << e;
        }
    }
    ASSERT_NE(source.parameters()[0].value->data()[0], soft_params[0].value->data()[0]);

    PolyakUpdate(source, destination, 1.0f);
    SyncTargetNetwork(source, hard_destination);

    for (size_t p = 0; p < soft_params.size(); ++p) {
        for (int64_t e = 0; e < soft_params[p].value->numel(); ++e) {
            EXPECT_EQ(soft_params[p].value->data()[e], hard_params[p].value->data()[e])
                << "tau == 1 diverged from SyncTargetNetwork at parameter " << p << ", element " << e;
        }
    }
    // And both really did move onto the source, so the agreement is not two shared no-ops.
    EXPECT_EQ(destination_a.weight().data()[0], 1.0f);
    EXPECT_EQ(hard_a.weight().data()[0], 1.0f);
}

// ---------------------------------------------------------------------------------------
// Buffer discipline -- inherited verbatim from SyncTargetNetwork's contract, and load-bearing
// in one extra way here: the destination's buffer must survive *between* calls for successive
// updates to compose into a moving average at all.
// ---------------------------------------------------------------------------------------

TEST_F(PolyakUpdateTest, UpdatePreservesTheDestinationsParameterStorageIdentity) {
    std::vector<ParamRef> before = destination.parameters();
    const Tensor* value_before = before[0].value;
    const float* buffer_before = before[0].value->data();

    PolyakUpdate(source, destination, 0.1f);

    std::vector<ParamRef> after = destination.parameters();
    EXPECT_EQ(after[0].value, value_before);
    EXPECT_EQ(after[0].value->data(), buffer_before) << "update replaced the destination's parameter buffer";
}

// The blend must be a real value computation, not an alias: the source is mutated *through its
// own buffer* afterwards (not via set_weight(), which replaces the Tensor object wholesale and
// so would mask a genuine aliasing bug).
TEST_F(PolyakUpdateTest, MutatingSourceAfterUpdateLeavesDestinationUnaffected) {
    PolyakUpdate(source, destination, 0.1f);
    const float blended = destination_a.weight().data()[0];

    for (ParamRef& param : source.parameters()) {
        for (int64_t e = 0; e < param.value->numel(); ++e) {
            param.value->data()[e] = -999.0f;
        }
    }

    EXPECT_FLOAT_EQ(destination_a.weight().data()[0], blended);
    EXPECT_FLOAT_EQ(source_a.weight().data()[0], -999.0f);  // ...and the source really changed
}

TEST_F(PolyakUpdateTest, UpdateDoesNotTouchGradients) {
    std::vector<ParamRef> source_params = source.parameters();
    std::vector<ParamRef> destination_params = destination.parameters();
    source_params[0].grad->data()[0] = 5.0f;
    destination_params[0].grad->data()[0] = 3.0f;

    PolyakUpdate(source, destination, 0.1f);

    EXPECT_FLOAT_EQ(destination_params[0].grad->data()[0], 3.0f) << "a soft update is a pure value blend";
    EXPECT_FLOAT_EQ(source_params[0].grad->data()[0], 5.0f);
}

TEST_F(PolyakUpdateTest, UpdatedDestinationAtTauOneProducesTheSameForwardOutputAsTheSource) {
    PolyakUpdate(source, destination, 1.0f);
    Tensor input(Shape({1, 2}), &backend, {0.5f, -1.5f});

    const Tensor from_source = source.forward(input);
    const Tensor from_destination = destination.forward(input);

    ASSERT_EQ(from_source.numel(), from_destination.numel());
    for (int64_t i = 0; i < from_source.numel(); ++i) {
        EXPECT_FLOAT_EQ(from_destination.data()[i], from_source.data()[i]) << "index " << i;
    }
}

// ---------------------------------------------------------------------------------------
// External-boundary validation.
// ---------------------------------------------------------------------------------------

// tau == 0 is rejected rather than accepted as a no-op: a target network that provably never
// moves is a real caller error (typically an uninitialized hyperparameter), and silently doing
// nothing forever is the worst possible way to report it.
TEST_F(PolyakUpdateTest, ThrowsOnTauZero) {
    EXPECT_THROW({ PolyakUpdate(source, destination, 0.0f); }, std::invalid_argument);
    // Rejected before anything was written.
    EXPECT_FLOAT_EQ(destination_a.weight().data()[0], -1.0f);
}

TEST_F(PolyakUpdateTest, ThrowsOnTauOutsideTheUnitInterval) {
    EXPECT_THROW({ PolyakUpdate(source, destination, -0.1f); }, std::invalid_argument);
    EXPECT_THROW({ PolyakUpdate(source, destination, 1.0001f); }, std::invalid_argument);
    EXPECT_THROW({ PolyakUpdate(source, destination, std::numeric_limits<float>::quiet_NaN()); },
                 std::invalid_argument);
    EXPECT_FLOAT_EQ(destination_a.weight().data()[0], -1.0f);
}

TEST_F(PolyakUpdateTest, ThrowsOnMismatchedParameterCounts) {
    LinearModule lonely(2, 3, &backend);

    // 4 parameters (two layers) vs 2 (one layer).
    EXPECT_THROW({ PolyakUpdate(source, lonely, 0.1f); }, std::invalid_argument);
    EXPECT_THROW({ PolyakUpdate(lonely, source, 0.1f); }, std::invalid_argument);
}

TEST_F(PolyakUpdateTest, ThrowsOnMismatchedParameterShapes) {
    LinearModule wrong_a(2, 3, &backend);
    LinearModule wrong_b(3, 5, &backend);  // same parameter *count*, different shapes
    SequentialModule wrong({&wrong_a, &wrong_b});

    EXPECT_THROW({ PolyakUpdate(source, wrong, 0.1f); }, std::invalid_argument);
}

// A rejected update must not leave the destination half-blended.
TEST_F(PolyakUpdateTest, AMismatchedShapeIsDetectedBeforeAnyPartialBlendOfThatParameter) {
    LinearModule wrong_a(2, 3, &backend);
    LinearModule wrong_b(3, 5, &backend);
    SequentialModule wrong({&wrong_a, &wrong_b});

    EXPECT_THROW({ PolyakUpdate(source, wrong, 0.1f); }, std::invalid_argument);
    // Parameter index 2 (wrong_b's weight) is where the shapes first disagree; it must be
    // untouched, even though indices 0 and 1 matched and were legitimately blended.
    EXPECT_FLOAT_EQ(wrong_b.weight().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(wrong_b.bias().data()[0], 0.0f);
}

TEST_F(PolyakUpdateTest, UpdatingAParameterlessModulePairIsALegalNoOp) {
    // Not every Module has parameters; an empty parameter list is a legitimate answer (see
    // Module::parameters()'s own note), not an architecture mismatch to reject.
    ReluModule a(&backend);
    ReluModule b(&backend);
    EXPECT_NO_THROW({ PolyakUpdate(a, b, 0.1f); });
}

// No death test, by design and for the identical reason SyncTargetNetwork has none: PolyakUpdate
// takes no caller-supplied Tensor, only Modules, whose parameter buffers were validated when
// those Modules were constructed. There is no EXAI_ASSERT to trip and a guard here would be
// untestable dead code, not a real safety net (mission_host_loop_guards.md).

}  // namespace
}  // namespace exai
