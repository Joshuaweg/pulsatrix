/** @file datalog_dual_semiring.hpp
 *  @brief Forward-mode-automatic-differentiation semiring (`DualNumber<T>`/`DualSemiring<T>`) --
 *         a second, differentiable real-valued instantiation of Mission 1's generic
 *         `Semiring` trait shape, carrying a value and its derivative w.r.t. one seeded
 *         scalar through every `⊕`/`⊗` the weighted engine performs. Phase 3 Mission 2 of
 *         campaign_exai_dl_library_neuro_symbolic (Neural-Predicate Integration).
 *  @ingroup neuro_symbolic
 */
#pragma once

#include <cmath>
#include <type_traits>

namespace pulsatrix::datalog {

/**
 * @brief A dual number `(value, grad)`: `value` is the ordinary real-valued semiring result,
 *        `grad` is its derivative w.r.t. whichever single base fact's weight was seeded with
 *        `grad = 1` (every other base fact's weight is seeded with `grad = 0`, i.e. treated as
 *        a constant).
 * @note This is forward-mode automatic differentiation, implemented as an ordinary algebraic
 *       (closed-form) construction -- not a new virtual/graph type. See
 *       datalog_semiring.hpp's own trait-shape note: any type supplying `Value`, `zero()`,
 *       `one()`, `add()`, `mul()` is a legal `Semiring` for `naive_evaluate_weighted`/
 *       `semi_naive_evaluate_weighted`, with zero changes to the evaluator itself.
 *       `add`/`mul` below are just the sum rule (`d(a+b) = da + db`) and product rule
 *       (`d(a*b) = da*b + a*db`) of ordinary calculus, applied at every semiring operation the
 *       already-proven-correct weighted engine performs -- so a whole fixpoint evaluation
 *       under `DualSemiring<T>` computes, for every derived fact, both its real-valued
 *       provenance-semiring weight *and* the exact partial derivative of that weight w.r.t.
 *       the one seeded base fact, in a single pass, with no separate symbolic-differentiation
 *       step and no numerical approximation.
 */
template <typename T>
struct DualNumber {
    T value;
    T grad;

    /**
     * @brief Epsilon-tolerant equality, mirroring `RealSemiring<T>`'s own reason for needing
     *        one (see datalog_weighted_engine.cpp's `values_equal`): the weighted engine's
     *        fixpoint-termination check compares a `Semiring::Value` produced by one round
     *        against the previous round's stored value, and floating-point summation order
     *        can differ (bit-for-bit) across `std::unordered_map` iteration orders even when
     *        mathematically identical. `values_equal<Value>` falls back to this `operator==`
     *        for any non-floating-point `Value` (a struct, here), so this type supplies its
     *        own epsilon comparison directly rather than relying on `values_equal`'s
     *        floating-point branch (which never fires for a struct `Value`).
     */
    [[nodiscard]] bool operator==(const DualNumber& other) const {
        return std::fabs(static_cast<double>(value) - static_cast<double>(other.value)) < 1e-9 &&
               std::fabs(static_cast<double>(grad) - static_cast<double>(other.grad)) < 1e-9;
    }
    [[nodiscard]] bool operator!=(const DualNumber& other) const { return !(*this == other); }
};

/**
 * @brief The dual-number semiring: `⊕`/`⊗` are ordinary dual-number addition/multiplication
 *        (sum rule / product rule), `zero = (0, 0)`, `one = (1, 0)` -- the multiplicative
 *        identity carries no derivative of its own, matching the fact that a constant
 *        contributes nothing to any derivative.
 * @note Stage 3 design decision 2 (this mission): differentiability flows through the
 *       Datalog evaluation via this closed-form/algebraic dual-number construction, *not*
 *       via `ComputationGraph`/`Autograd` node-level wiring -- see this file's own header
 *       comment and the mission file's Stage 3 write-up for the recon that established this
 *       is the only structurally sound option (`Node`/`OpType` carry no value payload and
 *       have no shape for a variable-arity, substitution-joined Datalog rule body; `Autograd`
 *       is scoped to single-parent `Module`-shaped nodes). The *transitive* half of the
 *       requirement -- flowing the resulting derivative into the neural predicate's own
 *       `Module` (`LinearModule`) parameters -- is handled separately, by seeding a real
 *       `Tensor` gradient and calling that `Module`'s own real `backward()`, exactly mirroring
 *       this codebase's own `ToyKnowledgeBase::backward()` precedent (hand-chained
 *       `Module::backward()` calls, no `ComputationGraph`/`Autograd` object involved there
 *       either).
 */
template <typename T>
struct DualSemiring {
    static_assert(std::is_floating_point_v<T>, "DualSemiring<T> requires a floating-point T (double or float)");

    using Value = DualNumber<T>;

    [[nodiscard]] static constexpr Value zero() { return Value{static_cast<T>(0), static_cast<T>(0)}; }
    [[nodiscard]] static constexpr Value one() { return Value{static_cast<T>(1), static_cast<T>(0)}; }
    [[nodiscard]] static constexpr Value add(Value a, Value b) { return Value{a.value + b.value, a.grad + b.grad}; }
    [[nodiscard]] static constexpr Value mul(Value a, Value b) {
        return Value{a.value * b.value, a.grad * b.value + a.value * b.grad};
    }
};

}  // namespace pulsatrix::datalog
