/** @file satisfaction_loss.hpp
 *  @brief Real-Logic-style knowledge-base satisfaction loss -- Phase 1 Mission 1 of
 *         campaign_exai_dl_library_neuro_symbolic.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/aggregator_module.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief `loss = 1 - agg_p(truth_values)` -- the standard LTN "Real Logic" training
 *        objective (research_2026_neuro_symbolic_ai.md §1/§2): maximizing a knowledge
 *        base's aggregated satisfaction via ordinary gradient descent is the same as
 *        minimizing this loss.
 *
 * @note **Design decision (mission_1_aggregator_satisfaction_loss.md Stage 3, resolved):
 *       not a Module subclass -- confirmed directly against mse_loss.hpp's own stated
 *       rationale, not assumed by analogy alone.** MSELoss's own header states losses are
 *       "the seed point relevance/gradient propagation starts from, not something a
 *       propagate_relevance rule is defined for -- LRP explains a model's prediction, not
 *       the loss function used to train it." That rationale is about *role* (a loss sits
 *       outside the relevance-bearing forward computation entirely, by definition), not
 *       about argument count -- so it applies identically here even though this loss's
 *       arity differs from MSELoss's. MSELoss takes exactly two tensors (prediction,
 *       target); SatisfactionLoss takes exactly **one** (a single formula's already-composed
 *       per-grounding truth degrees) -- a genuinely different input shape, checked here
 *       rather than silently assumed identical, but the same "loss, not Module"
 *       classification either way.
 * @note **Composes Mission 0's ConjunctionModule/DisjunctionModule/NegationModule with
 *       this mission's AggregatorModule the way every multi-module pipeline in this
 *       codebase composes** -- by chaining calls at the call site, not by one class owning
 *       every piece (e.g. XorNetwork chains LinearModule/ReluModule externally rather than
 *       a monolithic class inlining both). This class's own single responsibility is the
 *       aggregation-and-complement step (`1 - agg_p(...)`) -- it owns exactly one
 *       AggregatorModule instance. Grounding a specific first-order formula by chaining
 *       Conjunction/Disjunction/Negation calls over predicate outputs happens upstream, at
 *       the call site; which concrete formula that is is Phase 1 Mission 2's own scope
 *       (the toy knowledge base), not re-implemented or anticipated here.
 * @note **Scope restriction: `truth_values` must be rank 1** -- a single formula's
 *       per-grounding truth degrees, batch axis only, no other dimensions. A multi-formula
 *       knowledge base (several formulas' satisfaction degrees combined into one training
 *       signal) is Mission 2's scope, not this class's; documented explicitly here rather
 *       than silently generalized ahead of that mission's own design work.
 */
class SatisfactionLoss {
public:
    /**
     * @brief Constructs a satisfaction loss with an internally-owned p-mean aggregator.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this loss.
     * @param p Aggregator exponent, forwarded to AggregatorModule (see its own constructor
     *        note on p == 0 being rejected there). Defaults to 2.0, matching
     *        AggregatorModule's own default.
     */
    explicit SatisfactionLoss(DeviceBackend* backend, float p = 2.0f);

    /**
     * @brief Aggregates truth_values into a single scalar satisfaction degree and returns
     *        `1 - that`.
     * @param truth_values Rank-1 tensor of per-grounding formula-truth degrees. Must be
     *        rank 1 -- see the class note on scope.
     * @return `1 - agg_p(truth_values)`, the scalar loss to minimize via gradient descent.
     * @throws std::invalid_argument if truth_values is not rank 1 -- this method's own
     *        documented scope restriction (external boundary), checked before delegating
     *        to AggregatorModule (which is itself rank-agnostic and would not otherwise
     *        reject a higher-rank input).
     * @note Inherits AggregatorModule::forward()'s own empty-input rejection (via
     *       Module::forward's NVI precondition) for a zero-length truth_values -- not
     *       re-implemented here.
     */
    [[nodiscard]] float forward(const Tensor& truth_values);

    /**
     * @brief Gradient w.r.t. truth_values: `d(loss)/d(sat) == -1` exactly (the complement
     *        is affine), propagated back through the internally-owned AggregatorModule's
     *        own backward().
     * @return Gradient tensor, same shape as the truth_values passed to forward().
     * @throws std::logic_error if called before any forward().
     * @note Not `const` -- unlike MSELoss::backward(), which only reads cached members,
     *       this method delegates to AggregatorModule::backward(), which Module declares
     *       non-const (it is a virtual method every mutable-state subclass overrides).
     *       Documented explicitly as a deliberate deviation from MSELoss's own const
     *       signature, not an oversight.
     */
    [[nodiscard]] Tensor backward();

private:
    DeviceBackend* backend_;
    AggregatorModule aggregator_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
