/** @file neuro_symbolic_toy_kb.hpp
 *  @brief Toy knowledge-base training demo -- Phase 1 Mission 2 of
 *         campaign_exai_dl_library_neuro_symbolic (differentiable fuzzy-logic core's own
 *         correctness oracle: proves Missions 0-1's operators compose into a real,
 *         trainable Logic Tensor Network).
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/disjunction_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/negation_module.hpp"
#include "pulsatrix/satisfaction_loss.hpp"
#include "pulsatrix/sgd_optimizer.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief A small, hand-traceable knowledge base: two neural predicates `A(x)`/`B(x)`
 *        (each a sigmoid-squashed `LinearModule(1,1)`, producing a fuzzy truth degree in
 *        `(0,1)`), one logical rule `A(x) -> not(B(x))`, expressed via De Morgan
 *        (`not(A(x)) or not(B(x))`, Product t-conorm/t-negation) using only
 *        `NegationModule`/`DisjunctionModule` (Mission 0) -- no dedicated `Implication`
 *        `Module` -- aggregated across synthetic groundings into one scalar via
 *        `SatisfactionLoss` (Mission 1).
 *
 * @note **Not a `Module` subclass** -- directly analogous to `XorNetwork`'s own
 *       classification (mission_training_loop.md): a composing class that owns and chains
 *       real `Module`s/`SatisfactionLoss` but is not itself relevance-bearing in the
 *       `propagate_relevance` sense. See mission_2_toy_kb_training_demo.md's Stage 3
 *       "Interface/class design" note.
 * @note **De Morgan is exact, not approximate, for this rule under the Product family**:
 *       `A -> not(B) == not(A) or not(B)` is the ordinary propositional-logic identity
 *       (`P -> Q == not(P) or Q`, here `Q = not(B)`); with Product t-conorm/t-norm, the
 *       composed rule's truth degree collapses algebraically to the closed form
 *       `rule_i = 1 - a_i*b_i` (`disjunction_Product(not_a, not_b) = not_a+not_b-not_a*not_b
 *       = 1-a*b` after substituting `not_a=1-a`, `not_b=1-b`) -- verified as a dedicated
 *       test comparing the composed Negation->Negation->Disjunction pipeline against this
 *       closed form elementwise, not merely asserted. See mission file Stage 3, Decision 1.
 * @note **Sigmoid squashing is file-local glue, not a new library `Module`** -- pulsatrix
 *       has no standalone `Sigmoid` `Module` (checked); adding one for a single toy demo
 *       would be scope creep the mission file explicitly cautions against (by direct
 *       extension of its `Implication`-`Module` caution). `sigmoid`/`sigmoid_backward` are
 *       small anonymous-namespace free functions in neuro_symbolic_toy_kb.cpp, the same
 *       "small, per-file helper" convention `ConjunctionModule`/`DisjunctionModule`'s own
 *       `split_operands`/`combine_operands` already established (mission_0's Stage 3).
 * @note **No direct (label) supervision on `A`/`B`** -- the only training signal is the
 *       rule's own satisfaction loss, per the phase gate's literal "trainable end-to-end...
 *       on the toy knowledge base" wording. The honest degenerate optimum (push `a_i*b_i`
 *       toward 0 per grounding) is exactly what "satisfaction increases over epochs" is
 *       expected to show -- not disguised as a more semantically rich result than it is.
 */
class ToyKnowledgeBase {
public:
    /**
     * @brief Constructs the toy KB with fixed, small, sign-varied initial predicate
     *        weights (mission file Stage 3's own hand-picked values: `W_A=0.6, b_A=0.0`,
     *        `W_B=-0.4, b_B=0.0`), mirroring XorNetwork's documented non-zero-init
     *        rationale.
     * @param backend Backend to compute through. Not owned; must outlive this object.
     * @param p Aggregator exponent, forwarded to the owned SatisfactionLoss. Defaults to
     *        2.0 (RMS), matching SatisfactionLoss's own default.
     */
    explicit ToyKnowledgeBase(DeviceBackend* backend, float p = 2.0f);

    /**
     * @brief Runs the full forward pass: A(x)/B(x) (Linear+sigmoid) -> not(A)/not(B)
     *        (NegationModule) -> rule = not(A) or not(B) (DisjunctionModule, Product) ->
     *        loss = 1 - agg_p(rule) (SatisfactionLoss). Caches every intermediate backward()
     *        needs.
     * @param x Shape (N, 1) -- N synthetic groundings' shared feature value.
     * @return The scalar satisfaction loss (to minimize via gradient descent).
     * @throws std::invalid_argument if x is not rank 2 with a trailing dimension of 1, or
     *         if N == 0 -- external boundary, this class's own documented shape contract
     *         (not delegated silently to whichever downstream module would happen to
     *         reject it first).
     */
    [[nodiscard]] float forward(const Tensor& x);

    /**
     * @brief Backpropagates the loss through SatisfactionLoss -> DisjunctionModule ->
     *        NegationModule(x2) -> sigmoid(x2) -> LinearModule(x2), accumulating each
     *        predicate's weight/bias gradient (LinearModule::accumulate() convention).
     * @throws std::logic_error if called before forward().
     */
    void backward();

    /**
     * @brief One SGD step: zero_grad both predicates, forward(), backward(), optimizer
     *        step on both predicates.
     * @param x Shape (N, 1), as forward()'s own parameter.
     * @param optimizer Optimizer applied to both predicate LinearModules.
     * @return The loss value for this step, before the update (matches XorNetwork::train_step's
     *         own "before the update" convention).
     */
    float train_step(const Tensor& x, SGDOptimizer& optimizer);

    /** @brief Cached predicate A output from the most recent forward(), shape (N, 1). */
    [[nodiscard]] const Tensor& last_a() const { return last_a_; }

    /** @brief Cached predicate B output from the most recent forward(), shape (N, 1). */
    [[nodiscard]] const Tensor& last_b() const { return last_b_; }

    /** @brief Cached rule truth degrees from the most recent forward(), shape (N,). */
    [[nodiscard]] const Tensor& last_rule() const { return last_rule_; }

    /** @brief Predicate A's own LinearModule -- test/inspection accessor. */
    [[nodiscard]] LinearModule& predicate_a() { return linear_a_; }

    /** @brief Predicate B's own LinearModule -- test/inspection accessor. */
    [[nodiscard]] LinearModule& predicate_b() { return linear_b_; }

private:
    DeviceBackend* backend_;
    LinearModule linear_a_;
    LinearModule linear_b_;
    NegationModule neg_a_;
    NegationModule neg_b_;
    DisjunctionModule disj_;
    SatisfactionLoss sat_loss_;
    Tensor last_a_;
    Tensor last_b_;
    Tensor last_rule_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
