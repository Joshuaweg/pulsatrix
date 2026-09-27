/** @file datalog_weighted_engine.hpp
 *  @brief Semiring-parameterized bottom-up fixpoint evaluation -- the weighted counterparts of
 *         Mission 0's `naive_evaluate`/`semi_naive_evaluate`. Phase 3 Mission 1 of
 *         campaign_exai_dl_library_neuro_symbolic (Generic Provenance-Semiring Abstraction).
 *  @ingroup dl_modules
 */
#pragma once

#include <vector>

#include "pulsatrix/datalog_dual_semiring.hpp"
#include "pulsatrix/datalog_rule.hpp"
#include "pulsatrix/datalog_semiring.hpp"
#include "pulsatrix/datalog_weighted_fact_database.hpp"

namespace pulsatrix::datalog {

/**
 * @brief Naive bottom-up fixpoint evaluation, generalized over any `Semiring`-shaped `Semiring`
 *        (see datalog_semiring.hpp): each round, every rule/substitution combination is
 *        matched against the *entire* current weighted fact set, and every derivation's weight
 *        (the `⊗`-product of the weights of the facts it matched) is combined via `⊕` with
 *        every other derivation of the *same* head atom found in that same round. The atom's
 *        stored weight is then **replaced** with that round's freshly recomputed total (never
 *        added to the previous round's total) -- see Stage 3 design decision 2 below for why
 *        replace-with-a-fresh-full-recompute, not incremental accumulate-across-rounds, is the
 *        correct generalization.
 *
 * @note Stage 3 design decision 2 (multi-path weight accumulation): the correctness pitfall
 *       the mission flags -- "first write wins" vs. "accumulate" being indistinguishable under
 *       boolean `⊕ = OR` but not under real-valued `⊕ = +` -- is resolved as follows:
 *       - **Within one round**, every rule/substitution pair that derives the *same* head atom
 *         has its weight `⊕`-summed into that atom's `round_totals` entry before anything is
 *         written back to the fact database. This is what makes a genuine multi-path fact
 *         (e.g. `ancestor(a,d)` derivable via two different intermediate nodes in a diamond
 *         graph) correctly get the *sum* of both paths' products, not just one of them
 *         (verified by `WeightedDatalogEngineTest.MultiplePathsAccumulateViaSemiringAdd`).
 *       - **Across rounds**, the atom's weight is *replaced* by each round's freshly
 *         recomputed total, not added to the prior round's stored value. This is the subtle
 *         part: naive evaluation re-scans the *entire* fact set every round, so the same
 *         derivation combination is rediscovered every round for as long as its inputs remain
 *         unchanged -- if the round total were `⊕`-accumulated *onto* the existing stored
 *         weight every round (the naive-looking-but-wrong translation of "accumulate, don't
 *         overwrite"), the same path's contribution would be re-added every single round
 *         forever, diverging rather than reaching a fixpoint. Replacing with a freshly
 *         recomputed total instead means: the total is always "the full `⊕`-sum over every
 *         derivation visible in *this* round's snapshot of the database," which is monotone
 *         non-decreasing round-over-round (later rounds only ever have equal-or-larger
 *         dependency weights to draw on, for a nonnegative semiring) and stabilizes at exactly
 *         the sum-of-products-over-all-derivation-paths value once every dependency has itself
 *         stabilized -- proven against the hand-derived expected weights below, not merely
 *         asserted.
 *       - A fact derivable via **zero** paths never appears in `round_totals` at all and is
 *         therefore never written into the result database -- `weight_of()` reports the
 *         caller-supplied default (conventionally `Semiring::zero()`) for it, per
 *         `WeightedFactDatabase::weight_of`'s own contract.
 *       - A fact derivable via **exactly one** path gets `round_totals[head] = Semiring::add(
 *         Semiring::zero(), that_one_path's_product)`, which for every semiring here satisfies
 *         `add(zero(), x) == x` (the semiring identity law) -- so a single-path fact's stored
 *         weight is exactly that path's product, with no spurious contribution from anywhere
 *         else (verified by `WeightedDatalogEngineTest.SinglePathWeightEqualsThatPathsProductExactly`).
 * @param rules The Datalog program (each rule already validated safe by its own constructor).
 * @param initial_facts The starting weighted fact database (extensional database).
 * @returns The fixpoint: every atom reachable from `initial_facts` by any finite number of
 *          rule applications, each carrying the `⊕`-sum-of-`⊗`-products weight standard
 *          weighted-Datalog/provenance-semiring semantics assigns it.
 */
template <typename Semiring>
[[nodiscard]] WeightedFactDatabase<typename Semiring::Value> naive_evaluate_weighted(
    const std::vector<Rule>& rules, const WeightedFactDatabase<typename Semiring::Value>& initial_facts);

/**
 * @brief The semi-naive-named weighted counterpart of `naive_evaluate_weighted`.
 * @note **Deliberately scoped, logged limitation (not silently dropped)**: a genuinely
 *       *incremental* semi-naive optimization for weighted/provenance-semiring accumulation is
 *       materially harder than the boolean case's delta-restriction trick, because a derived
 *       atom's weight is not monotone-and-idempotent the way boolean presence is -- a later
 *       round can still add a *new, previously-undiscovered* derivation path's contribution to
 *       an atom whose weight had already stopped changing for several rounds (e.g. a longer
 *       path stabilizing later than a shorter one to the same destination), which breaks the
 *       boolean semi-naive algorithm's core assumption that "once an atom is in the delta set,
 *       and then leaves it, it never needs revisiting." Correctly generalizing semi-naive's
 *       delta-restriction to semirings (tracking *which* dependency weight changes require
 *       *which* downstream atoms to be recomputed) is real, uncommitted future work -- this
 *       mission's own scoping note explicitly warns against silently absorbing extra scope
 *       (that note names Mission 2's gradient work; the same discipline applies here to a
 *       correctness-vs-effort tradeoff this mission's own exit condition does not require).
 *       This function therefore delegates to the already-proven-correct
 *       `naive_evaluate_weighted<Semiring>` algorithm, preserving the requested API surface
 *       (parity with Mission 0's naive/semi-naive pairing) without *claiming* a performance
 *       optimization this mission did not actually implement. `WeightedDatalogEngineTest`'s
 *       equivalence test therefore checks this function returns *identically* what
 *       `naive_evaluate_weighted` returns, by construction, not as an emergent property.
 * @param rules The Datalog program.
 * @param initial_facts The starting weighted fact database.
 * @returns Exactly what `naive_evaluate_weighted<Semiring>(rules, initial_facts)` returns.
 */
template <typename Semiring>
[[nodiscard]] WeightedFactDatabase<typename Semiring::Value> semi_naive_evaluate_weighted(
    const std::vector<Rule>& rules, const WeightedFactDatabase<typename Semiring::Value>& initial_facts);

// Explicit instantiation declarations -- definitions + explicit instantiations for the two
// Semiring types Mission 1 shipped (BooleanSemiring, RealSemiring<double>) live in
// datalog_weighted_engine.cpp, mirroring datalog_weighted_fact_database.hpp/.cpp's own
// declaration/definition split.
extern template WeightedFactDatabase<bool> naive_evaluate_weighted<BooleanSemiring>(
    const std::vector<Rule>&, const WeightedFactDatabase<bool>&);
extern template WeightedFactDatabase<double> naive_evaluate_weighted<RealSemiring<double>>(
    const std::vector<Rule>&, const WeightedFactDatabase<double>&);
extern template WeightedFactDatabase<bool> semi_naive_evaluate_weighted<BooleanSemiring>(
    const std::vector<Rule>&, const WeightedFactDatabase<bool>&);
extern template WeightedFactDatabase<double> semi_naive_evaluate_weighted<RealSemiring<double>>(
    const std::vector<Rule>&, const WeightedFactDatabase<double>&);

// Phase 3 Mission 2 additive extension: a third Semiring instantiation, DualSemiring<double>
// (see datalog_dual_semiring.hpp) -- the forward-mode-AD differentiable case. Zero changes to
// naive_evaluate_weighted/semi_naive_evaluate_weighted's own template bodies; this is exactly
// what Mission 1's Decision 3 (templates, not a virtual interface) was for -- a brand new
// Semiring is "one more ordinary instantiation of one generic algorithm," not an engine change.
extern template WeightedFactDatabase<DualNumber<double>> naive_evaluate_weighted<DualSemiring<double>>(
    const std::vector<Rule>&, const WeightedFactDatabase<DualNumber<double>>&);
extern template WeightedFactDatabase<DualNumber<double>> semi_naive_evaluate_weighted<DualSemiring<double>>(
    const std::vector<Rule>&, const WeightedFactDatabase<DualNumber<double>>&);

}  // namespace pulsatrix::datalog
