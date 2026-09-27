/** @file datalog_lrp.hpp
 *  @brief LRP-style relevance propagation for the real-valued `(+, x)` provenance-semiring
 *         Datalog circuit built in Mission 1/2 -- Phase 3 Mission 3 of
 *         campaign_exai_dl_library_neuro_symbolic (LRP for the Datalog/Provenance-Semiring
 *         Circuit), extending Phase 1-2's fuzzy-logic LRP methodology (the weighted-sum/
 *         epsilon-rule split for `+`, the bilinear split for `x`) to derived-fact weights
 *         instead of `Module` output tensors.
 *  @ingroup dl_modules
 *
 * @note **Hand-derived rule (Stage 3 design question 1), worked before implementation, on
 *       Mission 1's diamond-graph toy program**: a derived fact's weight is a `⊕`-sum of
 *       `⊗`-products over derivation paths -- e.g. `ancestor(a,d) = edge(a,b)*ancestor(b,d) +
 *       edge(a,c)*ancestor(c,d)`, and (in this specific toy KB, since `b`/`c` each have exactly
 *       one outgoing edge) `ancestor(b,d) = edge(b,d)`, `ancestor(c,d) = edge(c,d)` -- so
 *       `ancestor(a,d) = edge(a,b)*edge(b,d) + edge(a,c)*edge(c,d)`. This is *structurally
 *       identical* to the `(+, x)` circuit shape Phase 1-2 already derived LRP rules for
 *       (`ConjunctionModule`'s Product-t-norm bilinear split for `x`, `LinearModule`'s
 *       epsilon-rule weighted-sum split for `+`) -- the rule below is that exact composition,
 *       reapplied one level at a time down the Datalog derivation DAG rather than down a
 *       `Module` chain, recursing at any body position that is itself a derived (non-
 *       extensional) atom, rather than assumed to generalize trivially from the single-path
 *       case without being re-derived for this specific multi-path circuit shape:
 *
 *       1. **`⊕` level (weighted-sum/epsilon-rule split across derivation paths).** If atom
 *          `h`'s fixpoint weight is `y = sum_k z_k` (over every (rule, substitution) match `k`
 *          deriving `h`, `z_k` that match's `⊗`-product weight), relevance `R_h` arriving at
 *          `h` splits across matches as `R_k = (z_k / (y + eps*sign(y))) * R_h` -- exactly
 *          `LinearModule::propagate_relevance`'s own epsilon rule, applied to derivation-path
 *          terms instead of a linear layer's summands.
 *       2. **`x` level (bilinear split across a match's body atoms).** Within match `k`,
 *          `z_k = x_1 * x_2` for a two-atom rule body (every rule body in this campaign's toy
 *          programs has arity 1 or 2 -- see the arity-2 restriction note below), so `R_k`
 *          splits via `ConjunctionModule`'s own Product-t-norm bilinear rule:
 *          `contribution = (x_1*x_2 / (2*z_k + eps*sign(z_k))) * R_k`, given to *both* `x_1`
 *          and `x_2` (near-exactly `R_k/2` each, exactly `R_k/2` each at `eps=0`, since
 *          `x_1*x_2 == z_k` makes the fraction `z_k/(2*z_k) = 1/2` identically regardless of
 *          the actual operand values). A one-atom body (Mission 0's base rule,
 *          `ancestor(X,Y):-edge(X,Y).`) is a trivial one-factor "product" -- `z_k` already
 *          equals that single factor, so it is a pass-through, not a new case: `R_k` flows to
 *          the one body atom unchanged, mirroring `NegationModule`/`ReluModule`'s own
 *          single-input pass-through precedent (Phase 2 Mission 0's Sigmoid Design Decision
 *          note makes the same move for exactly this reason).
 *       3. **Recursion, not a one-level formula.** If a body position's grounded atom is
 *          itself a derived (non-extensional) fact -- e.g. `ancestor(b,d)` at the recursive
 *          rule's `Z=b` match -- the relevance it receives from step 2 becomes *its own*
 *          `R_h` for another application of steps 1-2 against *its* derivations, recursing
 *          until every relevance-bearing quantity has been pushed back to an atom with **zero**
 *          matching derivations in the fixpoint database (an extensional/base fact -- e.g.
 *          `edge(b,d)`, never the head of any rule in this campaign's toy programs). This is
 *          the piece a naive "just apply the single-path formula" reading of Decision Point 2's
 *          rationale would have missed: `ancestor(a,d)`'s own two paths are NOT both flat
 *          products of four *base* facts in general -- only in this specific toy KB do they
 *          happen to collapse to that shape, because `ancestor(b,d)`/`ancestor(c,d)` are
 *          themselves single-path-derived from a base fact one level down. A KB where `b`/`c`
 *          have their own multi-path onward routes would need this same rule applied
 *          recursively at that next level too, which this implementation does unconditionally
 *          (not special-cased to a fixed depth of two).
 *
 * @note **Hand-worked numeric check** (Mission 1's diamond KB, constant weights,
 *       `edge(a,b)=0.5, edge(a,c)=0.4, edge(b,d)=0.6, edge(c,d)=0.3`, seed `R=1.0` at
 *       `ancestor(a,d)`, `eps=0` for exactness):
 *       ```
 *       ancestor(a,d) derivations: path1 (Z=b) weight=0.5*0.6=0.30, path2 (Z=c) weight=0.4*0.3=0.12
 *       total = 0.42 (matches the fixpoint weight)
 *       R_path1 = 0.30/0.42 = 0.714286   R_path2 = 0.12/0.42 = 0.285714   (sum = 1.0)
 *       path1 bilinear split (edge(a,b)=0.5, ancestor(b,d)=0.6): each gets R_path1/2 = 0.357143
 *         (a 2-term product's bilinear split is always exactly half-and-half at eps=0,
 *         independent of the operand values, since x1*x2/(2*x1*x2) == 1/2 identically)
 *       recurse into ancestor(b,d), r_out=0.357143: single derivation (edge(b,d)=0.6, arity 1)
 *         -> pass-through, edge(b,d) receives 0.357143
 *       path2 bilinear split (edge(a,c)=0.4, ancestor(c,d)=0.3): each gets R_path2/2 = 0.142857
 *       recurse into ancestor(c,d), r_out=0.142857: single derivation (edge(c,d)=0.3, arity 1)
 *         -> pass-through, edge(c,d) receives 0.142857
 *       Leaf relevances: edge(a,b)=0.357143, edge(b,d)=0.357143, edge(a,c)=0.142857, edge(c,d)=0.142857
 *       Sum = 1.0 == seed, exactly (eps=0) -- verified by
 *       DatalogLRPTest.MultiPathDiamondFactConservesRelevanceMatchingHandDerivedSplit.
 *       ```
 *       And the single-path boundary case, `ancestor(a,b)` (derivable only directly from
 *       `edge(a,b)`, arity-1 body): `R_out` flows entirely, unsplit, to `edge(a,b)` --
 *       verified by `DatalogLRPTest.SinglePathFactReceivesEntireRelevanceNoSplit`.
 *
 * @note **Scope restriction, deliberate and logged**: rule bodies of arity > 2 throw
 *       `std::logic_error` rather than silently approximating an n-ary bilinear split -- no
 *       toy program in Missions 0-3 has a rule body wider than two atoms (the recursive
 *       `ancestor` rule's body is always exactly `edge(X,Z), ancestor(Z,Y)`), so this
 *       generalization is genuinely undesigned-for rather than assumed-fine, per this
 *       project's "log the honest boundary, don't silently extend past what's been derived"
 *       convention (mirrors datalog_weighted_engine.cpp's own logged extensional-and-rule-head
 *       scope note).
 * @note **Acyclic-derivation-DAG assumption, deliberate and logged**: this function recurses
 *       down the fixpoint's own derivation structure with no cycle/memoization guard. Every
 *       toy program in this campaign (`ancestor` over a DAG of `edge` facts) has an acyclic
 *       derivation graph by construction (a transitive-closure relation over a finite DAG
 *       cannot derive an atom from itself), so this is not exercised as a defect here, but a
 *       Datalog program whose *derivation* graph (not just its underlying fact graph) contains
 *       a genuine cycle would recurse unboundedly -- flagged here rather than silently assumed
 *       safe in general, mirroring Mission 0's own empty-body-rule scope note convention.
 */
#pragma once

#include <unordered_map>
#include <vector>

#include "pulsatrix/datalog_atom.hpp"
#include "pulsatrix/datalog_rule.hpp"
#include "pulsatrix/datalog_weighted_fact_database.hpp"

namespace pulsatrix::datalog {

/** @brief A map from ground atom to its accumulated relevance (real-valued, per the
 *         `RealSemiring<double>` circuit this rule operates over). */
using RelevanceMap = std::unordered_map<Atom, double, AtomHash>;

/**
 * @brief The result of one `propagate_relevance_weighted` call: relevance recorded at every
 *        atom the recursion visited (`all_atoms`, useful for inspection/debugging), and
 *        relevance recorded only at atoms with zero matching derivations in `fixpoint` --
 *        the extensional/base facts (`base_facts`) -- which is what an end-to-end
 *        conservation check must sum against the seeded relevance, per ordinary LRP
 *        convention (intermediate/derived atoms are redistribution waypoints, not part of
 *        the conserved total, exactly as an ordinary `Module` chain's intermediate
 *        activations are not summed alongside its raw inputs).
 */
struct RelevanceResult {
    RelevanceMap all_atoms;
    RelevanceMap base_facts;
};

/**
 * @brief Propagates relevance from `query`'s derived weight back through every derivation
 *        path in `fixpoint` (a stable, already-evaluated `naive_evaluate_weighted<
 *        RealSemiring<double>>` result), down to every extensional/base fact that
 *        contributed to it -- see this file's header doc comment for the full hand-derived
 *        rule and worked numeric example.
 * @param rules The Datalog program (used to re-derive, from the stable fixpoint, exactly
 *        which (rule, substitution) matches produced `query`'s weight and every intermediate
 *        derived atom's weight along the way).
 * @param fixpoint A stable (already-converged) weighted fact database -- the output of
 *        `naive_evaluate_weighted<RealSemiring<double>>` (or equivalent), not an
 *        intermediate round's snapshot. Re-matching rule bodies against a non-fixpoint
 *        database would not reproduce the derivation structure that actually produced the
 *        atom's *stored* weight.
 * @param query The atom whose weight's relevance is being explained (typically the query
 *        fact a caller cares about, e.g. `ancestor(a,d)`).
 * @param relevance_seed The relevance value seeded at `query` (mirrors every other
 *        conservation test in this codebase's own convention of seeding an arbitrary value
 *        at a chain's output and checking it is conserved at the input).
 * @param epsilon Stabilizer for both the `⊕`-level weighted-sum split and the `⊗`-level
 *        bilinear split, matching `LRPRuleConfig::epsilon`'s own role and default
 *        (1e-6f, widened to double here since this circuit's own weights are `double`).
 * @throws std::logic_error if any rule body actually matched during propagation has arity
 *        greater than 2 -- see this file's own "Scope restriction" note; no toy program in
 *        this campaign needs a wider rule body, so this is an honest, logged boundary rather
 *        than a silently-approximated n-ary split.
 */
[[nodiscard]] RelevanceResult propagate_relevance_weighted(const std::vector<Rule>& rules,
                                                             const WeightedFactDatabase<double>& fixpoint,
                                                             const Atom& query, double relevance_seed,
                                                             double epsilon = 1e-6);

}  // namespace pulsatrix::datalog
