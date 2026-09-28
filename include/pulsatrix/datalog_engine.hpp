/** @file datalog_engine.hpp
 *  @brief Bottom-up fixpoint evaluation (naive and semi-naive), boolean semiring only.
 *         Phase 3 Mission 0 of campaign_exai_dl_library_neuro_symbolic (Datalog core).
 *  @ingroup neuro_symbolic
 */
#pragma once

#include <vector>

#include "pulsatrix/datalog_fact_database.hpp"
#include "pulsatrix/datalog_rule.hpp"

namespace pulsatrix::datalog {

/**
 * @brief Naive bottom-up fixpoint evaluation: repeatedly re-evaluates every rule against the
 *        *entire* accumulated fact set until a round derives no new facts.
 * @param rules The Datalog program (each rule already validated safe by its own constructor).
 * @param initial_facts The starting fact database (extensional database, e.g. `edge` facts).
 * @returns The fixpoint: `initial_facts` plus every fact derivable by any finite number of
 *          rule applications (the intensional database), boolean semiring (a fact is either
 *          present or absent -- no weight/provenance).
 * @note Always terminates on a function-symbol-free, range-restricted program over a finite
 *       initial fact set: every derived atom's constants come from the finite set of
 *       constants already present in `rules`/`initial_facts` (no function symbols to
 *       manufacture new ones), so the space of possible ground atoms is finite and facts
 *       only ever accumulate (monotone) -- the fixpoint is reached in a bounded number of
 *       rounds. This is Decision Point 1's whole rationale for choosing Datalog over Prolog.
 */
[[nodiscard]] FactDatabase naive_evaluate(const std::vector<Rule>& rules, const FactDatabase& initial_facts);

/**
 * @brief Semi-naive bottom-up fixpoint evaluation: each round only considers rule
 *        applications where at least one body atom is matched against facts newly derived
 *        in the *previous* round (the "delta"), rather than re-scanning the full
 *        accumulated fact set for every body-atom position every round.
 * @param rules The Datalog program.
 * @param initial_facts The starting fact database.
 * @returns The identical fixpoint `naive_evaluate` would produce on the same program and
 *          initial facts (proven by a direct equivalence test, not merely "also seems to
 *          work" -- see datalog_engine_test.cpp's
 *          `SemiNaiveEvaluationProducesIdenticalFixpointToNaiveEvaluation`).
 */
[[nodiscard]] FactDatabase semi_naive_evaluate(const std::vector<Rule>& rules, const FactDatabase& initial_facts);

}  // namespace pulsatrix::datalog
