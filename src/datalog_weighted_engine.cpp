#include "pulsatrix/datalog_weighted_engine.hpp"

#include <cmath>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "pulsatrix/assert.hpp"

namespace pulsatrix::datalog {

namespace {

/** @brief Same substitution shape as datalog_engine.cpp's (file-local, intentionally
 *         duplicated -- see datalog_weighted_engine.hpp's header comment on why this mission
 *         does not attempt to share Mission 0's own anonymous-namespace helpers). */
using Substitution = std::unordered_map<std::string, std::string>;

/** @brief A substitution paired with the accumulated `⊗`-product weight of every fact matched
 *         to reach it so far. */
template <typename Value>
struct WeightedSubstitution {
    Substitution sub;
    Value weight;
};

/** @brief Tries to extend `sub` (with running weight `sub_weight`) by matching `body_atom`
 *         against ground `fact` (whose own weight is `fact_weight`). Mirrors
 *         datalog_engine.cpp's `try_extend_substitution` exactly for the substitution part;
 *         additionally folds `fact_weight` into the running weight via `Semiring::mul`. */
template <typename Semiring>
std::optional<WeightedSubstitution<typename Semiring::Value>> try_extend_weighted(
    const Atom& body_atom, const Atom& fact, typename Semiring::Value fact_weight,
    const WeightedSubstitution<typename Semiring::Value>& sub) {
    PULSATRIX_ASSERT(fact.is_ground());

    if (body_atom.predicate_name() != fact.predicate_name()) return std::nullopt;
    if (body_atom.arity() != fact.arity()) return std::nullopt;

    Substitution extended = sub.sub;
    const std::vector<Term>& body_terms = body_atom.terms();
    const std::vector<Term>& fact_terms = fact.terms();
    for (std::size_t i = 0; i < body_terms.size(); ++i) {
        const Term& body_term = body_terms[i];
        const std::string& fact_value = fact_terms[i].value();

        if (body_term.is_constant()) {
            if (body_term.value() != fact_value) return std::nullopt;
            continue;
        }

        auto it = extended.find(body_term.value());
        if (it == extended.end()) {
            extended.emplace(body_term.value(), fact_value);
        } else if (it->second != fact_value) {
            return std::nullopt;
        }
    }
    return WeightedSubstitution<typename Semiring::Value>{std::move(extended), Semiring::mul(sub.weight, fact_weight)};
}

/** @brief Applies a (complete, per rule safety) substitution to produce a ground head atom --
 *         identical logic to datalog_engine.cpp's `apply_substitution`. */
Atom apply_substitution(const Atom& head, const Substitution& sub) {
    std::vector<Term> resolved;
    resolved.reserve(head.arity());
    for (const Term& term : head.terms()) {
        if (term.is_constant()) {
            resolved.push_back(term);
        } else {
            auto it = sub.find(term.value());
            PULSATRIX_ASSERT(it != sub.end());
            resolved.push_back(Term::make_constant(it->second));
        }
    }
    return Atom(head.predicate_name(), std::move(resolved));
}

/**
 * @brief Joins a rule's body atoms left-to-right against the given weighted fact set,
 *        accumulating each surviving substitution's `⊗`-product weight as it goes. Every
 *        body-atom position draws its candidate ground facts (and their weights) from
 *        `source` -- naive weighted evaluation always passes the full current fact set for
 *        every position, mirroring datalog_engine.cpp's own naive `join_body`.
 */
template <typename Semiring>
std::vector<WeightedSubstitution<typename Semiring::Value>> weighted_join_body(
    const std::vector<Atom>& body, const WeightedFactSet<typename Semiring::Value>& source) {
    using Value = typename Semiring::Value;
    std::vector<WeightedSubstitution<Value>> substitutions{WeightedSubstitution<Value>{Substitution{}, Semiring::one()}};

    for (const Atom& body_atom : body) {
        std::vector<WeightedSubstitution<Value>> next;
        for (const WeightedSubstitution<Value>& sub : substitutions) {
            for (const auto& [fact, fact_weight] : source) {
                std::optional<WeightedSubstitution<Value>> extended =
                    try_extend_weighted<Semiring>(body_atom, fact, fact_weight, sub);
                if (extended.has_value()) {
                    next.push_back(std::move(*extended));
                }
            }
        }
        substitutions = std::move(next);
        if (substitutions.empty()) break;
    }
    return substitutions;
}

/** @brief Equality check used only to decide whether a round changed anything (the fixpoint
 *         termination test) -- an epsilon compare for floating-point `Value`s, guarding
 *         against the (unlikely, but not structurally impossible) case where two rounds sum
 *         the same set of terms in a different `std::unordered_map` iteration order and land
 *         on a bit-different-but-mathematically-equal total; exact `==` for `bool`. */
template <typename Value>
bool values_equal(const Value& a, const Value& b) {
    if constexpr (std::is_floating_point_v<Value>) {
        return std::fabs(static_cast<double>(a) - static_cast<double>(b)) < 1e-9;
    } else {
        return a == b;
    }
}

}  // namespace

template <typename Semiring>
WeightedFactDatabase<typename Semiring::Value> naive_evaluate_weighted(
    const std::vector<Rule>& rules, const WeightedFactDatabase<typename Semiring::Value>& initial_facts) {
    using Value = typename Semiring::Value;
    WeightedFactDatabase<Value> facts = initial_facts;

    while (true) {
        std::unordered_map<Atom, Value, AtomHash> round_totals;

        for (const Rule& rule : rules) {
            if (rule.body().empty()) {
                // Ground fact expressed as a rule (`edge(a,b) :- .`) -- fires with weight
                // one() every round, mirroring datalog_engine.cpp's own empty-body handling
                // (Mission 0 Notes) generalized to the semiring's multiplicative identity.
                Value& total = round_totals.try_emplace(rule.head(), Semiring::zero()).first->second;
                total = Semiring::add(total, Semiring::one());
                continue;
            }

            std::vector<WeightedSubstitution<Value>> matches = weighted_join_body<Semiring>(rule.body(), facts.facts());
            for (const WeightedSubstitution<Value>& match : matches) {
                Atom head = apply_substitution(rule.head(), match.sub);
                Value& total = round_totals.try_emplace(head, Semiring::zero()).first->second;
                total = Semiring::add(total, match.weight);
            }
        }

        if (round_totals.empty()) break;

        // Scope note: only atoms that are the head of some rule application this round are
        // written here -- a purely-extensional atom (never a rule head, e.g. this mission's
        // toy KBs' `edge` facts) is never touched and keeps its initial weight forever, exact
        // parity with Mission 0's boolean case. An atom that is *both* an extensional fact
        // and a rule head in the same program (not exercised by either toy KB below) would
        // have its extensional weight replaced by the rule-derived total rather than
        // `⊕`-merged with it -- a real, deliberately undesigned-for scope edge that this
        // mission's toy KBs (edge/ancestor, always disjoint predicates) never exercise;
        // flagged here rather than silently mishandled.
        bool changed = false;
        for (auto& [atom, computed] : round_totals) {
            Value existing = facts.weight_of(atom, Semiring::zero());
            if (!values_equal(existing, computed)) changed = true;
            facts.set(atom, computed);
        }
        if (!changed) break;
    }

    return facts;
}

template <typename Semiring>
WeightedFactDatabase<typename Semiring::Value> semi_naive_evaluate_weighted(
    const std::vector<Rule>& rules, const WeightedFactDatabase<typename Semiring::Value>& initial_facts) {
    // See datalog_weighted_engine.hpp's header comment: this deliberately delegates to the
    // proven-correct naive algorithm rather than a not-yet-designed incremental weighted
    // delta optimization.
    return naive_evaluate_weighted<Semiring>(rules, initial_facts);
}

template WeightedFactDatabase<bool> naive_evaluate_weighted<BooleanSemiring>(
    const std::vector<Rule>&, const WeightedFactDatabase<bool>&);
template WeightedFactDatabase<double> naive_evaluate_weighted<RealSemiring<double>>(
    const std::vector<Rule>&, const WeightedFactDatabase<double>&);
template WeightedFactDatabase<bool> semi_naive_evaluate_weighted<BooleanSemiring>(
    const std::vector<Rule>&, const WeightedFactDatabase<bool>&);
template WeightedFactDatabase<double> semi_naive_evaluate_weighted<RealSemiring<double>>(
    const std::vector<Rule>&, const WeightedFactDatabase<double>&);

// Phase 3 Mission 2 additive extension -- see datalog_weighted_engine.hpp's own note.
template WeightedFactDatabase<DualNumber<double>> naive_evaluate_weighted<DualSemiring<double>>(
    const std::vector<Rule>&, const WeightedFactDatabase<DualNumber<double>>&);
template WeightedFactDatabase<DualNumber<double>> semi_naive_evaluate_weighted<DualSemiring<double>>(
    const std::vector<Rule>&, const WeightedFactDatabase<DualNumber<double>>&);

}  // namespace pulsatrix::datalog
