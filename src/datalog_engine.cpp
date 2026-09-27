#include "pulsatrix/datalog_engine.hpp"

#include <optional>
#include <unordered_map>

#include "pulsatrix/assert.hpp"

namespace pulsatrix::datalog {

namespace {

/**
 * @brief A partial variable substitution built up while matching a rule's body against the
 *        fact database: variable name -> the constant value it is bound to.
 * @note This is the mission's one genuinely nontrivial algorithm (Stage 3 design question
 *       3), designed explicitly rather than improvised inline:
 *       - A rule's body is matched atom-by-atom, left to right.
 *       - Matching starts from a single empty substitution (the "no variables bound yet"
 *         state) and, for each body atom in turn, tries to extend every substitution
 *         surviving so far against every candidate fact for that atom's position.
 *       - "Consistent across the whole body" (e.g. `X` in both `edge(X,Z)` and
 *         `ancestor(Z,Y)` must bind to the same value) falls out directly from this
 *         extend-not-replace design: extending an existing substitution checks any
 *         already-bound variable against the new candidate value and rejects the match if
 *         they disagree, rather than starting a fresh substitution per atom.
 *       - Because atoms/facts are function-symbol-free (no nested terms), matching one atom
 *         against one candidate fact is a single flat position-by-position walk -- no
 *         recursive term unification (the general Prolog case) is needed at all.
 */
using Substitution = std::unordered_map<std::string, std::string>;

/**
 * @brief Tries to extend `sub` by matching `body_atom` (may contain variables) against
 *        `fact` (must be ground). Returns std::nullopt if the predicate/arity/any constant
 *        term disagrees, or if a variable already bound in `sub` would need a different
 *        value than before.
 */
std::optional<Substitution> try_extend_substitution(const Atom& body_atom, const Atom& fact, const Substitution& sub) {
    PULSATRIX_ASSERT(fact.is_ground());

    if (body_atom.predicate_name() != fact.predicate_name()) return std::nullopt;
    if (body_atom.arity() != fact.arity()) return std::nullopt;

    Substitution extended = sub;
    const std::vector<Term>& body_terms = body_atom.terms();
    const std::vector<Term>& fact_terms = fact.terms();
    for (std::size_t i = 0; i < body_terms.size(); ++i) {
        const Term& body_term = body_terms[i];
        const std::string& fact_value = fact_terms[i].value();

        if (body_term.is_constant()) {
            if (body_term.value() != fact_value) return std::nullopt;
            continue;
        }

        // body_term is a variable.
        auto it = extended.find(body_term.value());
        if (it == extended.end()) {
            extended.emplace(body_term.value(), fact_value);
        } else if (it->second != fact_value) {
            return std::nullopt;
        }
    }
    return extended;
}

/** @brief Applies a (complete, per rule safety) substitution to produce a ground head atom. */
Atom apply_substitution(const Atom& head, const Substitution& sub) {
    std::vector<Term> resolved;
    resolved.reserve(head.arity());
    for (const Term& term : head.terms()) {
        if (term.is_constant()) {
            resolved.push_back(term);
        } else {
            auto it = sub.find(term.value());
            // Guaranteed present by Rule's own range-restriction check at construction.
            PULSATRIX_ASSERT(it != sub.end());
            resolved.push_back(Term::make_constant(it->second));
        }
    }
    return Atom(head.predicate_name(), std::move(resolved));
}

/**
 * @brief Joins a rule's body atoms left-to-right, where each body atom position may draw
 *        its candidate facts from a different source set -- naive evaluation passes the
 *        same full fact set for every position; semi-naive evaluation passes the "delta"
 *        set for exactly one position and the full accumulated set for the rest.
 */
std::vector<Substitution> join_body(const std::vector<Atom>& body, const std::vector<const FactSet*>& sources) {
    std::vector<Substitution> substitutions{Substitution{}};

    for (std::size_t i = 0; i < body.size(); ++i) {
        std::vector<Substitution> next;
        const FactSet& source = *sources[i];
        for (const Substitution& sub : substitutions) {
            for (const Atom& fact : source) {
                std::optional<Substitution> extended = try_extend_substitution(body[i], fact, sub);
                if (extended.has_value()) {
                    next.push_back(std::move(*extended));
                }
            }
        }
        substitutions = std::move(next);
        if (substitutions.empty()) break;  // no matches at all for this position -- whole body fails
    }
    return substitutions;
}

}  // namespace

FactDatabase naive_evaluate(const std::vector<Rule>& rules, const FactDatabase& initial_facts) {
    FactDatabase facts = initial_facts;

    while (true) {
        std::vector<Atom> newly_derived;

        for (const Rule& rule : rules) {
            std::vector<const FactSet*> sources(rule.body().size(), &facts.facts());
            std::vector<Substitution> substitutions = join_body(rule.body(), sources);
            for (const Substitution& sub : substitutions) {
                Atom derived = apply_substitution(rule.head(), sub);
                if (!facts.contains(derived)) {
                    newly_derived.push_back(std::move(derived));
                }
            }
        }

        if (newly_derived.empty()) break;

        for (Atom& fact : newly_derived) {
            facts.insert(std::move(fact));
        }
    }

    return facts;
}

FactDatabase semi_naive_evaluate(const std::vector<Rule>& rules, const FactDatabase& initial_facts) {
    FactDatabase facts = initial_facts;

    // Rules with an empty body are ground facts expressed as rules (`edge(a,b) :- .`) --
    // they don't participate in the delta-driven join below (there is no body atom
    // position to restrict to `delta`), so derive them once, up front, exactly as naive
    // evaluation's first round would.
    FactSet delta;
    for (const Rule& rule : rules) {
        if (rule.body().empty() && !facts.contains(rule.head())) {
            delta.insert(rule.head());
        }
    }
    for (const Atom& fact : delta) {
        facts.insert(fact);
    }
    for (const Atom& fact : initial_facts.facts()) {
        delta.insert(fact);  // round 0's "new" facts are the initial facts themselves.
    }

    while (!delta.empty()) {
        FactSet new_delta;

        for (const Rule& rule : rules) {
            const std::size_t body_size = rule.body().size();
            if (body_size == 0) continue;  // handled once, above.

            // For each body-atom position i, restrict position i to `delta` and every other
            // position to the full accumulated `facts` -- guarantees any fact produced this
            // round involves at least one fact newly derived last round, without re-deriving
            // facts that follow purely from already-known facts (already in `facts`).
            for (std::size_t i = 0; i < body_size; ++i) {
                std::vector<const FactSet*> sources(body_size, &facts.facts());
                sources[i] = &delta;

                std::vector<Substitution> substitutions = join_body(rule.body(), sources);
                for (const Substitution& sub : substitutions) {
                    Atom derived = apply_substitution(rule.head(), sub);
                    if (!facts.contains(derived)) {
                        new_delta.insert(std::move(derived));
                    }
                }
            }
        }

        for (const Atom& fact : new_delta) {
            facts.insert(fact);
        }
        delta = std::move(new_delta);
    }

    return facts;
}

}  // namespace pulsatrix::datalog
