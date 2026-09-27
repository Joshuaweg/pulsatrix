#include "pulsatrix/datalog_lrp.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace pulsatrix::datalog {

namespace {

using Substitution = std::unordered_map<std::string, std::string>;

/** @brief One grounded body-atom position of a specific (rule, substitution) match, paired
 *         with that fact's own fixpoint weight. */
struct GroundedBodyAtom {
    Atom atom;
    double weight;
};

/** @brief One (rule, substitution) match deriving some head atom: the match's own `⊗`-product
 *         weight, plus every body position's grounded atom + weight (needed so the caller can
 *         recurse into any body atom that is itself a derived, non-extensional fact). */
struct Derivation {
    double weight;
    std::vector<GroundedBodyAtom> body;
};

/** @brief Tries to unify `pattern`'s terms against `ground`'s terms under `sub`, extending
 *         `sub` in place. Mirrors datalog_engine.cpp's/datalog_weighted_engine.cpp's own
 *         substitution-matching logic (intentionally re-implemented file-local, per this
 *         subsystem's own established convention of not sharing anonymous-namespace helpers
 *         across files -- see datalog_weighted_engine.hpp's header comment). */
bool try_unify(const Atom& pattern, const Atom& ground, Substitution& sub) {
    if (pattern.predicate_name() != ground.predicate_name()) return false;
    if (pattern.arity() != ground.arity()) return false;

    Substitution extended = sub;
    const std::vector<Term>& pattern_terms = pattern.terms();
    const std::vector<Term>& ground_terms = ground.terms();
    for (std::size_t i = 0; i < pattern_terms.size(); ++i) {
        const Term& pt = pattern_terms[i];
        const std::string& gv = ground_terms[i].value();
        if (pt.is_constant()) {
            if (pt.value() != gv) return false;
            continue;
        }
        auto it = extended.find(pt.value());
        if (it == extended.end()) {
            extended.emplace(pt.value(), gv);
        } else if (it->second != gv) {
            return false;
        }
    }
    sub = std::move(extended);
    return true;
}

/** @brief Finds every (rule, substitution) match in `rules` whose head instantiates to
 *         exactly `head_atom`, matched against the stable `fixpoint` database -- the
 *         derivation structure that actually produced `head_atom`'s stored fixpoint weight.
 *         Returns an empty vector if `head_atom` is never the head of any matching rule
 *         instantiation (i.e. it is an extensional/base fact, or simply not derivable at
 *         all -- both cases are treated identically as "nothing further to redistribute
 *         into," per this file's own header note on leaf classification). */
std::vector<Derivation> find_derivations(const Atom& head_atom, const std::vector<Rule>& rules,
                                          const WeightedFactDatabase<double>& fixpoint) {
    std::vector<Derivation> result;

    for (const Rule& rule : rules) {
        if (rule.head().predicate_name() != head_atom.predicate_name()) continue;
        if (rule.head().arity() != head_atom.arity()) continue;

        Substitution seed;
        if (!try_unify(rule.head(), head_atom, seed)) continue;

        if (rule.body().empty()) {
            // Ground fact expressed as a rule (`edge(a,b) :- .`) -- fires with weight one(),
            // no body facts to recurse into. Mirrors datalog_weighted_engine.cpp's own
            // empty-body handling generalized to this file's Derivation shape.
            result.push_back(Derivation{1.0, {}});
            continue;
        }

        struct PartialMatch {
            Substitution sub;
            double weight;
            std::vector<GroundedBodyAtom> body;
        };
        std::vector<PartialMatch> partials{PartialMatch{seed, 1.0, {}}};

        for (const Atom& body_atom : rule.body()) {
            std::vector<PartialMatch> next;
            for (const PartialMatch& pm : partials) {
                for (const auto& [fact, fact_weight] : fixpoint.facts()) {
                    Substitution extended = pm.sub;
                    if (!try_unify(body_atom, fact, extended)) continue;

                    PartialMatch pm2;
                    pm2.sub = std::move(extended);
                    pm2.weight = pm.weight * fact_weight;
                    pm2.body = pm.body;
                    pm2.body.push_back(GroundedBodyAtom{fact, fact_weight});
                    next.push_back(std::move(pm2));
                }
            }
            partials = std::move(next);
            if (partials.empty()) break;
        }

        for (const PartialMatch& pm : partials) {
            result.push_back(Derivation{pm.weight, pm.body});
        }
    }

    return result;
}

void distribute(const Atom& atom, double r_out, const std::vector<Rule>& rules,
                 const WeightedFactDatabase<double>& fixpoint, double epsilon, RelevanceResult& out) {
    out.all_atoms[atom] += r_out;

    std::vector<Derivation> derivations = find_derivations(atom, rules, fixpoint);
    if (derivations.empty()) {
        // Leaf: an extensional/base fact (or an atom with no derivation at all) -- nothing
        // further to redistribute into. This is the "relevance terminates at the base
        // facts" half of Stage 3 design question 2; the neural-predicate-weighted base fact
        // is deliberately treated identically to a constant base fact *at this layer* --
        // continuing on into the predicate's own Module chain is the caller's job
        // (NeuralPredicateDatalogBridge::propagate_relevance), composed on top of this
        // function's result, not a case this function special-cases internally.
        out.base_facts[atom] += r_out;
        return;
    }

    double total = 0.0;
    for (const Derivation& d : derivations) total += d.weight;
    if (total == 0.0) return;  // degenerate: no positive weight to redistribute proportionally.

    const double sum_denom = total + epsilon * ((total >= 0.0) ? 1.0 : -1.0);

    for (const Derivation& d : derivations) {
        const double r_k = (d.weight / sum_denom) * r_out;  // ⊕-level weighted-sum/epsilon split

        if (d.body.empty()) continue;  // empty-body rule: nothing to recurse into.

        if (d.body.size() == 1) {
            // Single-factor "product" -- pass-through, mirrors NegationModule/ReluModule's
            // own single-input precedent (see this file's header doc comment).
            distribute(d.body[0].atom, r_k, rules, fixpoint, epsilon, out);
        } else if (d.body.size() == 2) {
            // Bilinear split -- ConjunctionModule's Product-t-norm rule, applied to two
            // Datalog atoms' weights instead of two Tensor operands.
            const double x1 = d.body[0].weight;
            const double x2 = d.body[1].weight;
            const double y = d.weight;  // == x1 * x2
            const double bilinear_denom = 2.0 * y + epsilon * ((y >= 0.0) ? 1.0 : -1.0);
            const double contribution = (x1 * x2 / bilinear_denom) * r_k;
            distribute(d.body[0].atom, contribution, rules, fixpoint, epsilon, out);
            distribute(d.body[1].atom, contribution, rules, fixpoint, epsilon, out);
        } else {
            throw std::logic_error(
                "propagate_relevance_weighted: rule bodies with arity > 2 are not yet designed for -- "
                "see datalog_lrp.hpp's own logged scope restriction");
        }
    }
}

}  // namespace

RelevanceResult propagate_relevance_weighted(const std::vector<Rule>& rules,
                                              const WeightedFactDatabase<double>& fixpoint, const Atom& query,
                                              double relevance_seed, double epsilon) {
    RelevanceResult out;
    distribute(query, relevance_seed, rules, fixpoint, epsilon, out);
    return out;
}

}  // namespace pulsatrix::datalog
