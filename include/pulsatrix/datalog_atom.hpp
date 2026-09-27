/** @file datalog_atom.hpp
 *  @brief A Datalog atom: a predicate name plus a tuple of terms, no function symbols.
 *         Phase 3 Mission 0 of campaign_exai_dl_library_neuro_symbolic (Datalog core).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "pulsatrix/datalog_term.hpp"

namespace pulsatrix::datalog {

/**
 * @brief An atom is a predicate name applied to a tuple of `Term`s, e.g. `edge(a, b)` or
 *        `ancestor(X, Y)` -- never a nested/compound structure, since `Term` itself cannot
 *        nest (see datalog_term.hpp).
 * @note An atom is "ground" (a fact, suitable for a FactDatabase) iff every one of its
 *       terms is a constant. An atom with any variable term is only valid inside a rule's
 *       head or body, never inside a FactDatabase.
 */
class Atom {
public:
    Atom(std::string predicate_name, std::vector<Term> terms)
        : predicate_name_(std::move(predicate_name)), terms_(std::move(terms)) {}

    [[nodiscard]] const std::string& predicate_name() const { return predicate_name_; }
    [[nodiscard]] const std::vector<Term>& terms() const { return terms_; }
    [[nodiscard]] std::size_t arity() const { return terms_.size(); }

    /** @brief True iff every term in this atom is a constant (no variables). */
    [[nodiscard]] bool is_ground() const {
        for (const Term& term : terms_) {
            if (!term.is_constant()) return false;
        }
        return true;
    }

    [[nodiscard]] bool operator==(const Atom& other) const {
        return predicate_name_ == other.predicate_name_ && terms_ == other.terms_;
    }
    [[nodiscard]] bool operator!=(const Atom& other) const { return !(*this == other); }

private:
    std::string predicate_name_;
    std::vector<Term> terms_;
};

/** @brief Hash functor for `Atom`, for use in `std::unordered_map`/`std::unordered_set`. */
struct AtomHash {
    [[nodiscard]] std::size_t operator()(const Atom& atom) const {
        std::size_t h = std::hash<std::string>{}(atom.predicate_name());
        for (const Term& term : atom.terms()) {
            h ^= (TermHash{}(term) + 0x9e3779b9U + (h << 6) + (h >> 2));
        }
        return h;
    }
};

}  // namespace pulsatrix::datalog
