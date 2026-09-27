/** @file datalog_term.hpp
 *  @brief Function-symbol-free Datalog term -- a constant or a variable, never a compound
 *         term. Phase 3 Mission 0 of campaign_exai_dl_library_neuro_symbolic (Datalog core).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstddef>
#include <functional>
#include <string>

namespace pulsatrix::datalog {

/** @brief Which of the two (and only two) term alternatives this Term is. */
enum class TermKind { Constant, Variable };

/**
 * @brief A Datalog term: either a constant (e.g. `a`) or a variable (e.g. `X`) -- never a
 *        compound/function-symbol term.
 * @note Enforced structurally, not just documented: the only state this class stores is a
 *       `TermKind` tag plus a single `std::string` payload (the constant's value or the
 *       variable's name). There is no member, constructor, or accessor through which a Term
 *       could ever hold another Term or a list of Terms -- the decidable-fragment restriction
 *       (campaign doc Risk Register: "no function symbols, no unbounded recursion over
 *       complex/nested terms") is a property of this type's shape, not a runtime check that
 *       could be bypassed.
 *
 *       Representation choice (mission Stage 3 design question 2): interned as plain
 *       `std::string`s, compared/hashed via `std::string`'s own `operator==`/`std::hash`,
 *       not small integer IDs. Rationale: this mission's stated scope is the boolean
 *       semiring on toy-scale knowledge bases (campaign Risk Register: "keep Phase 3's
 *       exit-gate knowledge base deliberately small ... until proven") -- correctness and
 *       readability of hand-verified fixpoint traces matter far more than string-comparison
 *       performance at this stage. An integer-interning table is a legitimate future
 *       optimization if Mission 1/2's real-valued semiring work hits an actual bottleneck,
 *       but building it now would be optimizing ahead of a need this mission doesn't have.
 */
class Term {
public:
    /** @brief Constructs a constant term with the given value (e.g. `"a"`). */
    static Term make_constant(std::string value) { return Term(TermKind::Constant, std::move(value)); }

    /** @brief Constructs a variable term with the given name (e.g. `"X"`). */
    static Term make_variable(std::string name) { return Term(TermKind::Variable, std::move(name)); }

    /** @brief Whether this term is a constant or a variable. */
    [[nodiscard]] TermKind kind() const { return kind_; }

    /** @brief True iff this term is a constant. */
    [[nodiscard]] bool is_constant() const { return kind_ == TermKind::Constant; }

    /** @brief True iff this term is a variable. */
    [[nodiscard]] bool is_variable() const { return kind_ == TermKind::Variable; }

    /** @brief The constant's value, or the variable's name -- whichever this term is. */
    [[nodiscard]] const std::string& value() const { return value_; }

    [[nodiscard]] bool operator==(const Term& other) const {
        return kind_ == other.kind_ && value_ == other.value_;
    }
    [[nodiscard]] bool operator!=(const Term& other) const { return !(*this == other); }

private:
    Term(TermKind kind, std::string value) : kind_(kind), value_(std::move(value)) {}

    TermKind kind_;
    std::string value_;
};

/** @brief Hash functor for `Term`, for use in `std::unordered_map`/`std::unordered_set`. */
struct TermHash {
    [[nodiscard]] std::size_t operator()(const Term& term) const {
        return std::hash<std::string>{}(term.value()) ^ (static_cast<std::size_t>(term.kind()) << 1);
    }
};

}  // namespace pulsatrix::datalog
