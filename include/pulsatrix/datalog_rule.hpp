/** @file datalog_rule.hpp
 *  @brief A Datalog rule: `head :- body1, body2, ...`, range-restricted (safe) by
 *         construction. Phase 3 Mission 0 of campaign_exai_dl_library_neuro_symbolic
 *         (Datalog core).
 *  @ingroup dl_modules
 */
#pragma once

#include <vector>

#include "pulsatrix/datalog_atom.hpp"

namespace pulsatrix::datalog {

/**
 * @brief A rule: a head atom entailed whenever every body atom is simultaneously
 *        satisfiable under one consistent variable substitution.
 * @note Enforces Datalog "range restriction"/safety at construction: every variable that
 *       appears in the head must also appear in at least one body atom. Without this, a
 *       rule like `p(X) :- q(a).` would be free to bind X to an unbounded/arbitrary
 *       constant, breaking the decidable-fragment termination guarantee this whole phase
 *       depends on (campaign doc Decision Point 1: "Datalog's decidable fragment always
 *       terminates via bottom-up fixpoint"). A rule with an empty body is never safe unless
 *       its head is also empty-arity (there is no such head shape here), so an empty body
 *       with a non-empty head always throws.
 */
class Rule {
public:
    /**
     * @brief Constructs a rule.
     * @throws std::invalid_argument if any variable in `head` does not appear in `body`
     *         (unsafe/unrestricted rule).
     */
    Rule(Atom head, std::vector<Atom> body);

    [[nodiscard]] const Atom& head() const { return head_; }
    [[nodiscard]] const std::vector<Atom>& body() const { return body_; }

private:
    Atom head_;
    std::vector<Atom> body_;
};

}  // namespace pulsatrix::datalog
