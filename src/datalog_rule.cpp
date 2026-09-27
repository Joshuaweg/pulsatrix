#include "pulsatrix/datalog_rule.hpp"

#include <stdexcept>
#include <unordered_set>

namespace pulsatrix::datalog {

Rule::Rule(Atom head, std::vector<Atom> body) : head_(std::move(head)), body_(std::move(body)) {
    std::unordered_set<std::string> body_variables;
    for (const Atom& body_atom : body_) {
        for (const Term& term : body_atom.terms()) {
            if (term.is_variable()) {
                body_variables.insert(term.value());
            }
        }
    }

    for (const Term& term : head_.terms()) {
        if (term.is_variable() && body_variables.find(term.value()) == body_variables.end()) {
            throw std::invalid_argument(
                "Rule: head variable '" + term.value() +
                "' does not appear in the body -- unsafe/unrestricted rule");
        }
    }
}

}  // namespace pulsatrix::datalog
