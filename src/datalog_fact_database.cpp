#include "pulsatrix/datalog_fact_database.hpp"

#include <stdexcept>

namespace pulsatrix::datalog {

FactDatabase::FactDatabase(FactSet facts) {
    for (const Atom& fact : facts) {
        if (!fact.is_ground()) {
            throw std::invalid_argument(
                "FactDatabase: predicate '" + fact.predicate_name() +
                "' has a non-ground term -- a FactDatabase holds only ground facts");
        }
    }
    facts_ = std::move(facts);
}

bool FactDatabase::insert(Atom fact) {
    if (!fact.is_ground()) {
        throw std::invalid_argument(
            "FactDatabase::insert: predicate '" + fact.predicate_name() +
            "' has a non-ground term -- a FactDatabase holds only ground facts");
    }
    return facts_.insert(std::move(fact)).second;
}

}  // namespace pulsatrix::datalog
