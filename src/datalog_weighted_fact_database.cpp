#include "pulsatrix/datalog_weighted_fact_database.hpp"

#include <stdexcept>

namespace pulsatrix::datalog {

template <typename T>
WeightedFactDatabase<T>::WeightedFactDatabase(WeightedFactSet<T> facts) {
    for (const auto& [fact, weight] : facts) {
        if (!fact.is_ground()) {
            throw std::invalid_argument(
                "WeightedFactDatabase: predicate '" + fact.predicate_name() +
                "' has a non-ground term -- a WeightedFactDatabase holds only ground facts");
        }
    }
    facts_ = std::move(facts);
}

template <typename T>
void WeightedFactDatabase<T>::set(Atom fact, T weight) {
    if (!fact.is_ground()) {
        throw std::invalid_argument(
            "WeightedFactDatabase::set: predicate '" + fact.predicate_name() +
            "' has a non-ground term -- a WeightedFactDatabase holds only ground facts");
    }
    facts_[std::move(fact)] = weight;
}

template class WeightedFactDatabase<bool>;
template class WeightedFactDatabase<double>;
template class WeightedFactDatabase<float>;
template class WeightedFactDatabase<DualNumber<double>>;

}  // namespace pulsatrix::datalog
