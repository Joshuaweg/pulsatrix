/** @file datalog_weighted_fact_database.hpp
 *  @brief A map from ground Datalog atom to a semiring-typed weight. Phase 3 Mission 1 of
 *         campaign_exai_dl_library_neuro_symbolic (Generic Provenance-Semiring Abstraction).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstddef>
#include <unordered_map>

#include "pulsatrix/datalog_atom.hpp"

namespace pulsatrix::datalog {

/** @brief The concrete map type a WeightedFactDatabase<T> stores facts in. */
template <typename T>
using WeightedFactSet = std::unordered_map<Atom, T, AtomHash>;

/**
 * @brief A weighted fact database: a map from ground atom to a semiring value type `T`
 *        (`bool` for the trivial/boolean instantiation, `double`/`float` for the real-valued
 *        `(+, x)` instantiation).
 * @note Stage 3 design decision 1 (FactDatabase generalization shape): this is a **new
 *       templated type introduced alongside** Mission 0's `FactDatabase`, not a rewrite of it
 *       in place. Verified achievable (not assumed) by direct inspection of Mission 0's
 *       shipped `FactDatabase`: its own public contract (`insert`/`contains`/`size`/`empty`/
 *       `facts()`/`operator==`, all over a *set*, no weight parameter anywhere) is a strictly
 *       narrower shape than a weighted map would need (`set(fact, weight)`/`weight_of(fact)`),
 *       so "generalize `FactDatabase` in place" would have meant either breaking its existing
 *       call sites (`naive_evaluate`/`semi_naive_evaluate`/all of `datalog_fact_database_test.cpp`)
 *       or bolting an unused weight parameter onto every boolean-only call site for no benefit.
 *       Introducing `WeightedFactDatabase<T>` alongside it instead makes the campaign doc's own
 *       phrasing -- "boolean semiring re-expressed as the trivial case" -- literally true:
 *       Mission 0's `FactDatabase`/`naive_evaluate`/`semi_naive_evaluate` are **untouched**
 *       (zero diff against Mission 0's shipped files), and the "trivial case" is demonstrated
 *       by running the *new* generic `naive_evaluate_weighted<BooleanSemiring>` against a
 *       `WeightedFactDatabase<bool>` built from the identical toy KB and checking it reproduces
 *       Mission 0's own `naive_evaluate` fixpoint exactly (the regression/equivalence test).
 */
template <typename T>
class WeightedFactDatabase {
public:
    WeightedFactDatabase() = default;
    explicit WeightedFactDatabase(WeightedFactSet<T> facts);

    /**
     * @brief Sets (overwrites) a fact's weight.
     * @throws std::invalid_argument if `fact` is not ground (contains a variable term).
     */
    void set(Atom fact, T weight);

    [[nodiscard]] bool contains(const Atom& fact) const { return facts_.find(fact) != facts_.end(); }

    /** @brief The fact's current weight, or `default_weight` (conventionally `Semiring::zero()`)
     *         if the fact is not present -- "zero derivation paths" reads as "absent from the
     *         map," never as an explicit zero-weight entry. */
    [[nodiscard]] T weight_of(const Atom& fact, T default_weight) const {
        auto it = facts_.find(fact);
        return it == facts_.end() ? default_weight : it->second;
    }

    [[nodiscard]] std::size_t size() const { return facts_.size(); }
    [[nodiscard]] bool empty() const { return facts_.empty(); }
    [[nodiscard]] const WeightedFactSet<T>& facts() const { return facts_; }

    [[nodiscard]] bool operator==(const WeightedFactDatabase& other) const { return facts_ == other.facts_; }
    [[nodiscard]] bool operator!=(const WeightedFactDatabase& other) const { return !(*this == other); }

private:
    WeightedFactSet<T> facts_;
};

// Explicit instantiation declarations -- definitions live in datalog_weighted_fact_database.cpp,
// instantiated there for the two Value types this mission ships (bool, double) plus float for
// the real-valued semiring's stated "double/float" scope. Mirrors Mission 0's own
// declaration/definition split (datalog_rule.hpp/.cpp, datalog_fact_database.hpp/.cpp) rather
// than making this a header-only template -- deliberate, since only a closed, known set of
// Value types is ever needed (no caller in this mission instantiates an arbitrary T).
extern template class WeightedFactDatabase<bool>;
extern template class WeightedFactDatabase<double>;
extern template class WeightedFactDatabase<float>;

}  // namespace pulsatrix::datalog
