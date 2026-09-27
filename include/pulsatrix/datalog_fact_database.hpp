/** @file datalog_fact_database.hpp
 *  @brief A set of ground (fully-constant) Datalog atoms. Phase 3 Mission 0 of
 *         campaign_exai_dl_library_neuro_symbolic (Datalog core).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstddef>
#include <unordered_set>

#include "pulsatrix/datalog_atom.hpp"

namespace pulsatrix::datalog {

/** @brief The concrete set type a FactDatabase stores facts in and the engine derives into. */
using FactSet = std::unordered_set<Atom, AtomHash>;

/**
 * @brief A fact database: a set of ground atoms. Boolean semiring only (Mission 0's scope)
 *        -- a fact is simply present or absent, with no associated weight/provenance
 *        (Mission 1's scope).
 */
class FactDatabase {
public:
    FactDatabase() = default;
    explicit FactDatabase(FactSet facts);

    /**
     * @brief Adds a fact.
     * @throws std::invalid_argument if `fact` is not ground (contains a variable term) --
     *         a FactDatabase holds only ground atoms by construction.
     * @returns true if the fact was newly inserted, false if it was already present.
     */
    bool insert(Atom fact);

    [[nodiscard]] bool contains(const Atom& fact) const { return facts_.find(fact) != facts_.end(); }
    [[nodiscard]] std::size_t size() const { return facts_.size(); }
    [[nodiscard]] bool empty() const { return facts_.empty(); }
    [[nodiscard]] const FactSet& facts() const { return facts_; }

    [[nodiscard]] bool operator==(const FactDatabase& other) const { return facts_ == other.facts_; }
    [[nodiscard]] bool operator!=(const FactDatabase& other) const { return !(*this == other); }

private:
    FactSet facts_;
};

}  // namespace pulsatrix::datalog
