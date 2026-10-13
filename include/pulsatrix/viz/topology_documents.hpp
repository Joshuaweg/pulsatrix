/** @file topology_documents.hpp
 *  @brief The persistence diagram document (TDA-9): the features of each homology dimension,
 *         with the certified error bound and any significance results, for the persistence
 *         diagram, barcode and Betti curve views in topology_views.hpp.
 *  @ingroup visualization
 *
 *  It follows the rules of every viz document (document.hpp): a `"schema"` of
 *  `pulsatrix.persistence_diagram.v1`, non-finite numbers (an essential feature's infinite death,
 *  an untested feature's NaN p-value) as null plus a `"nonfinite"` entry, and readers that name the
 *  JSON Pointer of the first problem.
 */
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/tda/persistence_diagram.hpp"
#include "pulsatrix/tda/rips_persistence.hpp"
#include "pulsatrix/tda/significance.hpp"

namespace pulsatrix {

/** @brief One homology dimension of a PersistenceDiagramDocument. */
struct PersistenceDiagramEntry {
    /** @brief 0 for components, 1 for loops, 2 for voids. */
    int dimension = 0;
    /** @brief One birth and one death per feature; a death of +infinity for an essential feature. */
    std::vector<double> births;
    std::vector<double> deaths;
    /** @brief PersistenceDiagram::error_bound: every point is within this distance of where the
     *         exact distances would put it, so a feature with persistence at most twice this
     *         could be noise in the distances alone. */
    double error_bound = 0;
    /** @brief Empty, or one p-value per feature; NaN for a feature that wasn't tested, such as an
     *         essential one. From AttachSignificance(). */
    std::vector<double> p_values;
    /** @brief A feature is significant when its p-value is below this; 0 when there are no
     *         p-values. UniversalNullResult::threshold, already corrected for the number tested. */
    double significance_threshold = 0;
    /** @brief A bootstrap confidence band (BootstrapBandResult::band); 0 for none. Features with
     *         persistence above twice the band are farther from the diagonal than it reaches. */
    double bootstrap_band = 0;

    [[nodiscard]] size_t size() const { return births.size(); }
};

/** @brief Persistence diagrams of one point cloud, one entry per homology dimension. */
struct PersistenceDiagramDocument {
    std::string title;
    /** @brief The filtration that made the diagrams, for example "rips". */
    std::string filtration = "rips";
    /** @brief What the points were, for example "SmolLM2-135M, layer 12, weekday tokens". */
    std::string source;
    /** @brief How many points; 0 when unknown. */
    int64_t num_points = 0;
    /** @brief The largest scale the filtration was computed to (RipsResult::threshold); 0 when
     *         unknown. Essential features are drawn at this scale. */
    double threshold = 0;
    /** @brief One per dimension, each dimension at most once. */
    std::vector<PersistenceDiagramEntry> diagrams;

    /** @brief The entry for @p dimension, or nullptr. */
    [[nodiscard]] const PersistenceDiagramEntry* find(int dimension) const;
    [[nodiscard]] PersistenceDiagramEntry* find(int dimension);
};

/** @brief The diagrams as a document: births, deaths and each diagram's error bound.
 *  @throws std::invalid_argument for a diagram the document can't hold (a NaN, a death before its
 *          birth, a dimension given twice). */
[[nodiscard]] PersistenceDiagramDocument ToPersistenceDiagramDocument(const std::vector<PersistenceDiagram>& diagrams,
                                                                      std::string title = "");
/** @brief The same, from ComputeRipsPersistence(), with its threshold. */
[[nodiscard]] PersistenceDiagramDocument ToPersistenceDiagramDocument(const RipsResult& result, std::string title = "");

/**
 * @brief Adds UniversalNullTest()'s p-values and corrected threshold to the entry for @p dimension.
 *        Features the test skipped get NaN.
 * @param test The test of that dimension's diagram, exactly as the document holds it.
 * @throws std::invalid_argument if the document has no such dimension, or a tested feature isn't
 *         in it: its index is out of range, or the pair there has a different death / birth.
 */
void AttachSignificance(PersistenceDiagramDocument& doc, int dimension, const UniversalNullResult& test);

/** @brief Adds BootstrapBand()'s band to the entry for @p dimension.
 *  @throws std::invalid_argument if the document has no such dimension. */
void AttachBootstrapBand(PersistenceDiagramDocument& doc, int dimension, const BootstrapBandResult& band);

/** @brief Whether feature @p i of @p entry passed the attached tests: its p-value is below the
 *         threshold, or, with only a bootstrap band, its persistence is above twice the band.
 *         True for every feature when no test is attached. */
[[nodiscard]] bool IsSignificant(const PersistenceDiagramEntry& entry, size_t i);

/** @brief Writes the document.
 *  @throws std::invalid_argument naming the JSON Pointer of the first problem. */
[[nodiscard]] std::string ToJson(const PersistenceDiagramDocument& doc);
/** @brief Reads a document ToJson() wrote.
 *  @throws std::invalid_argument naming the JSON Pointer of the first problem. */
[[nodiscard]] PersistenceDiagramDocument ParsePersistenceDiagramDocument(std::string_view json);

}  // namespace pulsatrix
