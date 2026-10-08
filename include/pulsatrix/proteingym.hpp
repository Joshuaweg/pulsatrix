/** @file proteingym.hpp
 *  @brief Running ProteinGym's deep-mutational-scanning substitution benchmarks (PLM-3): read an
 *         assay, score every variant with a protein language model, and compute ProteinGym's metrics.
 *  @ingroup interpretability
 */
#pragma once

#include <string>
#include <vector>

#include "pulsatrix/fitness_metrics.hpp"
#include "pulsatrix/variant_scoring.hpp"

namespace pulsatrix {

/** @brief One assay's row of ProteinGym's reference file (`DMS_substitutions.csv`). */
struct ProteinGymAssay {
    std::string id;        ///< `DMS_id`, for example `BLAT_ECOLX_Stiffler_2015`
    std::string filename;  ///< `DMS_filename`
    std::string target_seq;
    /** @brief The position of the sequence's first residue in the variant names (`start_idx`, else 1). */
    int64_t offset = 1;
};

/** @brief Every assay in a reference file. @throws std::runtime_error if it can't be read;
 *         std::invalid_argument if a needed column is missing. */
[[nodiscard]] std::vector<ProteinGymAssay> ReadProteinGymReference(const std::string& path);

/** @brief An assay's measured variants: `mutant`, `DMS_score` and `DMS_score_bin` columns. */
struct DmsVariants {
    std::vector<std::string> mutants;
    std::vector<double> scores;
    std::vector<int> bins;
};

/** @brief Reads an assay's CSV. @throws std::runtime_error / std::invalid_argument as above. */
[[nodiscard]] DmsVariants ReadDmsVariants(const std::string& path);

/** @brief How variants are scored. */
enum class VariantStrategy { MaskedMarginals, WildTypeMarginals };

/** @brief One assay's result. */
struct AssayResult {
    std::string id;
    int64_t length = 0;
    FitnessMetrics metrics;
    /** @brief One score per variant, in the assay's order. */
    std::vector<double> scores;
    double seconds = 0;
};

/** @brief Scores every variant of @p assay and evaluates them against the measurements.
 *  @throws std::invalid_argument if a variant doesn't match the assay's sequence. */
[[nodiscard]] AssayResult EvaluateAssay(VariantScorer& scorer, const ProteinGymAssay& assay, const DmsVariants& variants,
                                        VariantStrategy strategy = VariantStrategy::MaskedMarginals);

}  // namespace pulsatrix
