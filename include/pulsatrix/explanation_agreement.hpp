/** @file explanation_agreement.hpp
 *  @brief Whether two models explain the same inputs with the same evidence (roadmap KD-2): a
 *         distilled student against its teacher, a fine-tuned model against its base, or any two
 *         models that take the same input.
 *  @ingroup interpretability_agnostic
 *
 * A student can match its teacher's outputs while relying on different features. The two
 * models' attributions of the same inputs are compared input by input:
 *
 *  - **Spearman rank correlation** of the attributions, the comparison XAI-3's stability check
 *    and the randomization test use;
 *  - **top-k overlap:** the share of the teacher's k most relevant features that are also among
 *    the student's k most relevant (Quantus's TopKIntersection, with k a fraction of the
 *    features);
 *  - **cosine similarity** of the attributions;
 *
 * and across inputs by the **linear CKA** of the two (inputs × features) attribution matrices
 * (INT-2), which asks whether the maps vary from input to input in the same way.
 *
 * Two maps can agree only because every map looks alike: an image model's relevance gathers in
 * the middle of every image, a language model's on the last token. So the student's maps are
 * also compared with the teacher's maps *of other inputs*, and a permutation test asks whether
 * the matched pairs agree more than mismatched ones do. Only that excess is evidence of the same
 * reasoning about each input.
 *
 * tests/explanation_agreement_test.cpp checks every number against
 * tools/generate_explanation_agreement_reference_values.py (scipy and numpy, the same shuffles).
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/explanation_metrics.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/representation_similarity.hpp"

namespace pulsatrix {

/** @brief CompareExplanations()'s settings. */
struct AgreementOptions {
    /** @brief Compare |attribution|, as XAI-3's checks do (a feature that counts against a class
     *         is still evidence); false compares the signed values. */
    bool magnitudes = true;
    /** @brief The top-k overlap's k, as a fraction of the features: max(1, round(fraction × n)). */
    double top_fraction = 0.1;
    /** @brief Relabelings of the student's inputs for the permutation test; 0 skips it. */
    int64_t permutations = 999;
    uint64_t seed = 0;
    CkaEstimator estimator = CkaEstimator::Debiased;
};

/** @brief How one input's two attributions compare. */
struct InputAgreement {
    double spearman = 0.0;
    double top_k_overlap = 0.0;
    double cosine = 0.0;
};

/** @brief CompareExplanations()'s result. */
struct AgreementReport {
    std::vector<InputAgreement> per_input;
    /** @brief The per-input values' means. */
    double spearman = 0.0;
    double top_k_overlap = 0.0;
    double cosine = 0.0;
    /** @brief Linear CKA of the (inputs × features) matrices; NaN with too few inputs for the
     *         estimator (4 debiased, 2 biased) or when either set of maps doesn't vary. */
    double cka = 0.0;
    /** @brief The mean Spearman correlation between the student's map of one input and the
     *         teacher's map of another, over every ordered pair: the agreement that maps share
     *         whatever the input. NaN with one input. */
    double mismatched_spearman = 0.0;
    /** @brief One-sided: the share of relabelings of the student's inputs, counting the real
     *         labeling, whose mean Spearman reaches @ref spearman. NaN when skipped or with one
     *         input. Never below 1 / (permutations + 1). */
    double p_value = 0.0;
    int64_t permutations = 0;
    int64_t inputs = 0;
    int64_t features = 0;
    int64_t top_k = 0;
};

/**
 * @brief Compares two models' attributions of the same inputs, in the same order.
 * @throws std::invalid_argument if the lists are empty or of different lengths, an attribution's
 *         size differs from the first's, there are fewer than 2 features, top_fraction isn't in
 *         (0, 1], or permutations is negative.
 */
[[nodiscard]] AgreementReport CompareExplanations(const std::vector<Attribution>& teacher, const std::vector<Attribution>& student,
                                                  const AgreementOptions& options = {});

/** @brief Explains every input with both explainers, then CompareExplanations(). */
[[nodiscard]] AgreementReport ExplanationAgreement(const std::vector<Tensor>& inputs, const ExplainFn& teacher,
                                                   const ExplainFn& student, const AgreementOptions& options = {});

/** @brief The report as a `pulsatrix.explanation_agreement.v1` document. NaNs are written as null. */
[[nodiscard]] JsonValue ToJson(const AgreementReport& report);

}  // namespace pulsatrix
