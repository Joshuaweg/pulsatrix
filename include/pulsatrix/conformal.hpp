/** @file conformal.hpp
 *  @brief Split conformal prediction (KS-3): prediction sets for a classifier and intervals for a
 *         regressor that contain the truth with a chosen probability, whatever the model
 *         (Vovk, Gammerman and Shafer, 2005; Angelopoulos and Bates, arXiv 2107.07511).
 *  @ingroup interpretability
 *
 * A model's probability of 0.9 doesn't mean it is right 90% of the time. Conformal prediction
 * turns any model's scores into a guarantee. Score each example of a held-out calibration set by
 * how badly the model fits it, and take the ⌈(n + 1)(1 − α)⌉-th smallest score, q̂. A new input's
 * prediction set holds every answer that would score at most q̂, so it can be empty. If the calibration examples and
 * the new one are exchangeable, the set contains the true answer with probability at least
 * 1 − α, and at most 1 − α + 1/(n + 1).
 *
 * The guarantee is marginal: it holds on average over inputs, not for each one or each group.
 * Check the coverage of each slice with PerSliceError() on the misses (EvaluateCoverage()).
 *
 * Scores:
 *  - **LAC**, 1 − p(true class) (Sadinle, Lei and Wasserman, 2019): the smallest sets on average,
 *    but they can be empty, and hard inputs are undercovered;
 *  - **APS**, the probability mass of the classes ranked at or above the true one (Romano, Sesia
 *    and Candès, NeurIPS 2020): larger sets for harder inputs, closer to coverage for each input;
 *  - for regression, the absolute residual |y − ŷ|, giving ŷ ± q̂, or for quantile regressors
 *    CQR's max(lo − y, y − hi) (Romano, Patterson and Candès, NeurIPS 2019), giving [lo − q̂, hi + q̂].
 *
 * LAC sets and absolute-residual intervals match MAPIE 1.1.0's SplitConformalClassifier and
 * SplitConformalRegressor; tests/conformal_test.cpp checks them against
 * tools/generate_conformal_reference_values.py.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief How a conformal classifier scores an example. */
enum class ConformalScore {
    /// 1 − p(true class): Least Ambiguous set-valued Classifier.
    Lac,
    /// The mass of the classes ranked at or above the true one: Adaptive Prediction Sets.
    Aps
};

/** @brief CalibrateConformalClassifier()'s settings. */
struct ConformalOptions {
    /** @brief The miscoverage: sets miss the true class with probability at most alpha. */
    double alpha = 0.1;
    ConformalScore score = ConformalScore::Aps;
    /** @brief APS only: subtract u p(class) from each score, u uniform in [0, 1) and drawn once per
     *         example, as Romano et al. do, for coverage close to exactly 1 − alpha; the seed for u.
     *         Without it APS over-covers a little, since its score jumps by a whole class's
     *         probability. */
    bool randomized = false;
    uint64_t seed = 0;
};

/** @brief A calibrated conformal classifier. */
struct ConformalClassifier {
    ConformalScore score = ConformalScore::Aps;
    double alpha = 0.1;
    bool randomized = false;
    uint64_t seed = 0;
    /** @brief The calibration examples. */
    int64_t n = 0;
    /** @brief The ⌈(n + 1)(1 − alpha)⌉-th smallest calibration score; +infinity when that rank is
     *         above n, and every set then holds every class. */
    double quantile = 0.0;
};

/**
 * @brief Calibrates on held-out examples the model wasn't trained on.
 * @param probabilities (n, K), each row the model's probabilities for one calibration example.
 * @param labels The true class of each.
 * @throws std::invalid_argument if probabilities isn't (n, K) with n >= 1 and K >= 2, labels has
 *         another length or a class outside [0, K), a probability is negative or not finite,
 *         alpha isn't in (0, 1), or options.randomized is set for LAC.
 */
[[nodiscard]] ConformalClassifier CalibrateConformalClassifier(const Tensor& probabilities,
                                                               const std::vector<int64_t>& labels,
                                                               const ConformalOptions& options = {});

/**
 * @brief The prediction set of each input: the classes, most probable first.
 * @param probabilities (N, K), with the K of the calibration.
 * @throws std::invalid_argument as CalibrateConformalClassifier() does for probabilities.
 */
[[nodiscard]] std::vector<std::vector<int64_t>> PredictionSets(const ConformalClassifier& classifier,
                                                               const Tensor& probabilities);

/** @brief A calibrated conformal regressor. */
struct ConformalRegressor {
    /** @brief "absolute_residual" or "cqr". */
    std::string score;
    double alpha = 0.1;
    int64_t n = 0;
    /** @brief The ⌈(n + 1)(1 − alpha)⌉-th smallest score; +infinity when that rank is above n. */
    double quantile = 0.0;
};

/** @brief One prediction interval. */
struct PredictionInterval {
    double lower = 0.0;
    double upper = 0.0;
};

/**
 * @brief Calibrates intervals ŷ ± q̂ from a point regressor's predictions on held-out examples.
 * @throws std::invalid_argument if the two are empty, have different lengths or hold a value that
 *         isn't finite, or alpha isn't in (0, 1).
 */
[[nodiscard]] ConformalRegressor CalibrateConformalRegressor(const std::vector<double>& predictions,
                                                             const std::vector<double>& targets, double alpha = 0.1);

/**
 * @brief Calibrates CQR intervals [lo − q̂, hi + q̂] from a quantile regressor's lower and upper
 *        predictions, such as its alpha/2 and 1 − alpha/2 quantiles, on held-out examples.
 * @throws std::invalid_argument as CalibrateConformalRegressor() does, for all three.
 */
[[nodiscard]] ConformalRegressor CalibrateConformalQuantileRegressor(const std::vector<double>& lower,
                                                                     const std::vector<double>& upper,
                                                                     const std::vector<double>& targets,
                                                                     double alpha = 0.1);

/**
 * @brief The intervals for a point regressor's predictions (absolute_residual).
 * @throws std::invalid_argument if the regressor is CQR's, or a prediction isn't finite.
 */
[[nodiscard]] std::vector<PredictionInterval> PredictionIntervals(const ConformalRegressor& regressor,
                                                                  const std::vector<double>& predictions);

/**
 * @brief The intervals for a quantile regressor's predictions (cqr).
 * @throws std::invalid_argument if the regressor isn't CQR's, the two have different lengths, or a
 *         value isn't finite.
 */
[[nodiscard]] std::vector<PredictionInterval> PredictionIntervals(const ConformalRegressor& regressor,
                                                                  const std::vector<double>& lower,
                                                                  const std::vector<double>& upper);

/** @brief How often the sets or intervals held the truth, and how large they were. */
struct CoverageReport {
    int64_t n = 0;
    /** @brief The fraction that held the truth; compare with 1 − alpha. */
    double coverage = 0.0;
    /** @brief The mean set size, or the mean interval width. */
    double mean_size = 0.0;
    /** @brief Classification only: how many sets were empty. */
    int64_t empty = 0;
    /** @brief Classification only: size_counts[s] sets had s classes, and coverage_by_size[s] of
     *         them held the truth (0 where there were none). Undercoverage of the large sets is a
     *         sign the score doesn't adapt to hard inputs. */
    std::vector<int64_t> size_counts;
    std::vector<double> coverage_by_size;
    /** @brief Per input: 1 where the truth was missed, 0 where it was held; PerSliceError() on it
     *         tests each slice's coverage. */
    std::vector<double> misses;
};

/** @throws std::invalid_argument if there are no sets or labels has another length. */
[[nodiscard]] CoverageReport EvaluateCoverage(const std::vector<std::vector<int64_t>>& sets,
                                              const std::vector<int64_t>& labels);

/** @throws std::invalid_argument if there are no intervals or targets has another length. */
[[nodiscard]] CoverageReport EvaluateCoverage(const std::vector<PredictionInterval>& intervals,
                                              const std::vector<double>& targets);

/** @brief The report as a `pulsatrix.coverage.v1` document, without the per-input misses. */
[[nodiscard]] JsonValue ToJson(const CoverageReport& report);

}  // namespace pulsatrix
