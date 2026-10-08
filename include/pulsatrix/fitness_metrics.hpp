/** @file fitness_metrics.hpp
 *  @brief ProteinGym's metrics for how well predicted variant scores track measured fitness
 *         (PLM-3), computed exactly as ProteinGym's `performance_DMS_benchmarks.py` does.
 *  @ingroup interpretability
 */
#pragma once

#include <vector>

namespace pulsatrix {

/** @brief Spearman's rank correlation, ties given their average rank (scipy's `spearmanr`). NaN
 *         if either input is constant. @throws std::invalid_argument for sizes that differ or < 2. */
[[nodiscard]] double SpearmanCorrelation(const std::vector<double>& x, const std::vector<double>& y);

/** @brief Area under the ROC curve of @p scores for @p labels (0 or 1), ties counted half
 *         (sklearn's `roc_auc_score`). NaN if only one class is present. */
[[nodiscard]] double RocAuc(const std::vector<int>& labels, const std::vector<double>& scores);

/** @brief Matthews correlation of binary @p labels and @p predicted (sklearn's
 *         `matthews_corrcoef`); 0 when a row or column of the confusion matrix is empty. */
[[nodiscard]] double MatthewsCorrelation(const std::vector<int>& labels, const std::vector<int>& predicted);

/** @brief ProteinGym's MCC: @p scores at or above their median predict 1. */
[[nodiscard]] double MedianSplitMcc(const std::vector<int>& labels, const std::vector<double>& scores);

/**
 * @brief ProteinGym's NDCG: gains are the true scores min-max scaled to [0, 1]; the variants the
 *        model ranks in the top @p top_percent (floor of the count) contribute gain / log2(rank + 1),
 *        normalized by the same sum for the truly best ones. Zero gains are dropped.
 * @note Tied predicted scores are ranked in input order (a stable sort). numpy's default argsort
 *       isn't stable, so ProteinGym may order exact ties differently.
 */
[[nodiscard]] double Ndcg(const std::vector<double>& true_scores, const std::vector<double>& predicted, double top_percent = 10.0);

/** @brief ProteinGym's top recall: of the variants at or above the true scores' (100 - top)
 *         percentile, the fraction also at or above the predicted scores' percentile. Percentiles
 *         interpolate linearly, as numpy's do. */
[[nodiscard]] double TopRecall(const std::vector<double>& true_scores, const std::vector<double>& predicted, double top_true = 10.0,
                               double top_model = 10.0);

/** @brief The q-th percentile (0 to 100) with linear interpolation (numpy's default). */
[[nodiscard]] double Percentile(std::vector<double> values, double q);

/** @brief All five, as ProteinGym reports them. */
struct FitnessMetrics {
    double spearman = 0;
    double auc = 0;
    double mcc = 0;
    double ndcg = 0;
    double top_recall = 0;
};

/** @brief FitnessMetrics for one assay: measured scores and their binarized labels, and predictions. */
[[nodiscard]] FitnessMetrics EvaluateFitness(const std::vector<double>& dms_scores, const std::vector<int>& dms_bins,
                                             const std::vector<double>& predicted);

}  // namespace pulsatrix
