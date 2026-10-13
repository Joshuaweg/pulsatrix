/** @file performance.hpp
 *  @brief How well a model does (KS-3): a classifier's accuracy, log loss, Brier score, calibration
 *         and per-class scores; a regressor's errors; and the error on each slice of the data, with
 *         a test of whether a slice is worse than the rest.
 *  @ingroup interpretability
 *
 * These answer the Performance question of the XAI question bank. Every number here matches
 * scikit-learn's (or, for the tests, SciPy's) on the same data; tests/performance_test.cpp checks
 * them against values from tools/generate_performance_reference_values.py.
 *
 * Calibration is the expected calibration error of Guo, Pleiss, Sun and Weinberger (ICML 2017):
 * group the predictions into equal-width bins of confidence, the largest probability, and average
 * the gap between each bin's accuracy and its mean confidence, weighted by its size. A model whose
 * 0.8-confidence predictions are right 80% of the time is calibrated.
 *
 * A model can do well on average and fail one group. PerSliceError() reports the error on each
 * slice and tests each against the rest of the data, so a failure confined to one slice isn't
 * averaged away. It is the per-slice error check the Control reason requires.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief EvaluateClassifier()'s settings. */
struct ClassificationOptions {
    /** @brief Equal-width confidence bins for the calibration error; Guo et al. used 15. */
    int64_t bins = 15;
};

/** @brief One confidence bin, (lower, upper]; the first bin also holds a confidence of 0. */
struct ReliabilityBin {
    double lower = 0.0;
    double upper = 0.0;
    int64_t count = 0;
    /** @brief The mean confidence of the predictions in the bin; 0 when it is empty. */
    double confidence = 0.0;
    /** @brief The fraction of them that are right; 0 when it is empty. */
    double accuracy = 0.0;
};

/** @brief How well a classifier does on a labeled dataset. */
struct ClassificationReport {
    int64_t n = 0;
    int64_t classes = 0;
    /** @brief The fraction of inputs whose most probable class is the label. */
    double accuracy = 0.0;
    /** @brief The mean negative log probability of the label (scikit-learn's log_loss). A
     *         probability of 0 counts as 1e-15. */
    double log_loss = 0.0;
    /** @brief Brier's multi-class score: the mean over inputs of sum_k (p_k - y_k)^2, from 0 to 2
     *         (scikit-learn's brier_score_loss with scale_by_half=False). */
    double brier = 0.0;
    /** @brief The expected calibration error: sum over bins of count/n |accuracy - confidence|. */
    double ece = 0.0;
    /** @brief The maximum calibration error: the largest |accuracy - confidence| of a non-empty bin. */
    double mce = 0.0;
    /** @brief The reliability diagram: one entry per bin, from confidence 0 to 1. */
    std::vector<ReliabilityBin> bins;
    /** @brief confusion[true class][predicted class]: how many inputs. */
    std::vector<std::vector<int64_t>> confusion;
    /** @brief Per class: the fraction of its predictions that are right, of its inputs found, and
     *         their harmonic mean; 0 where there is nothing to divide by, as scikit-learn's
     *         zero_division=0. */
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1;
    /** @brief Per class: how many inputs have it as their label. */
    std::vector<int64_t> support;
    /** @brief The unweighted mean of f1 over the classes. */
    double macro_f1 = 0.0;
};

/**
 * @brief Scores a classifier's probabilities against the labels.
 * @param probabilities (N, K): each row the model's probabilities for one input, such as a softmax.
 *        Rows aren't renormalized; the predicted class is the most probable one, the lowest index
 *        on a tie.
 * @param labels The true class of each input, in [0, K).
 * @throws std::invalid_argument if probabilities isn't (N, K) with N >= 1 and K >= 2, labels has
 *         another length or a class outside [0, K), a probability is negative or not finite, or
 *         options.bins < 1.
 */
[[nodiscard]] ClassificationReport EvaluateClassifier(const Tensor& probabilities, const std::vector<int64_t>& labels,
                                                      const ClassificationOptions& options = {});

/** @brief How well a regressor does. */
struct RegressionReport {
    int64_t n = 0;
    double mse = 0.0;
    double rmse = 0.0;
    double mae = 0.0;
    /** @brief The coefficient of determination, 1 - SSE / SST (scikit-learn's r2_score). When the
     *         targets are constant it is 1 for a perfect fit and 0 otherwise, as scikit-learn's. */
    double r2 = 0.0;
};

/**
 * @brief Scores a regressor's predictions against the targets.
 * @throws std::invalid_argument if the two are empty, have different lengths or hold a value that
 *         isn't finite.
 */
[[nodiscard]] RegressionReport EvaluateRegressor(const std::vector<double>& predictions,
                                                 const std::vector<double>& targets);

/** @brief How PerSliceError() tests a slice against the rest. */
enum class SliceTest {
    /// Fisher's exact test when every error is 0 or 1, a permutation test otherwise.
    Auto,
    /// Fisher's exact test, one-sided: whether the slice's errors are more frequent than the rest's.
    /// Needs every error to be 0 or 1.
    FisherExact,
    /// A permutation test, one-sided, of the slice's mean error against the rest's.
    Permutation
};

/** @brief PerSliceError()'s settings. */
struct SliceErrorOptions {
    /** @brief The family-wise error rate: a slice is worse when its p-value is at most
     *         alpha / (number of slices), Bonferroni's correction. Also sets the intervals' level,
     *         1 - alpha. */
    double alpha = 0.05;
    SliceTest test = SliceTest::Auto;
    /** @brief Permutation test only: how many relabelings, and their seed. */
    int64_t permutations = 999;
    uint64_t seed = 0;
};

/** @brief One slice's error. */
struct SliceError {
    std::string name;
    int64_t count = 0;
    /** @brief The slice's mean error: its error rate when the errors are 0 or 1. */
    double error = 0.0;
    /** @brief An interval for it at level 1 - alpha: Wilson's score interval for 0/1 errors, the
     *         normal approximation mean +- z s / sqrt(n) otherwise. */
    double lower = 0.0;
    double upper = 0.0;
    /** @brief The mean error of every input outside the slice. */
    double rest_error = 0.0;
    /** @brief One-sided: how likely an error this much above the rest's is if slice membership
     *         doesn't matter. 1 when the slice is the whole dataset. */
    double p_value = 1.0;
    /** @brief p_value <= alpha / (number of slices). */
    bool worse = false;
};

/** @brief PerSliceError()'s result. */
struct SliceErrorReport {
    int64_t n = 0;
    /** @brief The mean error over the whole dataset. */
    double error = 0.0;
    /** @brief The test that ran: "fisher_exact" or "permutation". */
    std::string test;
    double alpha = 0.05;
    /** @brief Every slice, the highest error first; ties keep the order slices first appear in. */
    std::vector<SliceError> slices;

    /** @brief Whether any slice is worse than the rest. */
    [[nodiscard]] bool any_worse() const;
};

/**
 * @brief The error on each slice of a dataset, each tested against the rest.
 * @param errors Each input's error: 0 or 1 for right or wrong, or a loss.
 * @param slices The slice each input belongs to, such as a group or a data source.
 * @throws std::invalid_argument if errors is empty, slices has another length, an error isn't
 *         finite, options.test is FisherExact and an error isn't 0 or 1, alpha isn't in (0, 1), or
 *         permutations < 1.
 */
[[nodiscard]] SliceErrorReport PerSliceError(const std::vector<double>& errors, const std::vector<std::string>& slices,
                                             const SliceErrorOptions& options = {});

/**
 * @brief A classifier's error rate on each slice: an input is wrong when its most probable class
 *        isn't its label.
 * @throws std::invalid_argument as EvaluateClassifier() and PerSliceError() do.
 */
[[nodiscard]] SliceErrorReport PerSliceError(const Tensor& probabilities, const std::vector<int64_t>& labels,
                                             const std::vector<std::string>& slices,
                                             const SliceErrorOptions& options = {});

/** @brief The report as a `pulsatrix.classification_report.v1` document. */
[[nodiscard]] JsonValue ToJson(const ClassificationReport& report);
/** @brief The report as a `pulsatrix.regression_report.v1` document. */
[[nodiscard]] JsonValue ToJson(const RegressionReport& report);
/** @brief The report as a `pulsatrix.slice_error.v1` document. */
[[nodiscard]] JsonValue ToJson(const SliceErrorReport& report);

/** @brief The slices as a plain-text table, worst first, with the test and whether any is worse. */
[[nodiscard]] std::string DescribeSliceError(const SliceErrorReport& report);

}  // namespace pulsatrix
