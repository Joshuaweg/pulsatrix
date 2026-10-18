/** @file calibration.hpp
 *  @brief Calibrating a model's probabilities (KS-13): temperature scaling fitted on held-out
 *         data, and the error in log-odds against reference probabilities, by probability band.
 *  @ingroup interpretability
 *
 * **Temperature scaling** (Guo, Pleiss, Sun and Weinberger, ICML 2017) divides a model's logits by
 * one number T, fitted to minimize the log loss on held-out labeled data. It changes how confident
 * the model is, never which class it ranks first. The log loss is convex in 1/T, so the fit is a
 * safeguarded Newton iteration and finds the global minimum.
 *
 * **Log-odds error** measures probabilities against reference probabilities rather than labels:
 * |logit(p) - logit(q)|. It is meant for rare events. The expected calibration error (performance.hpp)
 * averages over bins and is dominated by common cases, so a model that says 1e-2 where the truth
 * is 1e-4 barely moves it; in log-odds that is an error of 4.6. LogOddsError() reports the error
 * separately in each band of the reference probability, with optional importance weights for test
 * sets that oversample rare cases.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/json.hpp"

namespace pulsatrix {

/** @brief A fitted temperature and the weighted mean log loss before and after it. */
struct TemperatureFit {
    double temperature = 1.0;
    double log_loss_before = 0.0;  ///< at T = 1
    double log_loss_after = 0.0;   ///< at the fitted T
    int64_t iterations = 0;
    /** @brief False when the minimum is at a bound of T in [1e-6, 1e6]: the loss keeps falling as
     *         T grows (the logits are no better than chance, or point the wrong way) or as T
     *         shrinks (the logits separate the labels perfectly). temperature is then that bound. */
    bool converged = true;
};

/**
 * @brief Fits T for binary log-odds: P(y = 1) = sigmoid(logit / T).
 * @param weights Per example, such as importance weights; all 1 when empty.
 * @throws std::invalid_argument if the inputs are empty or of different lengths, a label isn't 0
 *         or 1, a logit isn't finite, or a weight is negative, not finite, or all are 0.
 */
[[nodiscard]] TemperatureFit FitBinaryTemperature(const std::vector<double>& logits, const std::vector<int64_t>& labels,
                                                  const std::vector<double>& weights = {});

/**
 * @brief Fits T for multi-class logits: P(y = k) = softmax(logits / T)_k.
 * @param logits One row per example, every row the same length K >= 2.
 * @throws std::invalid_argument as FitBinaryTemperature(), or if a row has another length or a
 *         label is outside [0, K).
 */
[[nodiscard]] TemperatureFit FitTemperature(const std::vector<std::vector<double>>& logits,
                                            const std::vector<int64_t>& labels, const std::vector<double>& weights = {});

/** @brief Log-odds errors over the examples whose reference probability is in [lower, upper). */
struct LogOddsBand {
    double lower = 0.0;
    double upper = 1.0;  ///< the last band includes 1
    int64_t count = 0;
    double weight = 0.0;  ///< the band's total weight
    /** @brief Weighted means of |e| and e, and sqrt of the weighted mean of e^2, where e is
     *         logit(predicted) - logit(reference): bias > 0 means the predictions are too high. */
    double mean_abs = 0.0;
    double bias = 0.0;
    double rms = 0.0;
    /** @brief Weighted quantiles of |e|: the smallest |e| with at least half (or 90%) of the
     *         band's weight at or below it. */
    double median_abs = 0.0;
    double p90_abs = 0.0;
    double max_abs = 0.0;
};

struct LogOddsErrorReport {
    double clip = 0.0;               ///< probabilities were clipped to [clip, 1 - clip]
    LogOddsBand overall;             ///< every example
    std::vector<LogOddsBand> bands;  ///< one per band of the reference probability, low to high
};

/**
 * @brief The error in log-odds of predicted probabilities against reference ones.
 * @param band_edges Increasing edges strictly inside (0, 1): edges {1e-3} give the bands
 *        [0, 1e-3) and [1e-3, 1]. Empty gives one band.
 * @param clip Both probabilities are clipped to [clip, 1 - clip] first, so a reference of exactly
 *        0 or 1 gives a finite error. With the default, logit(1e-6) is about -13.8.
 * @throws std::invalid_argument if the inputs are empty or of different lengths, a probability is
 *         outside [0, 1] or not finite, a weight is negative or not finite or all are 0, the edges
 *         aren't increasing inside (0, 1), or clip isn't in (0, 0.5).
 */
[[nodiscard]] LogOddsErrorReport LogOddsError(const std::vector<double>& predicted, const std::vector<double>& reference,
                                              const std::vector<double>& weights = {},
                                              const std::vector<double>& band_edges = {1e-3}, double clip = 1e-6);

/** @brief The report as a JSON object: clip, overall, and bands. */
[[nodiscard]] JsonValue ToJson(const LogOddsErrorReport& report);

}  // namespace pulsatrix
