/** @file anchors.hpp
 *  @brief Anchors (Ribeiro, Singh and Guestrin, "Anchors: High-Precision Model-Agnostic
 *         Explanations", AAAI 2018): an if-then rule around one input that keeps the model's
 *         prediction with high probability, whatever the other features do (XAI-4).
 *  @ingroup interpretability_agnostic
 *
 * An anchor answers "how to still be this": which conditions, if they hold, keep the prediction.
 * `IF income > 0.71 AND debt <= -0.12 THEN class 1`, with
 *  - precision: the share of inputs satisfying the rule that the model gives class 1, estimated
 *    by sampling, with a confidence bound;
 *  - coverage: the share of the background data the rule applies to.
 *
 * The predicates are the reference implementation's for ordinal features: each feature is cut at
 * the background's percentiles (quartiles by default), and every half-line at a cut that contains
 * the input is a predicate, `x > e` for cuts below the input and `x <= e` for cuts above it. An
 * anchor may hold two on one feature, which make an interval. An input satisfying the anchor is
 * drawn by taking a random background row and, for each anchored feature, a background value
 * inside its interval; features are drawn independently of one another, as the reference
 * implementation does when too few rows satisfy every predicate at once.
 *
 * The search is the paper's: a beam search that adds one predicate at a time, using KL-LUCB to find
 * the most precise candidates with few model calls, and keeps, among the anchors whose precision
 * clears the threshold, the one with the highest coverage.
 *
 * @note Precision is estimated, not guaranteed (Anchor::precision_lower is the bound), and an
 *       anchor can cover a sliver of the data: read both.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief FindAnchor's settings. The defaults are the reference implementation's, except a beam of
 *         2 and epsilon_stop. */
struct AnchorOptions {
    /** @brief The precision an anchor must reach (the paper's τ). */
    double precision_threshold = 0.95;
    /** @brief The confidence bounds hold with probability 1 - delta. */
    double delta = 0.1;
    /** @brief KL-LUCB stops when the best candidates are separated from the rest to within this. */
    double epsilon = 0.15;
    /** @brief An anchor is accepted when its precision is at least the threshold and its lower bound
     *         above threshold - epsilon_stop.
     * @note The reference implementation uses 0.05, which accepts anchors whose true precision is
     *       below the threshold: on a model where one predicate has true precision 0.925, 22 of 50
     *       seeds returned an anchor below 0.95 (that predicate plus an irrelevant one, lifted over
     *       the threshold by sampling noise). With 0, none did, for about 2.5 times the model calls. */
    double epsilon_stop = 0.0;
    /** @brief Candidates kept at each anchor size. */
    int64_t beam_size = 2;
    /** @brief Samples drawn at a time for a candidate. */
    int64_t batch_size = 100;
    /** @brief The most samples drawn for one candidate. */
    int64_t max_samples_per_candidate = 10000;
    /** @brief The most predicates in an anchor, counting both sides of an interval; 0 for no limit. */
    int64_t max_anchor_size = 0;
    /** @brief Where the features are cut, in percent of the background. */
    std::vector<double> percentiles = {25.0, 50.0, 75.0};
    uint64_t seed = 0;
};

/** @brief `lower < x[feature] <= upper`; a missing side is infinite. */
struct AnchorPredicate {
    int64_t feature = 0;
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();

    [[nodiscard]] bool holds(float value) const { return lower < value && value <= upper; }
};

/** @brief FindAnchor()'s result. */
struct Anchor {
    /** @brief The class the model gives the input (its largest output), which the anchor keeps. */
    int64_t predicted_class = 0;
    /** @brief The rule, one interval per feature, in the order the search first constrained each.
     *         Empty: the model gives this class almost everywhere. */
    std::vector<AnchorPredicate> predicates;
    /** @brief The share of sampled inputs satisfying the rule that kept the class. */
    double precision = 0.0;
    /** @brief Its KL confidence bounds at 1 - delta. */
    double precision_lower = 0.0;
    double precision_upper = 1.0;
    /** @brief Samples behind the estimate. */
    int64_t samples = 0;
    /** @brief The share of the background the rule applies to. */
    double coverage = 0.0;
    /** @brief Whether it reached the threshold. When no anchor does, this is the most precise
     *         candidate found, and false. */
    bool meets_threshold = false;
    /** @brief Calls to the model over the whole search. */
    int64_t model_calls = 0;

    /** @brief Whether @p x (flattened) satisfies every predicate. */
    [[nodiscard]] bool applies_to(const std::vector<float>& x) const;
};

/**
 * @brief Finds an anchor for the model's prediction on @p input.
 * @param predict The model as a function of one input (a leading batch dimension of 1); its class
 *        is the index of its largest output.
 * @param input The input to explain; every element is a feature.
 * @param background Inputs shaped like @p input, from the data the model sees: the bins, the
 *        samples and the coverage all come from them.
 * @throws std::invalid_argument if background is empty, its inputs aren't the input's size,
 *         percentiles aren't increasing within (0, 100), or an option is out of range.
 */
[[nodiscard]] Anchor FindAnchor(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input,
                                const std::vector<Tensor>& background, const AnchorOptions& options = {});

/** @brief `IF income > 0.71 AND debt <= -0.12 THEN class 1 (precision 0.97, coverage 0.21)`.
 *  @param feature_names A name per feature; `x[3]` where missing. */
[[nodiscard]] std::string DescribeAnchor(const Anchor& anchor, const std::vector<std::string>& feature_names = {});

/** @brief The anchor as a `pulsatrix.anchor.v1` document; an infinite bound is null. */
[[nodiscard]] JsonValue ToJson(const Anchor& anchor, const std::vector<std::string>& feature_names = {});

/** @name The KL bounds KL-LUCB uses (Kaufmann and Kalyanakrishnan 2013), as in the reference code. */
///@{
/** @brief KL divergence between Bernoulli(p) and Bernoulli(q), with p and q clipped to (0, 1). */
[[nodiscard]] double BernoulliKl(double p, double q);
/** @brief The largest q >= p with BernoulliKl(p, q) <= level. */
[[nodiscard]] double KlUpperBound(double p, double level);
/** @brief The smallest q <= p with BernoulliKl(p, q) <= level. */
[[nodiscard]] double KlLowerBound(double p, double level);
///@}

}  // namespace pulsatrix
