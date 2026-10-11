/** @file probe_controls.hpp
 *  @brief Linear probes with controls (INT-3): a multi-class probe trainer, Hewitt and Liang's
 *         control task and selectivity, and a comparison against a baseline representation.
 *  @ingroup mech_interp
 *
 *  A probe's accuracy alone doesn't say the representation encodes a concept: the probe may have
 *  learned the task itself, for example by memorizing a label for each word. Hewitt and Liang
 *  ("Designing and Interpreting Probes with Control Tasks", EMNLP 2019) give each token type a
 *  random but fixed label and train the same probe on that. A control task can only be learned by
 *  memorizing types, so selectivity, the task accuracy minus the control accuracy, is what the
 *  representation adds beyond that.
 *
 *  When there are few types the control task is no control: a linear probe learns 20 amino acids'
 *  labels from any representation that encodes identity. Compare against a baseline representation
 *  instead, such as one-hot local sequence (`SequenceWindowFeatures()`) or the same layer of a
 *  randomly initialized model.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

struct ProbeOptions {
    int64_t epochs = 20;
    int64_t batch = 1024;
    float learning_rate = 1e-2f;
    float weight_decay = 1e-4f;
    uint64_t seed = 0;
};

/** @brief A probe's quality on held-out rows. */
struct ProbeResult {
    double accuracy = 0;
    /** @brief The mean recall over classes, so a rare class counts as much as a common one. */
    double balanced_accuracy = 0;
    /** @brief For two classes, the ROC AUC of class 1's probability; NaN otherwise. */
    double auc = 0;
    int64_t train = 0;
    int64_t test = 0;
};

/**
 * @brief Trains a linear softmax probe (standardized features, AdamW, cross-entropy) on the rows
 *        marked in @p train and scores it on those marked in @p test.
 * @param features `rows x dim`, row-major. @param labels one class in [0, classes) per row.
 * @throws std::invalid_argument if the sizes don't agree, classes < 2, or either set is empty.
 */
[[nodiscard]] ProbeResult TrainLinearProbe(const std::vector<float>& features, int64_t dim, const std::vector<int>& labels, int classes,
                                           const std::vector<bool>& train, const std::vector<bool>& test, DeviceBackend* backend,
                                           const ProbeOptions& options = {});

/**
 * @brief Hewitt and Liang's control labels: each distinct value in @p token_types gets one label,
 *        drawn from the distribution of @p labels, and every row of that type carries it.
 * @param token_types One id per row saying which token type (a word, an amino acid) the row is.
 * @param labels The real task's labels, one per row in [0, classes). Only their distribution is
 *        used.
 * @param seed The same seed, types and labels give the same control labels. The types are visited
 *        in increasing order, so the order of the rows doesn't matter.
 * @throws std::invalid_argument if the sizes differ, there are no rows, classes < 2, or a label is
 *         outside [0, classes).
 */
[[nodiscard]] std::vector<int> ControlTaskLabels(const std::vector<int64_t>& token_types, const std::vector<int>& labels, int classes,
                                                 uint64_t seed);

/** @brief A probe on the real task beside the same probe on the control task. */
struct ProbeSelectivity {
    ProbeResult task;
    ProbeResult control;
    /** @brief `task.accuracy - control.accuracy`, Hewitt and Liang's selectivity. */
    double selectivity = 0;
    /** @brief The same with balanced accuracy, for imbalanced labels. */
    double balanced_selectivity = 0;
    /** @brief How many token types the training rows hold. Few types make the control easy to
     *         learn; with no more types than dimensions a linear probe can learn any control. */
    int64_t train_types = 0;
};

/**
 * @brief Trains one probe on @p labels and one on ControlTaskLabels(@p token_types, @p labels,
 *        @p classes, `options.seed`), with the same features, split and options, and reports both.
 * @throws std::invalid_argument for what TrainLinearProbe() and ControlTaskLabels() reject.
 */
[[nodiscard]] ProbeSelectivity MeasureSelectivity(const std::vector<float>& features, int64_t dim, const std::vector<int>& labels, int classes,
                                                  const std::vector<int64_t>& token_types, const std::vector<bool>& train,
                                                  const std::vector<bool>& test, DeviceBackend* backend, const ProbeOptions& options = {});

/** @brief A probe on a representation beside the same probe on a baseline representation. */
struct ProbeGain {
    ProbeResult probe;
    ProbeResult baseline;
    /** @brief `probe.accuracy - baseline.accuracy`. */
    double gain = 0;
    /** @brief The same with balanced accuracy. */
    double balanced_gain = 0;
    /** @brief The same with ROC AUC, for two classes; NaN otherwise. */
    double auc_gain = 0;
};

/**
 * @brief Trains the same probe on @p features and on @p baseline, rows matched, and reports how
 *        much better the representation does. A gain near zero means the probe reads nothing from
 *        the representation that the baseline (local sequence, a random model's layer, the raw
 *        input) doesn't already hold.
 * @throws std::invalid_argument if @p baseline doesn't have the same rows as @p features, or for
 *         what TrainLinearProbe() rejects.
 */
[[nodiscard]] ProbeGain CompareToBaseline(const std::vector<float>& features, int64_t dim, const std::vector<float>& baseline,
                                          int64_t baseline_dim, const std::vector<int>& labels, int classes, const std::vector<bool>& train,
                                          const std::vector<bool>& test, DeviceBackend* backend, const ProbeOptions& options = {});

}  // namespace pulsatrix
