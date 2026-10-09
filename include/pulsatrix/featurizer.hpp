/** @file featurizer.hpp
 *  @brief The featurizer family's common interface (FEAT-1): models that rewrite activations as
 *         sparse codes over learned directions and back, such as sparse autoencoders, with a
 *         shared training step and metrics (L0, dead and dense latents).
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief One batch's loss, split into its parts. */
struct FeaturizerLoss {
    /** @brief What was minimized: reconstruction + sparsity, except that a Matryoshka SAE adds
     *         every prefix's reconstruction error (TopKSparseAutoencoder). */
    float total = 0.0f;
    /** @brief Mean squared reconstruction error over every element. */
    float reconstruction = 0.0f;
    /** @brief The sparsity penalty's contribution (0 for featurizers that set sparsity directly). */
    float sparsity = 0.0f;
};

/**
 * @brief A featurizer maps activations `x`, `(N, input_dim)`, to codes `f = encode(x)`,
 *        `(N, num_features)`, and reconstructs `x ≈ decode(f) = f D + b`: feature i writes
 *        along its decoder direction `D[i]`.
 *
 * Sparse autoencoders (SparseAutoencoder) were the first. TopK, BatchTopK and Matryoshka
 * (TopKSparseAutoencoder) and JumpReLU (JumpReLUSparseAutoencoder) follow, and transcoders and
 * block-sparse featurizers (FEAT-5, FEAT-6) will too, so training, metrics and views work for
 * all of them.
 *
 * A **transcoder** (FEAT-5) predicts another activation instead of reconstructing its input: an
 * MLP's output from its input. It has an output_dim(), predict() adds what doesn't go through the
 * codes (a skip connection), and it trains with loss_and_backward_with_target().
 *
 * @note A featurizer is a discovery tool: a low reconstruction error says nothing about whether
 *       its directions are meaningful or causal. Compare features with probes and baselines.
 */
class Featurizer {
public:
    virtual ~Featurizer() = default;

    [[nodiscard]] virtual int64_t input_dim() const = 0;
    [[nodiscard]] virtual int64_t num_features() const = 0;
    /** @brief What decode() produces: input_dim() for an autoencoder. */
    [[nodiscard]] virtual int64_t output_dim() const { return input_dim(); }
    /**
     * @brief Codes come in blocks of this many (FEAT-6): a block-sparse featurizer's feature is a
     *        block, active when any of its codes is nonzero, and its codes may be negative. 1 (the
     *        default) for featurizers whose features are single directions.
     */
    [[nodiscard]] virtual int64_t block_size() const { return 1; }

    /** @brief The codes for a batch, `(N, num_features)`. */
    [[nodiscard]] virtual Tensor encode(const Tensor& x) = 0;
    /** @brief The reconstruction from codes, `(N, output_dim)`. */
    [[nodiscard]] virtual Tensor decode(const Tensor& codes) = 0;
    /** @brief The full output for a batch, `(N, output_dim)`: decode(encode(x)), plus any part
     *         that bypasses the codes (a transcoder's skip connection). */
    [[nodiscard]] virtual Tensor predict(const Tensor& x) { return decode(encode(x)); }

    /**
     * @brief Computes the training loss on @p x and adds its gradients to the parameters'
     *        (zero them first). When @p codes is given, it receives the batch's codes, row-major
     *        `(N, num_features)`, for metrics.
     * @throws std::invalid_argument for a batch that isn't `(N, input_dim)` with N > 0.
     */
    virtual FeaturizerLoss loss_and_backward(const Tensor& x, std::vector<float>* codes = nullptr) = 0;
    /**
     * @brief The loss for predicting @p target, `(N, output_dim)`, from @p x: a transcoder's.
     * @throws std::invalid_argument for an autoencoder, whose target is its input (the default).
     */
    virtual FeaturizerLoss loss_and_backward_with_target(const Tensor& x, const Tensor& target, std::vector<float>* codes = nullptr);

    /**
     * @brief Rescales every feature so its decoder direction has unit L2 norm, scaling its code
     *        by the inverse where the encoder allows it, so reconstructions don't change. Training
     *        with unit-norm decoders keeps a sparsity penalty from being dodged by shrinking codes
     *        and growing directions, and makes code sizes comparable across features.
     */
    virtual void normalize_decoder() = 0;

    /** @brief Feature @p i's decoder direction, `output_dim` values. */
    [[nodiscard]] virtual std::vector<float> decoder_direction(int64_t i) = 0;

    /** @brief The trainable parameters, for an optimizer's `step()` and `zero_grad()`, and for
     *         checkpoints. */
    [[nodiscard]] virtual Module& parameters_module() = 0;
};

/**
 * @brief How often each feature fires, and how long since it last did. A feature that hasn't
 *        fired in a long window is **dead**: it no longer learns, and wastes capacity. One that
 *        fires on most inputs is **dense**, usually a sign it encodes a bias, not a feature.
 */
class FeatureActivityTracker {
public:
    /** @throws std::invalid_argument if @p num_features < 1. */
    explicit FeatureActivityTracker(int64_t num_features);

    /** @brief Records a batch of codes, row-major `(N, num_features)`; a code above
     *         @p threshold in magnitude counts as firing. @throws std::invalid_argument for a size that isn't
     *         a whole number of rows. */
    void observe(const std::vector<float>& codes, float threshold = 0.0f);

    /** @brief Inputs observed so far. */
    [[nodiscard]] int64_t inputs_seen() const { return inputs_; }
    /** @brief The share of observed inputs each feature fired on. */
    [[nodiscard]] std::vector<double> firing_rates() const;
    /** @brief Features that haven't fired in the last @p window inputs (or ever, if fewer were
     *         seen), in index order. */
    [[nodiscard]] std::vector<int64_t> dead(int64_t window) const;
    /** @brief dead(window)'s share of all features. */
    [[nodiscard]] double dead_fraction(int64_t window) const;
    /** @brief Features firing on more than @p rate of the inputs observed. */
    [[nodiscard]] std::vector<int64_t> dense(double rate = 0.1) const;
    /** @brief Forgets everything observed. */
    void reset();

private:
    int64_t inputs_ = 0;
    std::vector<int64_t> fired_;       ///< inputs each feature fired on
    std::vector<int64_t> last_fired_;  ///< the input count when it last fired, or -1
};

/** @brief The mean number of codes above @p threshold in magnitude per input (L0), for codes row-major
 *         `(N, num_features)`. @throws std::invalid_argument for a size that isn't a whole
 *         number of rows, or no rows. */
[[nodiscard]] double MeanL0(const std::vector<float>& codes, int64_t num_features, float threshold = 0.0f);
/** @brief MeanL0 of a featurizer's codes for @p x, in blocks: a block counts once when any of its
 *         codes is above @p threshold in magnitude. */
[[nodiscard]] double MeanL0(Featurizer& featurizer, const Tensor& x, float threshold = 0.0f);

/**
 * @brief One training step for any featurizer: zero the gradients, loss_and_backward(), one
 *        optimizer step, then normalize_decoder() when @p unit_norm_decoder. When @p tracker is
 *        given it observes the batch's codes.
 * @tparam Optimizer SGDOptimizer, AdamOptimizer or AdamWOptimizer: anything with
 *         `zero_grad(Module&)` and `step(Module&)`.
 */
template <typename Optimizer>
FeaturizerLoss TrainFeaturizer(Featurizer& featurizer, const Tensor& x, Optimizer& optimizer, bool unit_norm_decoder = true,
                               FeatureActivityTracker* tracker = nullptr) {
    optimizer.zero_grad(featurizer.parameters_module());
    std::vector<float> codes;
    const FeaturizerLoss loss = featurizer.loss_and_backward(x, tracker != nullptr ? &codes : nullptr);
    optimizer.step(featurizer.parameters_module());
    if (unit_norm_decoder) featurizer.normalize_decoder();
    if (tracker != nullptr) tracker->observe(codes);
    return loss;
}

/** @brief TrainFeaturizer for a transcoder: one step on predicting @p target from @p x. */
template <typename Optimizer>
FeaturizerLoss TrainFeaturizer(Featurizer& featurizer, const Tensor& x, const Tensor& target, Optimizer& optimizer,
                               bool unit_norm_decoder = true, FeatureActivityTracker* tracker = nullptr) {
    optimizer.zero_grad(featurizer.parameters_module());
    std::vector<float> codes;
    const FeaturizerLoss loss = featurizer.loss_and_backward_with_target(x, target, tracker != nullptr ? &codes : nullptr);
    optimizer.step(featurizer.parameters_module());
    if (unit_norm_decoder) featurizer.normalize_decoder();
    if (tracker != nullptr) tracker->observe(codes);
    return loss;
}

}  // namespace pulsatrix
