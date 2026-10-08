/** @file featurizer_metrics.hpp
 *  @brief How good a featurizer is (FEAT-3), after SAEBench (Karvonen et al., arXiv 2503.09532):
 *         reconstruction (explained variance, cosine, norm ratio, L0, dead and dense latents),
 *         loss recovered when the reconstruction is spliced back into the model, and feature
 *         absorption (Chanin et al., arXiv 2409.14507) with a linear probe as its baseline.
 *  @ingroup mech_interp
 *  @note SAEBench's authors find that these proxies don't reliably predict how useful a
 *        featurizer is. Report them with a baseline: the same metrics for a featurizer trained on
 *        a randomly initialized model's activations (NullModelBaseline, null_model_baseline.hpp),
 *        and, for a concept, a linear probe on the raw activations, which FeatureAbsorption
 *        always fits.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "pulsatrix/causal_lm.hpp"  // HiddenStateHook
#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/mlp_hook.hpp"

namespace pulsatrix {

/** @brief EvaluateReconstruction()'s result. */
struct ReconstructionMetrics {
    int64_t inputs = 0;
    /** @brief Mean squared error over every element. */
    double mse = 0;
    /** @brief `1 - Σ|x - x̂|² / Σ|x - mean(x)|²`: the share of the inputs' variance, about
     *         their mean, that the reconstructions keep. */
    double explained_variance = 0;
    /** @brief The mean cosine similarity of each input and its reconstruction. */
    double cosine = 0;
    /** @brief The mean of `|x̂| / |x|`: below 1 means codes are shrunk. */
    double norm_ratio = 0;
    /** @brief Active features per input. */
    double l0 = 0;
    /** @brief The share of features that fired on none of these inputs. */
    double dead_fraction = 0;
    /** @brief The share that fired on more than the dense rate of them. */
    double dense_fraction = 0;
    /** @brief Each feature's share of inputs it fired on. */
    std::vector<double> firing_rates;
};

/**
 * @brief Encodes and reconstructs @p x, `(N, input_dim)`, in batches of @p batch rows.
 * @param dense_rate A feature firing on more than this share of inputs counts as dense.
 * @throws std::invalid_argument for a batch that isn't `(N, input_dim)` with N > 0, or batch < 1.
 */
[[nodiscard]] ReconstructionMetrics EvaluateReconstruction(Featurizer& featurizer, const Tensor& x, double dense_rate = 0.1,
                                                           int64_t batch = 4096);
/**
 * @brief EvaluateReconstruction for a transcoder: predict(x) against @p target,
 *        `(N, output_dim)`. Explained variance is relative to the targets' variance.
 * @throws std::invalid_argument for mismatched shapes.
 */
[[nodiscard]] ReconstructionMetrics EvaluatePrediction(Featurizer& featurizer, const Tensor& x, const Tensor& target,
                                                       double dense_rate = 0.1, int64_t batch = 4096);

/**
 * @brief Splices the featurizer in at one position: a hook that replaces the hidden states there,
 *        `(N, L, input_dim)`, by their reconstructions, and leaves other positions alone.
 * @param rows Which of the `N * L` rows to replace (all when empty), for example only residues,
 *        not special tokens or padding, if that is what the featurizer was trained on.
 * @note The hook throws std::invalid_argument if the hidden size isn't input_dim or @p rows has
 *       the wrong length.
 */
[[nodiscard]] HiddenStateHook SpliceHook(Featurizer& featurizer, int64_t position, std::vector<bool> rows = {});
/** @brief Zero-ablates one position: the same rows set to 0 instead of reconstructed. */
[[nodiscard]] HiddenStateHook AblationHook(int64_t position, std::vector<bool> rows = {});

/**
 * @brief Splices a transcoder in for an MLP (install it with EncoderBlock::set_mlp_hook() or
 *        TransformerBlock::set_mlp_hook()): the MLP's output rows are replaced by predict() of
 *        its input rows.
 * @param rows As SpliceHook's: which of the `N * L` rows to replace, all when empty.
 */
[[nodiscard]] MlpHook MlpSpliceHook(Featurizer& transcoder, std::vector<bool> rows = {});
/** @brief Zero-ablates an MLP: its output rows set to 0. */
[[nodiscard]] MlpHook MlpAblationHook(std::vector<bool> rows = {});

/** @brief MeasureLossRecovered()'s result: the model's loss three ways. */
struct LossRecovered {
    double clean = 0;    ///< unchanged
    double spliced = 0;  ///< with the reconstructions spliced in
    double ablated = 0;  ///< with the position zeroed
    /** @brief `(ablated - spliced) / (ablated - clean)`: 1 when splicing costs nothing, 0 when it
     *         is as bad as deleting the position. NaN unless ablation raises the loss: then the
     *         position doesn't matter to the model, and there is nothing to recover. A randomly
     *         initialized model often gives NaN. */
    double recovered = 0;
};

/**
 * @brief Loss recovered, SAEBench's main downstream measure.
 * @param loss Runs the model with the hook it is given installed (for example with
 *        CausalLM::set_hidden_state_hook()), removes it, and returns the loss: cross-entropy on
 *        held-out text, or a masked-LM loss. It is called with an empty hook for the clean loss.
 */
[[nodiscard]] LossRecovered MeasureLossRecovered(Featurizer& featurizer, int64_t position,
                                                 const std::function<double(const HiddenStateHook&)>& loss,
                                                 const std::vector<bool>& rows = {});

/**
 * @brief Loss recovered for a transcoder spliced in for an MLP: @p loss installs the MlpHook it is
 *        given on the MLP's block, runs the model, removes it and returns the loss. Ablation zeroes
 *        the MLP's output.
 */
[[nodiscard]] LossRecovered MeasureMlpLossRecovered(Featurizer& transcoder, const std::function<double(const MlpHook&)>& loss,
                                                    const std::vector<bool>& rows = {});

struct AbsorptionOptions {
    /** @brief The share of inputs used to fit the probe and pick the main features; the rest are
     *         scored. */
    double train_fraction = 0.5;
    uint64_t seed = 0;
    /** @brief Main features are added, best first, while each raises F1 by at least this much
     *         (SAEBench's k-sparse probing threshold). */
    double min_f1_gain = 0.03;
    int64_t max_main_features = 10;
    /** @brief An absorbing feature's decoder direction must have at least this cosine with the
     *         probe direction (SAEBench: 0.025). */
    double min_cosine = 0.025;
    /** @brief And absorbing features must carry at least this share of the input's projection
     *         onto the probe direction (SAEBench: 0.4). */
    double min_share = 0.4;
    int64_t probe_epochs = 30;
    float probe_learning_rate = 1e-2f;
    int64_t batch = 4096;
};

/** @brief FeatureAbsorption()'s result, on the held-out inputs. */
struct AbsorptionResult {
    int64_t test_inputs = 0;
    int64_t positives = 0;
    /** @brief The baseline: a logistic probe on the raw activations. */
    double probe_f1 = 0;
    /** @brief The features that stand for the concept, and their F1 when any one firing predicts it. */
    std::vector<int64_t> main_features;
    double main_f1 = 0;
    /** @brief Positives the probe finds. */
    int64_t probe_hits = 0;
    /** @brief Of those, the ones on which no main feature fires. */
    int64_t missed = 0;
    /** @brief Of those, the ones that other features, aligned with the probe, carry instead. */
    int64_t absorbed = 0;
    double miss_rate = 0;        ///< missed / probe_hits
    double absorption_rate = 0;  ///< absorbed / probe_hits
    /** @brief The absorbing features and on how many inputs each absorbed the concept, most first. */
    std::vector<std::pair<int64_t, int64_t>> absorbing_features;
};

/**
 * @brief Feature absorption (Chanin et al.; SAEBench's version, in outline). A concept is
 *        absorbed when the features that stand for it stay silent on an input that has it,
 *        because a more specific feature fires instead and carries its direction: a "starts
 *        with S" feature that skips "short", which a "short" feature covers.
 *
 * 1. A logistic probe on the raw activations finds the concept's direction (and is the baseline).
 * 2. Features are ranked by their mean code on positives minus negatives; the main features are
 *    added, best first, while each raises F1 by min_f1_gain.
 * 3. On held-out positives the probe finds but no main feature fires on, the input is absorbed
 *    when firing features whose decoder directions align with the probe carry min_share of the
 *    input's projection onto it, measured from the mean of the negatives.
 *
 * @param x `(N, input_dim)`. @param labels 0 or 1 per input.
 * @note For autoencoders: the probe lives in the input space and the decoder directions must too,
 *       so a featurizer whose output_dim differs from its input_dim is refused. For a transcoder
 *       the spaces differ even when the sizes match; compare its features with MatchConcept-style
 *       F1 instead.
 * @throws std::invalid_argument for a featurizer whose output_dim isn't its input_dim, mismatched sizes, labels other than 0 and 1, or a train or
 *         test split without both classes.
 */
[[nodiscard]] AbsorptionResult FeatureAbsorption(Featurizer& featurizer, const Tensor& x, const std::vector<int>& labels,
                                                 const AbsorptionOptions& options = {});

}  // namespace pulsatrix
