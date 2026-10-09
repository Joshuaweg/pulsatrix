/** @file crosscoder.hpp
 *  @brief Crosscoders (FEAT-8): one sparse dictionary read from and written to several sources at
 *         once, such as the residual stream at several layers, or the same layer in a base model
 *         and its fine-tune (Lindsey et al., "Sparse Crosscoders for Cross-Layer Features and
 *         Model Diffing", Transformer Circuits, 2024). Model diffing reads which latents' decoders
 *         live in one model only, with Minder et al.'s BatchTopK crosscoder and Latent Scaling
 *         ("Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning", arXiv
 *         2504.02922), and Kassem et al.'s Delta-Crosscoder for narrow fine-tunes ("Delta-
 *         Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes", arXiv
 *         2603.04426).
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {

enum class CrosscoderSparsity {
    /** @brief Lindsey et al.: `λ · Σ_i f_i Σ_s |d_i^s|`, an L1 penalty on each code weighted by
     *         the sum (not the L2 norm) of its decoder norms across sources, so a latent pays for
     *         every source it writes to and a source-exclusive latent stays cheap. */
    L1,
    /** @brief Minder et al.: the batch's `N · k` largest values of `f_i · Σ_s |d_i^s|` are kept
     *         (the unscaled f_i is the code), with AuxK for dead latents. Inference uses a
     *         threshold on the scaled value (threshold()). */
    BatchTopK,
};

struct CrosscoderOptions {
    CrosscoderSparsity sparsity = CrosscoderSparsity::BatchTopK;
    /** @brief L1: λ. */
    float l1_coefficient = 1e-3f;
    /** @brief BatchTopK: mean latents kept per input (the delta partition's, with delta). */
    int64_t k = 32;
    /** @brief AuxK, as TopKSparseAutoencoder: the k_aux largest dead latents reconstruct the
     *         error (0: half the input size), weighted by aux_coefficient (0 turns it off). */
    int64_t k_aux = 0;
    float aux_coefficient = 1.0f / 32.0f;
    int64_t dead_after = 10'000'000;
    /** @brief BatchTopK's inference threshold: a moving average of the smallest kept scaled value
     *         per training batch, with this decay. */
    float threshold_decay = 0.99f;
    /** @brief Every latent's decoder starts at this norm in every source (0: 0.05 for L1, 1 for
     *         BatchTopK, Minder et al.'s values; they report the small L1 value is crucial). */
    float decoder_init_norm = 0.0f;
    /**
     * @brief The Delta-Crosscoder (Kassem et al.), for two sources: source 0 the base model and
     *        source 1 the fine-tune, BatchTopK only.
     * - The encoder averages the sources: `u = ½ Σ_s x^s W_enc^s + b_enc`.
     * - The first shared_fraction of the latents are **shared**, kept by their own BatchTopK at
     *   k_shared per input; the rest are **delta** latents, at k.
     * - A delta loss has the delta latents alone predict the fine-tune's change:
     *   `delta_coefficient · mean((z_Δ (W_dec^1 - W_dec^0) - (x^1 - x^0))²)`.
     */
    bool delta = false;
    double shared_fraction = 0.2;
    /** @brief The shared partition's latents per input (0: twice k, the paper's multiplier). */
    int64_t k_shared = 0;
    float delta_coefficient = 0.005f;
    uint64_t seed = 0;
};

/**
 * @brief A crosscoder over S sources of the same size d. Its input is the sources side by side,
 *        `(N, S·d)`, source s in columns `[s·d, (s+1)·d)`:
 *        `f = ReLU(Σ_s x^s W_enc^s + b_enc)`, `x̂^s = f W_dec^s + b_dec^s`.
 *
 * - **One code, a decoder per source.** A latent's decoder norm in each source says how much it
 *   writes there: about equal for a feature both models share, near zero in one source for a
 *   feature only the other has (CrosscoderLatents(), ClassifyLatent()).
 * - **Loss:** `mean((x̂ - x)²)` over every source's values, plus the L1 penalty or AuxK, plus the
 *   delta loss with delta. Lindsey et al. sum the squared errors instead; the mean keeps the loss
 *   on the scale of pulsatrix's other featurizers, and with one source and BatchTopK the
 *   crosscoder is a BatchTopK SAE without the encoder's bias subtraction.
 * - **Initialization** (Minder et al.): random decoder directions, the same in every source, at
 *   decoder_init_norm; the encoder their transpose; zero biases until initialize_bias().
 * - **Decoder norms carry the result,** so training doesn't normalize them: train with
 *   TrainFeaturizer with `unit_norm_decoder = false`, as the references do.
 *   normalize_decoder() is still exact: it scales each latent's whole decoder to unit norm and
 *   its encoder by the inverse, which changes no reconstruction, no penalty and no selection.
 * - **Activations:** normalize each source separately first, so each contributes comparably
 *   (Lindsey et al.); for example scale it so its mean squared norm is d.
 *
 * A Module from input to reconstruction, with parameters `encoder.*` and `decoder.*`;
 * BatchTopK's thresholds are a buffer, `threshold` (one value, two with delta: shared, delta).
 */
class Crosscoder : public Module, public Featurizer {
public:
    /** @throws std::invalid_argument if a size < 1, k outside [1, num_features], a negative
     *          coefficient, dead_after < 1, threshold_decay outside [0, 1), or delta without two
     *          sources and BatchTopK, or with a partition empty. */
    Crosscoder(int64_t num_sources, int64_t source_dim, int64_t num_features, DeviceBackend* backend, CrosscoderOptions options = {});

    /** @brief Sets `b_dec` to the mean of @p x, `(N, S·d)`. */
    void initialize_bias(const Tensor& x);

    [[nodiscard]] int64_t num_sources() const { return S_; }
    [[nodiscard]] int64_t source_dim() const { return d_; }
    /** @brief With delta: latents [0, num_shared()) are shared, the rest delta latents. */
    [[nodiscard]] int64_t num_shared() const { return shared_; }

    // ---- Featurizer ----------------------------------------------------------------------
    [[nodiscard]] int64_t input_dim() const override { return S_ * d_; }
    [[nodiscard]] int64_t num_features() const override { return m_; }
    /** @brief The codes: BatchTopK by threshold once one is set, else across this batch. */
    [[nodiscard]] Tensor encode(const Tensor& x) override;
    [[nodiscard]] Tensor decode(const Tensor& codes) override;
    /** @brief The loss and its gradients: `reconstruction` is the mean squared error over every
     *         source, `sparsity` the weighted L1 penalty or AuxK loss, `total` adds the delta loss
     *         (delta_loss()). Counts the batch toward dead latents and moves the thresholds. */
    FeaturizerLoss loss_and_backward(const Tensor& x, std::vector<float>* codes = nullptr) override;
    void normalize_decoder() override;
    /** @brief Latent @p i's decoder in every source, `S·d` values. */
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i) override;
    [[nodiscard]] Module& parameters_module() override { return *this; }

    /** @brief Latent @p i's decoder in source @p s, d values. */
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i, int64_t s);
    /** @brief The last loss_and_backward()'s weighted delta loss (0 without delta). */
    [[nodiscard]] float delta_loss() const { return last_delta_loss_; }

    [[nodiscard]] std::vector<int64_t> dead_latents() const;
    [[nodiscard]] const std::vector<int64_t>& inputs_since_fired() const { return since_fired_; }
    /** @throws std::invalid_argument unless it has num_features non-negative values. */
    void set_inputs_since_fired(const std::vector<int64_t>& since);
    /** @brief BatchTopK's inference thresholds on `f_i · Σ_s |d_i^s|` (shared, then delta, with
     *         delta); negative until a training batch sets them. */
    [[nodiscard]] std::vector<float> thresholds() const;
    /** @throws std::invalid_argument for the wrong count. */
    void set_thresholds(const std::vector<float>& thresholds);
    [[nodiscard]] const CrosscoderOptions& options() const { return options_; }
    [[nodiscard]] LinearModule& encoder() { return encoder_; }
    [[nodiscard]] LinearModule& decoder() { return decoder_; }

    // ---- Module: input to reconstruction -------------------------------------------------
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief Relevance of every source's input: the decoder's and the encoder's rules, through
     *         the selected latents (a fixed gate), as the other featurizers do. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override;
    void release_activations() override;

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    struct Pass {
        std::vector<float> pre;    ///< (N, m)
        std::vector<float> codes;  ///< (N, m)
        std::vector<float> min_kept;  ///< per partition, BatchTopK in training; -1 if none kept
    };
    Pass Encode(const Tensor& x, bool training);
    void Check(const Tensor& x, const char* who) const;
    /** @brief The encoder's input: x, halved with delta. */
    [[nodiscard]] Tensor EncoderInput(const Tensor& x) const;
    /** @brief Σ_s |d_i^s| per latent. */
    [[nodiscard]] std::vector<double> NormSums() const;

    int64_t S_, d_, m_, shared_ = 0;
    DeviceBackend* backend_;
    CrosscoderOptions options_;
    LinearModule encoder_;  ///< (S·d -> m) with b_enc
    LinearModule decoder_;  ///< (m -> S·d) with b_dec
    std::vector<int64_t> since_fired_;
    Tensor threshold_;  ///< (1,) or (2,)
    float last_delta_loss_ = 0;
    std::vector<float> last_codes_;
    int64_t last_rows_ = 0;
};

/** @brief One latent across two sources a and b. */
struct CrosscoderLatentStats {
    double norm_a = 0, norm_b = 0;
    /** @brief `|d^b| / (|d^a| + |d^b|)` (Lindsey et al., Kassem et al.): 0 a only, ½ even, 1 b only. */
    double relative_norm = 0;
    /** @brief Minder et al.'s Δnorm, `½ (1 + (|d^b| - |d^a|) / max(|d^a|, |d^b|))`: the same
     *         ends and middle, spread differently between them. */
    double delta_norm = 0;
    /** @brief The cosine between the two decoders (0 if either is zero). */
    double cosine = 0;
};

/** @brief Every latent's stats for sources @p a and @p b. @throws std::invalid_argument for a
 *         source out of range or a == b. */
[[nodiscard]] std::vector<CrosscoderLatentStats> CrosscoderLatents(Crosscoder& crosscoder, int64_t a = 0, int64_t b = 1);

enum class LatentClass {
    AOnly,   ///< Δnorm below 0.1
    BOnly,   ///< Δnorm above 0.9
    Shared,  ///< Δnorm in [0.4, 0.6]
    Other,
};
/** @brief Minder et al.'s bins on Δnorm. */
[[nodiscard]] LatentClass ClassifyLatent(const CrosscoderLatentStats& stats);

/**
 * @brief Latent Scaling (Minder et al., Eq. 4): how much of latent j's b-side decoder direction
 *        each source actually needs. For a target Y^m over the inputs where it's computed,
 *        `β^m = ⟨Y^m d, f⟩ / (|f|² |d|²)` with `d = d_j^b`, the best scale of `f d` against Y^m.
 * - **Error** targets are what the rest of the crosscoder leaves: `x^m - (x̂^m - f_j d_j^m)`.
 *   A b-only latent whose ν_error is high is **complete shrinkage**: source a needs it too, but
 *   the sparsity penalty zeroed its a decoder.
 * - **Reconstruction** targets are `x̂^m`. A high ν_reconstruction is **latent decoupling**: the
 *   same thing is also written to source a by other latents.
 * - **ν = β^a / β^b.** Minder et al. call a latent b-specific when ν_reconstruction < 0.5 and
 *   ν_error < 0.2; a negative β^b leaves ν meaningless, and they drop such latents.
 */
struct LatentScaling {
    int64_t latent = 0;
    double beta_error_a = 0, beta_error_b = 0, beta_reconstruction_a = 0, beta_reconstruction_b = 0;
    double nu_error = 0, nu_reconstruction = 0;
    /** @brief Inputs where the latent fired. */
    int64_t active = 0;
};
/** @brief Latent Scaling of each of @p latents on @p x, `(N, S·d)`, in batches of @p batch.
 *         β is 0 (and ν 0) for a latent that never fires. @throws std::invalid_argument for a
 *         latent or source out of range, or a == b. */
[[nodiscard]] std::vector<LatentScaling> MeasureLatentScaling(Crosscoder& crosscoder, const Tensor& x, const std::vector<int64_t>& latents,
                                                              int64_t a = 0, int64_t b = 1, int64_t batch = 4096);

/** @brief Each source's explained variance on @p x: `1 - Σ|x^s - x̂^s|² / Σ|x^s - mean(x^s)|²`. */
[[nodiscard]] std::vector<double> ExplainedVarianceBySource(Crosscoder& crosscoder, const Tensor& x, int64_t batch = 4096);

}  // namespace pulsatrix
