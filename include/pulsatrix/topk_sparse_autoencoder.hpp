/** @file topk_sparse_autoencoder.hpp
 *  @brief The TopK sparse autoencoder (FEAT-2; Gao et al., "Scaling and evaluating sparse
 *         autoencoders", arXiv 2406.04093): sparsity set directly by keeping each input's k
 *         largest latents, with an auxiliary loss that revives dead latents. Two variants
 *         (FEAT-4): BatchTopK (Bussmann et al., arXiv 2412.06410) and Matryoshka (Bussmann et
 *         al., arXiv 2503.17547).
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {

struct TopKSaeOptions {
    /** @brief Latents kept per input: L0 is exactly this (or less, where fewer are positive). */
    int64_t k = 32;
    /** @brief The auxiliary loss's latents: the k_aux largest among the dead ones. 0 means half
     *         the input dimension, Gao et al.'s choice. */
    int64_t k_aux = 0;
    /** @brief The auxiliary loss's weight, 1/32 in Gao et al.; 0 turns it off. */
    float aux_coefficient = 1.0f / 32.0f;
    /** @brief A latent is dead once it hasn't fired in this many training inputs (Gao et al.
     *         use 10 million tokens). */
    int64_t dead_after = 10'000'000;
    uint64_t seed = 0;
    /**
     * @brief BatchTopK: in training, keep the batch's `N * k` largest activations wherever they
     *        fall, so L0 is k on average and inputs that need more latents get them. Inference
     *        uses a threshold instead, an exponential moving average of the smallest kept value
     *        per training batch (threshold()), so an input's codes don't depend on its batch.
     */
    bool batch_topk = false;
    /** @brief The threshold's moving-average decay, per training batch. */
    float threshold_decay = 0.99f;
    /**
     * @brief Matryoshka: nested dictionary sizes, increasing (for example {m/16, m/4, m}); the
     *        full size is added if missing. Training minimizes the sum of every prefix's
     *        reconstruction error, the prefix using only latents [0, size), so the first latents
     *        learn general features on their own and the later ones refine them. Empty: off.
     */
    std::vector<int64_t> matryoshka_prefixes;
};

/**
 * @brief A TopK sparse autoencoder:
 *        `z = ReLU(TopK_k(W_enc (x - b_dec) + b_enc))`, `x̂ = z W_dec + b_dec`.
 *
 * - **No L1 penalty.** k fixes how many latents fire, so codes aren't shrunk toward zero the way
 *   an L1 penalty shrinks them (SparseAutoencoder).
 * - **Dead latents.** A latent that stops firing gets no gradient and stays dead. The auxiliary
 *   loss has the k_aux largest dead latents reconstruct the residual `x - x̂`; its
 *   normalized squared error `|ê - e|² / |e|²`, weighted by aux_coefficient, gives them a
 *   gradient. The training loss is `mean((x̂ - x)²) + aux_coefficient * that`.
 * - **Decoder directions** have unit norm after every normalize_decoder(). Rows are rescaled as
 *   Gao et al. do, without moving the scale into the encoder (which would change which latents
 *   are selected), and the decoder's gradient loses its component along each direction.
 * - **Initialization** (Gao et al.): random unit decoder directions, the encoder their transpose,
 *   `b_enc = 0`, and `b_dec = 0` until initialize_bias() sets it to the data's mean.
 *
 * - **BatchTopK** (batch_topk) selects across the batch in training and by threshold() at
 *   inference. Before any training batch has set the threshold, encode() selects across the
 *   batch it is given.
 * - **Matryoshka** (matryoshka_prefixes) adds the prefixes' losses: FeaturizerLoss::total is
 *   their sum plus the auxiliary loss, and `reconstruction` is the full dictionary's.
 *
 * It is a Module from input to reconstruction, with parameters named `encoder.*` and
 * `decoder.*`; the decoder's bias is `b_dec`. BatchTopK's threshold is a buffer, `threshold`,
 * so checkpoints keep it.
 */
class TopKSparseAutoencoder : public Module, public Featurizer {
public:
    /** @throws std::invalid_argument if dim or num_features < 1, k outside [1, num_features],
     *          k_aux < 0, aux_coefficient < 0, dead_after < 1, threshold_decay outside [0, 1),
     *          or prefixes that aren't increasing within [1, num_features]. */
    TopKSparseAutoencoder(int64_t dim, int64_t num_features, DeviceBackend* backend, TopKSaeOptions options = {});

    /** @brief Sets `b_dec` to the mean of @p x, `(N, dim)`, as Gao et al. start it (they use the
     *         geometric median; the mean is close for activations). */
    void initialize_bias(const Tensor& x);

    // ---- Featurizer ----------------------------------------------------------------------
    [[nodiscard]] int64_t input_dim() const override { return dim_; }
    [[nodiscard]] int64_t num_features() const override { return m_; }
    [[nodiscard]] Tensor encode(const Tensor& x) override;
    [[nodiscard]] Tensor decode(const Tensor& codes) override;
    /** @brief The loss and its gradients; `sparsity` holds the weighted auxiliary loss. Counts the
     *         batch toward dead-latent tracking. */
    FeaturizerLoss loss_and_backward(const Tensor& x, std::vector<float>* codes = nullptr) override;
    /** @brief Unit decoder rows (Gao et al.: the encoder isn't rescaled). */
    void normalize_decoder() override;
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i) override;
    [[nodiscard]] Module& parameters_module() override { return *this; }

    /** @brief Latents dead now: none fired in the last dead_after training inputs. */
    [[nodiscard]] std::vector<int64_t> dead_latents() const;
    /** @brief Training inputs since each latent last fired. */
    [[nodiscard]] const std::vector<int64_t>& inputs_since_fired() const { return since_fired_; }
    /** @brief Restores inputs_since_fired(), for example when resuming training from a checkpoint.
     *  @throws std::invalid_argument unless it has num_features non-negative values. */
    void set_inputs_since_fired(const std::vector<int64_t>& since);
    [[nodiscard]] const TopKSaeOptions& options() const { return options_; }
    /** @brief BatchTopK's inference threshold; negative until a training batch sets it. */
    [[nodiscard]] float threshold() const;
    /** @brief Sets the inference threshold (negative: select across the batch). */
    void set_threshold(float threshold);
    [[nodiscard]] LinearModule& encoder() { return encoder_; }
    [[nodiscard]] LinearModule& decoder() { return decoder_; }

    // ---- Module: input to reconstruction -------------------------------------------------
    /** @brief Backward from the reconstruction's gradient, through the decoder, the selected
     *         latents, the encoder and the bias subtraction, adding to the parameters' gradients. */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief Relevance of the centered input `x - b_dec`: the decoder's and the encoder's own
     *         rules, passed through the selected latents. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override;
    void release_activations() override;

protected:
    /** @brief The reconstruction. @throws std::invalid_argument unless the input is (N, dim), N > 0. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    /** @brief The pass shared by everything: pre-activations and codes. A latent passes gradient
     *         where its code is positive. */
    struct Pass {
        std::vector<float> pre;    ///< (N, m) pre-activations
        std::vector<float> codes;  ///< (N, m)
        float min_kept = 0;        ///< BatchTopK in training: the smallest positive kept value
    };
    /** @param training BatchTopK selects across the batch (and reports min_kept) when true. */
    Pass Encode(const Tensor& x, bool training);
    void Check(const Tensor& x, const char* who) const;
    /** @brief `x - b_dec`, row by row. */
    Tensor Centered(const Tensor& x);
    /** @brief The decoder's weight gradient, less its component along each (unit) direction. */
    void ProjectDecoderGradient();

    int64_t dim_, m_;
    DeviceBackend* backend_;
    TopKSaeOptions options_;
    LinearModule encoder_;  ///< (dim -> m) with b_enc
    LinearModule decoder_;  ///< (m -> dim) with b_dec
    std::vector<int64_t> since_fired_;
    Tensor threshold_;  ///< (1,): BatchTopK's inference threshold, negative until set
    std::vector<float> last_codes_;
    int64_t last_rows_ = 0;
};

}  // namespace pulsatrix
