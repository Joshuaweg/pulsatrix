/** @file topk_sparse_autoencoder.hpp
 *  @brief The TopK sparse autoencoder (FEAT-2; Gao et al., "Scaling and evaluating sparse
 *         autoencoders", arXiv 2406.04093): sparsity set directly by keeping each input's k
 *         largest latents, with an auxiliary loss that revives dead latents.
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
 * It is a Module from input to reconstruction, with parameters named `encoder.*` and
 * `decoder.*`; the decoder's bias is `b_dec`.
 */
class TopKSparseAutoencoder : public Module, public Featurizer {
public:
    /** @throws std::invalid_argument if dim or num_features < 1, k outside [1, num_features],
     *          k_aux < 0, aux_coefficient < 0 or dead_after < 1. */
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
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override;
    void release_activations() override;

protected:
    /** @brief The reconstruction. @throws std::invalid_argument unless the input is (N, dim), N > 0. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    /** @brief The pass shared by everything: centered input, pre-activations, codes, and which
     *         latents each row kept. */
    struct Pass {
        std::vector<float> pre;       ///< (N, m) pre-activations
        std::vector<float> codes;     ///< (N, m)
        std::vector<int64_t> kept;    ///< (N, k) the selected latents
    };
    Pass Encode(const Tensor& x, Tensor* centered = nullptr);
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
    std::vector<float> last_codes_;
    std::vector<int64_t> last_kept_;
    int64_t last_rows_ = 0;
};

}  // namespace pulsatrix
