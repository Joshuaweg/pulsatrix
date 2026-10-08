/** @file jumprelu_sparse_autoencoder.hpp
 *  @brief The JumpReLU sparse autoencoder (FEAT-4; Rajamanoharan et al., "Jumping Ahead:
 *         Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders",
 *         arXiv 2407.14435): a learned threshold per latent, trained against an L0 penalty
 *         through straight-through estimators.
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {

struct JumpReLUSaeOptions {
    /** @brief λ, the L0 penalty's weight: the loss is `mean((x̂ - x)²) + λ * L0`, L0 averaged over
     *         the batch. The paper sums the squared error over dimensions; with the mean, λ is
     *         smaller by the input dimension. */
    float l0_coefficient = 1e-3f;
    /** @brief ε, the bandwidth of the rectangle kernel that stands in for the step's derivative. */
    float bandwidth = 1e-3f;
    /** @brief Every latent's starting threshold θ (the paper's 0.001). */
    float initial_threshold = 1e-3f;
    uint64_t seed = 0;
};

/**
 * @brief A JumpReLU sparse autoencoder: `z = pre · H(pre - θ)` with `pre = x W_enc + b_enc`, and
 *        `x̂ = z W_dec + b_dec`. Each latent has its own threshold θ > 0, kept as `log θ`.
 *
 * - **Training.** The step H has no useful derivative, so the paper's straight-through
 *   estimators stand in: with K the rectangle kernel (1 on (-1/2, 1/2)),
 *   `∂z/∂pre = H(pre - θ)`, `∂z/∂θ = -(θ/ε) K((pre - θ)/ε)` and `∂H/∂θ = -(1/ε) K((pre - θ)/ε)`.
 *   They give θ a gradient from both the reconstruction and the L0 penalty, from inputs within
 *   ε/2 of it.
 * - **Sparsity** comes from the L0 penalty, not L1, so codes above the threshold aren't shrunk.
 * - **Decoder directions** are made unit by normalize_decoder(), which moves each scale into the
 *   latent's encoder column, bias and threshold, so codes and reconstructions don't change.
 * - **Initialization:** random unit decoder directions, the encoder their transpose, `b_enc = 0`,
 *   `b_dec = 0` (initialize_bias() sets it to the data's mean) and θ = initial_threshold.
 *
 * A Module from input to reconstruction, with parameters `encoder.*`, `decoder.*` and
 * `log_threshold`.
 */
class JumpReLUSparseAutoencoder : public Module, public Featurizer {
public:
    /** @throws std::invalid_argument if dim or num_features < 1, or l0_coefficient < 0,
     *          bandwidth <= 0 or initial_threshold <= 0. */
    JumpReLUSparseAutoencoder(int64_t dim, int64_t num_features, DeviceBackend* backend, JumpReLUSaeOptions options = {});

    /** @brief Sets `b_dec` to the mean of @p x, `(N, dim)`. */
    void initialize_bias(const Tensor& x);
    /** @brief Changes λ, for example to warm it up over the first steps as the paper does.
     *  @throws std::invalid_argument if negative. */
    void set_l0_coefficient(float lambda);
    /** @brief Each latent's threshold θ. */
    [[nodiscard]] std::vector<float> thresholds() const;
    [[nodiscard]] const JumpReLUSaeOptions& options() const { return options_; }
    [[nodiscard]] LinearModule& encoder() { return encoder_; }
    [[nodiscard]] LinearModule& decoder() { return decoder_; }

    // ---- Featurizer ----------------------------------------------------------------------
    [[nodiscard]] int64_t input_dim() const override { return dim_; }
    [[nodiscard]] int64_t num_features() const override { return m_; }
    [[nodiscard]] Tensor encode(const Tensor& x) override;
    [[nodiscard]] Tensor decode(const Tensor& codes) override;
    /** @brief The loss and its gradients; `sparsity` holds `λ * L0`. */
    FeaturizerLoss loss_and_backward(const Tensor& x, std::vector<float>* codes = nullptr) override;
    void normalize_decoder() override;
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i) override;
    [[nodiscard]] Module& parameters_module() override { return *this; }

    // ---- Module: input to reconstruction -------------------------------------------------
    /** @brief Backward through the decoder, the active latents and the encoder (the thresholds
     *         get the straight-through gradient too). */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief The decoder's and the encoder's rules, passed through the active latents. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override;
    void release_activations() override;

protected:
    /** @throws std::invalid_argument unless the input is (N, dim), N > 0. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    void Check(const Tensor& x, const char* who) const;
    /** @brief Codes for @p pre, `(N, m)` pre-activations. */
    [[nodiscard]] std::vector<float> Codes(const std::vector<float>& pre) const;
    /** @brief Back through the active latents to the encoder, adding the thresholds' gradient from
     *         @p g_codes (the reconstruction's) and from the L0 penalty when @p l0_weight > 0. */
    [[nodiscard]] Tensor BackwardFromCodes(const std::vector<float>& pre, const std::vector<float>& g_codes, double l0_weight);

    int64_t dim_, m_;
    DeviceBackend* backend_;
    JumpReLUSaeOptions options_;
    LinearModule encoder_;  ///< (dim -> m) with b_enc
    LinearModule decoder_;  ///< (m -> dim) with b_dec
    Tensor log_threshold_;       ///< (m,)
    Tensor log_threshold_grad_;  ///< (m,)
    std::vector<float> last_pre_;
    int64_t last_rows_ = 0;
};

}  // namespace pulsatrix
