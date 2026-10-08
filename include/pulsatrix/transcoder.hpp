/** @file transcoder.hpp
 *  @brief Transcoders and skip transcoders (FEAT-5): sparse, wide replacements for an MLP that
 *         predict its output from its input (Dunefsky et al., "Transcoders Find Interpretable LLM
 *         Feature Circuits", arXiv 2406.11944), optionally beside a linear skip connection (Paulo
 *         et al., "Transcoders Beat Sparse Autoencoders for Interpretability", arXiv 2501.18823).
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {

struct TranscoderOptions {
    /** @brief Latents kept per input (a TopK activation, as Paulo et al. use). */
    int64_t k = 32;
    /** @brief A skip transcoder: add `x W_skip`, a linear map from input to output that starts at
     *         zero, so the latents only need to explain what isn't linear. */
    bool skip = false;
    /** @brief AuxK, as TopKSparseAutoencoder: the k_aux largest dead latents predict the error
     *         (0: half the output dimension), weighted by aux_coefficient (0 turns it off). */
    int64_t k_aux = 0;
    float aux_coefficient = 1.0f / 32.0f;
    int64_t dead_after = 10'000'000;
    uint64_t seed = 0;
};

/**
 * @brief A transcoder: `z = ReLU(TopK_k(x W_enc + b_enc))`, `ŷ = z W_dec + b_dec [+ x W_skip]`,
 *        trained to predict an MLP's output y from its input x by `mean((ŷ - y)²)` plus AuxK.
 *
 * - **Why.** An SAE explains the residual stream; a transcoder explains a computation. Its
 *   latents read from the MLP's input and write to its output, so a circuit through the MLP can
 *   be traced latent to latent, and Paulo et al. find its latents more interpretable than an
 *   SAE's at the same sparsity.
 * - **The skip connection** takes the MLP's linear part off the latents. decode() leaves it out
 *   (it needs the input); predict() includes it.
 * - **Decoder directions** are unit vectors after normalize_decoder(), and the decoder's gradient
 *   loses its component along each, as in TopKSparseAutoencoder.
 * - **Initialization:** random unit decoder directions, the encoder their transpose (when the
 *   input and output sizes match; random otherwise), `b_enc = 0`, `W_skip = 0`, and `b_dec` 0
 *   until initialize_bias() sets it to the targets' mean.
 *
 * A Module from input to prediction, with parameters `encoder.*`, `decoder.*` and `skip.weight`.
 * Train it with loss_and_backward_with_target() or TrainFeaturizer(t, x, target, optimizer);
 * loss_and_backward() without a target throws.
 */
class Transcoder : public Module, public Featurizer {
public:
    /** @throws std::invalid_argument if a size < 1, k outside [1, num_features], k_aux < 0,
     *          aux_coefficient < 0 or dead_after < 1. */
    Transcoder(int64_t input_dim, int64_t output_dim, int64_t num_features, DeviceBackend* backend, TranscoderOptions options = {});

    /** @brief Sets `b_dec` to the mean of @p target, `(N, output_dim)`. */
    void initialize_bias(const Tensor& target);

    // ---- Featurizer ----------------------------------------------------------------------
    [[nodiscard]] int64_t input_dim() const override { return in_; }
    [[nodiscard]] int64_t output_dim() const override { return out_; }
    [[nodiscard]] int64_t num_features() const override { return m_; }
    [[nodiscard]] Tensor encode(const Tensor& x) override;
    /** @brief `codes W_dec + b_dec`: the prediction without the skip connection. */
    [[nodiscard]] Tensor decode(const Tensor& codes) override;
    /** @brief The full prediction, skip connection included. */
    [[nodiscard]] Tensor predict(const Tensor& x) override;
    /** @throws std::invalid_argument always: a transcoder needs a target. */
    FeaturizerLoss loss_and_backward(const Tensor& x, std::vector<float>* codes = nullptr) override;
    /** @brief The loss and its gradients; `reconstruction` is the prediction's mean squared error,
     *         `sparsity` the weighted AuxK loss. Counts the batch toward dead-latent tracking. */
    FeaturizerLoss loss_and_backward_with_target(const Tensor& x, const Tensor& target, std::vector<float>* codes = nullptr) override;
    void normalize_decoder() override;
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i) override;
    [[nodiscard]] Module& parameters_module() override { return *this; }

    [[nodiscard]] std::vector<int64_t> dead_latents() const;
    [[nodiscard]] const std::vector<int64_t>& inputs_since_fired() const { return since_fired_; }
    /** @throws std::invalid_argument unless it has num_features non-negative values. */
    void set_inputs_since_fired(const std::vector<int64_t>& since);
    [[nodiscard]] const TranscoderOptions& options() const { return options_; }
    [[nodiscard]] LinearModule& encoder() { return encoder_; }
    [[nodiscard]] LinearModule& decoder() { return decoder_; }
    /** @brief The skip connection; null unless options.skip. */
    [[nodiscard]] LinearModule* skip() { return skip_.get(); }

    // ---- Module: input to prediction -----------------------------------------------------
    /** @brief Backward through the latents (and the skip connection), adding to the parameters'
     *         gradients. */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief Relevance of the input: the output's relevance is split between the latents' part
     *         and the skip's in proportion to their contributions (epsilon rule), then each
     *         passes it back with its own layers' rules. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override;
    void release_activations() override;

protected:
    /** @throws std::invalid_argument unless the input is (N, input_dim), N > 0. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    void CheckInput(const Tensor& x, const char* who) const;
    [[nodiscard]] std::vector<float> Codes(const Tensor& pre) const;
    /** @brief Back through the active latents into the encoder, from the codes' gradient. */
    Tensor BackwardFromCodes(const std::vector<float>& codes, const std::vector<float>& g_codes);

    int64_t in_, out_, m_;
    DeviceBackend* backend_;
    TranscoderOptions options_;
    LinearModule encoder_;  ///< (in -> m) with b_enc
    LinearModule decoder_;  ///< (m -> out) with b_dec
    std::unique_ptr<LinearModule> skip_;  ///< (in -> out), no bias
    std::vector<int64_t> since_fired_;
    std::vector<float> last_codes_;
    std::vector<float> last_skip_out_;  ///< for relevance: the skip connection's output
    int64_t last_rows_ = 0;
};

}  // namespace pulsatrix
