/** @file loreft.hpp
 *  @brief LoReFT (Wu et al., 2024): fine-tune a frozen language model by editing its hidden states
 *         in a learned low-rank subspace, at a few positions after chosen layers, and explain the edit
 *         with LRP.
 *  @ingroup dl_modules
 *
 * At each chosen position, the residual stream h after a block becomes
 *
 *     h' = h + R^T (W h + b - R h)
 *
 * with R `(r, d)` orthonormal rows. Inside the r-dimensional subspace R spans, the edit replaces
 * the hidden state's coordinates R h by the learned W h + b; outside it, h is untouched. The model's
 * own weights stay frozen. The edit is affine in h, so the epsilon rule propagates relevance through
 * it exactly, and LoReFTIntervention splits the relevance into the part the edit carries and the part
 * that passes through untouched.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"

namespace pulsatrix {

/** @brief Where and how large LoReFT's interventions are. */
struct LoReFTConfig {
    /** @brief r, the dimension of the edited subspace. */
    int64_t rank = 4;
    /** @brief The blocks whose outputs are edited. Empty means every block. */
    std::vector<int64_t> layers;
    /** @brief Edit the first `first` and the last `last` positions of each prompt: pyreft's
     *         "f<first>+l<last>". During generation only the prompt's positions are edited. */
    int64_t first = 1;
    int64_t last = 1;
    /** @brief Seeds R, W and b, the same on every platform. */
    uint64_t seed = 0;
};

/**
 * @brief One block's LoReFT edit. Parameters, named as pyreft's state dict:
 *        - `rotate_layer`: R^T, `(d, r)` with orthonormal columns (pyreft's layout);
 *        - `learned_source.weight`: W^T, `(d, r)`, the transpose of pyreft's `(r, d)`, like every
 *          pulsatrix Linear weight;
 *        - `learned_source.bias`: b, `(r)`.
 * @note Gradient steps move R off the orthonormal set; call orthonormalize() after each optimizer
 *       step (OrthonormalizeLoReFT() does it for a model). pyreft keeps R orthonormal with torch's
 *       orthogonal parametrization instead, so training paths differ, but the edit, its gradient
 *       with respect to R and its relevance are the same for the same R.
 */
class LoReFTIntervention : public ResidualIntervention {
public:
    /**
     * @brief R orthonormal (QR of a Gaussian matrix), and W and b as an `nn.Linear(d, r)` starts,
     *        uniform on ±1/sqrt(d), as pyreft initializes them. Unlike LoRA, the edit isn't zero at
     *        the start.
     * @throws std::invalid_argument unless 1 <= rank <= d and first, last >= 0.
     */
    LoReFTIntervention(int64_t d_model, int64_t rank, int64_t first, int64_t last, DeviceBackend* backend, uint64_t seed);

    [[nodiscard]] Tensor forward_at(const Tensor& hidden, int64_t start, int64_t prompt_length) override;
    /** @brief The gradient with respect to the hidden states, accumulating R's, W's and b's. */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /**
     * @brief Identity at the positions it doesn't edit. At the ones it does, the edit is the affine
     *        map h (I + (W^T - R^T) R) + b R, and the rule runs on its two paths together, the
     *        identity and the edit, as for a split LoRA layer (TRN-8). The edit's part is kept in
     *        last_edit_relevance(). Every rule is supported; under epsilon the result is the
     *        relevance of the map as a whole.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] bool supports_lrp_rule(LRPRule) const override { return true; }
    [[nodiscard]] OpType op_type() const override { return OpType::Linear; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void release_activations() override;

    [[nodiscard]] int64_t rank() const { return rank_; }
    [[nodiscard]] int64_t first() const { return first_; }
    [[nodiscard]] int64_t last() const { return last_; }
    [[nodiscard]] const Tensor& rotation() const { return r_; }
    [[nodiscard]] const Tensor& source_weight() const { return w_; }
    [[nodiscard]] const Tensor& source_bias() const { return b_; }
    /** @brief Replaces R^T `(d, r)`, W^T `(d, r)` and b `(r)`, row-major. @throws std::invalid_argument on a wrong size. */
    void set_parameters(const std::vector<float>& rotation, const std::vector<float>& source_weight, const std::vector<float>& source_bias);
    /** @brief Makes R's rows orthonormal again (modified Gram-Schmidt in double, signs kept). */
    void orthonormalize();
    /** @brief The largest |R R^T - I| entry: 0 for an orthonormal R. */
    [[nodiscard]] double orthonormality_error() const;

    /** @brief The positions the last forward edited, as row indices into `(N * L)`. */
    [[nodiscard]] const std::vector<int64_t>& last_positions() const { return last_rows_; }
    /** @brief The last relevance pass's part carried by the edit, `(N, L, d)`, zero at unedited
     *         positions. @throws std::logic_error before a relevance pass, or after a new forward(). */
    [[nodiscard]] const Tensor& last_edit_relevance() const;

private:
    int64_t d_;
    int64_t rank_;
    int64_t first_;
    int64_t last_;
    DeviceBackend* backend_;
    Tensor r_, w_, b_, r_grad_, w_grad_, b_grad_;
    std::vector<int64_t> last_rows_;
    Tensor last_indices_;   // last_rows_ as floats, for gather_rows/scatter_add_rows
    Tensor last_selected_;  // h at the edited rows, (S, d)
    Tensor last_coords_;    // u = h (W^T - R^T) + b, (S, r)
    Shape last_shape_ = Shape({0});
    Tensor last_edit_relevance_;
};

/**
 * @brief Installs a LoReFTIntervention after each of config.layers' blocks and freezes the rest of
 *        the model: only the interventions train.
 * @return The interventions, in layer order; the model owns them.
 * @throws std::invalid_argument for a layer the model doesn't have, or invalid sizes; std::logic_error
 *         if a chosen layer already has an intervention.
 */
std::vector<LoReFTIntervention*> ApplyLoReFT(CausalLM& model, const LoReFTConfig& config = {});

/** @brief orthonormalize() on every LoReFT intervention of the model: call it after each optimizer
 *         step. */
void OrthonormalizeLoReFT(CausalLM& model);

/**
 * @brief Saves the model's LoReFT interventions as safetensors: per layer i, pyreft's state dict
 *        under `layer.<i>.rotate_layer`, `layer.<i>.learned_source.weight` and
 *        `layer.<i>.learned_source.bias`, in pyreft's layouts, with metadata `first` and `last`.
 * @throws std::invalid_argument if the model has none, or they edit different positions.
 */
void SaveLoReFT(CausalLM& model, const std::string& path);

/**
 * @brief Installs the interventions in a file SaveLoReFT() wrote, or one built from pyreft's
 *        per-intervention state dicts under those keys (convert its `.bin` files with
 *        tools/convert/pickle_to_safetensors.py), and freezes the rest of the model.
 * @throws std::invalid_argument for missing metadata, an unknown key, a layer the model doesn't have
 *         or already intervenes on, or a shape that doesn't fit. Nothing changes then.
 */
std::vector<LoReFTIntervention*> LoadLoReFT(CausalLM& model, const std::string& path);

}  // namespace pulsatrix
