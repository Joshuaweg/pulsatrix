/** @file parameter_decomposition.hpp
 *  @brief Parameter decomposition (FEAT-9): explaining a network in weight space. Each weight
 *         matrix is split into rank-one subcomponents, and training asks that, for every input,
 *         few of them be needed to compute the output. SPD (Bushnaq, Braun and Sharkey,
 *         "Stochastic Parameter Decomposition", arXiv 2506.20790) and VPD (Bushnaq et al.,
 *         "Interpreting Language Model Parameters", Goodfire, 2026: adversarial parameter
 *         decomposition) are both here, after their reference code (github.com/goodfire-ai/
 *         param-decomp: tag v1 for SPD, nano_param_decomp for VPD).
 *  @ingroup mech_interp
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief A LinearModule split into C rank-one subcomponents: `W ≈ V U`, V `(in, C)` reading the
 *        input and U `(C, out)` writing the output, so subcomponent c is `V[:, c] U[c, :]` and its
 *        inner activation is `h_c = (x V)_c`. The bias is the target's, frozen.
 *
 * It runs in one of two modes, chosen before each forward():
 * - **Target:** `y = x W + b` with the original weight, and the input is kept for causal
 *   importance (causal_importance()).
 * - **Components:** `y = ((x V) ⊙ m) U + b` under a mask m `(N, C)`. With a Δ mask `(N,)` (VPD),
 *   the rest of the weight, `Δ = W - V U`, is added back under it:
 *   `y = ((x V) ⊙ (m - m_Δ)) U + m_Δ (x W) + b`. Masks of 1 then give the target exactly.
 *
 * backward() returns the input's gradient, adds to V's and U's, and keeps the masks' gradients
 * (mask_grad(), delta_mask_grad()) for the decomposition to carry into the causal importances.
 *
 * **Causal importance** (SPD): each subcomponent has its own small MLP from its inner activation
 * to a scalar, `z_c = w_out,c · GELU(w_in,c h_c + b_in,c) + b_out,c`, hidden size gate_hidden.
 * Parameters: `V.weight`, `U.weight`, `gate.in_weight`, `gate.in_bias`, `gate.out_weight`,
 * `gate.out_bias`.
 */
class ComponentLinear : public Module {
public:
    /**
     * @brief Copies @p target's weight and bias, frozen.
     * - **SPD's initialization** (v1 `init_As_and_Bs_`): random unit V and U columns and rows, each
     *   U row then scaled by `V[:, c]ᵀ W U[c, :]ᵀ`, its overlap with the target.
     * - The gate MLPs start as the reference's: input weights `N(0, 2)`, output weights
     *   `N(0, 1/gate_hidden)`, zero biases.
     * @throws std::invalid_argument if num_components or gate_hidden < 1.
     */
    ComponentLinear(LinearModule& target, int64_t num_components, DeviceBackend* backend, int64_t gate_hidden = 16, uint64_t seed = 0);

    [[nodiscard]] int64_t in_features() const { return in_; }
    [[nodiscard]] int64_t out_features() const { return out_; }
    [[nodiscard]] int64_t num_components() const { return C_; }
    [[nodiscard]] int64_t gate_hidden() const { return G_; }

    /** @brief Later forward() calls use the target weight. */
    void use_target();
    /** @brief Later forward() calls use the subcomponents under @p masks `(N, C)`, and the Δ
     *         weight under @p delta_masks `(N,)` when given. Sizes are checked at forward(). */
    void use_components(std::vector<float> masks, std::vector<float> delta_masks = {});
    [[nodiscard]] bool uses_target() const { return target_mode_; }

    /** @brief dL/dm from the last backward() in components mode, `(N, C)`. */
    [[nodiscard]] const std::vector<float>& mask_grad() const { return mask_grad_; }
    /** @brief dL/dm_Δ from the last backward() with a Δ mask, `(N,)`. */
    [[nodiscard]] const std::vector<float>& delta_mask_grad() const { return delta_mask_grad_; }

    /** @brief The gates' outputs z `(N, C)` for the input of the last target-mode forward(); the
     *         decomposition turns them into causal importances. @throws std::logic_error before
     *         any target-mode forward(). */
    [[nodiscard]] std::vector<float> gate_outputs();
    /** @brief Backward from dL/dz `(N, C)` for the last gate_outputs(): into the gates'
     *         parameters and, through the inner activations, into V. */
    void gate_backward(const std::vector<float>& grad_z);

    /** @brief The subcomponents' sum `V U`, `(in, out)` as LinearModule stores weights. */
    [[nodiscard]] std::vector<float> component_weight();
    /** @brief The frozen target weight, `(in, out)`. */
    [[nodiscard]] const std::vector<float>& target_weight() const { return target_w_; }
    [[nodiscard]] LinearModule& V() { return v_; }
    [[nodiscard]] LinearModule& U() { return u_; }

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief Through the weight in use: the target's, or the masked subcomponents (and Δ). */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void release_activations() override;

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t in_, out_, C_, G_;
    DeviceBackend* backend_;
    std::vector<float> target_w_, bias_;
    LinearModule w_;  ///< (in -> out), the frozen target weight, no bias
    LinearModule v_;  ///< (in -> C)
    LinearModule u_;  ///< (C -> out)
    Tensor gate_in_w_, gate_in_b_, gate_out_w_, gate_out_b_;  ///< (C, G), (C, G), (C, G), (C)
    Tensor gate_in_w_grad_, gate_in_b_grad_, gate_out_w_grad_, gate_out_b_grad_;
    bool target_mode_ = true;
    std::vector<float> masks_, delta_masks_;
    // The last forward (components mode).
    int64_t rows_ = 0;
    std::vector<float> h_, xw_, scaled_;  ///< x V, x W (with Δ), the masks U saw
    std::vector<float> mask_grad_, delta_mask_grad_;
    // The last target-mode input and gate pass.
    std::optional<Tensor> target_input_;
    std::optional<Tensor> gate_input_;  ///< the input gate_outputs() used
    std::vector<float> gate_h_;  ///< (N, C) inner activations
    std::vector<float> gate_act_, gate_slope_;  ///< (N, C, G): GELU and its slope at each hidden unit
};

enum class DecompositionMethod {
    /** @brief SPD: masks `m = g + (1 - g) r`, r uniform; the losses are faithfulness,
     *         reconstruction with every layer masked, reconstruction with one layer masked at a
     *         time, and importance minimality `Σ g^p`. */
    SPD,
    /** @brief VPD: the Δ weight is part of the forward pass under its own mask; reconstruction
     *         routes each input through a random k of the L layers' components (k uniform), and
     *         again under adversarial masks (persistent PGD); importance minimality adds a
     *         frequency term; the causal importance's lower clamp is straight-through. */
    VPD,
};

/** @brief How the masked model's output is compared with the target's. */
enum class OutputDivergence {
    /** @brief mean((y - y*)²) over every value. */
    MeanSquaredError,
    /** @brief KL(softmax(y*) ‖ softmax(y)) over the last dimension, averaged over rows: for logits. */
    KlOnLogits,
};

struct DecompositionOptions {
    DecompositionMethod method = DecompositionMethod::SPD;
    OutputDivergence divergence = OutputDivergence::MeanSquaredError;
    /** @brief `Σ_l |W - V U|² / (number of weights)`: SPD's faithfulness, VPD's Δ loss (whose
     *         paper weights it 1e7). */
    float faithfulness_coefficient = 1.0f;
    /** @brief Reconstruction with every layer masked (SPD), or k of them (VPD; 0.5 there). */
    float stochastic_coefficient = 1.0f;
    /** @brief SPD: reconstruction with one layer masked at a time, averaged over layers. */
    float layerwise_coefficient = 1.0f;
    /** @brief VPD: reconstruction under adversarial masks. */
    float adversarial_coefficient = 0.5f;
    /** @brief Importance minimality's weight and exponent p (set_p() anneals it). */
    float importance_coefficient = 3e-3f;
    float p = 1.0f;
    /** @brief VPD: the frequency term's β in `Σ_c [mean_c + β mean_c log2(1 + sum_c)]`. */
    float frequency_beta = 0.5f;
    /** @brief The leaky slope of the causal importance's clamps. */
    float leak = 0.01f;
    /** @brief VPD's adversary: ascent steps per training step (before the one that uses the main
     *         loss's gradient), and Adam's learning rate and betas for the sources. */
    int64_t pgd_steps = 2;
    float pgd_lr = 0.01f, pgd_beta1 = 0.5f, pgd_beta2 = 0.99f;
    uint64_t seed = 0;
};

struct DecompositionLoss {
    float total = 0, faithfulness = 0, stochastic = 0, layerwise = 0, adversarial = 0, importance = 0;
};

/**
 * @brief Trains a decomposition of the ComponentLinear layers inside @p model against the model
 *        itself: the target is the model with every layer in target mode.
 *
 * - The model can be any Module built from them (SequentialModule, ResidualModule, ...) whose
 *   backward() reaches every layer. Only the layers' parameters train (parameters_module()).
 * - **Causal importance** comes from the target pass's inner activations. Masks use the lower
 *   clamp, `g = clamp(z, 0, 1)` with a leak below 0. SPD leaks in the forward pass too
 *   (`0.01 z`); VPD's is a straight-through leak only for gradients that raise g. The minimality
 *   loss uses the upper clamp, `z` in [0, 1] and `1 + leak (z - 1)` above.
 * - **The gate input isn't detached** (v1), so the importance losses reach V too.
 * - One mask sample per step (S = 1), from the decomposition's own generator or set_noise().
 */
class ParameterDecomposition {
public:
    /** @throws std::invalid_argument if @p layers is empty or holds a null. */
    ParameterDecomposition(Module& model, std::vector<ComponentLinear*> layers, DeviceBackend* backend, DecompositionOptions options = {});

    /** @brief Every loss and its gradients, added to the layers' (zero them first). */
    DecompositionLoss loss_and_backward(const Tensor& x);

    /** @brief The trainable parameters, for an optimizer's step() and zero_grad(). */
    [[nodiscard]] Module& parameters_module();
    /** @brief Importance minimality's exponent, for annealing (VPD: 2 down to 0.4). */
    void set_p(float p) { options_.p = p; }
    [[nodiscard]] const DecompositionOptions& options() const { return options_; }
    [[nodiscard]] const std::vector<ComponentLinear*>& layers() const { return layers_; }

    /** @brief The target model's output for @p x. */
    [[nodiscard]] Tensor target_output(const Tensor& x);
    /** @brief Each layer's causal importances `(N, C)` for @p x (lower clamp). */
    [[nodiscard]] std::vector<std::vector<float>> causal_importances(const Tensor& x);

    /**
     * @brief Replaces the uniform draws, for tests: called with a pass name ("stochastic",
     *        "layerwise", "delta", "route"), the layer (or -1), and how many values are needed;
     *        returns them in [0, 1). For "route" one value per row, read as k = 1 + floor(v L),
     *        then L more per row ranking the layers (the k smallest are routed).
     */
    using NoiseFn = std::function<std::vector<float>(const char* pass, int64_t layer, int64_t count)>;
    void set_noise(NoiseFn noise) { noise_ = std::move(noise); }
    /** @brief VPD's persistent adversarial sources per layer, `(N, C + 1)`, the last column the
     *         Δ mask's. Empty until the first step; set them to resume, or in tests. */
    [[nodiscard]] std::vector<std::vector<float>>& adversarial_sources() { return sources_; }

private:
    struct Importance {
        std::vector<float> z, lower, upper;
    };
    std::vector<float> Draw(const char* pass, int64_t layer, int64_t count);
    /** @brief The output loss and its gradient for @p pred against @p target. */
    float Divergence(const std::vector<float>& pred, const std::vector<float>& target, int64_t rows, std::vector<float>* grad) const;
    /** @brief Forward and backward with the layers already set; the loss times @p weight. */
    float Pass(const Tensor& x, const std::vector<float>& target, float weight);

    Module& model_;
    std::vector<ComponentLinear*> layers_;
    DeviceBackend* backend_;
    DecompositionOptions options_;
    std::unique_ptr<Module> params_;
    uint64_t rng_state_;
    NoiseFn noise_;
    std::vector<std::vector<float>> sources_, source_m_, source_v_;
    int64_t pgd_t_ = 0;
};

/** @brief One training step: zero the gradients, loss_and_backward(), one optimizer step. */
template <typename Optimizer>
DecompositionLoss TrainDecomposition(ParameterDecomposition& d, const Tensor& x, Optimizer& optimizer) {
    optimizer.zero_grad(d.parameters_module());
    const DecompositionLoss loss = d.loss_and_backward(x);
    optimizer.step(d.parameters_module());
    return loss;
}

/**
 * @brief How well a layer's subcomponents line up with its target weight's rows, the input
 *        features (SPD's toy-model measure): for each input feature j, the best cosine over c
 *        between subcomponent c's row j, `V[j, c] U[c, :]`, and the target's row j (MMCS), and
 *        that subcomponent's norm over the target row's (ML2R).
 */
struct ComponentAlignment {
    double mean_max_cosine = 0;
    double mean_norm_ratio = 0;
    /** @brief Per input feature: the best subcomponent. */
    std::vector<int64_t> best;
};
[[nodiscard]] ComponentAlignment AlignComponentsToRows(ComponentLinear& layer);

/** @brief Causal importance statistics of one layer over inputs. */
struct ImportanceStats {
    /** @brief Mean number of subcomponents per input with importance above 0.01 (SPD's ci_l0). */
    double l0 = 0;
    /** @brief Subcomponents above 0.1 on some input. */
    int64_t alive = 0;
    /** @brief Each subcomponent's largest importance. */
    std::vector<float> max_importance;
};
[[nodiscard]] std::vector<ImportanceStats> MeasureImportance(ParameterDecomposition& d, const Tensor& x);

}  // namespace pulsatrix
