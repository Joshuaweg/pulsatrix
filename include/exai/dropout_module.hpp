/** @file dropout_module.hpp
 *  @brief Inverted dropout -- the first module whose forward behavior genuinely differs
 *         between training and inference (Module::is_training()).
 */
#pragma once

#include <random>
#include <vector>

#include "exai/module.hpp"

namespace exai {

/**
 * @brief y = (mask_i ? x_i / (1 - p) : 0) at training time (inverted dropout -- scaling
 *        happens at training time so eval-time forward needs no rescaling); y = x at
 *        eval time or when p == 0. No parameters.
 * @note propagate_relevance is unconditional identity pass-through, reusing
 *       ReluModule's established precedent (Montavon et al. 2019: activation-/
 *       regularization-like pointwise operations pass relevance through unchanged) --
 *       the same masked-backward/unconditional-identity-relevance split ReluModule
 *       already uses, not a new rule invented for this module. Dropout is conventionally
 *       disabled during inference/explanation in every mainstream framework, so at
 *       is_training() == false (the expected state when running LRP) forward is already
 *       identity, making this the mathematically exact treatment for that case, not just
 *       an approximation carried over from ReLU.
 */
class DropoutModule : public Module {
public:
    /**
     * @brief Constructs a dropout layer.
     * @param p Drop probability, must be in [0, 1).
     * @param backend Backend to compute through. Not owned; must outlive this module.
     * @param seed RNG seed. Defaults to a fixed value for reproducibility -- callers
     *        needing independent randomness across instances should pass distinct seeds.
     * @throws std::invalid_argument if p < 0 or p >= 1 -- external boundary (p == 1 would
     *         make scale = 1/(1-p) diverge; construction arguments can originate from
     *         Phase 5's Python bindings with no upstream validation).
     * @note No device parameter -- unlike ReluModule, this module caches no Tensor members
     *       (only a Shape and a plain float mask vector), so there is no internal Tensor
     *       needing an initial device tag; forward_impl() tags its output with the input
     *       tensor's own device, same reasoning as ReluModule's forward_impl.
     */
    explicit DropoutModule(float p, DeviceBackend* backend, uint64_t seed = 42);

    /**
     * @brief Computes the gradient w.r.t. this module's input: grad_output * mask * scale,
     *        the true gradient of the actual (masked/scaled) forward computation. If the
     *        most recent forward() ran in eval mode, mask is all-ones and scale == 1, so
     *        this correctly reduces to identity with no special-casing needed here.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @return Gradient w.r.t. this module's input.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached
     *         forward output shape.
     * @note Not yet backend-generic -- raw host loop. EXAI_ASSERT(grad_output.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Elementwise per charter's closed OpType set -- a per-element scale-or-zero
     *         operation, an honest fit for the existing category (no new OpType needed). */
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    /**
     * @brief Unconditional identity LRP relevance propagation.
     * @param relevance_out Relevance at this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @param config Unused.
     * @return relevance_out, unchanged -- see the class-level note for why this is exact,
     *         not approximate, when is_training() == false.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

protected:
    /**
     * @brief The actual forward computation -- per-element RNG draw at training time,
     *        identity at eval time or p == 0.
     * @note Not yet backend-generic -- raw host loop. EXAI_ASSERT(input.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    float p_;
    float scale_;
    DeviceBackend* backend_;
    std::mt19937 rng_;
    Shape last_shape_ = Shape({0});
    std::vector<float> last_mask_;  // 1.0 (kept) or 0.0 (dropped), flat, numel() entries
    bool has_forwarded_ = false;
};

}  // namespace exai
