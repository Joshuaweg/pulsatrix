/** @file batch_norm_module.hpp
 *  @brief Batch normalization (Ioffe & Szegedy, 2015) -- per-channel statistics computed
 *         across the batch and spatial dimensions jointly, unlike GroupNormModule's
 *         per-(batch-row, group) statistics.
 *  @ingroup dl_modules
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief y_{n,c,h,w} = gamma_c * (x_{n,c,h,w} - mu_c)/std_c + beta_c, mu_c/std_c computed
 *        per channel c over every (n, h, w) element jointly -- BatchNorm's defining
 *        statistic, and the reason this module didn't exist before
 *        campaign_exai_dl_library_batch_dimension_support: it has nothing to compute over
 *        without a real batch dimension. Input/output are rank-4 (N, channels, H, W), the
 *        same convention Conv2DModule/GroupNormModule already establish.
 * @note propagate_relevance is an identity pass-through, cited to AttnLRP's
 *       normalization-layer treatment (Achtibat et al. 2024) -- the same rule, same
 *       citation, RMSNormModule/LayerNormModule/GroupNormModule already use; BatchNorm is
 *       architecturally the same normalization category, just a different statistic
 *       grouping (channel-over-batch-and-spatial instead of group-over-spatial-per-row).
 *       backward() is the real, undetached training gradient.
 * @note No running-mean/running-variance or train/eval mode -- this ships computing batch
 *       statistics from the current input on every call (the same behavior every other
 *       call site in this codebase already uses implicitly). A real BatchNorm eval-mode
 *       (exponential moving average statistics, frozen at inference) needs a train/eval
 *       toggle this codebase's Module hierarchy doesn't have yet (the same open design
 *       question already flagged for Dropout, campaign Decision Point 2) -- deliberately
 *       not built speculatively here; this is a real, documented scope cut, not an
 *       oversight.
 */
class BatchNormModule : public Module {
public:
    /**
     * @brief Constructs a BatchNorm layer with zero-initialized gamma and beta.
     * @param num_channels Number of channels (the statistic-bearing dimension).
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param device Which device every internal Tensor member is tagged as. Defaults to Cpu.
     * @param eps Stabilizer added inside the sqrt. Defaults to 1e-6, matching
     *        RMSNormModule/LayerNormModule/GroupNormModule's default for consistency
     *        within this codebase's normalization family.
     * @throws std::invalid_argument if num_channels <= 0 -- external boundary (construction
     *         arguments can originate from Phase 5's Python bindings with no upstream
     *         validation), per cpp_tdd/context_tdd_adversarial_boundary_testing.md.
     */
    BatchNormModule(int64_t num_channels, DeviceBackend* backend, DeviceType device = DeviceType::Cpu,
                     float eps = 1e-6f);

    /**
     * @brief Computes the gradient w.r.t. this module's input, and accumulates gamma's/
     *        beta's gradients internally.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @return Gradient w.r.t. this module's input.
     * @throws std::logic_error if forward() has never been called.
     * @note Not yet backend-generic -- raw host loop, PULSATRIX_ASSERT(grad_output.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor, matching
     *       every existing Module subclass's Phase 1.5 scope decision.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Normalization per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Normalization; }

    /** @brief Overwrites the per-channel gamma buffer -- test/initialization use only. */
    void set_gamma(std::initializer_list<float> values);
    /** @brief Overwrites the per-channel beta buffer -- test/initialization use only. */
    void set_beta(std::initializer_list<float> values);
    /** @brief Vector overload for runtime-sized sources -- see Tensor's own vector ctor. */
    void set_gamma(const std::vector<float>& values);
    /** @brief Vector overload for runtime-sized sources -- see Tensor's own vector ctor. */
    void set_beta(const std::vector<float>& values);

    [[nodiscard]] const Tensor& gamma() const { return gamma_; }
    [[nodiscard]] const Tensor& beta() const { return beta_; }
    [[nodiscard]] const Tensor& gamma_grad() const { return gamma_grad_; }
    [[nodiscard]] const Tensor& beta_grad() const { return beta_grad_; }

    /**
     * @brief Identity-rule LRP relevance propagation (AttnLRP, Achtibat et al. 2024).
     * @param relevance_out Relevance at this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @param config Unused -- the identity rule has no tunable parameter.
     * @return relevance_out, unchanged -- conservation holds trivially by construction.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's element count doesn't match the
     *         cached forward output's element count.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&gamma_, &gamma_grad_}, {&beta_, &beta_grad_}};
    }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t num_channels_;
    float eps_;
    DeviceBackend* backend_;
    Tensor gamma_;       // shape (num_channels,)
    Tensor beta_;        // shape (num_channels,)
    Tensor gamma_grad_;
    Tensor beta_grad_;
    Tensor last_input_;      // (N, num_channels, H, W)
    Tensor last_xhat_;       // (N, num_channels, H, W)
    std::vector<float> last_std_;  // one std per channel, cached for backward
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
