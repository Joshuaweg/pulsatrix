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

class BatchNormFold;

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
 * @note Training mode (the Module default) normalizes with the current batch's statistics
 *       and folds them into running statistics with PyTorch's rule: running = (1 - momentum) *
 *       running + momentum * batch, using the unbiased batch variance. Eval mode
 *       (set_training(false)) normalizes with the running statistics instead, so each sample's
 *       output depends only on that sample (roadmap FND-5, lrp_issues #8). Running statistics
 *       start at mean 0, variance 1. Put the model in eval mode before explaining it.
 * @note For LRP, fold an eval-mode BatchNorm into the Conv2D before it with BatchNormFold:
 *       the convolution's rule then distributes relevance through the combined affine map,
 *       and this module becomes an exact identity.
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
     * @param momentum Weight of each new batch in the running statistics, in (0, 1].
     *        Defaults to 0.1, PyTorch's default.
     * @throws std::invalid_argument if num_channels <= 0 or momentum is outside (0, 1] --
     *         external boundary (construction arguments can originate from Phase 5's Python
     *         bindings with no upstream validation), per
     *         cpp_tdd/context_tdd_adversarial_boundary_testing.md.
     */
    BatchNormModule(int64_t num_channels, DeviceBackend* backend, DeviceType device,
                     float eps = 1e-6f, float momentum = 0.1f);

    /** @brief On backend's own device (backend->device()), default eps. Previously the device
     *        defaulted to Cpu regardless of backend (GPU-native-kernels Mission 0). */
    BatchNormModule(int64_t num_channels, DeviceBackend* backend);

    /**
     * @brief Computes the gradient w.r.t. this module's input, and accumulates gamma's/
     *        beta's gradients internally.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @return Gradient w.r.t. this module's input.
     * @throws std::logic_error if forward() has never been called.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 4).
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

    /** @brief Per-channel running mean, used in eval mode. Shape (num_channels). */
    [[nodiscard]] const Tensor& running_mean() const { return running_mean_; }
    /** @brief Per-channel running variance, used in eval mode. Shape (num_channels). */
    [[nodiscard]] const Tensor& running_var() const { return running_var_; }
    /**
     * @brief Overwrites the running mean, e.g. when loading a pretrained model.
     * @throws std::invalid_argument on a size other than num_channels or a non-finite value.
     */
    void set_running_mean(const std::vector<float>& values);
    /**
     * @brief Overwrites the running variance, e.g. when loading a pretrained model.
     * @throws std::invalid_argument on a size other than num_channels, or a negative or
     *         non-finite value.
     */
    void set_running_var(const std::vector<float>& values);

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

    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override {
        return {{"weight", {&gamma_, &gamma_grad_}}, {"bias", {&beta_, &beta_grad_}}};
    }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    friend class BatchNormFold;

    // Which statistics the cached forward used, so backward() matches it even if the mode
    // changes in between.
    enum class Mode { Batch, Running, Folded };

    int64_t num_channels_;
    float eps_;
    float momentum_;
    DeviceBackend* backend_;
    Tensor gamma_;       // shape (num_channels,)
    Tensor beta_;        // shape (num_channels,)
    Tensor gamma_grad_;
    Tensor beta_grad_;
    Tensor last_input_;      // (N, num_channels, H, W)
    Tensor last_xhat_;       // (N, num_channels, H, W)
    Tensor last_std_;  // (num_channels,) one std per channel, on the module's device
    Tensor running_mean_;  // (num_channels,)
    Tensor running_var_;   // (num_channels,)
    bool has_forwarded_ = false;
    Mode last_mode_ = Mode::Batch;
    bool folded_ = false;  // set by BatchNormFold: forward/backward become the identity
};

}  // namespace pulsatrix
