/** @file group_norm_module.hpp
 *  @brief Group normalization (Wu & He, 2018) -- rank-3 (C, H, W), matching Conv2DModule's
 *         convention, unlike RMSNormModule/LayerNormModule's rank-1 feature-vector scope.
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "exai/module.hpp"

namespace exai {

/**
 * @brief Splits num_channels into num_groups equal-size groups; each group's mean/std is
 *        computed over every (channel-in-group, H, W) element jointly, per batch row n,
 *        then y_{n,c,h,w} = gamma_c * (x_{n,c,h,w} - mu_{n,g})/std_{n,g} + beta_c, gamma/beta
 *        per-channel (shape (num_channels,), not per-group, not per-batch-row). Batched --
 *        input/output are rank-4 (N, channels, H, W), migrated from the original unbatched
 *        (rank-3) scope by campaign_exai_dl_library_batch_dimension_support, matching
 *        Conv2DModule's own (not-yet-migrated) rank-3 convention plus a leading batch dim.
 *        H/W are not fixed at construction (only num_groups/num_channels are), so this
 *        module accepts any spatial size at forward() time, exactly like Conv2DModule does.
 * @note propagate_relevance is an identity pass-through, cited to AttnLRP's
 *       normalization-layer treatment (Achtibat et al. 2024) -- same rule, same citation,
 *       as RMSNormModule/LayerNormModule; GroupNorm is architecturally the same
 *       "normalization" category the citation covers, just with a different statistic
 *       grouping. backward() is the real, undetached training gradient.
 */
class GroupNormModule : public Module {
public:
    /**
     * @brief Constructs a GroupNorm layer with zero-initialized gamma and beta.
     * @param num_groups Number of groups to split num_channels into.
     * @param num_channels Total channel count. Must be evenly divisible by num_groups.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param device Which device every internal Tensor member is tagged as. Defaults to Cpu.
     * @param eps Stabilizer added inside the sqrt. Defaults to 1e-6, matching
     *        RMSNormModule/LayerNormModule's default for consistency within this codebase's
     *        normalization family.
     * @throws std::invalid_argument if num_groups <= 0, num_channels <= 0, or num_channels
     *         is not evenly divisible by num_groups -- external boundary (construction
     *         arguments can originate from Phase 5's Python bindings with no upstream
     *         validation), per cpp_tdd/context_tdd_adversarial_boundary_testing.md.
     */
    GroupNormModule(int64_t num_groups, int64_t num_channels, DeviceBackend* backend,
                     DeviceType device = DeviceType::Cpu, float eps = 1e-6f);

    /**
     * @brief Computes the gradient w.r.t. this module's input, and accumulates gamma's/
     *        beta's gradients internally.
     * @param grad_output Gradient w.r.t. this module's output. Must match the shape of the
     *        most recent forward() call's output.
     * @return Gradient w.r.t. this module's input.
     * @throws std::logic_error if forward() has never been called.
     * @note Not yet backend-generic -- raw host loop, EXAI_ASSERT(grad_output.device() ==
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
    int64_t num_groups_;
    int64_t num_channels_;
    int64_t group_size_;
    float eps_;
    DeviceBackend* backend_;
    Tensor gamma_;       // shape (num_channels,)
    Tensor beta_;        // shape (num_channels,)
    Tensor gamma_grad_;
    Tensor beta_grad_;
    Tensor last_input_;      // (N, num_channels, H, W)
    Tensor last_xhat_;       // (N, num_channels, H, W)
    std::vector<float> last_group_std_;  // N*num_groups entries, indexed n*num_groups+g
    int64_t last_h_ = 0;
    int64_t last_w_ = 0;
    bool has_forwarded_ = false;
};

}  // namespace exai
