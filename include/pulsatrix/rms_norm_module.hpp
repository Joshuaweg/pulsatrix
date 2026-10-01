/** @file rms_norm_module.hpp
 *  @brief RMS normalization layer (Zhang & Sennrich, 2019) -- this codebase's first
 *         normalization Module, establishing the Tier 1 pattern for GroupNorm/LayerNorm.
 *  @ingroup dl_modules
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief y_{n,i} = gamma_i * x_{n,i} / rms(x_n), rms(x_n) = sqrt(mean_i(x_{n,i}^2) + eps),
 *        computed independently per batch row n. Batched ((N, num_features)), migrated
 *        from the original unbatched (rank-1) scope by
 *        campaign_exai_dl_library_batch_dimension_support -- no mean-centering, no
 *        beta/bias term (RMSNorm's defining simplification vs. LayerNorm).
 * @note propagate_relevance is an identity pass-through, cited to AttnLRP's
 *       normalization-layer treatment (Achtibat et al. 2024, already named in this
 *       project's charter for Attention/Transformer work) -- the same pattern this
 *       codebase already uses for ReluModule/FlattenModule's parameterless/
 *       elementwise-treated operations, not a new pattern invented for this module.
 *       backward() is the real, undetached training gradient -- the identity-rule
 *       simplification applies only to relevance redistribution, never to the actual
 *       gradient used for parameter updates (mirrors LinearModule's own independent
 *       backward()/propagate_relevance() computations from the same cached forward state).
 */
class RMSNormModule : public Module {
public:
    /**
     * @brief Constructs an RMSNorm layer with zero-initialized gamma.
     * @param num_features Number of elements this layer normalizes over.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param device Which device every internal Tensor member is tagged as. Defaults to Cpu.
     * @param eps Stabilizer added inside the sqrt, avoiding division by zero for an
     *        all-zero input. Defaults to 1e-6, matching LRPRuleConfig's own default epsilon
     *        order of magnitude.
     * @throws std::invalid_argument if num_features <= 0 -- external boundary (construction
     *         arguments can originate from Phase 5's Python bindings with no upstream
     *         validation), per cpp_tdd/context_tdd_adversarial_boundary_testing.md.
     */
    RMSNormModule(int64_t num_features, DeviceBackend* backend, DeviceType device,
                  float eps = 1e-6f);

    /** @brief On backend's own device (backend->device()), default eps. Previously the device
     *        defaulted to Cpu regardless of backend (GPU-native-kernels Mission 0). */
    RMSNormModule(int64_t num_features, DeviceBackend* backend);

    /**
     * @brief Computes the gradient w.r.t. this module's input, and accumulates gamma's
     *        gradient internally.
     * @param grad_output Gradient w.r.t. this module's output. Must match num_features.
     * @return Gradient w.r.t. this module's input.
     * @throws std::logic_error if forward() has never been called.
     * @note Device-generic: runs on Cpu, Cuda or Hip tensors (GPU-native-kernels Mission 2).
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Normalization per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Normalization; }

    /** @brief Overwrites the gamma buffer -- test/initialization use only. */
    void set_gamma(std::initializer_list<float> values);

    /** @brief Vector overload for runtime-sized sources -- see Tensor's own vector ctor. */
    void set_gamma(const std::vector<float>& values);

    [[nodiscard]] const Tensor& gamma() const { return gamma_; }
    [[nodiscard]] const Tensor& gamma_grad() const { return gamma_grad_; }

    /**
     * @brief Identity-rule LRP relevance propagation (AttnLRP, Achtibat et al. 2024).
     * @param relevance_out Relevance at this module's output. Must match num_features.
     * @param config Unused -- the identity rule has no tunable parameter, unlike
     *        LinearModule's epsilon rule.
     * @return relevance_out, unchanged -- conservation holds trivially by construction.
     * @throws std::logic_error if forward() has never been called (sequencing violation,
     *         consistent with every other module's LRP-entry-point precedent).
     * @throws std::invalid_argument if relevance_out's element count doesn't match
     *         num_features.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override { return {{&gamma_, &gamma_grad_}}; }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t num_features_;
    float eps_;
    DeviceBackend* backend_;
    Tensor gamma_;
    Tensor gamma_grad_;
    Tensor last_input_;
    Tensor last_rms_;  // (N,) one rms per batch row, on the module's device
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
