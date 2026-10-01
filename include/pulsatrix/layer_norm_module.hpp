/** @file layer_norm_module.hpp
 *  @brief Layer normalization (Ba et al., 2016) -- mean-centered/scaled RMSNormModule sibling.
 *  @ingroup dl_modules
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief y_{n,i} = gamma_i * (x_{n,i} - mu_n)/std_n + beta_i, mu_n = mean_i(x_{n,i}),
 *        std_n = sqrt(var_i(x_{n,i}) + eps), computed independently per batch row n.
 *        Batched ((N, num_features)), migrated from the original unbatched (rank-1) scope
 *        by campaign_exai_dl_library_batch_dimension_support.
 * @note propagate_relevance is an identity pass-through, cited to AttnLRP's
 *       normalization-layer treatment (Achtibat et al. 2024) -- identical rationale to
 *       RMSNormModule's own identical rule; see that class's Doxygen for the full citation
 *       chain (Ali et al. 2022's detach-std treatment, simplified by AttnLRP to identity).
 *       backward() is the real, undetached training gradient -- the standard LayerNorm
 *       backward formula, independent of the LRP treatment.
 */
class LayerNormModule : public Module {
public:
    /**
     * @brief Constructs a LayerNorm layer with zero-initialized gamma and beta.
     * @param num_features Number of elements this layer normalizes over.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @param device Which device every internal Tensor member is tagged as. Defaults to Cpu.
     * @param eps Stabilizer added inside the sqrt. Defaults to 1e-6.
     * @throws std::invalid_argument if num_features <= 0 -- external boundary, mirrors
     *         RMSNormModule's identical constructor guard.
     */
    LayerNormModule(int64_t num_features, DeviceBackend* backend, DeviceType device,
                     float eps = 1e-6f);

    /** @brief On backend's own device (backend->device()), default eps. Previously the device
     *        defaulted to Cpu regardless of backend (GPU-native-kernels Mission 0). */
    LayerNormModule(int64_t num_features, DeviceBackend* backend);

    /**
     * @brief Computes the gradient w.r.t. this module's input, and accumulates gamma's/
     *        beta's gradients internally.
     * @param grad_output Gradient w.r.t. this module's output. Must match num_features.
     * @return Gradient w.r.t. this module's input.
     * @throws std::logic_error if forward() has never been called.
     * @note Not yet backend-generic -- raw host loop, PULSATRIX_ASSERT(grad_output.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor, matching
     *       every existing Module subclass's Phase 1.5 scope decision.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Normalization per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Normalization; }

    /** @brief Overwrites the gamma buffer -- test/initialization use only. */
    void set_gamma(std::initializer_list<float> values);
    /** @brief Overwrites the beta buffer -- test/initialization use only. */
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
     * @param relevance_out Relevance at this module's output. Must match num_features.
     * @param config Unused -- the identity rule has no tunable parameter.
     * @return relevance_out, unchanged -- conservation holds trivially by construction.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's element count doesn't match
     *         num_features.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override {
        return {{&gamma_, &gamma_grad_}, {&beta_, &beta_grad_}};
    }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t num_features_;
    float eps_;
    DeviceBackend* backend_;
    Tensor gamma_;
    Tensor beta_;
    Tensor gamma_grad_;
    Tensor beta_grad_;
    Tensor last_input_;
    Tensor last_xhat_;
    std::vector<float> last_std_;  // one std per batch row
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
