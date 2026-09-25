#include "pulsatrix/linear_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {
// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer.
// Used by backward()'s grad_x step (needs W^T) and, since the batch migration, also its
// weight-gradient step (needs X^T) -- CPUBackend::gemm has no transpose flag.
Tensor transpose(const Tensor& m, int64_t rows, int64_t cols, DeviceBackend* backend) {
    Tensor out(Shape({cols, rows}), backend);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.data()[c * rows + r] = m.data()[r * cols + c];
        }
    }
    return out;
}
}  // namespace

LinearModule::LinearModule(int64_t in_features, int64_t out_features, DeviceBackend* backend, DeviceType device)
    : in_features_(in_features),
      out_features_(out_features),
      backend_(backend),
      weight_(Shape({in_features, out_features}), backend, device),
      bias_(Shape({out_features}), backend, device),
      weight_grad_(Shape({in_features, out_features}), backend, device),
      bias_grad_(Shape({out_features}), backend, device),
      last_input_(Shape({1, in_features}), backend, device),
      last_pre_bias_output_(Shape({1, out_features}), backend, device) {}

void LinearModule::set_weight(std::initializer_list<float> values) {
    // Preserve weight_'s existing device tag -- reconstructing with the default (Cpu)
    // would pick the wrong CopyDirection for a non-Cpu module (see Tensor's own
    // device-based CopyDirection dispatch, Phase 1.5 Mission 3 Objective 1).
    weight_ = Tensor(weight_.shape(), backend_, values, weight_.device());
}

void LinearModule::set_bias(std::initializer_list<float> values) {
    bias_ = Tensor(bias_.shape(), backend_, values, bias_.device());
}

void LinearModule::set_weight(const std::vector<float>& values) {
    weight_ = Tensor(weight_.shape(), backend_, values, weight_.device());
}

void LinearModule::set_bias(const std::vector<float>& values) {
    bias_ = Tensor(bias_.shape(), backend_, values, bias_.device());
}

Tensor LinearModule::forward_impl(const Tensor& input) {
    // External boundary (campaign_exai_dl_library_adversarial_hardening.md, Mission 1,
    // findings 3/12; shape generalized to (N, in_features) by
    // campaign_exai_dl_library_batch_dimension_support): input can originate from Phase 5's
    // Python bindings with no upstream validation. Without this check, a wrong-shaped input
    // fed straight into gemm below is read as if it had (N, in_features_) elements -- a real
    // heap OOB read.
    if (input.rank() != 2 || input.shape().dim(1) != in_features_) {
        throw std::invalid_argument("LinearModule::forward: input must be rank-2 (N, in_features)");
    }
    const int64_t N = input.shape().dim(0);

    last_input_ = input;

    // Tagged with weight_'s device (this module's configured device, immutable after
    // construction) -- a fresh Tensor here defaults to Cpu otherwise, which would make the
    // last_pre_bias_output_ = pre_bias assignment below pick the wrong CopyDirection on a
    // non-Cpu module (Phase 1.5 Mission 3, Objective 2).
    Tensor pre_bias(Shape({N, out_features_}), backend_, weight_.device());
    backend_->gemm(input.data(), weight_.data(), pre_bias.data(), static_cast<size_t>(N),
                    static_cast<size_t>(in_features_), static_cast<size_t>(out_features_));
    last_pre_bias_output_ = pre_bias;  // cached for propagate_relevance's z_j

    // bias_ is (out_features,), broadcast-added per row -- Tensor::accumulate() requires
    // exact shape equality, so this is a raw loop, not accumulate(bias_) (which only worked
    // for the pre-migration N=1-shaped-as-rank-1 case).
    Tensor output(pre_bias);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t j = 0; j < out_features_; ++j) {
            output.data()[n * out_features_ + j] += bias_.data()[j];
        }
    }
    has_forwarded_ = true;
    return output;
}

Tensor LinearModule::backward(const Tensor& grad_output) {
    // Finding 12: calling backward() before any forward() previously silently computed a
    // meaningless answer from zero-initialized cached state (last_input_) instead of
    // erroring.
    if (!has_forwarded_) {
        throw std::logic_error("LinearModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    // Batch-size-mismatch is a new adversarial case introduced by the batch migration
    // (campaign_exai_dl_library_batch_dimension_support) -- grad_output must match the
    // batch size forward() actually cached, not just have the right feature count.
    if (grad_output.rank() != 2 || grad_output.shape().dim(1) != out_features_ ||
        grad_output.shape().dim(0) != N) {
        throw std::invalid_argument(
            "LinearModule::backward: grad_output must be rank-2 (N, out_features) matching the cached batch size");
    }

    // Dereferences Tensor::data() directly (via transpose()) -- not yet backend-generic.
    // See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu);

    // grad_W = X^T @ grad_Y = (in_features x N) @ (N x out_features) -- the batched
    // sum-of-outer-products reduces to a single gemm via X^T (Mission 0's design trace,
    // campaign_exai_dl_library_batch_dimension_support). The pre-migration N=1 case was
    // outer(x, grad_y); this generalizes it, not replaces it with new math.
    Tensor input_t = transpose(last_input_, N, in_features_, backend_);
    Tensor local_weight_grad(Shape({in_features_, out_features_}), backend_);
    backend_->gemm(input_t.data(), grad_output.data(), local_weight_grad.data(), static_cast<size_t>(in_features_),
                    static_cast<size_t>(N), static_cast<size_t>(out_features_));
    weight_grad_.accumulate(local_weight_grad);

    // grad_bias = sum over the batch of grad_output -- no new Tensor primitive needed
    // (Mission 0's design trace): a locally-built (out_features,) tensor via a raw host
    // loop, then the existing, unmodified accumulate().
    Tensor local_bias_grad(Shape({out_features_}), backend_);
    local_bias_grad.fill(0.0f);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t j = 0; j < out_features_; ++j) {
            local_bias_grad.data()[j] += grad_output.data()[n * out_features_ + j];
        }
    }
    bias_grad_.accumulate(local_bias_grad);

    // grad_x = grad_Y @ W^T = (N x out_features) @ (out_features x in_features)
    Tensor weight_t = transpose(weight_, in_features_, out_features_, backend_);
    Tensor grad_input(Shape({N, in_features_}), backend_);
    backend_->gemm(grad_output.data(), weight_t.data(), grad_input.data(), static_cast<size_t>(N),
                    static_cast<size_t>(out_features_), static_cast<size_t>(in_features_));
    return grad_input;
}

Tensor LinearModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    // Finding 12: see backward()'s identical guard above.
    if (!has_forwarded_) {
        throw std::logic_error("LinearModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    if (relevance_out.rank() != 2 || relevance_out.shape().dim(1) != out_features_ ||
        relevance_out.shape().dim(0) != N) {
        throw std::invalid_argument(
            "LinearModule::propagate_relevance: relevance_out must be rank-2 (N, out_features) matching the "
            "cached batch size");
    }

    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic.
    // See campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision and
    // mission_host_loop_guards.md.
    PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu);

    // Applied independently per example -- each row's relevance redistribution uses only
    // that row's own cached z_j/x_i, no cross-example coupling.
    Tensor relevance_in(Shape({N, in_features_}), backend_);
    relevance_in.fill(0.0f);

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t j = 0; j < out_features_; ++j) {
            float z_j = last_pre_bias_output_.data()[n * out_features_ + j];
            float sign = (z_j >= 0.0f) ? 1.0f : -1.0f;
            float denom = z_j + config.epsilon * sign;
            float r_j = relevance_out.data()[n * out_features_ + j];

            for (int64_t i = 0; i < in_features_; ++i) {
                float w_ij = weight_.data()[i * out_features_ + j];
                float x_i = last_input_.data()[n * in_features_ + i];
                relevance_in.data()[n * in_features_ + i] += (x_i * w_ij / denom) * r_j;
            }
        }
    }

    return relevance_in;
}

}  // namespace pulsatrix
