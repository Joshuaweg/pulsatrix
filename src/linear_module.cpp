#include "pulsatrix/linear_module.hpp"

#include <stdexcept>

#include "lrp_rules.hpp"
#include "pulsatrix/assert.hpp"

namespace pulsatrix {

LinearModule::LinearModule(int64_t in_features, int64_t out_features, DeviceBackend* backend)
    : LinearModule(in_features, out_features, backend, backend->device()) {}

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

    // bias_ is (out_features,), broadcast-added per row on the device. (The batch migration
    // originally used a raw host loop here, which silently dereferenced device pointers on a
    // Cuda/Hip module; Mission 0 of GPU-native-kernels routed it through a ones-gemm, now the
    // dedicated add_row_vector primitive.)
    Tensor output(pre_bias.shape(), backend_, weight_.device());
    backend_->add_row_vector(pre_bias.data(), bias_.data(), output.data(), static_cast<size_t>(N),
                             static_cast<size_t>(out_features_));
    has_forwarded_ = true;
    return output;
}

Tensor LinearModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "LinearModule::backward");
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

    // Device-generic (GPU-native-kernels Mission 1): gemm_ex reads X and W transposed in
    // place and accumulates straight into the gradient buffers (beta = 1), replacing the
    // host-side transpose copies and the bias-gradient host loop.
    const auto n = static_cast<size_t>(N);
    const auto in = static_cast<size_t>(in_features_);
    const auto out = static_cast<size_t>(out_features_);

    // grad_W += X^T @ grad_Y -- the batched sum of outer products is gemm's k-dimension sum.
    // Skipped entirely for a frozen weight (FND-2): this GEMM is the saving freezing exists for.
    if (weight_.requires_grad()) {
        backend_->gemm_ex(last_input_.data(), true, grad_output.data(), false, weight_grad_.data(), in, n, out,
                          1.0f);
    }

    // grad_bias += sum over the batch of grad_Y.
    if (bias_.requires_grad()) {
        backend_->column_sums(grad_output.data(), bias_grad_.data(), n, out, 1.0f);
    }

    // grad_X = grad_Y @ W^T
    Tensor grad_input(Shape({N, in_features_}), backend_, weight_.device());
    backend_->gemm_ex(grad_output.data(), false, weight_.data(), true, grad_input.data(), n, out, in, 0.0f);
    return grad_input;
}

Tensor LinearModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "LinearModule::propagate_relevance");
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

    lrp_rules::validate(config, "LinearModule");

    Tensor relevance_in(Shape({N, in_features_}), backend_, weight_.device());
    if (!lrp_rules::is_legacy_epsilon(config)) {
        // Zennit-compatible rules (LRP-rules Mission 2), composed from gemm/elementwise primitives.
        const auto n = static_cast<size_t>(N), in = static_cast<size_t>(in_features_),
                   out = static_cast<size_t>(out_features_);
        DeviceBackend* be = backend_;
        lrp_rules::AffineOp op;
        op.backend = be;
        op.device = weight_.device();
        op.input_numel = n * in;
        op.output_numel = n * out;
        op.weight_numel = in * out;
        op.bias_numel = out;
        op.forward = [=](const float* x, const float* w, float* y) { be->gemm(x, w, y, n, in, out); };
        op.backward = [=](const float* g, const float* w, float* gx) {
            be->gemm_ex(g, false, w, true, gx, n, out, in, 0.0f);
        };
        op.add_bias = [=](const float* y, const float* b, float* o) { be->add_row_vector(y, b, o, n, out); };
        lrp_rules::apply(op, last_input_.data(), weight_.data(), bias_.data(), last_pre_bias_output_.data(),
                         relevance_out.data(), relevance_in.data(), config);
        return relevance_in;
    }

    // Device-generic (GPU-native-kernels Mission 3): one output element per GPU thread, each
    // summing over out_features in the original loop's order.
    backend_->lrp_linear(last_input_.data(), weight_.data(), last_pre_bias_output_.data(), relevance_out.data(),
                         relevance_in.data(), static_cast<size_t>(N), static_cast<size_t>(in_features_),
                         static_cast<size_t>(out_features_), config.epsilon);
    return relevance_in;
}

}  // namespace pulsatrix
