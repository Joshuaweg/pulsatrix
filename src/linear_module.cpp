#include "pulsatrix/linear_module.hpp"

#include <stdexcept>
#include <utility>

#include "lrp_rules.hpp"
#include "pulsatrix/assert.hpp"
#include "pulsatrix/lora.hpp"
#include "pulsatrix/relu_module.hpp"

namespace pulsatrix {

LinearModule::LinearModule(int64_t in_features, int64_t out_features, DeviceBackend* backend, bool use_bias)
    : LinearModule(in_features, out_features, backend, backend->device(), use_bias) {}

LinearModule::LinearModule(int64_t in_features, int64_t out_features, DeviceBackend* backend, DeviceType device,
                           bool use_bias)
    : in_features_(in_features),
      out_features_(out_features),
      use_bias_(use_bias),
      backend_(backend),
      weight_(Shape({in_features, out_features}), backend, device),
      bias_(Shape({out_features}), backend, device),
      weight_grad_(Shape({in_features, out_features}), backend, device),
      bias_grad_(Shape({out_features}), backend, device),
      last_input_(Shape({1, in_features}), backend, device),
      last_pre_bias_output_(Shape({1, out_features}), backend, device) {}

LinearModule::~LinearModule() = default;
LinearModule::LinearModule(LinearModule&&) noexcept = default;
LinearModule& LinearModule::operator=(LinearModule&&) noexcept = default;

std::vector<NamedParamRef> LinearModule::named_parameters() {
    std::vector<NamedParamRef> out{{"weight", {&weight_, &weight_grad_}}};
    if (use_bias_) out.push_back({"bias", {&bias_, &bias_grad_}});
    if (lora_) {
        out.push_back({"lora_A.weight", {&lora_->a_, &lora_->a_grad_}});
        out.push_back({"lora_B.weight", {&lora_->b_, &lora_->b_grad_}});
    }
    return out;
}

void LinearModule::attach_lora(const LoRAConfig& config) {
    if (lora_) throw std::logic_error("LinearModule::attach_lora: the layer already has an adapter");
    lora_ = std::make_unique<LoRAAdapter>(in_features_, out_features_, config, backend_, weight_.device());
}

void LinearModule::detach_lora() { lora_.reset(); }

bool LinearModule::lora_active() const { return lora_ && lora_->enabled_ && !lora_->merged_; }

void LinearModule::merge_lora() {
    if (!lora_) throw std::logic_error("LinearModule::merge_lora: the layer has no adapter");
    if (lora_->merged_) throw std::logic_error("LinearModule::merge_lora: the adapter is already merged");
    if (!lora_->enabled_) throw std::logic_error("LinearModule::merge_lora: the adapter is disabled");
    lora_->add_update(weight_, 1.0f);
    lora_->merged_ = true;
}

void LinearModule::unmerge_lora() {
    if (!lora_ || !lora_->merged_) throw std::logic_error("LinearModule::unmerge_lora: the layer has no merged adapter");
    lora_->add_update(weight_, -1.0f);
    lora_->merged_ = false;
}

void LinearModule::set_lora_enabled(bool enabled) {
    if (!lora_) throw std::logic_error("LinearModule::set_lora_enabled: the layer has no adapter");
    if (lora_->merged_) throw std::logic_error("LinearModule::set_lora_enabled: the adapter is merged");
    lora_->enabled_ = enabled;
}

void LinearModule::set_weight(std::initializer_list<float> values) {
    // Preserve weight_'s existing device tag -- reconstructing with the default (Cpu)
    // would pick the wrong CopyDirection for a non-Cpu module (see Tensor's own
    // device-based CopyDirection dispatch, Phase 1.5 Mission 3 Objective 1).
    weight_ = Tensor(weight_.shape(), backend_, values, weight_.device());
}

void LinearModule::set_bias(std::initializer_list<float> values) {
    if (!use_bias_) {
        throw std::logic_error("LinearModule::set_bias: this layer was built without a bias");
    }
    bias_ = Tensor(bias_.shape(), backend_, values, bias_.device());
}

void LinearModule::set_weight(const std::vector<float>& values) {
    weight_ = Tensor(weight_.shape(), backend_, values, weight_.device());
}

void LinearModule::set_bias(const std::vector<float>& values) {
    if (!use_bias_) {
        throw std::logic_error("LinearModule::set_bias: this layer was built without a bias");
    }
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

    if (lora_) lora_->clear_relevance();
    if (lora_active()) lora_->add_forward(input, pre_bias, is_training());

    // bias_ is (out_features,), broadcast-added per row on the device. (The batch migration
    // originally used a raw host loop here, which silently dereferenced device pointers on a
    // Cuda/Hip module; Mission 0 of GPU-native-kernels routed it through a ones-gemm, now the
    // dedicated add_row_vector primitive.)
    has_forwarded_ = true;
    if (!use_bias_) {
        last_pre_bias_output_ = pre_bias;  // cached for propagate_relevance's z_j
        return pre_bias;
    }
    Tensor output(pre_bias.shape(), backend_, weight_.device());
    backend_->add_row_vector(pre_bias.data(), bias_.data(), output.data(), static_cast<size_t>(N),
                             static_cast<size_t>(out_features_));
    last_pre_bias_output_ = std::move(pre_bias);  // cached for propagate_relevance's z_j; no copy (HIP-6)
    return output;
}

Tensor LinearModule::forward_with_relu(const Tensor& input, ReluModule& relu) {
    if (!use_bias_ || lora_ || relu.compute_device() != std::optional<DeviceType>(weight_.device())) {
        return relu.forward(forward(input));
    }
    // Module::forward()'s checks on the input, then the gemm as forward_impl() does it.
    if (input.numel() <= 0) throw std::invalid_argument("Module::forward: input must not be empty");
    require_device(input, weight_.device(), "Module::forward");
    if (input.rank() != 2 || input.shape().dim(1) != in_features_) {
        throw std::invalid_argument("LinearModule::forward: input must be rank-2 (N, in_features)");
    }
    const int64_t N = input.shape().dim(0);
    last_input_ = input;
    Tensor pre_bias(Shape({N, out_features_}), backend_, weight_.device());
    backend_->gemm(input.data(), weight_.data(), pre_bias.data(), static_cast<size_t>(N),
                   static_cast<size_t>(in_features_), static_cast<size_t>(out_features_));
    // One kernel: this layer's output z (the ReLU's input) and the ReLU's output.
    Tensor z(pre_bias.shape(), backend_, weight_.device());
    Tensor out(pre_bias.shape(), backend_, weight_.device());
    backend_->add_row_vector_relu(pre_bias.data(), bias_.data(), z.data(), out.data(), static_cast<size_t>(N),
                                  static_cast<size_t>(out_features_));
    last_pre_bias_output_ = std::move(pre_bias);
    has_forwarded_ = true;
    relu.set_fused_forward_input(std::move(z));
    return out;
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
    if (use_bias_ && bias_.requires_grad()) {
        backend_->column_sums(grad_output.data(), bias_grad_.data(), n, out, 1.0f);
    }

    // grad_X = grad_Y @ W^T
    Tensor grad_input(Shape({N, in_features_}), backend_, weight_.device());
    backend_->gemm_ex(grad_output.data(), false, weight_.data(), true, grad_input.data(), n, out, in, 0.0f);
    if (lora_active()) lora_->backward(grad_output, grad_input);
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
    const auto n = static_cast<size_t>(N), in = static_cast<size_t>(in_features_), out = static_cast<size_t>(out_features_);
    if (!lora_active()) {
        if (lora_) lora_->clear_relevance();
        return lrp_rules::linear(backend_, weight_.device(), last_input_, weight_, bias_, last_pre_bias_output_, relevance_out,
                               n, in, out, config);
    }
    const Tensor update = lora_->update();
    if (config.lora_relevance == LoRARelevance::Merged) {
        Tensor adapted(weight_.shape(), backend_, weight_.device());
        backend_->add(weight_.data(), update.data(), adapted.data(), in * out);
        lora_->clear_relevance();
        return lrp_rules::linear(backend_, weight_.device(), last_input_, adapted, bias_, last_pre_bias_output_, relevance_out,
                               n, in, out, config);
    }
    // TRN-8: the base and adapter paths as one Linear layer on [x, x] with the weights [W; (alpha / r) A B]
    // stacked, so every rule's denominator sees both paths, then the input relevance split in two.
    const DeviceType device = weight_.device();
    Tensor doubled(Shape({N, 2 * in_features_}), backend_, device);
    backend_->copy_2d(doubled.data(), 2 * in, last_input_.data(), in, n, in);
    backend_->copy_2d(doubled.data() + in, 2 * in, last_input_.data(), in, n, in);
    Tensor stacked(Shape({2 * in_features_, out_features_}), backend_, device);
    backend_->copy(stacked.data(), weight_.data(), in * out * sizeof(float), CopyDirection::DeviceToDevice);
    backend_->copy(stacked.data() + in * out, update.data(), in * out * sizeof(float), CopyDirection::DeviceToDevice);
    const Tensor both = lrp_rules::linear(backend_, device, doubled, stacked, bias_, last_pre_bias_output_, relevance_out, n,
                                        2 * in, out, config);
    Tensor base(Shape({N, in_features_}), backend_, device), adapter(Shape({N, in_features_}), backend_, device);
    backend_->copy_2d(base.data(), in, both.data(), 2 * in, n, in);
    backend_->copy_2d(adapter.data(), in, both.data() + in, 2 * in, n, in);
    Tensor relevance_in(Shape({N, in_features_}), backend_, device);
    backend_->add(base.data(), adapter.data(), relevance_in.data(), n * in);
    lora_->last_base_relevance_ = std::move(base);
    lora_->last_adapter_relevance_ = std::move(adapter);
    return relevance_in;
}

void LinearModule::release_activations() {
    release_tensor(last_input_);
    release_tensor(last_pre_bias_output_);
    if (lora_) lora_->release_activations();
    has_forwarded_ = false;
}

}  // namespace pulsatrix
