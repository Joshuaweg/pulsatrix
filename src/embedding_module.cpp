#include "pulsatrix/embedding_module.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

EmbeddingModule::EmbeddingModule(int64_t num_embeddings, int64_t embedding_dim, DeviceBackend* backend)
    : num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim),
      backend_(backend),
      weight_(Shape({num_embeddings > 0 ? num_embeddings : 1, embedding_dim > 0 ? embedding_dim : 1}), backend),
      weight_grad_(Shape({num_embeddings > 0 ? num_embeddings : 1, embedding_dim > 0 ? embedding_dim : 1}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (num_embeddings <= 0) {
        throw std::invalid_argument("EmbeddingModule: num_embeddings must be positive");
    }
    if (embedding_dim <= 0) {
        throw std::invalid_argument("EmbeddingModule: embedding_dim must be positive");
    }
}

void EmbeddingModule::set_weight(std::initializer_list<float> values) {
    weight_ = Tensor(weight_.shape(), backend_, values);
}

void EmbeddingModule::set_weight(const std::vector<float>& values) {
    weight_ = Tensor(weight_.shape(), backend_, values);
}

Tensor EmbeddingModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    PULSATRIX_ASSERT(input.device() == DeviceType::Cpu);

    if (input.rank() != 2) {
        throw std::invalid_argument("EmbeddingModule::forward: input must be rank-2 (N, L)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t count = N * L;

    last_input_shape_ = input.shape();
    last_indices_.assign(static_cast<size_t>(count), 0);

    for (int64_t i = 0; i < count; ++i) {
        // Round-to-nearest, not truncation -- see the float-indices design decision.
        int64_t idx = static_cast<int64_t>(std::llround(static_cast<double>(input.data()[i])));
        if (idx < 0 || idx >= num_embeddings_) {
            throw std::invalid_argument("EmbeddingModule::forward: index out of range [0, num_embeddings)");
        }
        last_indices_[static_cast<size_t>(i)] = idx;
    }

    Tensor output(Shape({N, L, embedding_dim_}), backend_);
    for (int64_t i = 0; i < count; ++i) {
        const float* row = weight_.data() + last_indices_[static_cast<size_t>(i)] * embedding_dim_;
        float* out_row = output.data() + i * embedding_dim_;
        for (int64_t d = 0; d < embedding_dim_; ++d) {
            out_row[d] = row[d];
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor EmbeddingModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("EmbeddingModule::backward: called before any forward()");
    }
    const int64_t N = last_input_shape_.dim(0);
    const int64_t L = last_input_shape_.dim(1);
    if (grad_output.rank() != 3 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != L ||
        grad_output.shape().dim(2) != embedding_dim_) {
        throw std::invalid_argument(
            "EmbeddingModule::backward: grad_output must be (N, L, embedding_dim) matching the cached forward "
            "shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu);

    const int64_t count = N * L;
    Tensor local_weight_grad(weight_.shape(), backend_);
    local_weight_grad.fill(0.0f);

    // Scatter-add: multiple (n,l) positions can reference the same row -- each
    // contributes additively, not via overwrite. First module in this codebase needing
    // this pattern (LinearModule/Conv2DModule's gradients are dense-matmul sums).
    for (int64_t i = 0; i < count; ++i) {
        int64_t idx = last_indices_[static_cast<size_t>(i)];
        float* grad_row = local_weight_grad.data() + idx * embedding_dim_;
        const float* out_row = grad_output.data() + i * embedding_dim_;
        for (int64_t d = 0; d < embedding_dim_; ++d) {
            grad_row[d] += out_row[d];
        }
    }
    weight_grad_.accumulate(local_weight_grad);

    // Gradient w.r.t. discrete indices is undefined -- always zero, matching every
    // mainstream framework's nn.Embedding behavior.
    Tensor grad_input(last_input_shape_, backend_);
    grad_input.fill(0.0f);
    return grad_input;
}

Tensor EmbeddingModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) {
    if (!has_forwarded_) {
        throw std::logic_error("EmbeddingModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_shape_.dim(0);
    const int64_t L = last_input_shape_.dim(1);
    if (relevance_out.rank() != 3 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != L ||
        relevance_out.shape().dim(2) != embedding_dim_) {
        throw std::invalid_argument(
            "EmbeddingModule::propagate_relevance: relevance_out must be (N, L, embedding_dim) matching the "
            "cached forward shape");
    }
    PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu);

    const int64_t count = N * L;
    Tensor relevance_in(last_input_shape_, backend_);

    for (int64_t i = 0; i < count; ++i) {
        const float* row = relevance_out.data() + i * embedding_dim_;
        float sum = 0.0f;
        for (int64_t d = 0; d < embedding_dim_; ++d) {
            sum += row[d];
        }
        relevance_in.data()[i] = sum;
    }

    return relevance_in;
}

}  // namespace pulsatrix
