#include "pulsatrix/embedding_module.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

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
    if (input.rank() != 2) {
        throw std::invalid_argument("EmbeddingModule::forward: input must be rank-2 (N, L)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t count = N * L;

    last_input_shape_ = input.shape();

    // Index validation can throw per element, so it runs on the host: the N*L index values are
    // read back (a small transfer next to the N*L*embedding_dim gather) and the validated,
    // rounded indices are uploaded for the device kernels (GPU-native-kernels Mission 2).
    std::vector<float> raw(static_cast<size_t>(count));
    input.backend()->copy(raw.data(), input.data(), raw.size() * sizeof(float),
                          input.device() == DeviceType::Cpu ? CopyDirection::HostToHost : CopyDirection::DeviceToHost);
    std::vector<float> indices(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        int64_t idx = static_cast<int64_t>(std::llround(static_cast<double>(raw[static_cast<size_t>(i)])));
        if (idx < 0 || idx >= num_embeddings_) {
            throw std::invalid_argument("EmbeddingModule::forward: index out of range [0, num_embeddings)");
        }
        indices[static_cast<size_t>(i)] = static_cast<float>(idx);
    }
    last_indices_ = Tensor(Shape({count}), backend_, indices, weight_.device());

    Tensor output(Shape({N, L, embedding_dim_}), backend_, weight_.device());
    backend_->gather_rows(weight_.data(), last_indices_.data(), output.data(), static_cast<size_t>(count),
                          static_cast<size_t>(embedding_dim_));

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

    // Scatter-add into a zeroed local, then accumulate -- the original association. The
    // scatter walks tokens in order on every backend (deterministic; no atomics).
    Tensor local_weight_grad(weight_.shape(), backend_, weight_.device());
    local_weight_grad.fill(0.0f);
    backend_->scatter_add_rows(grad_output.data(), last_indices_.data(), local_weight_grad.data(),
                               static_cast<size_t>(N * L), static_cast<size_t>(embedding_dim_));
    weight_grad_.accumulate(local_weight_grad);

    // Indices are not differentiable: the input gradient is zero by definition.
    Tensor grad_input(last_input_shape_, backend_, weight_.device());
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
    PULSATRIX_REQUIRE_HOST(relevance_out);

    const int64_t count = N * L;
    Tensor relevance_in(last_input_shape_, backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(relevance_in);

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
