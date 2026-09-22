#include "exai/rnn_module.hpp"

#include <cmath>
#include <stdexcept>

#include "exai/assert.hpp"

namespace exai {

namespace {
// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer --
// same helper shape as LinearModule's/Conv2DModule's own transpose() (CPUBackend::gemm
// has no transpose flag).
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

RNNModule::RNNModule(int64_t input_size, int64_t hidden_size, DeviceBackend* backend)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      backend_(backend),
      weight_xh_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_hh_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      weight_xh_grad_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_hh_grad_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_grad_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      last_input_(Shape({0}), backend),
      last_hidden_states_(Shape({0}), backend),
      last_pre_activation_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (input_size <= 0) {
        throw std::invalid_argument("RNNModule: input_size must be positive");
    }
    if (hidden_size <= 0) {
        throw std::invalid_argument("RNNModule: hidden_size must be positive");
    }
}

void RNNModule::set_weight_xh(std::initializer_list<float> values) {
    weight_xh_ = Tensor(weight_xh_.shape(), backend_, values);
}

void RNNModule::set_weight_hh(std::initializer_list<float> values) {
    weight_hh_ = Tensor(weight_hh_.shape(), backend_, values);
}

void RNNModule::set_bias(std::initializer_list<float> values) {
    bias_ = Tensor(bias_.shape(), backend_, values);
}

Tensor RNNModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    EXAI_ASSERT(input.device() == DeviceType::Cpu);

    if (input.rank() != 3 || input.shape().dim(2) != input_size_) {
        throw std::invalid_argument("RNNModule::forward: input must be rank-3 (N, L, input_size)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);

    last_input_ = input;
    last_L_ = L;
    last_hidden_states_ = Tensor(Shape({N, L + 1, hidden_size_}), backend_);
    last_hidden_states_.fill(0.0f);  // h_0 = 0 for every example
    last_pre_activation_ = Tensor(Shape({N, L, hidden_size_}), backend_);

    Tensor output(Shape({N, L, hidden_size_}), backend_);

    for (int64_t t = 0; t < L; ++t) {
        Tensor x_t(Shape({N, input_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                x_t.data()[n * input_size_ + i] = input.data()[(n * L + t) * input_size_ + i];
            }
        }
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                h_prev.data()[n * hidden_size_ + k] = last_hidden_states_.data()[(n * (L + 1) + t) * hidden_size_ + k];
            }
        }

        Tensor z1(Shape({N, hidden_size_}), backend_);
        backend_->gemm(x_t.data(), weight_xh_.data(), z1.data(), static_cast<size_t>(N),
                        static_cast<size_t>(input_size_), static_cast<size_t>(hidden_size_));
        Tensor z2(Shape({N, hidden_size_}), backend_);
        backend_->gemm(h_prev.data(), weight_hh_.data(), z2.data(), static_cast<size_t>(N),
                        static_cast<size_t>(hidden_size_), static_cast<size_t>(hidden_size_));

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                int64_t idx = n * hidden_size_ + k;
                // z_no_bias is what the LRP epsilon-rule denominator uses (cached below,
                // excluding bias -- same convention as LinearModule/Conv2DModule: bias has
                // no associated input feature to redistribute relevance to, so it's
                // excluded from z entirely, which is what makes conservation exact rather
                // than merely approximate). The actual tanh activation still needs the
                // real bias-included pre-activation.
                float z_no_bias = z1.data()[idx] + z2.data()[idx];
                last_pre_activation_.data()[(n * L + t) * hidden_size_ + k] = z_no_bias;
                float h = std::tanh(z_no_bias + bias_.data()[k]);
                last_hidden_states_.data()[(n * (L + 1) + t + 1) * hidden_size_ + k] = h;
                output.data()[(n * L + t) * hidden_size_ + k] = h;
            }
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor RNNModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("RNNModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    if (grad_output.rank() != 3 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != L ||
        grad_output.shape().dim(2) != hidden_size_) {
        throw std::invalid_argument(
            "RNNModule::backward: grad_output must be (N, L, hidden_size) matching the cached forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    Tensor grad_input(last_input_.shape(), backend_);
    grad_input.fill(0.0f);
    Tensor local_wxh_grad(weight_xh_.shape(), backend_);
    local_wxh_grad.fill(0.0f);
    Tensor local_whh_grad(weight_hh_.shape(), backend_);
    local_whh_grad.fill(0.0f);
    Tensor local_bh_grad(bias_.shape(), backend_);
    local_bh_grad.fill(0.0f);

    Tensor dh_next(Shape({N, hidden_size_}), backend_);
    dh_next.fill(0.0f);

    for (int64_t t = L - 1; t >= 0; --t) {
        Tensor dh_t(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                int64_t idx = n * hidden_size_ + k;
                dh_t.data()[idx] = grad_output.data()[(n * L + t) * hidden_size_ + k] + dh_next.data()[idx];
            }
        }

        Tensor dz(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                int64_t idx = n * hidden_size_ + k;
                float h = last_hidden_states_.data()[(n * (L + 1) + t + 1) * hidden_size_ + k];
                dz.data()[idx] = dh_t.data()[idx] * (1.0f - h * h);
            }
        }

        Tensor x_t(Shape({N, input_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                x_t.data()[n * input_size_ + i] = last_input_.data()[(n * L + t) * input_size_ + i];
            }
        }
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                h_prev.data()[n * hidden_size_ + k] =
                    last_hidden_states_.data()[(n * (L + 1) + t) * hidden_size_ + k];
            }
        }

        // grad_Wxh += x_t^T @ dz
        Tensor x_t_T = transpose(x_t, N, input_size_, backend_);
        Tensor gwxh(weight_xh_.shape(), backend_);
        backend_->gemm(x_t_T.data(), dz.data(), gwxh.data(), static_cast<size_t>(input_size_),
                        static_cast<size_t>(N), static_cast<size_t>(hidden_size_));
        local_wxh_grad.accumulate(gwxh);

        // grad_Whh += h_prev^T @ dz
        Tensor h_prev_T = transpose(h_prev, N, hidden_size_, backend_);
        Tensor gwhh(weight_hh_.shape(), backend_);
        backend_->gemm(h_prev_T.data(), dz.data(), gwhh.data(), static_cast<size_t>(hidden_size_),
                        static_cast<size_t>(N), static_cast<size_t>(hidden_size_));
        local_whh_grad.accumulate(gwhh);

        // grad_bh += sum over batch of dz
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                local_bh_grad.data()[k] += dz.data()[n * hidden_size_ + k];
            }
        }

        // grad_x_t = dz @ Wxh^T
        Tensor wxh_T = transpose(weight_xh_, input_size_, hidden_size_, backend_);
        Tensor gx(Shape({N, input_size_}), backend_);
        backend_->gemm(dz.data(), wxh_T.data(), gx.data(), static_cast<size_t>(N), static_cast<size_t>(hidden_size_),
                        static_cast<size_t>(input_size_));
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                grad_input.data()[(n * L + t) * input_size_ + i] = gx.data()[n * input_size_ + i];
            }
        }

        // dh_next (for t-1) = dz @ Whh^T
        Tensor whh_T = transpose(weight_hh_, hidden_size_, hidden_size_, backend_);
        Tensor dh_prev(Shape({N, hidden_size_}), backend_);
        backend_->gemm(dz.data(), whh_T.data(), dh_prev.data(), static_cast<size_t>(N),
                        static_cast<size_t>(hidden_size_), static_cast<size_t>(hidden_size_));
        dh_next = dh_prev;
    }

    weight_xh_grad_.accumulate(local_wxh_grad);
    weight_hh_grad_.accumulate(local_whh_grad);
    bias_grad_.accumulate(local_bh_grad);

    return grad_input;
}

Tensor RNNModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("RNNModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    if (relevance_out.rank() != 3 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != L ||
        relevance_out.shape().dim(2) != hidden_size_) {
        throw std::invalid_argument(
            "RNNModule::propagate_relevance: relevance_out must be (N, L, hidden_size) matching the cached "
            "forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    EXAI_ASSERT(relevance_out.device() == DeviceType::Cpu);

    Tensor relevance_in(last_input_.shape(), backend_);
    relevance_in.fill(0.0f);

    Tensor R_h_next(Shape({N, hidden_size_}), backend_);
    R_h_next.fill(0.0f);

    for (int64_t t = L - 1; t >= 0; --t) {
        Tensor R_z(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                int64_t idx = n * hidden_size_ + k;
                R_z.data()[idx] = relevance_out.data()[(n * L + t) * hidden_size_ + k] + R_h_next.data()[idx];
            }
        }

        Tensor x_t(Shape({N, input_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                x_t.data()[n * input_size_ + i] = last_input_.data()[(n * L + t) * input_size_ + i];
            }
        }
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                h_prev.data()[n * hidden_size_ + k] =
                    last_hidden_states_.data()[(n * (L + 1) + t) * hidden_size_ + k];
            }
        }

        Tensor R_x(Shape({N, input_size_}), backend_);
        R_x.fill(0.0f);
        Tensor R_hprev(Shape({N, hidden_size_}), backend_);
        R_hprev.fill(0.0f);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                float z = last_pre_activation_.data()[(n * L + t) * hidden_size_ + k];
                float sign = (z >= 0.0f) ? 1.0f : -1.0f;
                float denom = z + config.epsilon * sign;
                float r = R_z.data()[n * hidden_size_ + k];

                for (int64_t i = 0; i < input_size_; ++i) {
                    float w = weight_xh_.data()[i * hidden_size_ + k];
                    float x = x_t.data()[n * input_size_ + i];
                    R_x.data()[n * input_size_ + i] += (x * w / denom) * r;
                }
                for (int64_t kk = 0; kk < hidden_size_; ++kk) {
                    float w2 = weight_hh_.data()[kk * hidden_size_ + k];
                    float hp = h_prev.data()[n * hidden_size_ + kk];
                    R_hprev.data()[n * hidden_size_ + kk] += (hp * w2 / denom) * r;
                }
            }
        }

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                relevance_in.data()[(n * L + t) * input_size_ + i] = R_x.data()[n * input_size_ + i];
            }
        }
        R_h_next = R_hprev;
    }

    return relevance_in;
}

}  // namespace exai
