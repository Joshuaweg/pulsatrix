#include "pulsatrix/mamba_module.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {
// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer --
// same helper shape as LinearModule's/Conv2DModule's/RNNModule's/LSTMModule's/GRUModule's
// own transpose() (CPUBackend::gemm has no transpose flag).
Tensor transpose(const Tensor& m, int64_t rows, int64_t cols, DeviceBackend* backend) {
    Tensor out(Shape({cols, rows}), backend);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.data()[c * rows + r] = m.data()[r * cols + c];
        }
    }
    return out;
}

// softplus(z) = log(1 + exp(z)), in the numerically stable branch form: for large z the
// naive exp(z) overflows while log1p(exp(z)) -> z to within float precision. No
// DeviceBackend::elementwise op exists for this (first consumer -- see the class note),
// so it is a raw host function, PULSATRIX_ASSERT-guarded at every entry point that calls it.
float softplus(float z) {
    return (z > 20.0f) ? z : std::log1p(std::exp(z));
}

// d(softplus)/dz = sigmoid(z). Computed from the pre-activation rather than from the
// cached forward output, because softplus is not invertible cheaply the way tanh's
// 1 - h^2 identity is.
float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

// Epsilon-stabilized LRP denominator, sign-preserving -- the exact same shape every other
// module here uses (RNNModule::propagate_relevance included).
float stabilize(float value, float epsilon) {
    return value + epsilon * ((value >= 0.0f) ? 1.0f : -1.0f);
}
}  // namespace

MambaModule::MambaModule(int64_t d_model, int64_t state_size, DeviceBackend* backend)
    : d_model_(d_model),
      state_size_(state_size),
      backend_(backend),
      w_delta_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      bias_delta_(Shape({d_model > 0 ? d_model : 1}), backend),
      w_b_(Shape({d_model > 0 ? d_model : 1, state_size > 0 ? state_size : 1}), backend),
      w_c_(Shape({d_model > 0 ? d_model : 1, state_size > 0 ? state_size : 1}), backend),
      a_(Shape({d_model > 0 ? d_model : 1, state_size > 0 ? state_size : 1}), backend),
      d_(Shape({d_model > 0 ? d_model : 1}), backend),
      w_delta_grad_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      bias_delta_grad_(Shape({d_model > 0 ? d_model : 1}), backend),
      w_b_grad_(Shape({d_model > 0 ? d_model : 1, state_size > 0 ? state_size : 1}), backend),
      w_c_grad_(Shape({d_model > 0 ? d_model : 1, state_size > 0 ? state_size : 1}), backend),
      a_grad_(Shape({d_model > 0 ? d_model : 1, state_size > 0 ? state_size : 1}), backend),
      d_grad_(Shape({d_model > 0 ? d_model : 1}), backend),
      last_input_(Shape({0}), backend),
      last_states_(Shape({0}), backend),
      last_abar_(Shape({0}), backend),
      last_bbar_(Shape({0}), backend),
      last_z_delta_(Shape({0}), backend),
      last_delta_(Shape({0}), backend),
      last_b_(Shape({0}), backend),
      last_c_(Shape({0}), backend),
      last_output_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from the Python bindings
    // with no upstream validation).
    if (d_model <= 0) {
        throw std::invalid_argument("MambaModule: d_model must be positive");
    }
    if (state_size <= 0) {
        throw std::invalid_argument("MambaModule: state_size must be positive");
    }
}

void MambaModule::set_W_delta(std::initializer_list<float> values) {
    w_delta_ = Tensor(w_delta_.shape(), backend_, values);
}
void MambaModule::set_bias_delta(std::initializer_list<float> values) {
    bias_delta_ = Tensor(bias_delta_.shape(), backend_, values);
}
void MambaModule::set_W_B(std::initializer_list<float> values) {
    w_b_ = Tensor(w_b_.shape(), backend_, values);
}
void MambaModule::set_W_C(std::initializer_list<float> values) {
    w_c_ = Tensor(w_c_.shape(), backend_, values);
}
void MambaModule::set_A(std::initializer_list<float> values) {
    a_ = Tensor(a_.shape(), backend_, values);
}
void MambaModule::set_D(std::initializer_list<float> values) {
    d_ = Tensor(d_.shape(), backend_, values);
}

void MambaModule::set_W_delta(const std::vector<float>& values) {
    w_delta_ = Tensor(w_delta_.shape(), backend_, values);
}
void MambaModule::set_bias_delta(const std::vector<float>& values) {
    bias_delta_ = Tensor(bias_delta_.shape(), backend_, values);
}
void MambaModule::set_W_B(const std::vector<float>& values) {
    w_b_ = Tensor(w_b_.shape(), backend_, values);
}
void MambaModule::set_W_C(const std::vector<float>& values) {
    w_c_ = Tensor(w_c_.shape(), backend_, values);
}
void MambaModule::set_A(const std::vector<float>& values) {
    a_ = Tensor(a_.shape(), backend_, values);
}
void MambaModule::set_D(const std::vector<float>& values) {
    d_ = Tensor(d_.shape(), backend_, values);
}

Tensor MambaModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly, and computes softplus/exp in raw host loops
    // (no DeviceBackend primitive exists for either) -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(input);

    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("MambaModule::forward: input must be rank-3 (N, L, d_model)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t D = d_model_;
    const int64_t S = state_size_;

    last_input_ = input;
    last_L_ = L;
    last_states_ = Tensor(Shape({N, L + 1, D, S}), backend_);  // zero-filled: h_0 = 0
    last_abar_ = Tensor(Shape({N, L, D, S}), backend_);
    last_bbar_ = Tensor(Shape({N, L, D, S}), backend_);
    last_z_delta_ = Tensor(Shape({N, L, D}), backend_);
    last_delta_ = Tensor(Shape({N, L, D}), backend_);
    last_b_ = Tensor(Shape({N, L, S}), backend_);
    last_c_ = Tensor(Shape({N, L, S}), backend_);

    Tensor output(Shape({N, L, D}), backend_);

    for (int64_t t = 0; t < L; ++t) {
        // x_t, gathered into a contiguous (N, d_model) buffer so the three selective
        // projections can go through backend_->gemm rather than hand-rolled loops.
        Tensor x_t(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                x_t.data()[b * D + e] = input.data()[(b * L + t) * D + e];
            }
        }

        Tensor z_delta(Shape({N, D}), backend_);
        backend_->gemm(x_t.data(), w_delta_.data(), z_delta.data(), static_cast<size_t>(N),
                       static_cast<size_t>(D), static_cast<size_t>(D));
        Tensor b_proj(Shape({N, S}), backend_);
        backend_->gemm(x_t.data(), w_b_.data(), b_proj.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(S));
        Tensor c_proj(Shape({N, S}), backend_);
        backend_->gemm(x_t.data(), w_c_.data(), c_proj.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(S));

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                // Delta_t's projection is the one selective projection WITH a bias.
                float z = z_delta.data()[b * D + d] + bias_delta_.data()[d];
                last_z_delta_.data()[(b * L + t) * D + d] = z;
                last_delta_.data()[(b * L + t) * D + d] = softplus(z);
            }
            for (int64_t s = 0; s < S; ++s) {
                last_b_.data()[(b * L + t) * S + s] = b_proj.data()[b * S + s];
                last_c_.data()[(b * L + t) * S + s] = c_proj.data()[b * S + s];
            }
        }

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const float delta = last_delta_.data()[(b * L + t) * D + d];
                const float x = x_t.data()[b * D + d];
                float y = d_.data()[d] * x;  // the D skip/feedthrough path
                for (int64_t s = 0; s < S; ++s) {
                    const int64_t scan_idx = ((b * L + t) * D + d) * S + s;
                    const float abar = std::exp(delta * a_.data()[d * S + s]);
                    const float bbar = delta * last_b_.data()[(b * L + t) * S + s];
                    last_abar_.data()[scan_idx] = abar;
                    last_bbar_.data()[scan_idx] = bbar;

                    const float h_prev = last_states_.data()[((b * (L + 1) + t) * D + d) * S + s];
                    const float h = abar * h_prev + bbar * x;
                    last_states_.data()[((b * (L + 1) + t + 1) * D + d) * S + s] = h;
                    y += last_c_.data()[(b * L + t) * S + s] * h;
                }
                output.data()[(b * L + t) * D + d] = y;
            }
        }
    }

    // y_t is propagate_relevance's step-1 denominator, so the output is cached in its own
    // right rather than recomputed.
    last_output_ = output;
    has_forwarded_ = true;
    return output;
}

Tensor MambaModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("MambaModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    const int64_t D = d_model_;
    const int64_t S = state_size_;
    if (grad_output.rank() != 3 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != L ||
        grad_output.shape().dim(2) != D) {
        throw std::invalid_argument(
            "MambaModule::backward: grad_output must be (N, L, d_model) matching the cached forward shape");
    }
    // Dereferences Tensor::data() directly, and computes exp/sigmoid in raw host loops --
    // not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(grad_output);

    Tensor grad_input(last_input_.shape(), backend_);
    Tensor local_w_delta_grad(w_delta_.shape(), backend_);
    Tensor local_bias_delta_grad(bias_delta_.shape(), backend_);
    Tensor local_w_b_grad(w_b_.shape(), backend_);
    Tensor local_w_c_grad(w_c_.shape(), backend_);
    Tensor local_a_grad(a_.shape(), backend_);
    Tensor local_d_grad(d_.shape(), backend_);

    // The carried state-gradient accumulator, dh_carry[b,d,n], threaded from t+1 back to t.
    Tensor dh_carry(Shape({N, D, S}), backend_);

    for (int64_t t = L - 1; t >= 0; --t) {
        Tensor x_t(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                x_t.data()[b * D + e] = last_input_.data()[(b * L + t) * D + e];
            }
        }

        Tensor dx_t(Shape({N, D}), backend_);
        Tensor d_b(Shape({N, S}), backend_);
        Tensor d_c(Shape({N, S}), backend_);
        Tensor d_delta(Shape({N, D}), backend_);
        Tensor dh_carry_next(Shape({N, D, S}), backend_);

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const float gy = grad_output.data()[(b * L + t) * D + d];
                const float x = x_t.data()[b * D + d];
                const float delta = last_delta_.data()[(b * L + t) * D + d];

                // Step 1: y_t = sum_n(C_t*h_t) + D*x_t.
                local_d_grad.data()[d] += gy * x;
                dx_t.data()[b * D + d] += gy * d_.data()[d];

                for (int64_t s = 0; s < S; ++s) {
                    const int64_t scan_idx = ((b * L + t) * D + d) * S + s;
                    const float h_t = last_states_.data()[((b * (L + 1) + t + 1) * D + d) * S + s];
                    const float h_prev = last_states_.data()[((b * (L + 1) + t) * D + d) * S + s];
                    const float abar = last_abar_.data()[scan_idx];
                    const float bbar = last_bbar_.data()[scan_idx];
                    const float c_val = last_c_.data()[(b * L + t) * S + s];
                    const float b_val = last_b_.data()[(b * L + t) * S + s];

                    d_c.data()[b * S + s] += gy * h_t;
                    const float dh_total = gy * c_val + dh_carry.data()[(b * D + d) * S + s];

                    // Step 2: h_t = Abar_t*h_{t-1} + Bbar_t*x_t.
                    dh_carry_next.data()[(b * D + d) * S + s] = dh_total * abar;
                    const float d_abar = dh_total * h_prev;
                    const float d_bbar = dh_total * x;
                    dx_t.data()[b * D + d] += dh_total * bbar;

                    // Step 3: Bbar_t = Delta_t*B_t.
                    d_delta.data()[b * D + d] += d_bbar * b_val;
                    d_b.data()[b * S + s] += d_bbar * delta;

                    // Step 4: Abar_t = exp(Delta_t*A), so d(Abar)/d(Delta) = Abar*A and
                    // d(Abar)/d(A) = Abar*Delta.
                    d_delta.data()[b * D + d] += d_abar * abar * a_.data()[d * S + s];
                    local_a_grad.data()[d * S + s] += d_abar * abar * delta;
                }
            }
        }

        // Step 5: Delta_t = softplus(z_delta_t), d(softplus)/dz = sigmoid(z).
        Tensor dz_delta(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                dz_delta.data()[b * D + d] =
                    d_delta.data()[b * D + d] * sigmoid(last_z_delta_.data()[(b * L + t) * D + d]);
            }
        }

        // Step 6: the three linear projections' parameter gradients and their shares of dx_t.
        Tensor x_t_T = transpose(x_t, N, D, backend_);

        Tensor gw_delta(w_delta_.shape(), backend_);
        backend_->gemm(x_t_T.data(), dz_delta.data(), gw_delta.data(), static_cast<size_t>(D),
                       static_cast<size_t>(N), static_cast<size_t>(D));
        local_w_delta_grad.accumulate(gw_delta);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                local_bias_delta_grad.data()[d] += dz_delta.data()[b * D + d];
            }
        }

        Tensor gw_b(w_b_.shape(), backend_);
        backend_->gemm(x_t_T.data(), d_b.data(), gw_b.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(S));
        local_w_b_grad.accumulate(gw_b);

        Tensor gw_c(w_c_.shape(), backend_);
        backend_->gemm(x_t_T.data(), d_c.data(), gw_c.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(S));
        local_w_c_grad.accumulate(gw_c);

        Tensor w_delta_T = transpose(w_delta_, D, D, backend_);
        Tensor gx_delta(Shape({N, D}), backend_);
        backend_->gemm(dz_delta.data(), w_delta_T.data(), gx_delta.data(), static_cast<size_t>(N),
                       static_cast<size_t>(D), static_cast<size_t>(D));
        dx_t.accumulate(gx_delta);

        Tensor w_b_T = transpose(w_b_, D, S, backend_);
        Tensor gx_b(Shape({N, D}), backend_);
        backend_->gemm(d_b.data(), w_b_T.data(), gx_b.data(), static_cast<size_t>(N), static_cast<size_t>(S),
                       static_cast<size_t>(D));
        dx_t.accumulate(gx_b);

        Tensor w_c_T = transpose(w_c_, D, S, backend_);
        Tensor gx_c(Shape({N, D}), backend_);
        backend_->gemm(d_c.data(), w_c_T.data(), gx_c.data(), static_cast<size_t>(N), static_cast<size_t>(S),
                       static_cast<size_t>(D));
        dx_t.accumulate(gx_c);

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                grad_input.data()[(b * L + t) * D + e] = dx_t.data()[b * D + e];
            }
        }
        dh_carry = dh_carry_next;
    }

    w_delta_grad_.accumulate(local_w_delta_grad);
    bias_delta_grad_.accumulate(local_bias_delta_grad);
    w_b_grad_.accumulate(local_w_b_grad);
    w_c_grad_.accumulate(local_w_c_grad);
    a_grad_.accumulate(local_a_grad);
    d_grad_.accumulate(local_d_grad);

    return grad_input;
}

Tensor MambaModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("MambaModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    const int64_t D = d_model_;
    const int64_t S = state_size_;
    if (relevance_out.rank() != 3 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != L ||
        relevance_out.shape().dim(2) != D) {
        throw std::invalid_argument(
            "MambaModule::propagate_relevance: relevance_out must be (N, L, d_model) matching the cached "
            "forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(relevance_out);

    Tensor relevance_in(last_input_.shape(), backend_);

    // The carried state-relevance accumulator, R(h_t)[b,d,n] arriving from step t+1. Zero
    // at t = L-1, and (by h_0 == 0) it carries nothing out past t = 0.
    Tensor r_h_carry(Shape({N, D, S}), backend_);

    // NOTE (MambaLRP, and grep-confirmed in this mission's close-out): the selective
    // projections' weights -- w_delta_, bias_delta_, w_b_, w_c_ -- are deliberately NOT
    // referenced anywhere below. Delta_t/B_t/C_t are pure conductors, consumed only through
    // the cached, *detached* Abar_t/Bbar_t/C_t forward values. Only a_ (via Abar_t) and d_
    // (the skip path, a genuine weighted connection from x_t to y_t) participate.
    for (int64_t t = L - 1; t >= 0; --t) {
        Tensor r_x(Shape({N, D}), backend_);
        Tensor r_h_prev(Shape({N, D, S}), backend_);

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const float x = last_input_.data()[(b * L + t) * D + d];
                const float y = last_output_.data()[(b * L + t) * D + d];
                const float denom_y = stabilize(y, config.epsilon);
                const float r_y = relevance_out.data()[(b * L + t) * D + d];

                // Step 1: y_t = sum_n(C_t*h_t) + D*x_t -- an (state_size + 1)-way weighted
                // sum sharing one output. The D skip term's share lands straight on R(x_t).
                r_x.data()[b * D + d] += (d_.data()[d] * x / denom_y) * r_y;

                for (int64_t s = 0; s < S; ++s) {
                    const int64_t scan_idx = ((b * L + t) * D + d) * S + s;
                    const float h_t = last_states_.data()[((b * (L + 1) + t + 1) * D + d) * S + s];
                    const float h_prev = last_states_.data()[((b * (L + 1) + t) * D + d) * S + s];
                    const float abar = last_abar_.data()[scan_idx];
                    const float bbar = last_bbar_.data()[scan_idx];
                    const float c_val = last_c_.data()[(b * L + t) * S + s];

                    // R(h_t) = this step's share of R(y_t), plus whatever step t+1 carried
                    // back into this same state element.
                    const float r_h = (c_val * h_t / denom_y) * r_y + r_h_carry.data()[(b * D + d) * S + s];

                    // Step 2: h_t = Abar_t*h_{t-1} + Bbar_t*x_t -- the two-weighted-source
                    // epsilon/z-rule, denominator h_t itself.
                    const float denom_h = stabilize(h_t, config.epsilon);
                    r_h_prev.data()[(b * D + d) * S + s] = (abar * h_prev / denom_h) * r_h;
                    r_x.data()[b * D + d] += (bbar * x / denom_h) * r_h;
                }
            }
        }

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                relevance_in.data()[(b * L + t) * D + e] = r_x.data()[b * D + e];
            }
        }
        r_h_carry = r_h_prev;
    }

    return relevance_in;
}

}  // namespace pulsatrix
