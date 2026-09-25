#include "pulsatrix/retnet_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {
// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer --
// same helper shape as LinearModule's/RNNModule's/MambaModule's/RWKVModule's own transpose()
// (CPUBackend::gemm has no transpose flag).
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

RetNetModule::RetNetModule(int64_t d_model, int64_t key_dim, float gamma, DeviceBackend* backend)
    : d_model_(d_model),
      key_dim_(key_dim),
      gamma_(gamma),
      backend_(backend),
      // The ternaries keep Shape construction legal while the invalid-argument checks below
      // are what actually reject the bad arguments -- same pattern as MambaModule/RWKVModule.
      w_q_(Shape({d_model > 0 ? d_model : 1, key_dim > 0 ? key_dim : 1}), backend),
      w_k_(Shape({d_model > 0 ? d_model : 1, key_dim > 0 ? key_dim : 1}), backend),
      w_v_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_q_grad_(Shape({d_model > 0 ? d_model : 1, key_dim > 0 ? key_dim : 1}), backend),
      w_k_grad_(Shape({d_model > 0 ? d_model : 1, key_dim > 0 ? key_dim : 1}), backend),
      w_v_grad_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      last_input_(Shape({0}), backend),
      last_q_(Shape({0}), backend),
      last_k_(Shape({0}), backend),
      last_v_(Shape({0}), backend),
      last_states_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from the Python bindings
    // with no upstream validation). gamma is deliberately unconstrained -- see the header.
    if (d_model <= 0) {
        throw std::invalid_argument("RetNetModule: d_model must be positive");
    }
    if (key_dim <= 0) {
        throw std::invalid_argument("RetNetModule: key_dim must be positive");
    }
}

void RetNetModule::set_W_Q(std::initializer_list<float> values) {
    w_q_ = Tensor(w_q_.shape(), backend_, values);
}
void RetNetModule::set_W_K(std::initializer_list<float> values) {
    w_k_ = Tensor(w_k_.shape(), backend_, values);
}
void RetNetModule::set_W_V(std::initializer_list<float> values) {
    w_v_ = Tensor(w_v_.shape(), backend_, values);
}

void RetNetModule::set_W_Q(const std::vector<float>& values) {
    w_q_ = Tensor(w_q_.shape(), backend_, values);
}
void RetNetModule::set_W_K(const std::vector<float>& values) {
    w_k_ = Tensor(w_k_.shape(), backend_, values);
}
void RetNetModule::set_W_V(const std::vector<float>& values) {
    w_v_ = Tensor(w_v_.shape(), backend_, values);
}

Tensor RetNetModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly for the state recurrence -- not yet
    // backend-generic.
    PULSATRIX_ASSERT(input.device() == DeviceType::Cpu);

    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("RetNetModule::forward: input must be rank-3 (N, L, d_model)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t D = d_model_;
    const int64_t Kd = key_dim_;

    last_input_ = input;
    last_L_ = L;
    last_q_ = Tensor(Shape({N, L, Kd}), backend_);
    last_k_ = Tensor(Shape({N, L, Kd}), backend_);
    last_v_ = Tensor(Shape({N, L, D}), backend_);
    last_states_ = Tensor(Shape({N, L + 1, Kd, D}), backend_);  // zero-filled: S_0 = 0

    Tensor output(Shape({N, L, D}), backend_);

    for (int64_t t = 0; t < L; ++t) {
        // Gather this timestep's slice into a contiguous (N, d_model) buffer so all three
        // projections can go through backend_->gemm rather than hand-rolled loops.
        Tensor x_t(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                x_t.data()[b * D + e] = input.data()[(b * L + t) * D + e];
            }
        }

        Tensor q_t(Shape({N, Kd}), backend_);
        backend_->gemm(x_t.data(), w_q_.data(), q_t.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(Kd));
        Tensor k_t(Shape({N, Kd}), backend_);
        backend_->gemm(x_t.data(), w_k_.data(), k_t.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(Kd));
        Tensor v_t(Shape({N, D}), backend_);
        backend_->gemm(x_t.data(), w_v_.data(), v_t.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t i = 0; i < Kd; ++i) {
                last_q_.data()[(b * L + t) * Kd + i] = q_t.data()[b * Kd + i];
                last_k_.data()[(b * L + t) * Kd + i] = k_t.data()[b * Kd + i];
            }
            for (int64_t j = 0; j < D; ++j) {
                last_v_.data()[(b * L + t) * D + j] = v_t.data()[b * D + j];
            }

            // The retention state: the decayed carry plus this token's outer product
            // K_t (x) V_t, then read out through Q_t.
            for (int64_t j = 0; j < D; ++j) {
                float acc = 0.0f;
                for (int64_t i = 0; i < Kd; ++i) {
                    const float s_prev = last_states_.data()[((b * (L + 1) + t) * Kd + i) * D + j];
                    const float s_cur = gamma_ * s_prev + k_t.data()[b * Kd + i] * v_t.data()[b * D + j];
                    last_states_.data()[((b * (L + 1) + t + 1) * Kd + i) * D + j] = s_cur;
                    acc += q_t.data()[b * Kd + i] * s_cur;
                }
                output.data()[(b * L + t) * D + j] = acc;
            }
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor RetNetModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("RetNetModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    const int64_t D = d_model_;
    const int64_t Kd = key_dim_;
    if (grad_output.rank() != 3 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != L ||
        grad_output.shape().dim(2) != D) {
        throw std::invalid_argument(
            "RetNetModule::backward: grad_output must be (N, L, d_model) matching the cached forward shape");
    }
    // Dereferences Tensor::data() directly for the state recurrence -- not yet
    // backend-generic.
    PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu);

    Tensor grad_input(last_input_.shape(), backend_);
    Tensor local_w_q_grad(w_q_.shape(), backend_);
    Tensor local_w_k_grad(w_k_.shape(), backend_);
    Tensor local_w_v_grad(w_v_.shape(), backend_);

    // The carried state-gradient accumulator, threaded from t+1 back to t: the gradient
    // arriving on S_t from step t+1's use of it as its own S_{t-1}. Zero-initialized, which
    // is exactly right at t = L (nothing follows the last step).
    Tensor ds_carry(Shape({N, Kd, D}), backend_);

    Tensor w_q_T = transpose(w_q_, D, Kd, backend_);
    Tensor w_k_T = transpose(w_k_, D, Kd, backend_);
    Tensor w_v_T = transpose(w_v_, D, D, backend_);

    for (int64_t t = L - 1; t >= 0; --t) {
        Tensor dq(Shape({N, Kd}), backend_);
        Tensor dk(Shape({N, Kd}), backend_);
        Tensor dv(Shape({N, D}), backend_);
        Tensor ds_next(Shape({N, Kd, D}), backend_);

        for (int64_t b = 0; b < N; ++b) {
            // Step 1: o_t[b,j] = sum_i Q_t[b,i]*S_t[b,i,j]. The readout hands gradient to
            // Q_t and to S_t; S_t's total gradient is that direct share plus whatever step
            // t+1 already carried back.
            for (int64_t i = 0; i < Kd; ++i) {
                const float q = last_q_.data()[(b * L + t) * Kd + i];
                float dq_acc = 0.0f;
                for (int64_t j = 0; j < D; ++j) {
                    const float g_o = grad_output.data()[(b * L + t) * D + j];
                    const float s_cur = last_states_.data()[((b * (L + 1) + t + 1) * Kd + i) * D + j];
                    dq_acc += g_o * s_cur;
                    ds_next.data()[(b * Kd + i) * D + j] = g_o * q + ds_carry.data()[(b * Kd + i) * D + j];
                }
                dq.data()[b * Kd + i] = dq_acc;
            }

            // Step 2: S_t = gamma*S_{t-1} + K_t (x) V_t. gamma is a fixed hyperparameter,
            // not a Tensor -- no gradient is accumulated for it; it only scales the carry.
            for (int64_t i = 0; i < Kd; ++i) {
                const float k = last_k_.data()[(b * L + t) * Kd + i];
                float dk_acc = 0.0f;
                for (int64_t j = 0; j < D; ++j) {
                    const float ds = ds_next.data()[(b * Kd + i) * D + j];
                    dk_acc += ds * last_v_.data()[(b * L + t) * D + j];
                    dv.data()[b * D + j] += ds * k;
                }
                dk.data()[b * Kd + i] = dk_acc;
            }

            // The decayed remainder becomes t-1's carry. Written into ds_next in place,
            // after both dK and dV have consumed the undecayed dS.
            for (int64_t i = 0; i < Kd; ++i) {
                for (int64_t j = 0; j < D; ++j) {
                    ds_next.data()[(b * Kd + i) * D + j] *= gamma_;
                }
            }
        }
        ds_carry = ds_next;

        // Step 3: the three projections' parameter gradients and their shares of x_t
        // (standard no-bias Linear backward). All three land on the same grad_input slot.
        Tensor x_t(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                x_t.data()[b * D + e] = last_input_.data()[(b * L + t) * D + e];
            }
        }
        Tensor x_t_T = transpose(x_t, N, D, backend_);

        Tensor gw_q(w_q_.shape(), backend_);
        backend_->gemm(x_t_T.data(), dq.data(), gw_q.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(Kd));
        local_w_q_grad.accumulate(gw_q);
        Tensor gw_k(w_k_.shape(), backend_);
        backend_->gemm(x_t_T.data(), dk.data(), gw_k.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(Kd));
        local_w_k_grad.accumulate(gw_k);
        Tensor gw_v(w_v_.shape(), backend_);
        backend_->gemm(x_t_T.data(), dv.data(), gw_v.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(D));
        local_w_v_grad.accumulate(gw_v);

        Tensor dx_q(Shape({N, D}), backend_);
        backend_->gemm(dq.data(), w_q_T.data(), dx_q.data(), static_cast<size_t>(N), static_cast<size_t>(Kd),
                       static_cast<size_t>(D));
        Tensor dx_k(Shape({N, D}), backend_);
        backend_->gemm(dk.data(), w_k_T.data(), dx_k.data(), static_cast<size_t>(N), static_cast<size_t>(Kd),
                       static_cast<size_t>(D));
        Tensor dx_v(Shape({N, D}), backend_);
        backend_->gemm(dv.data(), w_v_T.data(), dx_v.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                grad_input.data()[(b * L + t) * D + e] +=
                    dx_q.data()[b * D + e] + dx_k.data()[b * D + e] + dx_v.data()[b * D + e];
            }
        }
    }

    w_q_grad_.accumulate(local_w_q_grad);
    w_k_grad_.accumulate(local_w_k_grad);
    w_v_grad_.accumulate(local_w_v_grad);

    return grad_input;
}

Tensor RetNetModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    // Deliberately NOT implemented -- see the class-level note and the campaign's logged
    // Phase 5 charter deviation. This override exists so Module's pure-virtual LRP contract
    // is satisfied honestly; it redistributes nothing, and must never be made to silently
    // return zeros or a gradient-based substitute.
    //
    // The device guard is kept ahead of the throw so this entry point is consistent with
    // forward()/backward() and keeps firing under the death-test convention every module
    // here follows; it becomes load-bearing unchanged the moment a real rule lands.
    PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu);
    (void)relevance_out;  // NDEBUG builds compile PULSATRIX_ASSERT away entirely.
    (void)config;
    throw std::logic_error(
        "RetNetModule::propagate_relevance: LRP rule not yet implemented -- see campaign Decision Point 2");
}

}  // namespace pulsatrix
