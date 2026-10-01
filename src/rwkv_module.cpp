#include "pulsatrix/rwkv_module.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {
// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer --
// same helper shape as LinearModule's/RNNModule's/MambaModule's own transpose()
// (CPUBackend::gemm has no transpose flag).
Tensor transpose(const Tensor& m, int64_t rows, int64_t cols, DeviceBackend* backend) {
    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4). out is allocated through backend, so a GPU
    // backend tags it Cuda/Hip; callers' own guards cannot cover it.
    PULSATRIX_REQUIRE_HOST(m);
    Tensor out(Shape({cols, rows}), backend);
    PULSATRIX_REQUIRE_HOST(out);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.data()[c * rows + r] = m.data()[r * cols + c];
        }
    }
    return out;
}

// The receptance gate's activation. No DeviceBackend::elementwise op exists for it
// (MambaModule needed the same for softplus/exp), so it is a raw host function,
// PULSATRIX_ASSERT-guarded at every entry point that calls it.
float sigmoid(float z) {
    return 1.0f / (1.0f + std::exp(-z));
}

// Same additive epsilon-rule stabilizer every propagate_relevance() in this codebase uses
// (MambaModule's own local stabilize(), duplicated per the per-module-owns-its-helpers
// convention).
float stabilize(float value, float epsilon) {
    return value + epsilon * ((value >= 0.0f) ? 1.0f : -1.0f);
}
}  // namespace

RWKVModule::RWKVModule(int64_t d_model, DeviceBackend* backend)
    : d_model_(d_model),
      backend_(backend),
      w_r_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_k_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_v_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_o_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_(Shape({d_model > 0 ? d_model : 1}), backend),
      u_(Shape({d_model > 0 ? d_model : 1}), backend),
      mu_r_(Shape({d_model > 0 ? d_model : 1}), backend),
      mu_k_(Shape({d_model > 0 ? d_model : 1}), backend),
      mu_v_(Shape({d_model > 0 ? d_model : 1}), backend),
      w_r_grad_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_k_grad_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_v_grad_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_o_grad_(Shape({d_model > 0 ? d_model : 1, d_model > 0 ? d_model : 1}), backend),
      w_grad_(Shape({d_model > 0 ? d_model : 1}), backend),
      u_grad_(Shape({d_model > 0 ? d_model : 1}), backend),
      mu_r_grad_(Shape({d_model > 0 ? d_model : 1}), backend),
      mu_k_grad_(Shape({d_model > 0 ? d_model : 1}), backend),
      mu_v_grad_(Shape({d_model > 0 ? d_model : 1}), backend),
      last_input_(Shape({0}), backend),
      last_xr_(Shape({0}), backend),
      last_xk_(Shape({0}), backend),
      last_xv_(Shape({0}), backend),
      last_r_(Shape({0}), backend),
      last_k_(Shape({0}), backend),
      last_v_(Shape({0}), backend),
      last_e_(Shape({0}), backend),
      last_kk_(Shape({0}), backend),
      last_num_(Shape({0}), backend),
      last_den_(Shape({0}), backend),
      last_wkv_(Shape({0}), backend),
      last_a_(Shape({0}), backend),
      last_b_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from the Python bindings
    // with no upstream validation).
    if (d_model <= 0) {
        throw std::invalid_argument("RWKVModule: d_model must be positive");
    }
}

void RWKVModule::set_W_r(std::initializer_list<float> values) {
    w_r_ = Tensor(w_r_.shape(), backend_, values);
}
void RWKVModule::set_W_k(std::initializer_list<float> values) {
    w_k_ = Tensor(w_k_.shape(), backend_, values);
}
void RWKVModule::set_W_v(std::initializer_list<float> values) {
    w_v_ = Tensor(w_v_.shape(), backend_, values);
}
void RWKVModule::set_W_o(std::initializer_list<float> values) {
    w_o_ = Tensor(w_o_.shape(), backend_, values);
}
void RWKVModule::set_w(std::initializer_list<float> values) {
    w_ = Tensor(w_.shape(), backend_, values);
}
void RWKVModule::set_u(std::initializer_list<float> values) {
    u_ = Tensor(u_.shape(), backend_, values);
}
void RWKVModule::set_mu_r(std::initializer_list<float> values) {
    mu_r_ = Tensor(mu_r_.shape(), backend_, values);
}
void RWKVModule::set_mu_k(std::initializer_list<float> values) {
    mu_k_ = Tensor(mu_k_.shape(), backend_, values);
}
void RWKVModule::set_mu_v(std::initializer_list<float> values) {
    mu_v_ = Tensor(mu_v_.shape(), backend_, values);
}

void RWKVModule::set_W_r(const std::vector<float>& values) {
    w_r_ = Tensor(w_r_.shape(), backend_, values);
}
void RWKVModule::set_W_k(const std::vector<float>& values) {
    w_k_ = Tensor(w_k_.shape(), backend_, values);
}
void RWKVModule::set_W_v(const std::vector<float>& values) {
    w_v_ = Tensor(w_v_.shape(), backend_, values);
}
void RWKVModule::set_W_o(const std::vector<float>& values) {
    w_o_ = Tensor(w_o_.shape(), backend_, values);
}
void RWKVModule::set_w(const std::vector<float>& values) {
    w_ = Tensor(w_.shape(), backend_, values);
}
void RWKVModule::set_u(const std::vector<float>& values) {
    u_ = Tensor(u_.shape(), backend_, values);
}
void RWKVModule::set_mu_r(const std::vector<float>& values) {
    mu_r_ = Tensor(mu_r_.shape(), backend_, values);
}
void RWKVModule::set_mu_k(const std::vector<float>& values) {
    mu_k_ = Tensor(mu_k_.shape(), backend_, values);
}
void RWKVModule::set_mu_v(const std::vector<float>& values) {
    mu_v_ = Tensor(mu_v_.shape(), backend_, values);
}

Tensor RWKVModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly, and computes exp/sigmoid in raw host loops
    // (no DeviceBackend primitive exists for either) -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(input);

    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("RWKVModule::forward: input must be rank-3 (N, L, d_model)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t D = d_model_;

    last_input_ = input;
    last_L_ = L;
    last_xr_ = Tensor(Shape({N, L, D}), backend_);
    last_xk_ = Tensor(Shape({N, L, D}), backend_);
    last_xv_ = Tensor(Shape({N, L, D}), backend_);
    last_r_ = Tensor(Shape({N, L, D}), backend_);
    last_k_ = Tensor(Shape({N, L, D}), backend_);
    last_v_ = Tensor(Shape({N, L, D}), backend_);
    last_e_ = Tensor(Shape({N, L, D}), backend_);
    last_kk_ = Tensor(Shape({N, L, D}), backend_);
    last_num_ = Tensor(Shape({N, L, D}), backend_);
    last_den_ = Tensor(Shape({N, L, D}), backend_);
    last_wkv_ = Tensor(Shape({N, L, D}), backend_);
    last_a_ = Tensor(Shape({N, L + 1, D}), backend_);  // zero-filled: a_0 = 0
    last_b_ = Tensor(Shape({N, L + 1, D}), backend_);  // zero-filled: b_0 = 0

    Tensor output(Shape({N, L, D}), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(output);

    for (int64_t t = 0; t < L; ++t) {
        // The three token-shift mixes, gathered into contiguous (N, d_model) buffers so the
        // projections can go through backend_->gemm rather than hand-rolled loops. x_{-1} is
        // zero (this module's documented zero-init convention), so at t = 0 the shifted
        // term simply drops out.
        Tensor xr(Shape({N, D}), backend_);
        Tensor xk(Shape({N, D}), backend_);
        Tensor xv(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                const float x_cur = input.data()[(b * L + t) * D + e];
                const float x_prev = (t > 0) ? input.data()[(b * L + t - 1) * D + e] : 0.0f;
                xr.data()[b * D + e] = mu_r_.data()[e] * x_cur + (1.0f - mu_r_.data()[e]) * x_prev;
                xk.data()[b * D + e] = mu_k_.data()[e] * x_cur + (1.0f - mu_k_.data()[e]) * x_prev;
                xv.data()[b * D + e] = mu_v_.data()[e] * x_cur + (1.0f - mu_v_.data()[e]) * x_prev;
                last_xr_.data()[(b * L + t) * D + e] = xr.data()[b * D + e];
                last_xk_.data()[(b * L + t) * D + e] = xk.data()[b * D + e];
                last_xv_.data()[(b * L + t) * D + e] = xv.data()[b * D + e];
            }
        }

        Tensor z_r(Shape({N, D}), backend_);
        backend_->gemm(xr.data(), w_r_.data(), z_r.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));
        Tensor k_proj(Shape({N, D}), backend_);
        backend_->gemm(xk.data(), w_k_.data(), k_proj.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));
        Tensor v_proj(Shape({N, D}), backend_);
        backend_->gemm(xv.data(), w_v_.data(), v_proj.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));

        // r_t*wkv_t, the output projection's input.
        Tensor gated(Shape({N, D}), backend_);

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const int64_t idx = (b * L + t) * D + d;
                const float r = sigmoid(z_r.data()[b * D + d]);
                const float k = k_proj.data()[b * D + d];
                const float v = v_proj.data()[b * D + d];
                last_r_.data()[idx] = r;
                last_k_.data()[idx] = k;
                last_v_.data()[idx] = v;

                // The WKV quotient: the bonus-weighted current token on top of the decayed
                // running state.
                const float a_prev = last_a_.data()[(b * (L + 1) + t) * D + d];
                const float b_prev = last_b_.data()[(b * (L + 1) + t) * D + d];
                const float e_t = std::exp(u_.data()[d] + k);
                const float num = a_prev + e_t * v;
                const float den = b_prev + e_t;
                const float wkv = num / den;
                last_e_.data()[idx] = e_t;
                last_num_.data()[idx] = num;
                last_den_.data()[idx] = den;
                last_wkv_.data()[idx] = wkv;

                // The state carry itself -- decayed, then extended by this token.
                const float decay = std::exp(-w_.data()[d]);
                const float kk = std::exp(k);
                last_kk_.data()[idx] = kk;
                last_a_.data()[(b * (L + 1) + t + 1) * D + d] = decay * a_prev + kk * v;
                last_b_.data()[(b * (L + 1) + t + 1) * D + d] = decay * b_prev + kk;

                gated.data()[b * D + d] = r * wkv;
            }
        }

        Tensor o_t(Shape({N, D}), backend_);
        backend_->gemm(gated.data(), w_o_.data(), o_t.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                output.data()[(b * L + t) * D + d] = o_t.data()[b * D + d];
            }
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor RWKVModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("RWKVModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    const int64_t D = d_model_;
    if (grad_output.rank() != 3 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != L ||
        grad_output.shape().dim(2) != D) {
        throw std::invalid_argument(
            "RWKVModule::backward: grad_output must be (N, L, d_model) matching the cached forward shape");
    }
    // Dereferences Tensor::data() directly, and computes exp/sigmoid math in raw host loops
    // -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(grad_output);

    Tensor grad_input(last_input_.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(grad_input);
    Tensor local_w_r_grad(w_r_.shape(), backend_);
    Tensor local_w_k_grad(w_k_.shape(), backend_);
    Tensor local_w_v_grad(w_v_.shape(), backend_);
    Tensor local_w_o_grad(w_o_.shape(), backend_);
    Tensor local_w_grad(w_.shape(), backend_);
    Tensor local_u_grad(u_.shape(), backend_);
    Tensor local_mu_r_grad(mu_r_.shape(), backend_);
    Tensor local_mu_k_grad(mu_k_.shape(), backend_);
    Tensor local_mu_v_grad(mu_v_.shape(), backend_);

    // The carried state-gradient accumulators, threaded from t+1 back to t: the gradient
    // arriving on a_t/b_t from step t+1's use of them as its own a_{t-1}/b_{t-1}.
    Tensor da_carry(Shape({N, D}), backend_);
    Tensor db_carry(Shape({N, D}), backend_);

    for (int64_t t = L - 1; t >= 0; --t) {
        // Step 1: the output projection o_t = (r_t*wkv_t) @ W_o.
        Tensor g_o(Shape({N, D}), backend_);
        Tensor gated(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                g_o.data()[b * D + d] = grad_output.data()[(b * L + t) * D + d];
                gated.data()[b * D + d] = last_r_.data()[(b * L + t) * D + d] * last_wkv_.data()[(b * L + t) * D + d];
            }
        }

        Tensor gated_T = transpose(gated, N, D, backend_);
        Tensor gw_o(w_o_.shape(), backend_);
        backend_->gemm(gated_T.data(), g_o.data(), gw_o.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(D));
        local_w_o_grad.accumulate(gw_o);

        Tensor w_o_T = transpose(w_o_, D, D, backend_);
        Tensor g_gated(Shape({N, D}), backend_);
        backend_->gemm(g_o.data(), w_o_T.data(), g_gated.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));

        Tensor dz_r(Shape({N, D}), backend_);
        Tensor dk(Shape({N, D}), backend_);
        Tensor dv(Shape({N, D}), backend_);
        Tensor da_next(Shape({N, D}), backend_);
        Tensor db_next(Shape({N, D}), backend_);

        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const int64_t idx = (b * L + t) * D + d;
                const float r = last_r_.data()[idx];
                const float v = last_v_.data()[idx];
                const float e_t = last_e_.data()[idx];
                const float kk = last_kk_.data()[idx];
                const float num = last_num_.data()[idx];
                const float den = last_den_.data()[idx];
                const float a_prev = last_a_.data()[(b * (L + 1) + t) * D + d];
                const float b_prev = last_b_.data()[(b * (L + 1) + t) * D + d];
                const float decay = std::exp(-w_.data()[d]);

                const float g_rwkv = g_gated.data()[b * D + d];
                const float dr = g_rwkv * last_wkv_.data()[idx];
                const float dwkv = g_rwkv * r;

                // Step 2: wkv_t = num_t/den_t.
                const float dnum = dwkv / den;
                const float dden = -dwkv * num / (den * den);

                // Step 3: num_t = a_{t-1} + e_t*v_t, den_t = b_{t-1} + e_t.
                float da_prev = dnum;
                float db_prev = dden;
                const float de = dnum * v + dden;
                float dv_local = dnum * e_t;

                // Step 4: e_t = exp(u + k_t).
                float dk_local = de * e_t;
                local_u_grad.data()[d] += de * e_t;

                // Step 5: the recurrence a_t = decay*a_{t-1} + kk_t*v_t (and b_t likewise),
                // driven by the *total* gradient arriving on a_t/b_t from step t+1.
                const float ga = da_carry.data()[b * D + d];
                const float gb = db_carry.data()[b * D + d];
                const float d_decay_from_a = ga * a_prev;
                da_prev += ga * decay;
                float dkk = ga * v;
                dv_local += ga * kk;
                const float d_decay_from_b = gb * b_prev;
                db_prev += gb * decay;
                dkk += gb;

                // Step 6: decay = exp(-w), so d(decay)/dw = -decay.
                local_w_grad.data()[d] += (d_decay_from_a + d_decay_from_b) * (-decay);

                // Step 7: kk_t = exp(k_t) -- onto the same dk_t step 4 started.
                dk_local += dkk * kk;

                // Step 8: what steps 3 and 5 accumulated becomes t-1's carry.
                da_next.data()[b * D + d] = da_prev;
                db_next.data()[b * D + d] = db_prev;

                dk.data()[b * D + d] = dk_local;
                dv.data()[b * D + d] = dv_local;
                // The receptance gate's sigmoid derivative, from its cached output.
                dz_r.data()[b * D + d] = dr * r * (1.0f - r);
            }
        }
        da_carry = da_next;
        db_carry = db_next;

        // Step 9: the three input projections' parameter gradients and their shares of the
        // token-shifted inputs (standard no-bias Linear backward).
        Tensor xr(Shape({N, D}), backend_);
        Tensor xk(Shape({N, D}), backend_);
        Tensor xv(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                xr.data()[b * D + e] = last_xr_.data()[(b * L + t) * D + e];
                xk.data()[b * D + e] = last_xk_.data()[(b * L + t) * D + e];
                xv.data()[b * D + e] = last_xv_.data()[(b * L + t) * D + e];
            }
        }

        Tensor xr_T = transpose(xr, N, D, backend_);
        Tensor xk_T = transpose(xk, N, D, backend_);
        Tensor xv_T = transpose(xv, N, D, backend_);

        Tensor gw_r(w_r_.shape(), backend_);
        backend_->gemm(xr_T.data(), dz_r.data(), gw_r.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(D));
        local_w_r_grad.accumulate(gw_r);
        Tensor gw_k(w_k_.shape(), backend_);
        backend_->gemm(xk_T.data(), dk.data(), gw_k.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(D));
        local_w_k_grad.accumulate(gw_k);
        Tensor gw_v(w_v_.shape(), backend_);
        backend_->gemm(xv_T.data(), dv.data(), gw_v.data(), static_cast<size_t>(D), static_cast<size_t>(N),
                       static_cast<size_t>(D));
        local_w_v_grad.accumulate(gw_v);

        Tensor w_r_T = transpose(w_r_, D, D, backend_);
        Tensor w_k_T = transpose(w_k_, D, D, backend_);
        Tensor w_v_T = transpose(w_v_, D, D, backend_);
        Tensor dxr(Shape({N, D}), backend_);
        backend_->gemm(dz_r.data(), w_r_T.data(), dxr.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));
        Tensor dxk(Shape({N, D}), backend_);
        backend_->gemm(dk.data(), w_k_T.data(), dxk.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));
        Tensor dxv(Shape({N, D}), backend_);
        backend_->gemm(dv.data(), w_v_T.data(), dxv.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));

        // Step 10: the token-shift itself. grad_input is a full pre-allocated (N, L, D)
        // buffer, so the x_{t-1} share is a direct random-access += into the t-1 slot (which
        // a later loop iteration will add its own x_t share onto) rather than a carried
        // scalar accumulator.
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const float x_cur = last_input_.data()[(b * L + t) * D + d];
                const float x_prev = (t > 0) ? last_input_.data()[(b * L + t - 1) * D + d] : 0.0f;
                const float diff = x_cur - x_prev;

                const float g_xr = dxr.data()[b * D + d];
                const float g_xk = dxk.data()[b * D + d];
                const float g_xv = dxv.data()[b * D + d];
                local_mu_r_grad.data()[d] += g_xr * diff;
                local_mu_k_grad.data()[d] += g_xk * diff;
                local_mu_v_grad.data()[d] += g_xv * diff;

                grad_input.data()[(b * L + t) * D + d] +=
                    g_xr * mu_r_.data()[d] + g_xk * mu_k_.data()[d] + g_xv * mu_v_.data()[d];
                if (t > 0) {
                    grad_input.data()[(b * L + t - 1) * D + d] += g_xr * (1.0f - mu_r_.data()[d]) +
                                                                  g_xk * (1.0f - mu_k_.data()[d]) +
                                                                  g_xv * (1.0f - mu_v_.data()[d]);
                }
            }
        }
    }

    w_r_grad_.accumulate(local_w_r_grad);
    w_k_grad_.accumulate(local_w_k_grad);
    w_v_grad_.accumulate(local_w_v_grad);
    w_o_grad_.accumulate(local_w_o_grad);
    w_grad_.accumulate(local_w_grad);
    u_grad_.accumulate(local_u_grad);
    mu_r_grad_.accumulate(local_mu_r_grad);
    mu_k_grad_.accumulate(local_mu_k_grad);
    mu_v_grad_.accumulate(local_mu_v_grad);

    return grad_input;
}

Tensor RWKVModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("RWKVModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    const int64_t D = d_model_;
    if (relevance_out.rank() != 3 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != L ||
        relevance_out.shape().dim(2) != D) {
        throw std::invalid_argument(
            "RWKVModule::propagate_relevance: relevance_out must be (N, L, d_model) matching the cached "
            "forward shape");
    }
    // Dereferences Tensor::data() directly, and computes exp math in raw host loops -- not
    // yet backend-generic.
    PULSATRIX_REQUIRE_HOST(relevance_out);

    const float eps = config.epsilon;
    Tensor relevance_in(last_input_.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(relevance_in);
    Tensor w_o_T = transpose(w_o_, D, D, backend_);
    Tensor w_v_T = transpose(w_v_, D, D, backend_);

    // R(A[t+1]) -- the WKV numerator state's total relevance, threaded backward from t+1 to
    // t exactly the way MambaModule's own r_h_carry is threaded (see the header's
    // derivation note: A[t] is used at BOTH num_t's readout split and A[t+1]'s own state
    // split, the same dual-use shape as Mamba's h_t). Zero at t = L (nothing reads A[L]),
    // and (by A[0] == 0) it carries nothing out past t = 0.
    Tensor r_a_next(Shape({N, D}), backend_);

    for (int64_t t = L - 1; t >= 0; --t) {
        // --- Step 1' (output projection): standard no-bias weighted-connection z-rule,
        // denominator o_t itself (recomputed here -- forward_impl() didn't cache it
        // separately, only its own gated = r_t*wkv_t input). Gives r_gated, the relevance of
        // "gated" = r_t*wkv_t.
        Tensor gated(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                const int64_t idx = (b * L + t) * D + e;
                gated.data()[b * D + e] = last_r_.data()[idx] * last_wkv_.data()[idx];
            }
        }
        Tensor o_t(Shape({N, D}), backend_);
        backend_->gemm(gated.data(), w_o_.data(), o_t.data(), static_cast<size_t>(N), static_cast<size_t>(D),
                       static_cast<size_t>(D));
        Tensor scaled_r_o(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const float denom = stabilize(o_t.data()[b * D + d], eps);
                scaled_r_o.data()[b * D + d] = relevance_out.data()[(b * L + t) * D + d] / denom;
            }
        }
        Tensor r_gated_raw(Shape({N, D}), backend_);
        backend_->gemm(scaled_r_o.data(), w_o_T.data(), r_gated_raw.data(), static_cast<size_t>(N),
                       static_cast<size_t>(D), static_cast<size_t>(D));
        Tensor r_gated(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                r_gated.data()[b * D + e] = gated.data()[b * D + e] * r_gated_raw.data()[b * D + e];
            }
        }

        // --- Step 2' (receptance detach): r_t is a genuine data-dependent gate, detached
        // as a constant (MambaLRP's own technique, applied here to r_t exactly as it is to
        // Mamba's Abar/Bbar/C) -- a single-term rescaling under which R(wkv_t) == R(gated)
        // directly, no formula needed (same reasoning as RetNetModule's gamma^(t-s)
        // pass-through, except here the detachment IS the approximation, unlike RetNet's
        // exact hyperparameter scaling).
        //
        // --- Step 3' (WKV quotient, MambaLRP-style detached weighted sum): wkv_t =
        // num_t/den_t = (1/den_t)*A[t] + (e_t/den_t)*v_t, with e_t and den_t detached as
        // constants -- the standard weighted-sum epsilon/z-rule, denominator wkv_t itself.
        //
        // --- Step 4' (state carry, same technique): A[t+1] = decay*A[t] + kk_t*v_t, with
        // decay and kk_t detached -- the same weighted-sum rule, denominator A[t+1] itself,
        // consuming r_a_next (R(A[t+1])) and producing this step's own R(A[t]).
        Tensor v_relevance(Shape({N, D}), backend_);
        Tensor r_a_cur(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const int64_t idx = (b * L + t) * D + d;
                const float r_wkv = r_gated.data()[b * D + d];
                const float a_prev = last_a_.data()[(b * (L + 1) + t) * D + d];      // A[t]
                const float a_next_val = last_a_.data()[(b * (L + 1) + t + 1) * D + d];  // A[t+1]
                const float den = last_den_.data()[idx];
                const float e_t = last_e_.data()[idx];
                const float v_val = last_v_.data()[idx];
                const float wkv_val = last_wkv_.data()[idx];
                const float decay = std::exp(-w_.data()[d]);
                const float kk_t = last_kk_.data()[idx];

                const float denom_wkv = stabilize(wkv_val, eps);
                const float contrib_At_readout = (a_prev / den / denom_wkv) * r_wkv;
                const float contrib_vt_readout = (e_t * v_val / den / denom_wkv) * r_wkv;

                const float r_a_next_val = r_a_next.data()[b * D + d];
                const float denom_a_next = stabilize(a_next_val, eps);
                const float contrib_At_state = (decay * a_prev / denom_a_next) * r_a_next_val;
                const float contrib_vt_state = (kk_t * v_val / denom_a_next) * r_a_next_val;

                r_a_cur.data()[b * D + d] = contrib_At_readout + contrib_At_state;
                v_relevance.data()[b * D + d] = contrib_vt_readout + contrib_vt_state;
            }
        }
        r_a_next = r_a_cur;

        // --- Step 5' (value projection): standard no-bias weighted-connection z-rule,
        // denominator v_t itself, following exactly LinearModule's own gemm-based backward
        // shape (transpose W_v, scale relevance by 1/denom, gemm back through W_v^T,
        // elementwise-multiply by the input).
        Tensor scaled_r_v(Shape({N, D}), backend_);
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t d = 0; d < D; ++d) {
                const int64_t idx = (b * L + t) * D + d;
                const float denom = stabilize(last_v_.data()[idx], eps);
                scaled_r_v.data()[b * D + d] = v_relevance.data()[b * D + d] / denom;
            }
        }
        Tensor r_xv_raw(Shape({N, D}), backend_);
        backend_->gemm(scaled_r_v.data(), w_v_T.data(), r_xv_raw.data(), static_cast<size_t>(N),
                       static_cast<size_t>(D), static_cast<size_t>(D));

        // --- Step 6' (value token-shift): xv_t = mu_v*x_cur + (1-mu_v)*x_prev, a genuine
        // (not detached -- mu_v is a real learned weight, treated normally) two-term
        // weighted sum, denominator xv_t itself. w_r_/mu_r_/w_k_/mu_k_/u_ are deliberately
        // never referenced anywhere in this function (only their cached, detached forward
        // values last_r_/last_e_/last_kk_ are); w_ is referenced only to reconstruct the
        // detached decay value above -- see the class-level note. Only w_v_/w_o_/mu_v_ (and
        // w_ for decay) participate.
        for (int64_t b = 0; b < N; ++b) {
            for (int64_t e = 0; e < D; ++e) {
                const float xv_val = last_xv_.data()[(b * L + t) * D + e];
                // Step 5's z-rule is r_xv[e] = xv_t[e] * (W_v @ scaled_r)[e] -- r_xv_raw
                // above is only the (W_v @ scaled_r)[e] factor; the elementwise multiply by
                // the projection's own input (xv_t) -- exactly like Step 1's gated * (W_o @
                // scaled_r) -- was missing here (caught by this test's non-conservation).
                const float r_xv = xv_val * r_xv_raw.data()[b * D + e];
                const float x_cur = last_input_.data()[(b * L + t) * D + e];
                const float x_prev = (t > 0) ? last_input_.data()[(b * L + t - 1) * D + e] : 0.0f;
                const float denom = stabilize(xv_val, eps);
                const float mu = mu_v_.data()[e];

                relevance_in.data()[(b * L + t) * D + e] += (mu * x_cur / denom) * r_xv;
                if (t > 0) {
                    relevance_in.data()[(b * L + t - 1) * D + e] += ((1.0f - mu) * x_prev / denom) * r_xv;
                }
            }
        }
    }

    return relevance_in;
}

}  // namespace pulsatrix
