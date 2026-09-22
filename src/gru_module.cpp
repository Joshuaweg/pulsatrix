#include "exai/gru_module.hpp"

#include <cmath>
#include <stdexcept>

#include "exai/assert.hpp"

namespace exai {

namespace {
// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer --
// same helper shape as LinearModule's/Conv2DModule's/RNNModule's/LSTMModule's own
// transpose() (CPUBackend::gemm has no transpose flag).
Tensor transpose(const Tensor& m, int64_t rows, int64_t cols, DeviceBackend* backend) {
    Tensor out(Shape({cols, rows}), backend);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            out.data()[c * rows + r] = m.data()[r * cols + c];
        }
    }
    return out;
}

// No DeviceBackend::Sigmoid primitive exists (ElementwiseOp has only Relu/Neg) -- raw host
// helper, exactly like LSTMModule's. Third occurrence of this workaround; see the
// class-level note in gru_module.hpp for why it was evaluated and deliberately kept here.
float sigmoid(float z) { return 1.0f / (1.0f + std::exp(-z)); }
}  // namespace

GRUModule::GRUModule(int64_t input_size, int64_t hidden_size, DeviceBackend* backend)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      backend_(backend),
      weight_xz_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_hz_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_z_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      weight_xr_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_hr_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_r_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      weight_xn_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_hn_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_n_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      weight_xz_grad_(weight_xz_.shape(), backend),
      weight_hz_grad_(weight_hz_.shape(), backend),
      bias_z_grad_(bias_z_.shape(), backend),
      weight_xr_grad_(weight_xr_.shape(), backend),
      weight_hr_grad_(weight_hr_.shape(), backend),
      bias_r_grad_(bias_r_.shape(), backend),
      weight_xn_grad_(weight_xn_.shape(), backend),
      weight_hn_grad_(weight_hn_.shape(), backend),
      bias_n_grad_(bias_n_.shape(), backend),
      last_input_(Shape({0}), backend),
      last_hidden_states_(Shape({0}), backend),
      last_gate_z_(Shape({0}), backend),
      last_gate_r_(Shape({0}), backend),
      last_candidate_n_(Shape({0}), backend),
      last_hn_prev_(Shape({0}), backend),
      last_pre_activation_n_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (input_size <= 0) {
        throw std::invalid_argument("GRUModule: input_size must be positive");
    }
    if (hidden_size <= 0) {
        throw std::invalid_argument("GRUModule: hidden_size must be positive");
    }
}

void GRUModule::set_weight_xz(std::initializer_list<float> values) {
    weight_xz_ = Tensor(weight_xz_.shape(), backend_, values);
}
void GRUModule::set_weight_hz(std::initializer_list<float> values) {
    weight_hz_ = Tensor(weight_hz_.shape(), backend_, values);
}
void GRUModule::set_bias_z(std::initializer_list<float> values) {
    bias_z_ = Tensor(bias_z_.shape(), backend_, values);
}
void GRUModule::set_weight_xr(std::initializer_list<float> values) {
    weight_xr_ = Tensor(weight_xr_.shape(), backend_, values);
}
void GRUModule::set_weight_hr(std::initializer_list<float> values) {
    weight_hr_ = Tensor(weight_hr_.shape(), backend_, values);
}
void GRUModule::set_bias_r(std::initializer_list<float> values) {
    bias_r_ = Tensor(bias_r_.shape(), backend_, values);
}
void GRUModule::set_weight_xn(std::initializer_list<float> values) {
    weight_xn_ = Tensor(weight_xn_.shape(), backend_, values);
}
void GRUModule::set_weight_hn(std::initializer_list<float> values) {
    weight_hn_ = Tensor(weight_hn_.shape(), backend_, values);
}
void GRUModule::set_bias_n(std::initializer_list<float> values) {
    bias_n_ = Tensor(bias_n_.shape(), backend_, values);
}

Tensor GRUModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    EXAI_ASSERT(input.device() == DeviceType::Cpu);

    if (input.rank() != 3 || input.shape().dim(2) != input_size_) {
        throw std::invalid_argument("GRUModule::forward: input must be rank-3 (N, L, input_size)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);

    last_input_ = input;
    last_L_ = L;
    last_hidden_states_ = Tensor(Shape({N, L + 1, hidden_size_}), backend_);
    last_hidden_states_.fill(0.0f);  // h_0 = 0 for every example
    last_gate_z_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_gate_r_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_candidate_n_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_hn_prev_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_pre_activation_n_ = Tensor(Shape({N, L, hidden_size_}), backend_);

    Tensor output(Shape({N, L, hidden_size_}), backend_);

    // z = x_t @ Wx + h_prev @ Wh, EXCLUDING bias -- one gate's pre-activation.
    auto gate_preactivation = [&](const Tensor& x_t, const Tensor& h_prev, const Tensor& wx, const Tensor& wh) {
        Tensor z1(Shape({N, hidden_size_}), backend_);
        backend_->gemm(x_t.data(), wx.data(), z1.data(), static_cast<size_t>(N), static_cast<size_t>(input_size_),
                       static_cast<size_t>(hidden_size_));
        Tensor z2(Shape({N, hidden_size_}), backend_);
        backend_->gemm(h_prev.data(), wh.data(), z2.data(), static_cast<size_t>(N),
                       static_cast<size_t>(hidden_size_), static_cast<size_t>(hidden_size_));
        z1.accumulate(z2);
        return z1;
    };

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

        Tensor z_z = gate_preactivation(x_t, h_prev, weight_xz_, weight_hz_);
        Tensor z_r = gate_preactivation(x_t, h_prev, weight_xr_, weight_hr_);

        // The candidate's two sources are NOT symmetric: x_t goes through W_xn directly,
        // while h_{t-1} goes through its own projection W_hn whose *output* the reset gate
        // multiplies. Both halves are computed separately for exactly that reason.
        Tensor xn(Shape({N, hidden_size_}), backend_);
        backend_->gemm(x_t.data(), weight_xn_.data(), xn.data(), static_cast<size_t>(N),
                       static_cast<size_t>(input_size_), static_cast<size_t>(hidden_size_));
        Tensor hn_prev(Shape({N, hidden_size_}), backend_);
        backend_->gemm(h_prev.data(), weight_hn_.data(), hn_prev.data(), static_cast<size_t>(N),
                       static_cast<size_t>(hidden_size_), static_cast<size_t>(hidden_size_));

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                const int64_t idx = n * hidden_size_ + k;
                const int64_t seq_idx = (n * L + t) * hidden_size_ + k;

                const float z_t = sigmoid(z_z.data()[idx] + bias_z_.data()[k]);
                const float r_t = sigmoid(z_r.data()[idx] + bias_r_.data()[k]);
                const float hn_p = hn_prev.data()[idx];
                // n_pre EXCLUDING bias is what the LRP epsilon-rule denominator uses (cached
                // below -- same convention as LinearModule/RNNModule/LSTMModule: bias has no
                // associated input feature to redistribute relevance to, so it's excluded
                // from the denominator entirely). The actual tanh activation still needs the
                // real bias-included pre-activation.
                const float n_pre_no_bias = xn.data()[idx] + r_t * hn_p;
                const float n_t = std::tanh(n_pre_no_bias + bias_n_.data()[k]);

                const float h_prev_v = h_prev.data()[idx];
                const float h_t = (1.0f - z_t) * h_prev_v + z_t * n_t;

                last_gate_z_.data()[seq_idx] = z_t;
                last_gate_r_.data()[seq_idx] = r_t;
                last_candidate_n_.data()[seq_idx] = n_t;
                last_hn_prev_.data()[seq_idx] = hn_p;
                last_pre_activation_n_.data()[seq_idx] = n_pre_no_bias;
                last_hidden_states_.data()[(n * (L + 1) + t + 1) * hidden_size_ + k] = h_t;
                output.data()[seq_idx] = h_t;
            }
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor GRUModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("GRUModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    if (grad_output.rank() != 3 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != L ||
        grad_output.shape().dim(2) != hidden_size_) {
        throw std::invalid_argument(
            "GRUModule::backward: grad_output must be (N, L, hidden_size) matching the cached forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    EXAI_ASSERT(grad_output.device() == DeviceType::Cpu);

    Tensor grad_input(last_input_.shape(), backend_);
    grad_input.fill(0.0f);

    auto zeroed_like = [&](const Tensor& t) {
        Tensor out(t.shape(), backend_);
        out.fill(0.0f);
        return out;
    };
    Tensor local_wxz_grad = zeroed_like(weight_xz_);
    Tensor local_whz_grad = zeroed_like(weight_hz_);
    Tensor local_bz_grad = zeroed_like(bias_z_);
    Tensor local_wxr_grad = zeroed_like(weight_xr_);
    Tensor local_whr_grad = zeroed_like(weight_hr_);
    Tensor local_br_grad = zeroed_like(bias_r_);
    Tensor local_wxn_grad = zeroed_like(weight_xn_);
    Tensor local_whn_grad = zeroed_like(weight_hn_);
    Tensor local_bn_grad = zeroed_like(bias_n_);

    Tensor dh_next(Shape({N, hidden_size_}), backend_);
    dh_next.fill(0.0f);

    for (int64_t t = L - 1; t >= 0; --t) {
        Tensor x_t(Shape({N, input_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                x_t.data()[n * input_size_ + i] = last_input_.data()[(n * L + t) * input_size_ + i];
            }
        }
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                h_prev.data()[n * hidden_size_ + k] = last_hidden_states_.data()[(n * (L + 1) + t) * hidden_size_ + k];
            }
        }

        Tensor dz_pre(Shape({N, hidden_size_}), backend_);
        Tensor dr_pre(Shape({N, hidden_size_}), backend_);
        Tensor dn_pre(Shape({N, hidden_size_}), backend_);
        Tensor dhn_prev(Shape({N, hidden_size_}), backend_);
        Tensor dh_prev(Shape({N, hidden_size_}), backend_);
        dh_prev.fill(0.0f);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                const int64_t idx = n * hidden_size_ + k;
                const int64_t seq_idx = (n * L + t) * hidden_size_ + k;

                const float z_t = last_gate_z_.data()[seq_idx];
                const float r_t = last_gate_r_.data()[seq_idx];
                const float n_t = last_candidate_n_.data()[seq_idx];
                const float hn_p = last_hn_prev_.data()[seq_idx];
                const float h_prev_v = h_prev.data()[idx];

                const float dh = grad_output.data()[seq_idx] + dh_next.data()[idx];

                // h_t = (1-z_t)*h_{t-1} + z_t*n_t -- three consumers of dh: the direct
                // convex carry onto h_{t-1}, the candidate, and the update gate itself
                // (which sees n_t - h_{t-1}).
                dh_prev.data()[idx] += dh * (1.0f - z_t);
                const float dn = dh * z_t;
                dz_pre.data()[idx] = dh * (n_t - h_prev_v) * z_t * (1.0f - z_t);

                // n_t = tanh(n_pre): tanh' written as 1 - n^2 from the cached activation.
                const float dnp = dn * (1.0f - n_t * n_t);
                dn_pre.data()[idx] = dnp;
                // n_pre's term B = r_t * hn_prev_t: a genuine product of two computed
                // quantities, so it splits into a reset-gate branch and a projection branch.
                dr_pre.data()[idx] = dnp * hn_p * r_t * (1.0f - r_t);
                dhn_prev.data()[idx] = dnp * r_t;
            }
        }

        Tensor x_t_T = transpose(x_t, N, input_size_, backend_);
        Tensor h_prev_T = transpose(h_prev, N, hidden_size_, backend_);

        Tensor dx_t(Shape({N, input_size_}), backend_);
        dx_t.fill(0.0f);

        // One x-side weight matrix's contribution: its parameter gradient plus its share of
        // dx_t. Used by both gates and by the candidate's W_xn.
        auto accumulate_x_path = [&](const Tensor& dz, const Tensor& wx, Tensor& gwx_acc) {
            Tensor gwx(wx.shape(), backend_);
            backend_->gemm(x_t_T.data(), dz.data(), gwx.data(), static_cast<size_t>(input_size_),
                           static_cast<size_t>(N), static_cast<size_t>(hidden_size_));
            gwx_acc.accumulate(gwx);

            Tensor wx_T = transpose(wx, input_size_, hidden_size_, backend_);
            Tensor gx(Shape({N, input_size_}), backend_);
            backend_->gemm(dz.data(), wx_T.data(), gx.data(), static_cast<size_t>(N),
                           static_cast<size_t>(hidden_size_), static_cast<size_t>(input_size_));
            dx_t.accumulate(gx);
        };

        // One h-side weight matrix's contribution: its parameter gradient plus its share of
        // dh_prev. Used by both gates and by the candidate's W_hn projection -- the latter
        // is the second of the three paths that feed the same dh_prev accumulator (the
        // gradient-side analogue of propagate_relevance's two-path R(h_{t-1})).
        auto accumulate_h_path = [&](const Tensor& dz, const Tensor& wh, Tensor& gwh_acc) {
            Tensor gwh(wh.shape(), backend_);
            backend_->gemm(h_prev_T.data(), dz.data(), gwh.data(), static_cast<size_t>(hidden_size_),
                           static_cast<size_t>(N), static_cast<size_t>(hidden_size_));
            gwh_acc.accumulate(gwh);

            Tensor wh_T = transpose(wh, hidden_size_, hidden_size_, backend_);
            Tensor gh(Shape({N, hidden_size_}), backend_);
            backend_->gemm(dz.data(), wh_T.data(), gh.data(), static_cast<size_t>(N),
                           static_cast<size_t>(hidden_size_), static_cast<size_t>(hidden_size_));
            dh_prev.accumulate(gh);
        };

        auto accumulate_bias = [&](const Tensor& dz, Tensor& gb_acc) {
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t k = 0; k < hidden_size_; ++k) {
                    gb_acc.data()[k] += dz.data()[n * hidden_size_ + k];
                }
            }
        };

        accumulate_x_path(dz_pre, weight_xz_, local_wxz_grad);
        accumulate_h_path(dz_pre, weight_hz_, local_whz_grad);
        accumulate_bias(dz_pre, local_bz_grad);

        accumulate_x_path(dr_pre, weight_xr_, local_wxr_grad);
        accumulate_h_path(dr_pre, weight_hr_, local_whr_grad);
        accumulate_bias(dr_pre, local_br_grad);

        // The candidate's x-side sees dn_pre directly; its h-side sees dn_pre gated by r_t,
        // because the reset gate multiplies the projection's OUTPUT.
        accumulate_x_path(dn_pre, weight_xn_, local_wxn_grad);
        accumulate_h_path(dhn_prev, weight_hn_, local_whn_grad);
        accumulate_bias(dn_pre, local_bn_grad);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                grad_input.data()[(n * L + t) * input_size_ + i] = dx_t.data()[n * input_size_ + i];
            }
        }
        dh_next = dh_prev;
    }

    weight_xz_grad_.accumulate(local_wxz_grad);
    weight_hz_grad_.accumulate(local_whz_grad);
    bias_z_grad_.accumulate(local_bz_grad);
    weight_xr_grad_.accumulate(local_wxr_grad);
    weight_hr_grad_.accumulate(local_whr_grad);
    bias_r_grad_.accumulate(local_br_grad);
    weight_xn_grad_.accumulate(local_wxn_grad);
    weight_hn_grad_.accumulate(local_whn_grad);
    bias_n_grad_.accumulate(local_bn_grad);

    return grad_input;
}

Tensor GRUModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("GRUModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    if (relevance_out.rank() != 3 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != L ||
        relevance_out.shape().dim(2) != hidden_size_) {
        throw std::invalid_argument(
            "GRUModule::propagate_relevance: relevance_out must be (N, L, hidden_size) matching the cached "
            "forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    EXAI_ASSERT(relevance_out.device() == DeviceType::Cpu);

    Tensor relevance_in(last_input_.shape(), backend_);
    relevance_in.fill(0.0f);

    // One carried accumulator, mirroring backward()'s dh_next: the relevance a later
    // timestep assigned to h_{t-1}. Unlike RNNModule's (one source) and LSTMModule's (one
    // source per accumulator), this one is written by TWO distinct paths per timestep --
    // see the class-level note, step 7.
    Tensor R_h_next(Shape({N, hidden_size_}), backend_);
    R_h_next.fill(0.0f);

    for (int64_t t = L - 1; t >= 0; --t) {
        Tensor x_t(Shape({N, input_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                x_t.data()[n * input_size_ + i] = last_input_.data()[(n * L + t) * input_size_ + i];
            }
        }
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                h_prev.data()[n * hidden_size_ + k] = last_hidden_states_.data()[(n * (L + 1) + t) * hidden_size_ + k];
            }
        }

        Tensor R_x(Shape({N, input_size_}), backend_);
        R_x.fill(0.0f);
        Tensor R_hprev(Shape({N, hidden_size_}), backend_);
        R_hprev.fill(0.0f);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                const int64_t idx = n * hidden_size_ + k;
                const int64_t seq_idx = (n * L + t) * hidden_size_ + k;

                const float z_t = last_gate_z_.data()[seq_idx];
                const float r_t = last_gate_r_.data()[seq_idx];
                const float n_t = last_candidate_n_.data()[seq_idx];
                const float hn_p = last_hn_prev_.data()[seq_idx];
                const float h_prev_v = h_prev.data()[idx];
                const float h_t = last_hidden_states_.data()[(n * (L + 1) + t + 1) * hidden_size_ + k];

                const float R_h = relevance_out.data()[seq_idx] + R_h_next.data()[idx];

                // Step 1: h_t = (1-z_t)*h_{t-1} + z_t*n_t -- two-term weighted sum whose
                // denominator is exactly h_t. z_t is a multiplicative weight here, NOT a
                // relevance recipient; only the signals h_{t-1} and n_t receive.
                const float h_sign = (h_t >= 0.0f) ? 1.0f : -1.0f;
                const float h_denom = h_t + config.epsilon * h_sign;
                const float R_hprev_direct = ((1.0f - z_t) * h_prev_v / h_denom) * R_h;
                const float R_n = (z_t * n_t / h_denom) * R_h;

                // Step 7, contribution 1 of 2: the DIRECT split. The candidate path below
                // adds its own contribution onto the same accumulator slot -- these must
                // SUM, hence `+=` here and `+=` there, never assignment.
                R_hprev.data()[idx] += R_hprev_direct;

                // Step 2: n_t = tanh(n_pre) is identity pass-through, so R(n_pre) = R(n_t).
                // Steps 3+4: n_pre (bias excluded) = x_t@W_xn + r_t*hn_prev_t. Splitting
                // R(n_pre) between the two terms by value and then redistributing the
                // x_t@W_xn share across x_t's features composes into a single
                // epsilon-stabilized redistribution over the shared n_pre denominator (the
                // intermediate term-A denominator cancels), which is also strictly better
                // for conservation than stabilizing twice.
                const float n_pre = last_pre_activation_n_.data()[seq_idx];
                const float n_sign = (n_pre >= 0.0f) ? 1.0f : -1.0f;
                const float n_denom = n_pre + config.epsilon * n_sign;

                for (int64_t i = 0; i < input_size_; ++i) {
                    const float w = weight_xn_.data()[i * hidden_size_ + k];
                    const float x = x_t.data()[n * input_size_ + i];
                    R_x.data()[n * input_size_ + i] += (x * w / n_denom) * R_n;
                }

                // Step 5: term B = r_t * hn_prev_t. r_t is a pure gate, so ALL of term B's
                // relevance passes to hn_prev_t and none to r_t.
                const float R_term_b = (r_t * hn_p / n_denom) * R_n;

                // Step 6: hn_prev_t = h_{t-1} @ W_hn (no bias) -- a single-source linear
                // combination, epsilon rule over its own value.
                const float hn_sign = (hn_p >= 0.0f) ? 1.0f : -1.0f;
                const float hn_denom = hn_p + config.epsilon * hn_sign;
                for (int64_t kk = 0; kk < hidden_size_; ++kk) {
                    const float w2 = weight_hn_.data()[kk * hidden_size_ + k];
                    const float hp = h_prev.data()[n * hidden_size_ + kk];
                    // Step 7, contribution 2 of 2: the CANDIDATE-PATH split, accumulated
                    // onto the same R_hprev the direct split above wrote to.
                    R_hprev.data()[n * hidden_size_ + kk] += (hp * w2 / hn_denom) * R_term_b;
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
