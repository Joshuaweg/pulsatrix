#include "pulsatrix/lstm_module.hpp"

#include <stdexcept>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {
// Transposes a (rows x cols) row-major buffer into a (cols x rows) row-major buffer --
// same helper shape as LinearModule's/Conv2DModule's/RNNModule's own transpose()
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
}  // namespace

LSTMModule::LSTMModule(int64_t input_size, int64_t hidden_size, DeviceBackend* backend)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      backend_(backend),
      weight_xi_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_hi_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_i_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      weight_xf_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_hf_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_f_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      weight_xg_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_hg_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_g_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      weight_xo_(Shape({input_size > 0 ? input_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      weight_ho_(Shape({hidden_size > 0 ? hidden_size : 1, hidden_size > 0 ? hidden_size : 1}), backend),
      bias_o_(Shape({hidden_size > 0 ? hidden_size : 1}), backend),
      weight_xi_grad_(weight_xi_.shape(), backend),
      weight_hi_grad_(weight_hi_.shape(), backend),
      bias_i_grad_(bias_i_.shape(), backend),
      weight_xf_grad_(weight_xf_.shape(), backend),
      weight_hf_grad_(weight_hf_.shape(), backend),
      bias_f_grad_(bias_f_.shape(), backend),
      weight_xg_grad_(weight_xg_.shape(), backend),
      weight_hg_grad_(weight_hg_.shape(), backend),
      bias_g_grad_(bias_g_.shape(), backend),
      weight_xo_grad_(weight_xo_.shape(), backend),
      weight_ho_grad_(weight_ho_.shape(), backend),
      bias_o_grad_(bias_o_.shape(), backend),
      last_input_(Shape({0}), backend),
      last_hidden_states_(Shape({0}), backend),
      last_cell_states_(Shape({0}), backend),
      last_gate_i_(Shape({0}), backend),
      last_gate_f_(Shape({0}), backend),
      last_gate_g_(Shape({0}), backend),
      last_gate_o_(Shape({0}), backend),
      last_cell_tanh_(Shape({0}), backend),
      last_pre_activation_g_(Shape({0}), backend) {
    // External boundary (construction arguments can originate from Phase 5's Python
    // bindings with no upstream validation).
    if (input_size <= 0) {
        throw std::invalid_argument("LSTMModule: input_size must be positive");
    }
    if (hidden_size <= 0) {
        throw std::invalid_argument("LSTMModule: hidden_size must be positive");
    }
}

void LSTMModule::set_weight_xi(std::initializer_list<float> values) {
    weight_xi_ = Tensor(weight_xi_.shape(), backend_, values);
}
void LSTMModule::set_weight_hi(std::initializer_list<float> values) {
    weight_hi_ = Tensor(weight_hi_.shape(), backend_, values);
}
void LSTMModule::set_bias_i(std::initializer_list<float> values) {
    bias_i_ = Tensor(bias_i_.shape(), backend_, values);
}
void LSTMModule::set_weight_xf(std::initializer_list<float> values) {
    weight_xf_ = Tensor(weight_xf_.shape(), backend_, values);
}
void LSTMModule::set_weight_hf(std::initializer_list<float> values) {
    weight_hf_ = Tensor(weight_hf_.shape(), backend_, values);
}
void LSTMModule::set_bias_f(std::initializer_list<float> values) {
    bias_f_ = Tensor(bias_f_.shape(), backend_, values);
}
void LSTMModule::set_weight_xg(std::initializer_list<float> values) {
    weight_xg_ = Tensor(weight_xg_.shape(), backend_, values);
}
void LSTMModule::set_weight_hg(std::initializer_list<float> values) {
    weight_hg_ = Tensor(weight_hg_.shape(), backend_, values);
}
void LSTMModule::set_bias_g(std::initializer_list<float> values) {
    bias_g_ = Tensor(bias_g_.shape(), backend_, values);
}
void LSTMModule::set_weight_xo(std::initializer_list<float> values) {
    weight_xo_ = Tensor(weight_xo_.shape(), backend_, values);
}
void LSTMModule::set_weight_ho(std::initializer_list<float> values) {
    weight_ho_ = Tensor(weight_ho_.shape(), backend_, values);
}
void LSTMModule::set_bias_o(std::initializer_list<float> values) {
    bias_o_ = Tensor(bias_o_.shape(), backend_, values);
}

Tensor LSTMModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(input);

    if (input.rank() != 3 || input.shape().dim(2) != input_size_) {
        throw std::invalid_argument("LSTMModule::forward: input must be rank-3 (N, L, input_size)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);

    last_input_ = input;
    last_L_ = L;
    last_hidden_states_ = Tensor(Shape({N, L + 1, hidden_size_}), backend_);
    last_hidden_states_.fill(0.0f);  // h_0 = 0 for every example
    last_cell_states_ = Tensor(Shape({N, L + 1, hidden_size_}), backend_);
    last_cell_states_.fill(0.0f);  // c_0 = 0 for every example
    last_gate_i_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_gate_f_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_gate_g_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_gate_o_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_cell_tanh_ = Tensor(Shape({N, L, hidden_size_}), backend_);
    last_pre_activation_g_ = Tensor(Shape({N, L, hidden_size_}), backend_);

    Tensor output(Shape({N, L, hidden_size_}), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(output);

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

        Tensor z_i = gate_preactivation(x_t, h_prev, weight_xi_, weight_hi_);
        Tensor z_f = gate_preactivation(x_t, h_prev, weight_xf_, weight_hf_);
        Tensor z_g = gate_preactivation(x_t, h_prev, weight_xg_, weight_hg_);
        Tensor z_o = gate_preactivation(x_t, h_prev, weight_xo_, weight_ho_);

        // Adds the gate's bias into a fresh buffer, leaving the bias-free pre-activation
        // (which the LRP epsilon-rule denominator needs) intact in z.
        auto add_bias = [&](const Tensor& z, const Tensor& bias) {
            Tensor pre(Shape({N, hidden_size_}), backend_);
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t k = 0; k < hidden_size_; ++k) {
                    pre.data()[n * hidden_size_ + k] = z.data()[n * hidden_size_ + k] + bias.data()[k];
                }
            }
            return pre;
        };

        // Gate activations via the backend's Sigmoid/Tanh primitives (applied in place)
        // rather than raw host loops. z_g EXCLUDING bias is what the LRP epsilon-rule
        // denominator uses (cached below -- same convention as LinearModule/RNNModule: bias
        // has no associated input feature to redistribute relevance to, so it's excluded
        // from z entirely, which is what makes conservation exact rather than merely
        // approximate). The actual tanh activation still needs the real bias-included
        // pre-activation, which is what add_bias() builds.
        const size_t gate_n = static_cast<size_t>(N * hidden_size_);
        Tensor gate_i = add_bias(z_i, bias_i_);
        Tensor gate_f = add_bias(z_f, bias_f_);
        Tensor gate_g = add_bias(z_g, bias_g_);
        Tensor gate_o = add_bias(z_o, bias_o_);
        backend_->elementwise(ElementwiseOp::Sigmoid, gate_i.data(), gate_i.data(), gate_n);
        backend_->elementwise(ElementwiseOp::Sigmoid, gate_f.data(), gate_f.data(), gate_n);
        backend_->elementwise(ElementwiseOp::Tanh, gate_g.data(), gate_g.data(), gate_n);
        backend_->elementwise(ElementwiseOp::Sigmoid, gate_o.data(), gate_o.data(), gate_n);

        // c_t = f_t*c_{t-1} + i_t*g_t, then tanh(c_t) through the same backend primitive.
        Tensor cell(Shape({N, hidden_size_}), backend_);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                const int64_t idx = n * hidden_size_ + k;
                const float c_prev = last_cell_states_.data()[(n * (L + 1) + t) * hidden_size_ + k];
                cell.data()[idx] = gate_f.data()[idx] * c_prev + gate_i.data()[idx] * gate_g.data()[idx];
            }
        }
        Tensor cell_tanh(Shape({N, hidden_size_}), backend_);
        backend_->elementwise(ElementwiseOp::Tanh, cell.data(), cell_tanh.data(), gate_n);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                const int64_t idx = n * hidden_size_ + k;
                const int64_t seq_idx = (n * L + t) * hidden_size_ + k;

                const float i_t = gate_i.data()[idx];
                const float f_t = gate_f.data()[idx];
                const float g_t = gate_g.data()[idx];
                const float o_t = gate_o.data()[idx];

                const float c_t = cell.data()[idx];
                const float tanh_c = cell_tanh.data()[idx];
                const float h_t = o_t * tanh_c;

                last_gate_i_.data()[seq_idx] = i_t;
                last_gate_f_.data()[seq_idx] = f_t;
                last_gate_g_.data()[seq_idx] = g_t;
                last_gate_o_.data()[seq_idx] = o_t;
                last_cell_tanh_.data()[seq_idx] = tanh_c;
                last_pre_activation_g_.data()[seq_idx] = z_g.data()[idx];
                last_cell_states_.data()[(n * (L + 1) + t + 1) * hidden_size_ + k] = c_t;
                last_hidden_states_.data()[(n * (L + 1) + t + 1) * hidden_size_ + k] = h_t;
                output.data()[seq_idx] = h_t;
            }
        }
    }

    has_forwarded_ = true;
    return output;
}

Tensor LSTMModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("LSTMModule::backward: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    if (grad_output.rank() != 3 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != L ||
        grad_output.shape().dim(2) != hidden_size_) {
        throw std::invalid_argument(
            "LSTMModule::backward: grad_output must be (N, L, hidden_size) matching the cached forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(grad_output);

    Tensor grad_input(last_input_.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(grad_input);
    grad_input.fill(0.0f);

    auto zeroed_like = [&](const Tensor& t) {
        Tensor out(t.shape(), backend_);
        out.fill(0.0f);
        return out;
    };
    Tensor local_wxi_grad = zeroed_like(weight_xi_);
    Tensor local_whi_grad = zeroed_like(weight_hi_);
    Tensor local_bi_grad = zeroed_like(bias_i_);
    Tensor local_wxf_grad = zeroed_like(weight_xf_);
    Tensor local_whf_grad = zeroed_like(weight_hf_);
    Tensor local_bf_grad = zeroed_like(bias_f_);
    Tensor local_wxg_grad = zeroed_like(weight_xg_);
    Tensor local_whg_grad = zeroed_like(weight_hg_);
    Tensor local_bg_grad = zeroed_like(bias_g_);
    Tensor local_wxo_grad = zeroed_like(weight_xo_);
    Tensor local_who_grad = zeroed_like(weight_ho_);
    Tensor local_bo_grad = zeroed_like(bias_o_);

    Tensor dh_next(Shape({N, hidden_size_}), backend_);
    dh_next.fill(0.0f);
    Tensor dc_next(Shape({N, hidden_size_}), backend_);
    dc_next.fill(0.0f);

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

        // Per-gate pre-activation gradients. dz_* = dgate_* * d(activation)/dz, with the
        // sigmoid derivative written as s*(1-s) and tanh's as 1-g^2 -- both computed from
        // the cached activation itself, so no pre-activation re-evaluation is needed.
        Tensor dz_i(Shape({N, hidden_size_}), backend_);
        Tensor dz_f(Shape({N, hidden_size_}), backend_);
        Tensor dz_g(Shape({N, hidden_size_}), backend_);
        Tensor dz_o(Shape({N, hidden_size_}), backend_);
        Tensor dc_prev(Shape({N, hidden_size_}), backend_);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                const int64_t idx = n * hidden_size_ + k;
                const int64_t seq_idx = (n * L + t) * hidden_size_ + k;

                const float i_t = last_gate_i_.data()[seq_idx];
                const float f_t = last_gate_f_.data()[seq_idx];
                const float g_t = last_gate_g_.data()[seq_idx];
                const float o_t = last_gate_o_.data()[seq_idx];
                const float tanh_c = last_cell_tanh_.data()[seq_idx];
                const float c_prev = last_cell_states_.data()[(n * (L + 1) + t) * hidden_size_ + k];

                const float dh = grad_output.data()[seq_idx] + dh_next.data()[idx];
                const float dc = dh * o_t * (1.0f - tanh_c * tanh_c) + dc_next.data()[idx];

                dz_o.data()[idx] = dh * tanh_c * o_t * (1.0f - o_t);
                dz_f.data()[idx] = dc * c_prev * f_t * (1.0f - f_t);
                dz_i.data()[idx] = dc * g_t * i_t * (1.0f - i_t);
                dz_g.data()[idx] = dc * i_t * (1.0f - g_t * g_t);
                dc_prev.data()[idx] = dc * f_t;
            }
        }

        Tensor x_t_T = transpose(x_t, N, input_size_, backend_);
        Tensor h_prev_T = transpose(h_prev, N, hidden_size_, backend_);

        Tensor dx_t(Shape({N, input_size_}), backend_);
        dx_t.fill(0.0f);
        Tensor dh_prev(Shape({N, hidden_size_}), backend_);
        dh_prev.fill(0.0f);

        // One gate's contribution: parameter gradients plus its share of dx_t / dh_prev.
        // Every gate is structurally the same two-source affine map, so this runs four
        // times rather than being written out four times.
        auto accumulate_gate = [&](const Tensor& dz, const Tensor& wx, const Tensor& wh, Tensor& gwx_acc,
                                   Tensor& gwh_acc, Tensor& gb_acc) {
            Tensor gwx(wx.shape(), backend_);
            backend_->gemm(x_t_T.data(), dz.data(), gwx.data(), static_cast<size_t>(input_size_),
                           static_cast<size_t>(N), static_cast<size_t>(hidden_size_));
            gwx_acc.accumulate(gwx);

            Tensor gwh(wh.shape(), backend_);
            backend_->gemm(h_prev_T.data(), dz.data(), gwh.data(), static_cast<size_t>(hidden_size_),
                           static_cast<size_t>(N), static_cast<size_t>(hidden_size_));
            gwh_acc.accumulate(gwh);

            for (int64_t n = 0; n < N; ++n) {
                for (int64_t k = 0; k < hidden_size_; ++k) {
                    gb_acc.data()[k] += dz.data()[n * hidden_size_ + k];
                }
            }

            Tensor wx_T = transpose(wx, input_size_, hidden_size_, backend_);
            Tensor gx(Shape({N, input_size_}), backend_);
            backend_->gemm(dz.data(), wx_T.data(), gx.data(), static_cast<size_t>(N),
                           static_cast<size_t>(hidden_size_), static_cast<size_t>(input_size_));
            dx_t.accumulate(gx);

            Tensor wh_T = transpose(wh, hidden_size_, hidden_size_, backend_);
            Tensor gh(Shape({N, hidden_size_}), backend_);
            backend_->gemm(dz.data(), wh_T.data(), gh.data(), static_cast<size_t>(N),
                           static_cast<size_t>(hidden_size_), static_cast<size_t>(hidden_size_));
            dh_prev.accumulate(gh);
        };

        accumulate_gate(dz_i, weight_xi_, weight_hi_, local_wxi_grad, local_whi_grad, local_bi_grad);
        accumulate_gate(dz_f, weight_xf_, weight_hf_, local_wxf_grad, local_whf_grad, local_bf_grad);
        accumulate_gate(dz_g, weight_xg_, weight_hg_, local_wxg_grad, local_whg_grad, local_bg_grad);
        accumulate_gate(dz_o, weight_xo_, weight_ho_, local_wxo_grad, local_who_grad, local_bo_grad);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                grad_input.data()[(n * L + t) * input_size_ + i] = dx_t.data()[n * input_size_ + i];
            }
        }
        dh_next = dh_prev;
        dc_next = dc_prev;
    }

    weight_xi_grad_.accumulate(local_wxi_grad);
    weight_hi_grad_.accumulate(local_whi_grad);
    bias_i_grad_.accumulate(local_bi_grad);
    weight_xf_grad_.accumulate(local_wxf_grad);
    weight_hf_grad_.accumulate(local_whf_grad);
    bias_f_grad_.accumulate(local_bf_grad);
    weight_xg_grad_.accumulate(local_wxg_grad);
    weight_hg_grad_.accumulate(local_whg_grad);
    bias_g_grad_.accumulate(local_bg_grad);
    weight_xo_grad_.accumulate(local_wxo_grad);
    weight_ho_grad_.accumulate(local_who_grad);
    bias_o_grad_.accumulate(local_bo_grad);

    return grad_input;
}

Tensor LSTMModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("LSTMModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    if (relevance_out.rank() != 3 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != L ||
        relevance_out.shape().dim(2) != hidden_size_) {
        throw std::invalid_argument(
            "LSTMModule::propagate_relevance: relevance_out must be (N, L, hidden_size) matching the cached "
            "forward shape");
    }
    // Dereferences Tensor::data() directly -- not yet backend-generic.
    PULSATRIX_REQUIRE_HOST(relevance_out);

    Tensor relevance_in(last_input_.shape(), backend_);
    // Allocated through backend_ with no device tag, so a GPU backend tags it Cuda/Hip; every
    // later backend_-allocated temporary here shares that device. The raw host loops below
    // would be UB (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(relevance_in);
    relevance_in.fill(0.0f);

    // Two carried accumulators, mirroring backward()'s dh_next/dc_next: the relevance a
    // later timestep assigned to h_{t-1} and to c_{t-1} respectively.
    Tensor R_h_next(Shape({N, hidden_size_}), backend_);
    R_h_next.fill(0.0f);
    Tensor R_c_next(Shape({N, hidden_size_}), backend_);
    R_c_next.fill(0.0f);

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
        Tensor R_cprev(Shape({N, hidden_size_}), backend_);
        R_cprev.fill(0.0f);

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t k = 0; k < hidden_size_; ++k) {
                const int64_t idx = n * hidden_size_ + k;
                const int64_t seq_idx = (n * L + t) * hidden_size_ + k;

                const float i_t = last_gate_i_.data()[seq_idx];
                const float f_t = last_gate_f_.data()[seq_idx];
                const float g_t = last_gate_g_.data()[seq_idx];
                const float c_prev = last_cell_states_.data()[(n * (L + 1) + t) * hidden_size_ + k];
                const float c_t = last_cell_states_.data()[(n * (L + 1) + t + 1) * hidden_size_ + k];

                // h_t = o_t * tanh(c_t): o_t is a pure gate (Arras et al. 2019), so ALL of
                // R(h_t) passes to the single signal tanh(c_t), and tanh is an identity
                // pass-through onto c_t. Nothing is assigned to o_t at any point.
                const float R_h = relevance_out.data()[seq_idx] + R_h_next.data()[idx];
                const float R_c = R_h + R_c_next.data()[idx];

                // c_t = f_t*c_{t-1} + i_t*g_t: two-term weighted sum whose denominator is
                // exactly c_t. The gate values f_t/i_t are multiplicative weights here, not
                // relevance recipients -- only the signals c_{t-1} and g_t receive.
                const float cell_sign = (c_t >= 0.0f) ? 1.0f : -1.0f;
                const float cell_denom = c_t + config.epsilon * cell_sign;
                const float R_c_from_prev = (f_t * c_prev / cell_denom) * R_c;
                const float R_g = (i_t * g_t / cell_denom) * R_c;
                R_cprev.data()[idx] += R_c_from_prev;

                // g_t = tanh(z_g): tanh identity pass-through, then the same epsilon/z-rule
                // over two weighted sources (x_t, h_{t-1}) RNNModule uses. z_g excludes
                // bias, so this step conserves exactly.
                const float z_g = last_pre_activation_g_.data()[seq_idx];
                const float sign = (z_g >= 0.0f) ? 1.0f : -1.0f;
                const float denom = z_g + config.epsilon * sign;

                for (int64_t i = 0; i < input_size_; ++i) {
                    const float w = weight_xg_.data()[i * hidden_size_ + k];
                    const float x = x_t.data()[n * input_size_ + i];
                    R_x.data()[n * input_size_ + i] += (x * w / denom) * R_g;
                }
                for (int64_t kk = 0; kk < hidden_size_; ++kk) {
                    const float w2 = weight_hg_.data()[kk * hidden_size_ + k];
                    const float hp = h_prev.data()[n * hidden_size_ + kk];
                    R_hprev.data()[n * hidden_size_ + kk] += (hp * w2 / denom) * R_g;
                }
            }
        }

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t i = 0; i < input_size_; ++i) {
                relevance_in.data()[(n * L + t) * input_size_ + i] = R_x.data()[n * input_size_ + i];
            }
        }
        R_h_next = R_hprev;
        R_c_next = R_cprev;
    }

    return relevance_in;
}

}  // namespace pulsatrix
