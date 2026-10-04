#include "pulsatrix/lstm_module.hpp"

#include <stdexcept>

namespace pulsatrix {

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

    // Device-generic (GPU-native-kernels Mission 5): every step below is a DeviceBackend
    // primitive reproducing the former host loop's per-element expression and order.
    const auto n = static_cast<size_t>(N);
    const auto d = static_cast<size_t>(input_size_);
    const auto h = static_cast<size_t>(hidden_size_);
    const auto lu = static_cast<size_t>(L);
    const size_t gate_n = n * h;

    Tensor output(Shape({N, L, hidden_size_}), backend_);

    // z = x_t @ Wx + h_prev @ Wh, EXCLUDING bias -- one gate's pre-activation.
    auto gate_preactivation = [&](const Tensor& x_t, const Tensor& h_prev, const Tensor& wx, const Tensor& wh) {
        Tensor z1(Shape({N, hidden_size_}), backend_);
        backend_->gemm(x_t.data(), wx.data(), z1.data(), n, d, h);
        Tensor z2(Shape({N, hidden_size_}), backend_);
        backend_->gemm(h_prev.data(), wh.data(), z2.data(), n, h, h);
        z1.accumulate(z2);
        return z1;
    };

    // Adds the gate's bias into a fresh buffer, leaving the bias-free pre-activation (which
    // the LRP epsilon-rule denominator needs) intact in z.
    auto add_bias = [&](const Tensor& z, const Tensor& bias) {
        Tensor pre(Shape({N, hidden_size_}), backend_);
        backend_->add_row_vector(z.data(), bias.data(), pre.data(), n, h);
        return pre;
    };

    // Writes a contiguous (N, H) timestep buffer into slice t of an (N, L, H) cache.
    auto store_step = [&](Tensor& seq, const Tensor& step, size_t tu) {
        backend_->copy_2d(seq.data() + tu * h, lu * h, step.data(), h, n, h);
    };

    for (int64_t t = 0; t < L; ++t) {
        const auto tu = static_cast<size_t>(t);
        Tensor x_t(Shape({N, input_size_}), backend_);
        backend_->copy_2d(x_t.data(), d, input.data() + tu * d, lu * d, n, d);
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(h_prev.data(), h, last_hidden_states_.data() + tu * h, (lu + 1) * h, n, h);

        Tensor z_i = gate_preactivation(x_t, h_prev, weight_xi_, weight_hi_);
        Tensor z_f = gate_preactivation(x_t, h_prev, weight_xf_, weight_hf_);
        Tensor z_g = gate_preactivation(x_t, h_prev, weight_xg_, weight_hg_);
        Tensor z_o = gate_preactivation(x_t, h_prev, weight_xo_, weight_ho_);

        // Gate activations via the backend's Sigmoid/Tanh primitives (applied in place). z_g
        // EXCLUDING bias is what the LRP epsilon-rule denominator uses (cached below -- same
        // convention as LinearModule/RNNModule: bias has no associated input feature to
        // redistribute relevance to, so it's excluded from z entirely, which is what makes
        // conservation exact rather than merely approximate). The actual tanh activation still
        // needs the real bias-included pre-activation, which is what add_bias() builds.
        Tensor gate_i = add_bias(z_i, bias_i_);
        Tensor gate_f = add_bias(z_f, bias_f_);
        Tensor gate_g = add_bias(z_g, bias_g_);
        Tensor gate_o = add_bias(z_o, bias_o_);
        backend_->elementwise(ElementwiseOp::Sigmoid, gate_i.data(), gate_i.data(), gate_n);
        backend_->elementwise(ElementwiseOp::Sigmoid, gate_f.data(), gate_f.data(), gate_n);
        backend_->elementwise(ElementwiseOp::Tanh, gate_g.data(), gate_g.data(), gate_n);
        backend_->elementwise(ElementwiseOp::Sigmoid, gate_o.data(), gate_o.data(), gate_n);

        // c_t = f_t*c_{t-1} + i_t*g_t, tanh(c_t), h_t = o_t*tanh(c_t) in one fused pass.
        Tensor c_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(c_prev.data(), h, last_cell_states_.data() + tu * h, (lu + 1) * h, n, h);
        Tensor cell(Shape({N, hidden_size_}), backend_);
        Tensor cell_tanh(Shape({N, hidden_size_}), backend_);
        Tensor h_t(Shape({N, hidden_size_}), backend_);
        RecurrentCellArgs args;
        args.in[0] = gate_i.data();
        args.in[1] = gate_f.data();
        args.in[2] = gate_g.data();
        args.in[3] = gate_o.data();
        args.in[4] = c_prev.data();
        args.out[0] = cell.data();
        args.out[1] = cell_tanh.data();
        args.out[2] = h_t.data();
        backend_->recurrent_cell(RecurrentCellOp::LstmForward, args, gate_n);

        store_step(last_gate_i_, gate_i, tu);
        store_step(last_gate_f_, gate_f, tu);
        store_step(last_gate_g_, gate_g, tu);
        store_step(last_gate_o_, gate_o, tu);
        store_step(last_cell_tanh_, cell_tanh, tu);
        store_step(last_pre_activation_g_, z_g, tu);
        backend_->copy_2d(last_cell_states_.data() + (tu + 1) * h, (lu + 1) * h, cell.data(), h, n, h);
        backend_->copy_2d(last_hidden_states_.data() + (tu + 1) * h, (lu + 1) * h, h_t.data(), h, n, h);
        store_step(output, h_t, tu);
    }

    has_forwarded_ = true;
    return output;
}

Tensor LSTMModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "LSTMModule::backward");
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

    const auto n = static_cast<size_t>(N);
    const auto d = static_cast<size_t>(input_size_);
    const auto h = static_cast<size_t>(hidden_size_);
    const auto lu = static_cast<size_t>(L);

    Tensor grad_input(last_input_.shape(), backend_);
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

    // Contiguous (N, H) copy of timestep t of an (N, L, H) cache.
    auto load_step = [&](const Tensor& seq, size_t tu) {
        Tensor out(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(out.data(), h, seq.data() + tu * h, lu * h, n, h);
        return out;
    };

    for (int64_t t = L - 1; t >= 0; --t) {
        const auto tu = static_cast<size_t>(t);
        Tensor x_t(Shape({N, input_size_}), backend_);
        backend_->copy_2d(x_t.data(), d, last_input_.data() + tu * d, lu * d, n, d);
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(h_prev.data(), h, last_hidden_states_.data() + tu * h, (lu + 1) * h, n, h);
        Tensor c_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(c_prev.data(), h, last_cell_states_.data() + tu * h, (lu + 1) * h, n, h);

        Tensor g_t = load_step(grad_output, tu);
        Tensor i_t = load_step(last_gate_i_, tu);
        Tensor f_t = load_step(last_gate_f_, tu);
        Tensor gg_t = load_step(last_gate_g_, tu);
        Tensor o_t = load_step(last_gate_o_, tu);
        Tensor tanh_c = load_step(last_cell_tanh_, tu);

        // Per-gate pre-activation gradients. dz_* = dgate_* * d(activation)/dz, with the
        // sigmoid derivative written as s*(1-s) and tanh's as 1-g^2 -- both computed from
        // the cached activation itself, so no pre-activation re-evaluation is needed.
        Tensor dz_i(Shape({N, hidden_size_}), backend_);
        Tensor dz_f(Shape({N, hidden_size_}), backend_);
        Tensor dz_g(Shape({N, hidden_size_}), backend_);
        Tensor dz_o(Shape({N, hidden_size_}), backend_);
        Tensor dc_prev(Shape({N, hidden_size_}), backend_);
        RecurrentCellArgs args;
        args.in[0] = g_t.data();
        args.in[1] = dh_next.data();
        args.in[2] = dc_next.data();
        args.in[3] = i_t.data();
        args.in[4] = f_t.data();
        args.in[5] = gg_t.data();
        args.in[6] = o_t.data();
        args.in[7] = tanh_c.data();
        args.in[8] = c_prev.data();
        args.out[0] = dz_i.data();
        args.out[1] = dz_f.data();
        args.out[2] = dz_g.data();
        args.out[3] = dz_o.data();
        args.out[4] = dc_prev.data();
        backend_->recurrent_cell(RecurrentCellOp::LstmBackward, args, n * h);

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
            backend_->gemm_ex(x_t.data(), true, dz.data(), false, gwx.data(), d, n, h, 0.0f);
            gwx_acc.accumulate(gwx);

            Tensor gwh(wh.shape(), backend_);
            backend_->gemm_ex(h_prev.data(), true, dz.data(), false, gwh.data(), h, n, h, 0.0f);
            gwh_acc.accumulate(gwh);

            backend_->accumulate_rows(dz.data(), gb_acc.data(), n, h);

            Tensor gx(Shape({N, input_size_}), backend_);
            backend_->gemm_ex(dz.data(), false, wx.data(), true, gx.data(), n, h, d, 0.0f);
            dx_t.accumulate(gx);

            Tensor gh(Shape({N, hidden_size_}), backend_);
            backend_->gemm_ex(dz.data(), false, wh.data(), true, gh.data(), n, h, h, 0.0f);
            dh_prev.accumulate(gh);
        };

        accumulate_gate(dz_i, weight_xi_, weight_hi_, local_wxi_grad, local_whi_grad, local_bi_grad);
        accumulate_gate(dz_f, weight_xf_, weight_hf_, local_wxf_grad, local_whf_grad, local_bf_grad);
        accumulate_gate(dz_g, weight_xg_, weight_hg_, local_wxg_grad, local_whg_grad, local_bg_grad);
        accumulate_gate(dz_o, weight_xo_, weight_ho_, local_wxo_grad, local_who_grad, local_bo_grad);

        backend_->copy_2d(grad_input.data() + tu * d, lu * d, dx_t.data(), d, n, d);
        dh_next = dh_prev;
        dc_next = dc_prev;
    }

    // A frozen parameter (FND-2) accumulates nothing. Its local gradient is still computed
    // above: here it is cheap, or entangled with the input gradient's own recurrence.
    if (weight_xi_.requires_grad()) {
        weight_xi_grad_.accumulate(local_wxi_grad);
    }
    if (weight_hi_.requires_grad()) {
        weight_hi_grad_.accumulate(local_whi_grad);
    }
    if (bias_i_.requires_grad()) {
        bias_i_grad_.accumulate(local_bi_grad);
    }
    if (weight_xf_.requires_grad()) {
        weight_xf_grad_.accumulate(local_wxf_grad);
    }
    if (weight_hf_.requires_grad()) {
        weight_hf_grad_.accumulate(local_whf_grad);
    }
    if (bias_f_.requires_grad()) {
        bias_f_grad_.accumulate(local_bf_grad);
    }
    if (weight_xg_.requires_grad()) {
        weight_xg_grad_.accumulate(local_wxg_grad);
    }
    if (weight_hg_.requires_grad()) {
        weight_hg_grad_.accumulate(local_whg_grad);
    }
    if (bias_g_.requires_grad()) {
        bias_g_grad_.accumulate(local_bg_grad);
    }
    if (weight_xo_.requires_grad()) {
        weight_xo_grad_.accumulate(local_wxo_grad);
    }
    if (weight_ho_.requires_grad()) {
        weight_ho_grad_.accumulate(local_who_grad);
    }
    if (bias_o_.requires_grad()) {
        bias_o_grad_.accumulate(local_bo_grad);
    }

    return grad_input;
}

Tensor LSTMModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "LSTMModule::propagate_relevance");
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

    const auto n = static_cast<size_t>(N);
    const auto d = static_cast<size_t>(input_size_);
    const auto h = static_cast<size_t>(hidden_size_);
    const auto lu = static_cast<size_t>(L);

    Tensor relevance_in(last_input_.shape(), backend_);
    relevance_in.fill(0.0f);

    // Two carried accumulators, mirroring backward()'s dh_next/dc_next: the relevance a
    // later timestep assigned to h_{t-1} and to c_{t-1} respectively.
    Tensor R_h_next(Shape({N, hidden_size_}), backend_);
    R_h_next.fill(0.0f);
    Tensor R_c_next(Shape({N, hidden_size_}), backend_);
    R_c_next.fill(0.0f);

    // Contiguous (N, H) copy of timestep t of an (N, L, H) cache.
    auto load_step = [&](const Tensor& seq, size_t tu) {
        Tensor out(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(out.data(), h, seq.data() + tu * h, lu * h, n, h);
        return out;
    };

    for (int64_t t = L - 1; t >= 0; --t) {
        const auto tu = static_cast<size_t>(t);
        Tensor x_t(Shape({N, input_size_}), backend_);
        backend_->copy_2d(x_t.data(), d, last_input_.data() + tu * d, lu * d, n, d);
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(h_prev.data(), h, last_hidden_states_.data() + tu * h, (lu + 1) * h, n, h);
        Tensor c_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(c_prev.data(), h, last_cell_states_.data() + tu * h, (lu + 1) * h, n, h);
        Tensor c_t(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(c_t.data(), h, last_cell_states_.data() + (tu + 1) * h, (lu + 1) * h, n, h);

        Tensor r_t = load_step(relevance_out, tu);
        Tensor i_t = load_step(last_gate_i_, tu);
        Tensor f_t = load_step(last_gate_f_, tu);
        Tensor g_t = load_step(last_gate_g_, tu);
        Tensor z_g = load_step(last_pre_activation_g_, tu);

        // h_t = o_t * tanh(c_t): o_t is a pure gate (Arras et al. 2019), so ALL of R(h_t)
        // passes to the single signal tanh(c_t), and tanh is an identity pass-through onto
        // c_t. Nothing is assigned to o_t at any point. c_t = f_t*c_{t-1} + i_t*g_t is a
        // two-term weighted sum whose denominator is exactly c_t; the gate values f_t/i_t are
        // multiplicative weights there, not relevance recipients -- only the signals c_{t-1}
        // and g_t receive (LstmLrp).
        Tensor R_g(Shape({N, hidden_size_}), backend_);
        Tensor R_cprev(Shape({N, hidden_size_}), backend_);
        RecurrentCellArgs args;
        args.in[0] = r_t.data();
        args.in[1] = R_h_next.data();
        args.in[2] = R_c_next.data();
        args.in[3] = i_t.data();
        args.in[4] = f_t.data();
        args.in[5] = g_t.data();
        args.in[6] = c_prev.data();
        args.in[7] = c_t.data();
        args.out[0] = R_g.data();
        args.out[1] = R_cprev.data();
        args.eps = config.epsilon;
        backend_->recurrent_cell(RecurrentCellOp::LstmLrp, args, n * h);

        // g_t = tanh(z_g): tanh identity pass-through, then the same epsilon/z-rule over two
        // weighted sources (x_t, h_{t-1}) RNNModule uses. z_g excludes bias, so this step
        // conserves exactly.
        Tensor R_x(Shape({N, input_size_}), backend_);
        backend_->lrp_linear(x_t.data(), weight_xg_.data(), z_g.data(), R_g.data(), R_x.data(), n, d, h,
                             config.epsilon);
        Tensor R_hprev(Shape({N, hidden_size_}), backend_);
        backend_->lrp_linear(h_prev.data(), weight_hg_.data(), z_g.data(), R_g.data(), R_hprev.data(), n, h, h,
                             config.epsilon);

        backend_->copy_2d(relevance_in.data() + tu * d, lu * d, R_x.data(), d, n, d);
        R_h_next = R_hprev;
        R_c_next = R_cprev;
    }

    return relevance_in;
}

}  // namespace pulsatrix
