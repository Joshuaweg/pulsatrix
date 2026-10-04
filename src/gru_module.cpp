#include "pulsatrix/gru_module.hpp"

#include <stdexcept>

namespace pulsatrix {

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

    // Device-generic (GPU-native-kernels Mission 5): every step below is a DeviceBackend
    // primitive reproducing the former host loop's per-element expression and order.
    const auto n = static_cast<size_t>(N);
    const auto d = static_cast<size_t>(input_size_);
    const auto h = static_cast<size_t>(hidden_size_);
    const auto lu = static_cast<size_t>(L);
    const size_t gate_n = n * h;

    Tensor output(Shape({N, L, hidden_size_}), backend_);

    // Constant 1s, so axpby(-1, z, 1, ones) yields exactly 1 - z for the update-gate blend.
    Tensor ones(Shape({N, hidden_size_}), backend_);
    ones.fill(1.0f);

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

        Tensor z_z = gate_preactivation(x_t, h_prev, weight_xz_, weight_hz_);
        Tensor z_r = gate_preactivation(x_t, h_prev, weight_xr_, weight_hr_);

        // The candidate's two sources are NOT symmetric: x_t goes through W_xn directly,
        // while h_{t-1} goes through its own projection W_hn whose *output* the reset gate
        // multiplies. Both halves are computed separately for exactly that reason.
        Tensor xn(Shape({N, hidden_size_}), backend_);
        backend_->gemm(x_t.data(), weight_xn_.data(), xn.data(), n, d, h);
        Tensor hn_prev(Shape({N, hidden_size_}), backend_);
        backend_->gemm(h_prev.data(), weight_hn_.data(), hn_prev.data(), n, h, h);

        // Update/reset gates via the backend's Sigmoid primitive (in place).
        Tensor gate_z = add_bias(z_z, bias_z_);
        Tensor gate_r = add_bias(z_r, bias_r_);
        backend_->elementwise(ElementwiseOp::Sigmoid, gate_z.data(), gate_z.data(), gate_n);
        backend_->elementwise(ElementwiseOp::Sigmoid, gate_r.data(), gate_r.data(), gate_n);

        // The candidate's pre-activation depends on the reset gate, so it can only be built
        // once gate_r exists -- hence a second activation stage rather than one batched with
        // the gates above. n_pre EXCLUDING bias is what the LRP epsilon-rule denominator uses
        // (cached below -- same convention as LinearModule/RNNModule/LSTMModule: bias has no
        // associated input feature to redistribute relevance to, so it's excluded from the
        // denominator entirely). The actual tanh activation still needs the real
        // bias-included pre-activation. n_pre_no_bias = xn + (r * hn).
        Tensor r_hn(Shape({N, hidden_size_}), backend_);
        backend_->mul(gate_r.data(), hn_prev.data(), r_hn.data(), gate_n);
        Tensor n_pre_no_bias(Shape({N, hidden_size_}), backend_);
        backend_->add(xn.data(), r_hn.data(), n_pre_no_bias.data(), gate_n);
        Tensor candidate = add_bias(n_pre_no_bias, bias_n_);
        backend_->elementwise(ElementwiseOp::Tanh, candidate.data(), candidate.data(), gate_n);

        // h_t = ((1 - z_t) * h_{t-1}) + (z_t * n_t), one rounding per operation as before.
        Tensor one_minus_z(Shape({N, hidden_size_}), backend_);
        backend_->axpby(-1.0f, gate_z.data(), 1.0f, ones.data(), one_minus_z.data(), gate_n);
        Tensor carry(Shape({N, hidden_size_}), backend_);
        backend_->mul(one_minus_z.data(), h_prev.data(), carry.data(), gate_n);
        Tensor update(Shape({N, hidden_size_}), backend_);
        backend_->mul(gate_z.data(), candidate.data(), update.data(), gate_n);
        Tensor h_t(Shape({N, hidden_size_}), backend_);
        backend_->add(carry.data(), update.data(), h_t.data(), gate_n);

        store_step(last_gate_z_, gate_z, tu);
        store_step(last_gate_r_, gate_r, tu);
        store_step(last_candidate_n_, candidate, tu);
        store_step(last_hn_prev_, hn_prev, tu);
        store_step(last_pre_activation_n_, n_pre_no_bias, tu);
        backend_->copy_2d(last_hidden_states_.data() + (tu + 1) * h, (lu + 1) * h, h_t.data(), h, n, h);
        store_step(output, h_t, tu);
    }

    has_forwarded_ = true;
    return output;
}

Tensor GRUModule::backward(const Tensor& grad_output) {
    require_device(grad_output, *compute_device(), "GRUModule::backward");
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

        Tensor g_t = load_step(grad_output, tu);
        Tensor z_t = load_step(last_gate_z_, tu);
        Tensor r_t = load_step(last_gate_r_, tu);
        Tensor n_t = load_step(last_candidate_n_, tu);
        Tensor hn_p = load_step(last_hn_prev_, tu);

        // h_t = (1-z_t)*h_{t-1} + z_t*n_t -- three consumers of dh: the direct convex carry
        // onto h_{t-1} (which starts dh_prev), the candidate, and the update gate itself
        // (which sees n_t - h_{t-1}). n_t = tanh(n_pre): tanh' written as 1 - n^2 from the
        // cached activation. n_pre's term B = r_t * hn_prev_t: a genuine product of two
        // computed quantities, so it splits into a reset-gate branch and a projection branch.
        Tensor dh_prev(Shape({N, hidden_size_}), backend_);
        Tensor dz_pre(Shape({N, hidden_size_}), backend_);
        Tensor dn_pre(Shape({N, hidden_size_}), backend_);
        Tensor dr_pre(Shape({N, hidden_size_}), backend_);
        Tensor dhn_prev(Shape({N, hidden_size_}), backend_);
        RecurrentCellArgs args;
        args.in[0] = g_t.data();
        args.in[1] = dh_next.data();
        args.in[2] = z_t.data();
        args.in[3] = r_t.data();
        args.in[4] = n_t.data();
        args.in[5] = hn_p.data();
        args.in[6] = h_prev.data();
        args.out[0] = dh_prev.data();
        args.out[1] = dz_pre.data();
        args.out[2] = dn_pre.data();
        args.out[3] = dr_pre.data();
        args.out[4] = dhn_prev.data();
        backend_->recurrent_cell(RecurrentCellOp::GruBackward, args, n * h);

        Tensor dx_t(Shape({N, input_size_}), backend_);
        dx_t.fill(0.0f);

        // One x-side weight matrix's contribution: its parameter gradient plus its share of
        // dx_t. Used by both gates and by the candidate's W_xn.
        auto accumulate_x_path = [&](const Tensor& dz, const Tensor& wx, Tensor& gwx_acc) {
            Tensor gwx(wx.shape(), backend_);
            backend_->gemm_ex(x_t.data(), true, dz.data(), false, gwx.data(), d, n, h, 0.0f);
            gwx_acc.accumulate(gwx);

            Tensor gx(Shape({N, input_size_}), backend_);
            backend_->gemm_ex(dz.data(), false, wx.data(), true, gx.data(), n, h, d, 0.0f);
            dx_t.accumulate(gx);
        };

        // One h-side weight matrix's contribution: its parameter gradient plus its share of
        // dh_prev. Used by both gates and by the candidate's W_hn projection -- the latter
        // is the second of the three paths that feed the same dh_prev accumulator (the
        // gradient-side analogue of propagate_relevance's two-path R(h_{t-1})).
        auto accumulate_h_path = [&](const Tensor& dz, const Tensor& wh, Tensor& gwh_acc) {
            Tensor gwh(wh.shape(), backend_);
            backend_->gemm_ex(h_prev.data(), true, dz.data(), false, gwh.data(), h, n, h, 0.0f);
            gwh_acc.accumulate(gwh);

            Tensor gh(Shape({N, hidden_size_}), backend_);
            backend_->gemm_ex(dz.data(), false, wh.data(), true, gh.data(), n, h, h, 0.0f);
            dh_prev.accumulate(gh);
        };

        auto accumulate_bias = [&](const Tensor& dz, Tensor& gb_acc) {
            backend_->accumulate_rows(dz.data(), gb_acc.data(), n, h);
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

        backend_->copy_2d(grad_input.data() + tu * d, lu * d, dx_t.data(), d, n, d);
        dh_next = dh_prev;
    }

    // A frozen parameter (FND-2) accumulates nothing. Its local gradient is still computed
    // above: here it is cheap, or entangled with the input gradient's own recurrence.
    if (weight_xz_.requires_grad()) {
        weight_xz_grad_.accumulate(local_wxz_grad);
    }
    if (weight_hz_.requires_grad()) {
        weight_hz_grad_.accumulate(local_whz_grad);
    }
    if (bias_z_.requires_grad()) {
        bias_z_grad_.accumulate(local_bz_grad);
    }
    if (weight_xr_.requires_grad()) {
        weight_xr_grad_.accumulate(local_wxr_grad);
    }
    if (weight_hr_.requires_grad()) {
        weight_hr_grad_.accumulate(local_whr_grad);
    }
    if (bias_r_.requires_grad()) {
        bias_r_grad_.accumulate(local_br_grad);
    }
    if (weight_xn_.requires_grad()) {
        weight_xn_grad_.accumulate(local_wxn_grad);
    }
    if (weight_hn_.requires_grad()) {
        weight_hn_grad_.accumulate(local_whn_grad);
    }
    if (bias_n_.requires_grad()) {
        bias_n_grad_.accumulate(local_bn_grad);
    }

    return grad_input;
}

Tensor GRUModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "GRUModule::propagate_relevance");
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

    const auto n = static_cast<size_t>(N);
    const auto d = static_cast<size_t>(input_size_);
    const auto h = static_cast<size_t>(hidden_size_);
    const auto lu = static_cast<size_t>(L);

    Tensor relevance_in(last_input_.shape(), backend_);
    relevance_in.fill(0.0f);

    // One carried accumulator, mirroring backward()'s dh_next: the relevance a later
    // timestep assigned to h_{t-1}. Unlike RNNModule's (one source) and LSTMModule's (one
    // source per accumulator), this one is written by TWO distinct paths per timestep --
    // see the class-level note, step 7.
    Tensor R_h_next(Shape({N, hidden_size_}), backend_);
    R_h_next.fill(0.0f);

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
        Tensor h_t(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(h_t.data(), h, last_hidden_states_.data() + (tu + 1) * h, (lu + 1) * h, n, h);

        Tensor r_out = load_step(relevance_out, tu);
        Tensor z_t = load_step(last_gate_z_, tu);
        Tensor r_t = load_step(last_gate_r_, tu);
        Tensor n_t = load_step(last_candidate_n_, tu);
        Tensor hn_p = load_step(last_hn_prev_, tu);
        Tensor n_pre = load_step(last_pre_activation_n_, tu);

        // Step 1: h_t = (1-z_t)*h_{t-1} + z_t*n_t -- two-term weighted sum whose denominator
        // is exactly h_t; z_t is a multiplicative weight, NOT a relevance recipient (direct
        // split + R_n). Steps 2-5: tanh pass-through onto n_pre, then term B = r_t*hn_prev_t
        // over the shared n_pre denominator, all of it passing to hn_prev_t (R_term_b).
        Tensor R_hprev_direct(Shape({N, hidden_size_}), backend_);
        Tensor R_n(Shape({N, hidden_size_}), backend_);
        Tensor R_term_b(Shape({N, hidden_size_}), backend_);
        RecurrentCellArgs args;
        args.in[0] = r_out.data();
        args.in[1] = R_h_next.data();
        args.in[2] = z_t.data();
        args.in[3] = r_t.data();
        args.in[4] = n_t.data();
        args.in[5] = hn_p.data();
        args.in[6] = h_prev.data();
        args.in[7] = h_t.data();
        args.in[8] = n_pre.data();
        args.out[0] = R_hprev_direct.data();
        args.out[1] = R_n.data();
        args.out[2] = R_term_b.data();
        args.eps = config.epsilon;
        backend_->recurrent_cell(RecurrentCellOp::GruLrp, args, n * h);

        // Steps 3+4: the x_t@W_xn share of R(n_pre), redistributed across x_t's features over
        // the shared n_pre denominator (the intermediate term-A denominator cancels).
        Tensor R_x(Shape({N, input_size_}), backend_);
        backend_->lrp_linear(x_t.data(), weight_xn_.data(), n_pre.data(), R_n.data(), R_x.data(), n, d, h,
                             config.epsilon);

        // Steps 6+7: hn_prev_t = h_{t-1} @ W_hn (no bias), epsilon rule over its own value,
        // summed onto the direct split -- each element's direct term inserted at its own k,
        // exactly where the original loop added it.
        Tensor R_hprev(Shape({N, hidden_size_}), backend_);
        backend_->gru_lrp_hprev(h_prev.data(), weight_hn_.data(), hn_p.data(), R_term_b.data(),
                                R_hprev_direct.data(), R_hprev.data(), n, h, config.epsilon);

        backend_->copy_2d(relevance_in.data() + tu * d, lu * d, R_x.data(), d, n, d);
        R_h_next = R_hprev;
    }

    return relevance_in;
}

}  // namespace pulsatrix
