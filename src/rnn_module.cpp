#include "pulsatrix/rnn_module.hpp"

#include <stdexcept>

namespace pulsatrix {

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

    // Device-generic (GPU-native-kernels Mission 5): every step below is a DeviceBackend
    // primitive reproducing the former host loop's per-element expression and order.
    const auto n = static_cast<size_t>(N);
    const auto d = static_cast<size_t>(input_size_);
    const auto h = static_cast<size_t>(hidden_size_);
    const auto lu = static_cast<size_t>(L);
    const size_t nh = n * h;

    Tensor output(Shape({N, L, hidden_size_}), backend_);

    for (int64_t t = 0; t < L; ++t) {
        const auto tu = static_cast<size_t>(t);
        Tensor x_t(Shape({N, input_size_}), backend_);
        backend_->copy_2d(x_t.data(), d, input.data() + tu * d, lu * d, n, d);
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(h_prev.data(), h, last_hidden_states_.data() + tu * h, (lu + 1) * h, n, h);

        Tensor z1(Shape({N, hidden_size_}), backend_);
        backend_->gemm(x_t.data(), weight_xh_.data(), z1.data(), n, d, h);
        Tensor z2(Shape({N, hidden_size_}), backend_);
        backend_->gemm(h_prev.data(), weight_hh_.data(), z2.data(), n, h, h);

        // z_no_bias is what the LRP epsilon-rule denominator uses (cached below, excluding
        // bias -- same convention as LinearModule/Conv2DModule: bias has no associated input
        // feature to redistribute relevance to, so it's excluded from z entirely, which is
        // what makes conservation exact rather than merely approximate). The actual tanh
        // activation still needs the real bias-included pre-activation.
        Tensor z_no_bias(Shape({N, hidden_size_}), backend_);
        backend_->add(z1.data(), z2.data(), z_no_bias.data(), nh);
        backend_->copy_2d(last_pre_activation_.data() + tu * h, lu * h, z_no_bias.data(), h, n, h);
        Tensor pre_act(Shape({N, hidden_size_}), backend_);
        backend_->add_row_vector(z_no_bias.data(), bias_.data(), pre_act.data(), n, h);

        // h = tanh(pre_act), in place.
        backend_->elementwise(ElementwiseOp::Tanh, pre_act.data(), pre_act.data(), nh);

        backend_->copy_2d(last_hidden_states_.data() + (tu + 1) * h, (lu + 1) * h, pre_act.data(), h, n, h);
        backend_->copy_2d(output.data() + tu * h, lu * h, pre_act.data(), h, n, h);
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

    const auto n = static_cast<size_t>(N);
    const auto d = static_cast<size_t>(input_size_);
    const auto h = static_cast<size_t>(hidden_size_);
    const auto lu = static_cast<size_t>(L);

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
        const auto tu = static_cast<size_t>(t);
        Tensor g_t(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(g_t.data(), h, grad_output.data() + tu * h, lu * h, n, h);
        Tensor h_t(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(h_t.data(), h, last_hidden_states_.data() + (tu + 1) * h, (lu + 1) * h, n, h);

        // dz = (g + dh_next) * (1 - h*h)
        Tensor dz(Shape({N, hidden_size_}), backend_);
        RecurrentCellArgs args;
        args.in[0] = g_t.data();
        args.in[1] = dh_next.data();
        args.in[2] = h_t.data();
        args.out[0] = dz.data();
        backend_->recurrent_cell(RecurrentCellOp::RnnBackward, args, n * h);

        Tensor x_t(Shape({N, input_size_}), backend_);
        backend_->copy_2d(x_t.data(), d, last_input_.data() + tu * d, lu * d, n, d);
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(h_prev.data(), h, last_hidden_states_.data() + tu * h, (lu + 1) * h, n, h);

        // grad_Wxh += x_t^T @ dz
        Tensor gwxh(weight_xh_.shape(), backend_);
        backend_->gemm_ex(x_t.data(), true, dz.data(), false, gwxh.data(), d, n, h, 0.0f);
        local_wxh_grad.accumulate(gwxh);

        // grad_Whh += h_prev^T @ dz
        Tensor gwhh(weight_hh_.shape(), backend_);
        backend_->gemm_ex(h_prev.data(), true, dz.data(), false, gwhh.data(), h, n, h, 0.0f);
        local_whh_grad.accumulate(gwhh);

        // grad_bh += sum over batch of dz (row by row, n ascending)
        backend_->accumulate_rows(dz.data(), local_bh_grad.data(), n, h);

        // grad_x_t = dz @ Wxh^T
        Tensor gx(Shape({N, input_size_}), backend_);
        backend_->gemm_ex(dz.data(), false, weight_xh_.data(), true, gx.data(), n, h, d, 0.0f);
        backend_->copy_2d(grad_input.data() + tu * d, lu * d, gx.data(), d, n, d);

        // dh_next (for t-1) = dz @ Whh^T
        Tensor dh_prev(Shape({N, hidden_size_}), backend_);
        backend_->gemm_ex(dz.data(), false, weight_hh_.data(), true, dh_prev.data(), n, h, h, 0.0f);
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

    const auto n = static_cast<size_t>(N);
    const auto d = static_cast<size_t>(input_size_);
    const auto h = static_cast<size_t>(hidden_size_);
    const auto lu = static_cast<size_t>(L);

    Tensor relevance_in(last_input_.shape(), backend_);
    relevance_in.fill(0.0f);

    Tensor R_h_next(Shape({N, hidden_size_}), backend_);
    R_h_next.fill(0.0f);

    for (int64_t t = L - 1; t >= 0; --t) {
        const auto tu = static_cast<size_t>(t);
        Tensor r_t(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(r_t.data(), h, relevance_out.data() + tu * h, lu * h, n, h);
        Tensor R_z(Shape({N, hidden_size_}), backend_);
        backend_->add(r_t.data(), R_h_next.data(), R_z.data(), n * h);

        Tensor x_t(Shape({N, input_size_}), backend_);
        backend_->copy_2d(x_t.data(), d, last_input_.data() + tu * d, lu * d, n, d);
        Tensor h_prev(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(h_prev.data(), h, last_hidden_states_.data() + tu * h, (lu + 1) * h, n, h);
        Tensor z_t(Shape({N, hidden_size_}), backend_);
        backend_->copy_2d(z_t.data(), h, last_pre_activation_.data() + tu * h, lu * h, n, h);

        // Epsilon rule over the shared bias-free pre-activation, once per source (x_t through
        // W_xh, h_{t-1} through W_hh); each output sums over k ascending as the original loop.
        Tensor R_x(Shape({N, input_size_}), backend_);
        backend_->lrp_linear(x_t.data(), weight_xh_.data(), z_t.data(), R_z.data(), R_x.data(), n, d, h,
                             config.epsilon);
        Tensor R_hprev(Shape({N, hidden_size_}), backend_);
        backend_->lrp_linear(h_prev.data(), weight_hh_.data(), z_t.data(), R_z.data(), R_hprev.data(), n, h, h,
                             config.epsilon);

        backend_->copy_2d(relevance_in.data() + tu * d, lu * d, R_x.data(), d, n, d);
        R_h_next = R_hprev;
    }

    return relevance_in;
}

}  // namespace pulsatrix
