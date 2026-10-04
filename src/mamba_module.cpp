#include "pulsatrix/mamba_module.hpp"

#include <stdexcept>

namespace pulsatrix {

namespace {
// Operand dims for every SsmPassOp this module issues.
SsmPassArgs pass_dims(int64_t N, int64_t L, int64_t D, int64_t S) {
    SsmPassArgs args;
    args.n = N;
    args.l = L;
    args.d = D;
    args.s = S;
    return args;
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
    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("MambaModule::forward: input must be rank-3 (N, L, d_model)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t D = d_model_;
    const int64_t S = state_size_;
    const auto rows = static_cast<size_t>(N * L);
    const auto d = static_cast<size_t>(D);
    const auto s = static_cast<size_t>(S);

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

    // The three selective projections, over every (b, t) row at once: gemm's per-element dot
    // product does not depend on how many rows it is given, so this equals the per-timestep form.
    // Delta_t's projection is the one selective projection WITH a bias.
    Tensor z_raw(Shape({N, L, D}), backend_);
    backend_->gemm(input.data(), w_delta_.data(), z_raw.data(), rows, d, d);
    backend_->add_row_vector(z_raw.data(), bias_delta_.data(), last_z_delta_.data(), rows, d);
    backend_->gemm(input.data(), w_b_.data(), last_b_.data(), rows, d, s);
    backend_->gemm(input.data(), w_c_.data(), last_c_.data(), rows, d, s);

    // Delta_t = softplus(z), Abar_t = exp(Delta_t*A), Bbar_t = Delta_t*B_t,
    // h_t = Abar_t*h_{t-1} + Bbar_t*x_t, y_t = sum_n(C_t*h_t) + D*x_t -- one lane per (b, d),
    // sequential over t.
    SsmPassArgs args = pass_dims(N, L, D, S);
    args.in[0] = input.data();
    args.in[1] = last_z_delta_.data();
    args.in[2] = last_b_.data();
    args.in[3] = last_c_.data();
    args.in[4] = a_.data();
    args.in[5] = d_.data();
    args.out[0] = last_delta_.data();
    args.out[1] = last_abar_.data();
    args.out[2] = last_bbar_.data();
    args.out[3] = last_states_.data();
    args.out[4] = output.data();
    backend_->ssm_pass(SsmPassOp::MambaForward, args);

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
    const auto n = static_cast<size_t>(N);
    const auto lu = static_cast<size_t>(L);
    const auto d = static_cast<size_t>(D);
    const auto s = static_cast<size_t>(S);
    const size_t rows = n * lu;

    Tensor grad_input(last_input_.shape(), backend_);
    Tensor local_w_delta_grad(w_delta_.shape(), backend_);
    Tensor local_bias_delta_grad(bias_delta_.shape(), backend_);
    Tensor local_w_b_grad(w_b_.shape(), backend_);
    Tensor local_w_c_grad(w_c_.shape(), backend_);
    Tensor local_a_grad(a_.shape(), backend_);
    Tensor local_d_grad(d_.shape(), backend_);

    // Steps 1-5, one lane per (b, d) over t descending: y_t = sum_n(C_t*h_t) + D*x_t, then
    // h_t = Abar_t*h_{t-1} + Bbar_t*x_t, Bbar_t = Delta_t*B_t, Abar_t = exp(Delta_t*A) and
    // Delta_t = softplus(z_delta_t) (d/dz = sigmoid(z)). dh_carry is the carried state-gradient
    // accumulator dh_carry[b,d,n], threaded from t+1 back to t.
    Tensor dx_direct(Shape({N, L, D}), backend_);
    Tensor dz_delta(Shape({N, L, D}), backend_);
    Tensor dh_total(Shape({N, L, D, S}), backend_);
    Tensor a_terms(Shape({N, L, D, S}), backend_);
    Tensor dh_carry(Shape({N, D, S}), backend_);
    SsmPassArgs args = pass_dims(N, L, D, S);
    args.in[0] = grad_output.data();
    args.in[1] = last_input_.data();
    args.in[2] = last_delta_.data();
    args.in[3] = last_z_delta_.data();
    args.in[4] = last_states_.data();
    args.in[5] = last_abar_.data();
    args.in[6] = last_bbar_.data();
    args.in[7] = last_b_.data();
    args.in[8] = last_c_.data();
    args.in[9] = a_.data();
    args.in[10] = d_.data();
    args.out[0] = dx_direct.data();
    args.out[1] = dz_delta.data();
    args.out[2] = dh_total.data();
    args.out[3] = a_terms.data();
    args.out[4] = dh_carry.data();
    backend_->ssm_pass(SsmPassOp::MambaBackward, args);

    // dB_t and dC_t, each summed over d by its own lane.
    Tensor d_b(Shape({N, L, S}), backend_);
    Tensor d_c(Shape({N, L, S}), backend_);
    SsmPassArgs bc = pass_dims(N, L, D, S);
    bc.in[0] = grad_output.data();
    bc.in[1] = last_input_.data();
    bc.in[2] = last_delta_.data();
    bc.in[3] = last_states_.data();
    bc.in[4] = dh_total.data();
    bc.out[0] = d_b.data();
    bc.out[1] = d_c.data();
    backend_->ssm_pass(SsmPassOp::MambaGradBC, bc);

    // A's and D's gradients: per-step terms summed in the original (t descending, b ascending)
    // accumulation order.
    SsmPassArgs a_sum = pass_dims(N, L, D * S, 0);
    a_sum.in[0] = a_terms.data();
    a_sum.out[0] = local_a_grad.data();
    backend_->ssm_pass(SsmPassOp::ReverseTimeSum, a_sum);
    Tensor d_terms(Shape({N, L, D}), backend_);
    backend_->mul(grad_output.data(), last_input_.data(), d_terms.data(), rows * d);
    SsmPassArgs d_sum = pass_dims(N, L, D, 0);
    d_sum.in[0] = d_terms.data();
    d_sum.out[0] = local_d_grad.data();
    backend_->ssm_pass(SsmPassOp::ReverseTimeSum, d_sum);

    // Step 6: the three linear projections' parameter gradients, per timestep in the original
    // t-descending accumulation order.
    for (int64_t t = L - 1; t >= 0; --t) {
        const auto tu = static_cast<size_t>(t);
        Tensor x_t(Shape({N, D}), backend_);
        backend_->copy_2d(x_t.data(), d, last_input_.data() + tu * d, lu * d, n, d);
        Tensor dz_t(Shape({N, D}), backend_);
        backend_->copy_2d(dz_t.data(), d, dz_delta.data() + tu * d, lu * d, n, d);
        Tensor db_t(Shape({N, S}), backend_);
        backend_->copy_2d(db_t.data(), s, d_b.data() + tu * s, lu * s, n, s);
        Tensor dc_t(Shape({N, S}), backend_);
        backend_->copy_2d(dc_t.data(), s, d_c.data() + tu * s, lu * s, n, s);

        Tensor gw_delta(w_delta_.shape(), backend_);
        backend_->gemm_ex(x_t.data(), true, dz_t.data(), false, gw_delta.data(), d, n, d, 0.0f);
        local_w_delta_grad.accumulate(gw_delta);
        backend_->accumulate_rows(dz_t.data(), local_bias_delta_grad.data(), n, d);

        Tensor gw_b(w_b_.shape(), backend_);
        backend_->gemm_ex(x_t.data(), true, db_t.data(), false, gw_b.data(), d, n, s, 0.0f);
        local_w_b_grad.accumulate(gw_b);

        Tensor gw_c(w_c_.shape(), backend_);
        backend_->gemm_ex(x_t.data(), true, dc_t.data(), false, gw_c.data(), d, n, s, 0.0f);
        local_w_c_grad.accumulate(gw_c);
    }

    // ...and their shares of dx_t, over every (b, t) row at once, added onto the direct share in
    // the original order (delta, then B, then C).
    Tensor gx_delta(Shape({N, L, D}), backend_);
    backend_->gemm_ex(dz_delta.data(), false, w_delta_.data(), true, gx_delta.data(), rows, d, d, 0.0f);
    Tensor gx_b(Shape({N, L, D}), backend_);
    backend_->gemm_ex(d_b.data(), false, w_b_.data(), true, gx_b.data(), rows, s, d, 0.0f);
    Tensor gx_c(Shape({N, L, D}), backend_);
    backend_->gemm_ex(d_c.data(), false, w_c_.data(), true, gx_c.data(), rows, s, d, 0.0f);
    backend_->add(dx_direct.data(), gx_delta.data(), grad_input.data(), rows * d);
    backend_->add(grad_input.data(), gx_b.data(), grad_input.data(), rows * d);
    backend_->add(grad_input.data(), gx_c.data(), grad_input.data(), rows * d);

    // A frozen parameter (FND-2) accumulates nothing. Its local gradient is still computed
    // above: here it is cheap, or entangled with the input gradient's own recurrence.
    if (w_delta_.requires_grad()) {
        w_delta_grad_.accumulate(local_w_delta_grad);
    }
    if (bias_delta_.requires_grad()) {
        bias_delta_grad_.accumulate(local_bias_delta_grad);
    }
    if (w_b_.requires_grad()) {
        w_b_grad_.accumulate(local_w_b_grad);
    }
    if (w_c_.requires_grad()) {
        w_c_grad_.accumulate(local_w_c_grad);
    }
    if (a_.requires_grad()) {
        a_grad_.accumulate(local_a_grad);
    }
    if (d_.requires_grad()) {
        d_grad_.accumulate(local_d_grad);
    }

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

    Tensor relevance_in(last_input_.shape(), backend_);

    // The carried state-relevance accumulator, R(h_t)[b,d,n] arriving from step t+1. Zero
    // at t = L-1, and (by h_0 == 0) it carries nothing out past t = 0.
    Tensor r_h_carry(Shape({N, D, S}), backend_);

    // NOTE (MambaLRP, and grep-confirmed in this mission's close-out): the selective
    // projections' weights -- w_delta_, bias_delta_, w_b_, w_c_ -- are deliberately NOT
    // referenced anywhere below. Delta_t/B_t/C_t are pure conductors, consumed only through
    // the cached, *detached* Abar_t/Bbar_t/C_t forward values. Only a_ (via Abar_t) and d_
    // (the skip path, a genuine weighted connection from x_t to y_t) participate.
    //
    // Per (b, d) lane, t descending: step 1 splits y_t = sum_n(C_t*h_t) + D*x_t (the D skip
    // share lands straight on R(x_t)); step 2 splits h_t = Abar_t*h_{t-1} + Bbar_t*x_t with the
    // two-weighted-source epsilon/z-rule, denominator h_t itself.
    SsmPassArgs args = pass_dims(N, L, D, S);
    args.eps = config.epsilon;
    args.in[0] = last_input_.data();
    args.in[1] = last_output_.data();
    args.in[2] = relevance_out.data();
    args.in[3] = last_states_.data();
    args.in[4] = last_abar_.data();
    args.in[5] = last_bbar_.data();
    args.in[6] = last_c_.data();
    args.in[7] = d_.data();
    args.out[0] = relevance_in.data();
    args.out[1] = r_h_carry.data();
    backend_->ssm_pass(SsmPassOp::MambaLrp, args);

    return relevance_in;
}

}  // namespace pulsatrix
