#include "pulsatrix/rwkv_module.hpp"

#include <stdexcept>

namespace pulsatrix {

namespace {
// Operand dims for every SsmPassOp this module issues.
SsmPassArgs pass_dims(int64_t N, int64_t L, int64_t D) {
    SsmPassArgs args;
    args.n = N;
    args.l = L;
    args.d = D;
    return args;
}

// local[c] = sum over t descending, b ascending of terms[b, t, c] -- the order the original
// backward loop accumulated a per-channel parameter gradient in.
void reverse_time_sum(DeviceBackend* backend, const Tensor& terms, Tensor& out, int64_t N, int64_t L, int64_t D) {
    SsmPassArgs args = pass_dims(N, L, D);
    args.in[0] = terms.data();
    args.out[0] = out.data();
    backend->ssm_pass(SsmPassOp::ReverseTimeSum, args);
}

// out = r / (z + eps*sign(z)) elementwise over (N, L, D).
void stabilized_div(DeviceBackend* backend, const Tensor& r, const Tensor& z, Tensor& out, float eps) {
    SsmPassArgs args = pass_dims(r.shape().dim(0), r.shape().dim(1), r.shape().dim(2));
    args.eps = eps;
    args.in[0] = r.data();
    args.in[1] = z.data();
    args.out[0] = out.data();
    backend->ssm_pass(SsmPassOp::StabilizedDiv, args);
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
    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("RWKVModule::forward: input must be rank-3 (N, L, d_model)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t D = d_model_;
    const auto rows = static_cast<size_t>(N * L);
    const auto d = static_cast<size_t>(D);

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

    // The three token-shift mixes. x_{-1} is zero (this module's documented zero-init
    // convention), so at t = 0 the shifted term simply drops out.
    SsmPassArgs shift = pass_dims(N, L, D);
    shift.in[0] = input.data();
    shift.in[1] = mu_r_.data();
    shift.in[2] = mu_k_.data();
    shift.in[3] = mu_v_.data();
    shift.out[0] = last_xr_.data();
    shift.out[1] = last_xk_.data();
    shift.out[2] = last_xv_.data();
    backend_->ssm_pass(SsmPassOp::RwkvTokenShift, shift);

    // The projections, over every (b, t) row at once (gemm's per-element dot product does not
    // depend on the row count, so this equals the per-timestep form).
    Tensor z_r(Shape({N, L, D}), backend_);
    backend_->gemm(last_xr_.data(), w_r_.data(), z_r.data(), rows, d, d);
    backend_->gemm(last_xk_.data(), w_k_.data(), last_k_.data(), rows, d, d);
    backend_->gemm(last_xv_.data(), w_v_.data(), last_v_.data(), rows, d, d);

    // r_t = sigmoid(z_r); the WKV quotient (the bonus-weighted current token on top of the
    // decayed running state) and the state carry itself -- one lane per (b, d), sequential
    // over t. gated = r_t*wkv_t, the output projection's input.
    Tensor gated(Shape({N, L, D}), backend_);
    SsmPassArgs args = pass_dims(N, L, D);
    args.in[0] = z_r.data();
    args.in[1] = last_k_.data();
    args.in[2] = last_v_.data();
    args.in[3] = u_.data();
    args.in[4] = w_.data();
    args.out[0] = last_r_.data();
    args.out[1] = last_e_.data();
    args.out[2] = last_num_.data();
    args.out[3] = last_den_.data();
    args.out[4] = last_wkv_.data();
    args.out[5] = last_kk_.data();
    args.out[6] = last_a_.data();
    args.out[7] = last_b_.data();
    args.out[8] = gated.data();
    backend_->ssm_pass(SsmPassOp::RwkvForward, args);

    backend_->gemm(gated.data(), w_o_.data(), output.data(), rows, d, d);

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
    const auto n = static_cast<size_t>(N);
    const auto lu = static_cast<size_t>(L);
    const auto d = static_cast<size_t>(D);
    const size_t rows = n * lu;

    Tensor grad_input(last_input_.shape(), backend_);
    Tensor local_w_r_grad(w_r_.shape(), backend_);
    Tensor local_w_k_grad(w_k_.shape(), backend_);
    Tensor local_w_v_grad(w_v_.shape(), backend_);
    Tensor local_w_o_grad(w_o_.shape(), backend_);
    Tensor local_w_grad(w_.shape(), backend_);
    Tensor local_u_grad(u_.shape(), backend_);
    Tensor local_mu_r_grad(mu_r_.shape(), backend_);
    Tensor local_mu_k_grad(mu_k_.shape(), backend_);
    Tensor local_mu_v_grad(mu_v_.shape(), backend_);

    // Step 1: the output projection o_t = (r_t*wkv_t) @ W_o -- W_o's gradient per timestep in
    // the original t-descending accumulation order, the gated input's gradient over every row.
    Tensor gated(Shape({N, L, D}), backend_);
    backend_->mul(last_r_.data(), last_wkv_.data(), gated.data(), rows * d);
    for (int64_t t = L - 1; t >= 0; --t) {
        const auto tu = static_cast<size_t>(t);
        Tensor g_o(Shape({N, D}), backend_);
        backend_->copy_2d(g_o.data(), d, grad_output.data() + tu * d, lu * d, n, d);
        Tensor gated_t(Shape({N, D}), backend_);
        backend_->copy_2d(gated_t.data(), d, gated.data() + tu * d, lu * d, n, d);
        Tensor gw_o(w_o_.shape(), backend_);
        backend_->gemm_ex(gated_t.data(), true, g_o.data(), false, gw_o.data(), d, n, d, 0.0f);
        local_w_o_grad.accumulate(gw_o);
    }
    Tensor g_gated(Shape({N, L, D}), backend_);
    backend_->gemm_ex(grad_output.data(), false, w_o_.data(), true, g_gated.data(), rows, d, d, 0.0f);

    // Steps 2-8, one lane per (b, d) over t descending: wkv_t = num_t/den_t; num_t = a_{t-1} +
    // e_t*v_t, den_t = b_{t-1} + e_t; e_t = exp(u + k_t); the recurrence a_t = decay*a_{t-1} +
    // kk_t*v_t (and b_t likewise), driven by the total gradient carried back from step t+1;
    // decay = exp(-w); kk_t = exp(k_t); and the receptance gate's sigmoid derivative from its
    // cached output. u's and w's per-step terms are summed in the original (t descending,
    // b ascending) accumulation order.
    Tensor dz_r(Shape({N, L, D}), backend_);
    Tensor dk(Shape({N, L, D}), backend_);
    Tensor dv(Shape({N, L, D}), backend_);
    Tensor u_terms(Shape({N, L, D}), backend_);
    Tensor w_terms(Shape({N, L, D}), backend_);
    SsmPassArgs args = pass_dims(N, L, D);
    args.in[0] = g_gated.data();
    args.in[1] = last_r_.data();
    args.in[2] = last_v_.data();
    args.in[3] = last_e_.data();
    args.in[4] = last_kk_.data();
    args.in[5] = last_num_.data();
    args.in[6] = last_den_.data();
    args.in[7] = last_a_.data();
    args.in[8] = last_b_.data();
    args.in[9] = last_wkv_.data();
    args.in[10] = w_.data();
    args.out[0] = dz_r.data();
    args.out[1] = dk.data();
    args.out[2] = dv.data();
    args.out[3] = u_terms.data();
    args.out[4] = w_terms.data();
    backend_->ssm_pass(SsmPassOp::RwkvBackward, args);
    reverse_time_sum(backend_, u_terms, local_u_grad, N, L, D);
    reverse_time_sum(backend_, w_terms, local_w_grad, N, L, D);

    // Step 9: the three input projections' parameter gradients (per timestep, t descending) and
    // their shares of the token-shifted inputs (over every row at once).
    for (int64_t t = L - 1; t >= 0; --t) {
        const auto tu = static_cast<size_t>(t);
        const size_t offset = tu * d;
        Tensor xr(Shape({N, D}), backend_);
        backend_->copy_2d(xr.data(), d, last_xr_.data() + offset, lu * d, n, d);
        Tensor xk(Shape({N, D}), backend_);
        backend_->copy_2d(xk.data(), d, last_xk_.data() + offset, lu * d, n, d);
        Tensor xv(Shape({N, D}), backend_);
        backend_->copy_2d(xv.data(), d, last_xv_.data() + offset, lu * d, n, d);
        Tensor dz_r_t(Shape({N, D}), backend_);
        backend_->copy_2d(dz_r_t.data(), d, dz_r.data() + offset, lu * d, n, d);
        Tensor dk_t(Shape({N, D}), backend_);
        backend_->copy_2d(dk_t.data(), d, dk.data() + offset, lu * d, n, d);
        Tensor dv_t(Shape({N, D}), backend_);
        backend_->copy_2d(dv_t.data(), d, dv.data() + offset, lu * d, n, d);

        Tensor gw_r(w_r_.shape(), backend_);
        backend_->gemm_ex(xr.data(), true, dz_r_t.data(), false, gw_r.data(), d, n, d, 0.0f);
        local_w_r_grad.accumulate(gw_r);
        Tensor gw_k(w_k_.shape(), backend_);
        backend_->gemm_ex(xk.data(), true, dk_t.data(), false, gw_k.data(), d, n, d, 0.0f);
        local_w_k_grad.accumulate(gw_k);
        Tensor gw_v(w_v_.shape(), backend_);
        backend_->gemm_ex(xv.data(), true, dv_t.data(), false, gw_v.data(), d, n, d, 0.0f);
        local_w_v_grad.accumulate(gw_v);
    }
    Tensor dxr(Shape({N, L, D}), backend_);
    backend_->gemm_ex(dz_r.data(), false, w_r_.data(), true, dxr.data(), rows, d, d, 0.0f);
    Tensor dxk(Shape({N, L, D}), backend_);
    backend_->gemm_ex(dk.data(), false, w_k_.data(), true, dxk.data(), rows, d, d, 0.0f);
    Tensor dxv(Shape({N, L, D}), backend_);
    backend_->gemm_ex(dv.data(), false, w_v_.data(), true, dxv.data(), rows, d, d, 0.0f);

    // Step 10: the token-shift itself. Each input slot gathers step t+1's x_{t-1} share and then
    // its own step's x_t share -- the order the original t-descending loop added them in.
    Tensor mu_r_terms(Shape({N, L, D}), backend_);
    Tensor mu_k_terms(Shape({N, L, D}), backend_);
    Tensor mu_v_terms(Shape({N, L, D}), backend_);
    SsmPassArgs shift = pass_dims(N, L, D);
    shift.in[0] = last_input_.data();
    shift.in[1] = dxr.data();
    shift.in[2] = dxk.data();
    shift.in[3] = dxv.data();
    shift.in[4] = mu_r_.data();
    shift.in[5] = mu_k_.data();
    shift.in[6] = mu_v_.data();
    shift.out[0] = grad_input.data();
    shift.out[1] = mu_r_terms.data();
    shift.out[2] = mu_k_terms.data();
    shift.out[3] = mu_v_terms.data();
    backend_->ssm_pass(SsmPassOp::RwkvShiftBackward, shift);
    reverse_time_sum(backend_, mu_r_terms, local_mu_r_grad, N, L, D);
    reverse_time_sum(backend_, mu_k_terms, local_mu_k_grad, N, L, D);
    reverse_time_sum(backend_, mu_v_terms, local_mu_v_grad, N, L, D);

    // A frozen parameter (FND-2) accumulates nothing. Its local gradient is still computed
    // above: here it is cheap, or entangled with the input gradient's own recurrence.
    if (w_r_.requires_grad()) {
        w_r_grad_.accumulate(local_w_r_grad);
    }
    if (w_k_.requires_grad()) {
        w_k_grad_.accumulate(local_w_k_grad);
    }
    if (w_v_.requires_grad()) {
        w_v_grad_.accumulate(local_w_v_grad);
    }
    if (w_o_.requires_grad()) {
        w_o_grad_.accumulate(local_w_o_grad);
    }
    if (w_.requires_grad()) {
        w_grad_.accumulate(local_w_grad);
    }
    if (u_.requires_grad()) {
        u_grad_.accumulate(local_u_grad);
    }
    if (mu_r_.requires_grad()) {
        mu_r_grad_.accumulate(local_mu_r_grad);
    }
    if (mu_k_.requires_grad()) {
        mu_k_grad_.accumulate(local_mu_k_grad);
    }
    if (mu_v_.requires_grad()) {
        mu_v_grad_.accumulate(local_mu_v_grad);
    }

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
    const auto rows = static_cast<size_t>(N * L);
    const auto d = static_cast<size_t>(D);

    const float eps = config.epsilon;
    Tensor relevance_in(last_input_.shape(), backend_);

    // --- Step 1' (output projection): standard no-bias weighted-connection z-rule,
    // denominator o_t itself (recomputed here -- forward_impl() didn't cache it separately,
    // only its own gated = r_t*wkv_t input). Gives r_gated, the relevance of "gated" =
    // r_t*wkv_t. Every timestep at once: none of it depends on the carried state.
    Tensor gated(Shape({N, L, D}), backend_);
    backend_->mul(last_r_.data(), last_wkv_.data(), gated.data(), rows * d);
    Tensor o(Shape({N, L, D}), backend_);
    backend_->gemm(gated.data(), w_o_.data(), o.data(), rows, d, d);
    Tensor scaled_r_o(Shape({N, L, D}), backend_);
    stabilized_div(backend_, relevance_out, o, scaled_r_o, eps);
    Tensor r_gated_raw(Shape({N, L, D}), backend_);
    backend_->gemm_ex(scaled_r_o.data(), false, w_o_.data(), true, r_gated_raw.data(), rows, d, d, 0.0f);
    Tensor r_gated(Shape({N, L, D}), backend_);
    backend_->mul(gated.data(), r_gated_raw.data(), r_gated.data(), rows * d);

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
    // consuming R(A[t+1]) and producing this step's own R(A[t]). R(A[t+1]) -- the WKV
    // numerator state's total relevance -- is threaded backward from t+1 to t exactly the way
    // MambaModule's own r_h_carry is threaded (see the header's derivation note: A[t] is used
    // at BOTH num_t's readout split and A[t+1]'s own state split, the same dual-use shape as
    // Mamba's h_t). Zero at t = L (nothing reads A[L]), and (by A[0] == 0) it carries nothing
    // out past t = 0. One lane per (b, d), sequential over t.
    Tensor v_relevance(Shape({N, L, D}), backend_);
    SsmPassArgs args = pass_dims(N, L, D);
    args.eps = eps;
    args.in[0] = r_gated.data();
    args.in[1] = last_a_.data();
    args.in[2] = last_den_.data();
    args.in[3] = last_e_.data();
    args.in[4] = last_v_.data();
    args.in[5] = last_wkv_.data();
    args.in[6] = last_kk_.data();
    args.in[7] = w_.data();
    args.out[0] = v_relevance.data();
    backend_->ssm_pass(SsmPassOp::RwkvLrp, args);

    // --- Step 5' (value projection): standard no-bias weighted-connection z-rule,
    // denominator v_t itself, following exactly LinearModule's own gemm-based backward
    // shape (scale relevance by 1/denom, gemm back through W_v^T, elementwise-multiply by the
    // input -- the multiply happens inside Step 6').
    Tensor scaled_r_v(Shape({N, L, D}), backend_);
    stabilized_div(backend_, v_relevance, last_v_, scaled_r_v, eps);
    Tensor r_xv_raw(Shape({N, L, D}), backend_);
    backend_->gemm_ex(scaled_r_v.data(), false, w_v_.data(), true, r_xv_raw.data(), rows, d, d, 0.0f);

    // --- Step 6' (value token-shift): xv_t = mu_v*x_cur + (1-mu_v)*x_prev, a genuine
    // (not detached -- mu_v is a real learned weight, treated normally) two-term
    // weighted sum, denominator xv_t itself. w_r_/mu_r_/w_k_/mu_k_/u_ are deliberately
    // never referenced anywhere in this function (only their cached, detached forward
    // values last_r_/last_e_/last_kk_ are); w_ is referenced only to reconstruct the
    // detached decay value above -- see the class-level note. Only w_v_/w_o_/mu_v_ (and
    // w_ for decay) participate. Step 5's z-rule is r_xv[e] = xv_t[e] * (W_v @ scaled_r)[e]
    // -- the elementwise multiply by the projection's own input (xv_t), exactly like Step 1's
    // gated * (W_o @ scaled_r), was once missing (caught by this module's conservation test).
    // Each input slot gathers step t+1's x_{t-1} share, then its own step's x_t share -- the
    // original t-descending order.
    SsmPassArgs shift = pass_dims(N, L, D);
    shift.eps = eps;
    shift.in[0] = last_input_.data();
    shift.in[1] = last_xv_.data();
    shift.in[2] = r_xv_raw.data();
    shift.in[3] = mu_v_.data();
    shift.out[0] = relevance_in.data();
    backend_->ssm_pass(SsmPassOp::RwkvShiftLrp, shift);

    return relevance_in;
}

}  // namespace pulsatrix
