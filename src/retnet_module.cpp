#include "pulsatrix/retnet_module.hpp"

#include <stdexcept>

namespace pulsatrix {

namespace {
// Operand dims for every SsmPassOp this module issues.
SsmPassArgs pass_dims(int64_t N, int64_t L, int64_t D, int64_t Kd, float gamma) {
    SsmPassArgs args;
    args.n = N;
    args.l = L;
    args.d = D;
    args.s = Kd;
    args.gamma = gamma;
    return args;
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
    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("RetNetModule::forward: input must be rank-3 (N, L, d_model)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t D = d_model_;
    const int64_t Kd = key_dim_;
    const auto rows = static_cast<size_t>(N * L);
    const auto d = static_cast<size_t>(D);
    const auto kd = static_cast<size_t>(Kd);

    last_input_ = input;
    last_L_ = L;
    last_q_ = Tensor(Shape({N, L, Kd}), backend_);
    last_k_ = Tensor(Shape({N, L, Kd}), backend_);
    last_v_ = Tensor(Shape({N, L, D}), backend_);
    last_states_ = Tensor(Shape({N, L + 1, Kd, D}), backend_);  // zero-filled: S_0 = 0

    Tensor output(Shape({N, L, D}), backend_);

    // The three projections, over every (b, t) row at once (gemm's per-element dot product does
    // not depend on the row count, so this equals the per-timestep form).
    backend_->gemm(input.data(), w_q_.data(), last_q_.data(), rows, d, kd);
    backend_->gemm(input.data(), w_k_.data(), last_k_.data(), rows, d, kd);
    backend_->gemm(input.data(), w_v_.data(), last_v_.data(), rows, d, d);

    // The retention state: the decayed carry plus this token's outer product K_t (x) V_t, then
    // read out through Q_t -- one lane per (b, j), sequential over t.
    SsmPassArgs args = pass_dims(N, L, D, Kd, gamma_);
    args.in[0] = last_q_.data();
    args.in[1] = last_k_.data();
    args.in[2] = last_v_.data();
    args.out[0] = last_states_.data();
    args.out[1] = output.data();
    backend_->ssm_pass(SsmPassOp::RetnetForward, args);

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
    const auto n = static_cast<size_t>(N);
    const auto lu = static_cast<size_t>(L);
    const auto d = static_cast<size_t>(D);
    const auto kd = static_cast<size_t>(Kd);
    const size_t rows = n * lu;

    Tensor grad_input(last_input_.shape(), backend_);
    Tensor local_w_q_grad(w_q_.shape(), backend_);
    Tensor local_w_k_grad(w_k_.shape(), backend_);
    Tensor local_w_v_grad(w_v_.shape(), backend_);

    // Step 1: o_t[b,j] = sum_i Q_t[b,i]*S_t[b,i,j]. The readout hands gradient to Q_t and to S_t;
    // S_t's total gradient is that direct share plus whatever step t+1 carried back -- the
    // decayed remainder gamma*dS_{t+1}, threaded per (b, i, j) lane over t descending (zero at
    // t = L: nothing follows the last step).
    Tensor ds(Shape({N, L, Kd, D}), backend_);
    SsmPassArgs state = pass_dims(N, L, D, Kd, gamma_);
    state.in[0] = grad_output.data();
    state.in[1] = last_q_.data();
    state.out[0] = ds.data();
    backend_->ssm_pass(SsmPassOp::RetnetStateGrad, state);

    // Step 2: S_t = gamma*S_{t-1} + K_t (x) V_t. gamma is a fixed hyperparameter, not a Tensor
    // -- no gradient is accumulated for it; it only scales the carry. dQ, dK and dV each summed
    // by their own lane.
    Tensor dq(Shape({N, L, Kd}), backend_);
    Tensor dk(Shape({N, L, Kd}), backend_);
    SsmPassArgs qk = pass_dims(N, L, D, Kd, gamma_);
    qk.in[0] = grad_output.data();
    qk.in[1] = last_states_.data();
    qk.in[2] = ds.data();
    qk.in[3] = last_v_.data();
    qk.out[0] = dq.data();
    qk.out[1] = dk.data();
    backend_->ssm_pass(SsmPassOp::RetnetGradQK, qk);
    Tensor dv(Shape({N, L, D}), backend_);
    SsmPassArgs gv = pass_dims(N, L, D, Kd, gamma_);
    gv.in[0] = ds.data();
    gv.in[1] = last_k_.data();
    gv.out[0] = dv.data();
    backend_->ssm_pass(SsmPassOp::RetnetGradV, gv);

    // Step 3: the three projections' parameter gradients (per timestep, in the original
    // t-descending accumulation order) and their shares of x_t (standard no-bias Linear
    // backward, over every row at once). All three land on the same grad_input slot.
    for (int64_t t = L - 1; t >= 0; --t) {
        const auto tu = static_cast<size_t>(t);
        Tensor x_t(Shape({N, D}), backend_);
        backend_->copy_2d(x_t.data(), d, last_input_.data() + tu * d, lu * d, n, d);
        Tensor dq_t(Shape({N, Kd}), backend_);
        backend_->copy_2d(dq_t.data(), kd, dq.data() + tu * kd, lu * kd, n, kd);
        Tensor dk_t(Shape({N, Kd}), backend_);
        backend_->copy_2d(dk_t.data(), kd, dk.data() + tu * kd, lu * kd, n, kd);
        Tensor dv_t(Shape({N, D}), backend_);
        backend_->copy_2d(dv_t.data(), d, dv.data() + tu * d, lu * d, n, d);

        Tensor gw_q(w_q_.shape(), backend_);
        backend_->gemm_ex(x_t.data(), true, dq_t.data(), false, gw_q.data(), d, n, kd, 0.0f);
        local_w_q_grad.accumulate(gw_q);
        Tensor gw_k(w_k_.shape(), backend_);
        backend_->gemm_ex(x_t.data(), true, dk_t.data(), false, gw_k.data(), d, n, kd, 0.0f);
        local_w_k_grad.accumulate(gw_k);
        Tensor gw_v(w_v_.shape(), backend_);
        backend_->gemm_ex(x_t.data(), true, dv_t.data(), false, gw_v.data(), d, n, d, 0.0f);
        local_w_v_grad.accumulate(gw_v);
    }

    Tensor dx_q(Shape({N, L, D}), backend_);
    backend_->gemm_ex(dq.data(), false, w_q_.data(), true, dx_q.data(), rows, kd, d, 0.0f);
    Tensor dx_k(Shape({N, L, D}), backend_);
    backend_->gemm_ex(dk.data(), false, w_k_.data(), true, dx_k.data(), rows, kd, d, 0.0f);
    Tensor dx_v(Shape({N, L, D}), backend_);
    backend_->gemm_ex(dv.data(), false, w_v_.data(), true, dx_v.data(), rows, d, d, 0.0f);
    // grad_input (zero-filled) += (dx_q + dx_k) + dx_v -- the original expression's association.
    Tensor dx_sum(Shape({N, L, D}), backend_);
    backend_->add(dx_q.data(), dx_k.data(), dx_sum.data(), rows * d);
    backend_->add(dx_sum.data(), dx_v.data(), dx_sum.data(), rows * d);
    backend_->add(grad_input.data(), dx_sum.data(), grad_input.data(), rows * d);

    // A frozen parameter (FND-2) accumulates nothing. Its local gradient is still computed
    // above: here it is cheap, or entangled with the input gradient's own recurrence.
    if (w_q_.requires_grad()) {
        w_q_grad_.accumulate(local_w_q_grad);
    }
    if (w_k_.requires_grad()) {
        w_k_grad_.accumulate(local_w_k_grad);
    }
    if (w_v_.requires_grad()) {
        w_v_grad_.accumulate(local_w_v_grad);
    }

    return grad_input;
}

Tensor RetNetModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("RetNetModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_input_.shape().dim(0);
    const int64_t L = last_L_;
    const int64_t D = d_model_;
    const int64_t Kd = key_dim_;
    if (relevance_out.rank() != 3 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != L ||
        relevance_out.shape().dim(2) != D) {
        throw std::invalid_argument(
            "RetNetModule::propagate_relevance: relevance_out must be (N, L, d_model) matching the cached "
            "forward shape");
    }
    const auto n = static_cast<size_t>(N);
    const auto lu = static_cast<size_t>(L);
    const auto d = static_cast<size_t>(D);
    const auto kd = static_cast<size_t>(Kd);

    const float eps = config.epsilon;
    Tensor relevance_in(last_input_.shape(), backend_);

    // Per batch row: reconstruct the causal, gamma-decay-gated "attention-like" matrix
    // G[t,s] = gamma^(t-s)*(Q_t.K_s) for s <= t (0 above the diagonal -- see the header's
    // derivation note for why this is exactly the module's own unrolled forward math, not a
    // fresh approximation), then apply AttnLRP Eq. 15 twice -- once for Y = G @ V, once for
    // QK = Q @ K^T -- with an exact constant-scale identity pass-through for gamma^(t-s) in
    // between, composed the same way MultiHeadAttentionModule's own Eq. 15 chain composes
    // across its Q@K^T -> softmax -> Attn@V pipeline (RetNet has no softmax in the middle,
    // so there is one fewer stage and no non-conserving DTD approximation anywhere in it).
    //
    // QK[t,s] = Q_t.K_s for s <= t, and 0 above the diagonal, where the Eq. 15 denominator is
    // the only reader (a[i,j] = G is exactly 0 there too). G[t,s] = gamma^(t-s)*QK[t,s], the
    // decay built by repeated multiplication by gamma (mirrors forward_impl()'s own incremental
    // decay application; avoids std::pow entirely, so there is no risk of a
    // negative-base/non-integer-exponent edge case even though gamma is deliberately
    // unconstrained).
    Tensor qk(Shape({N, L, L}), backend_);
    Tensor g(Shape({N, L, L}), backend_);
    SsmPassArgs scores = pass_dims(N, L, D, Kd, gamma_);
    scores.in[0] = last_q_.data();
    scores.in[1] = last_k_.data();
    scores.out[0] = qk.data();
    scores.out[1] = g.data();
    backend_->ssm_pass(SsmPassOp::RetnetScores, scores);

    // Y[t,j] = sum_{s<=t} G[t,s]*V[s,j] -- exactly forward_impl()'s own o_t, recomputed from the
    // unrolled form rather than cached separately (see the header's derivation note: this
    // equals the recurrence's output up to floating-point summation order).
    Tensor y(Shape({N, L, D}), backend_);
    SsmPassArgs readout = pass_dims(N, L, D, Kd, gamma_);
    readout.in[0] = g.data();
    readout.in[1] = last_v_.data();
    readout.out[0] = y.data();
    backend_->ssm_pass(SsmPassOp::RetnetReadout, readout);

    // Step 1: Eq. 15 on Y = G @ V -- gives r_g (relevance of the decay-gated scores) and r_v
    // (relevance of V), each conserving exactly against r_y via the rule's factor-2
    // denominator.
    Tensor r_g(Shape({N, L, L}), backend_);
    Tensor r_v(Shape({N, L, D}), backend_);
    backend_->lrp_bilinear_matmul(g.data(), last_v_.data(), y.data(), relevance_out.data(), r_g.data(), r_v.data(),
                                  n, lu, lu, d, eps, false);

    // Step 2: gamma^(t-s) is a fixed, known scalar hyperparameter (not merely
    // detached-as-if-constant, unlike Mamba's Abar/Bbar or RWKV's decay/kk) scaling
    // QK[t,s] into G[t,s] -- a single-term rescaling under which the epsilon rule is
    // exactly the identity (same reasoning as MultiHeadAttentionModule's own
    // 1/sqrt(head_dim)-scale note): R(QK[t,s]) == R(G[t,s]) exactly. r_g is already
    // exactly 0 wherever G is exactly 0 (including every s > t entry), so there is no
    // 0/0 case to guard and no stabilizer is needed for this step at all.
    const Tensor& r_qk = r_g;

    // Step 3: Eq. 15 on QK = Q @ K^T, K read in place as the stored-transposed operand (r_k
    // comes back in K's own (L, Kd) layout).
    Tensor r_q(Shape({N, L, Kd}), backend_);
    Tensor r_k(Shape({N, L, Kd}), backend_);
    backend_->lrp_bilinear_matmul(last_q_.data(), last_k_.data(), qk.data(), r_qk.data(), r_q.data(), r_k.data(), n,
                                  lu, kd, lu, eps, true);

    // Step 4: the three no-bias linear projections Q_t = x_t@W_Q (and K_t/V_t alike) -- the
    // standard weighted-connection epsilon/z-rule, denominator the projection's own pre-output
    // value (there is no bias to exclude, unlike LinearModule). All three land on the same
    // relevance_in slot via fan-in accumulation, matching MultiHeadAttentionModule's own Step
    // 1' treatment of its Q/K/V projections.
    SsmPassArgs input = pass_dims(N, L, D, Kd, gamma_);
    input.eps = eps;
    input.in[0] = last_input_.data();
    input.in[1] = w_q_.data();
    input.in[2] = w_k_.data();
    input.in[3] = w_v_.data();
    input.in[4] = last_q_.data();
    input.in[5] = last_k_.data();
    input.in[6] = last_v_.data();
    input.in[7] = r_q.data();
    input.in[8] = r_k.data();
    input.in[9] = r_v.data();
    input.out[0] = relevance_in.data();
    backend_->ssm_pass(SsmPassOp::RetnetLrpInput, input);

    return relevance_in;
}

}  // namespace pulsatrix
