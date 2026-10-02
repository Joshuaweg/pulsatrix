// State-space / linear-recurrence math (MambaModule, RWKVModule, RetNetModule) shared verbatim by
// CPUBackend (a loop over lanes) and the GPU kernels (one thread per lane). Each routine is the
// pre-campaign module host loop for one independent lane, unchanged in expression and order:
// - scans run one lane per independent (batch, channel) recurrence, walking time in the original
//   direction inside the lane (parallel over lanes, sequential over time);
// - every reduction is owned by one lane and adds its terms in the order the original loop added
//   them. Sums the original scattered from several loop nests (parameter gradients over batch and
//   time, the token-shift's two writers per input slot) are inverted into gathers that visit the
//   same contributions in the same order. Deterministic, no atomics.
// Slot meanings per op are documented on SsmPassOp in device_backend.hpp. Private to src/.
// GPU-native-kernels Mission 6.
#pragma once

#include <cstdint>

#include "lrp_math.hpp"        // lrp::stabilize
#include "pointwise_math.hpp"  // PULSATRIX_HOST_DEVICE
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace ssm {

// softplus(z) = log(1 + exp(z)) in the overflow-safe branch form MambaModule always used.
PULSATRIX_HOST_DEVICE inline float softplus(float z) { return (z > 20.0f) ? z : log1pf(expf(z)); }

// The plain logistic form the SSM modules always used (not pointwise::stable_sigmoid, whose
// x < 0 branch rounds differently).
PULSATRIX_HOST_DEVICE inline float sigmoid(float z) { return 1.0f / (1.0f + expf(-z)); }

// Number of independent lanes op runs over.
PULSATRIX_HOST_DEVICE inline int64_t lanes(SsmPassOp op, const SsmPassArgs& a) {
    switch (op) {
        case SsmPassOp::MambaForward:
        case SsmPassOp::MambaBackward:
        case SsmPassOp::MambaLrp:
        case SsmPassOp::RwkvForward:
        case SsmPassOp::RwkvBackward:
        case SsmPassOp::RwkvLrp:
        case SsmPassOp::RetnetForward:
            return a.n * a.d;
        case SsmPassOp::MambaGradBC:
        case SsmPassOp::RetnetGradQK:
            return a.n * a.l * a.s;
        case SsmPassOp::RwkvTokenShift:
        case SsmPassOp::RwkvShiftBackward:
        case SsmPassOp::RwkvShiftLrp:
        case SsmPassOp::StabilizedDiv:
        case SsmPassOp::RetnetGradV:
        case SsmPassOp::RetnetReadout:
        case SsmPassOp::RetnetLrpInput:
            return a.n * a.l * a.d;
        case SsmPassOp::RetnetStateGrad:
            return a.n * a.s * a.d;
        case SsmPassOp::RetnetScores:
            return a.n * a.l * a.l;
        case SsmPassOp::ReverseTimeSum:
            return a.d;
    }
    return 0;
}

// ---- MambaModule ---------------------------------------------------------------------------------

// Lane (b, d): Delta = softplus(z), then the selective scan over t ascending, s ascending.
PULSATRIX_HOST_DEVICE inline void mamba_forward(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d, S = a.s;
    const int64_t b = lane / D, d = lane % D;
    const float* input = a.in[0];
    const float* z_delta = a.in[1];
    const float* b_proj = a.in[2];
    const float* c_proj = a.in[3];
    const float* A = a.in[4];
    const float* Dv = a.in[5];
    float* states = a.out[3];
    for (int64_t t = 0; t < L; ++t) {
        const int64_t idx = (b * L + t) * D + d;
        const float delta = softplus(z_delta[idx]);
        a.out[0][idx] = delta;
        const float x = input[idx];
        float y = Dv[d] * x;  // the D skip/feedthrough path
        for (int64_t s = 0; s < S; ++s) {
            const int64_t scan_idx = idx * S + s;
            const float abar = expf(delta * A[d * S + s]);
            const float bbar = delta * b_proj[(b * L + t) * S + s];
            a.out[1][scan_idx] = abar;
            a.out[2][scan_idx] = bbar;
            const float h_prev = states[((b * (L + 1) + t) * D + d) * S + s];
            const float h = abar * h_prev + bbar * x;
            states[((b * (L + 1) + t + 1) * D + d) * S + s] = h;
            y += c_proj[(b * L + t) * S + s] * h;
        }
        a.out[4][idx] = y;
    }
}

// Lane (b, d): BPTT over t descending with the per-state carry dh_carry[b, d, :] (out[4],
// zero-filled by the caller). Writes dx's direct share, dz_delta, every dh_total, and A's
// per-step gradient terms (summed later by ReverseTimeSum in the original (t desc, b) order).
PULSATRIX_HOST_DEVICE inline void mamba_backward(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d, S = a.s;
    const int64_t b = lane / D, d = lane % D;
    const float* grad_output = a.in[0];
    const float* input = a.in[1];
    const float* delta_buf = a.in[2];
    const float* z_delta = a.in[3];
    const float* states = a.in[4];
    const float* abar_buf = a.in[5];
    const float* bbar_buf = a.in[6];
    const float* b_buf = a.in[7];
    const float* c_buf = a.in[8];
    const float* A = a.in[9];
    const float* Dv = a.in[10];
    float* carry = a.out[4] + (b * D + d) * S;
    for (int64_t t = L - 1; t >= 0; --t) {
        const int64_t idx = (b * L + t) * D + d;
        const float gy = grad_output[idx];
        const float x = input[idx];
        const float delta = delta_buf[idx];
        float dx = 0.0f;
        dx += gy * Dv[d];
        float d_delta = 0.0f;
        for (int64_t s = 0; s < S; ++s) {
            const int64_t scan_idx = idx * S + s;
            const float h_prev = states[((b * (L + 1) + t) * D + d) * S + s];
            const float abar = abar_buf[scan_idx];
            const float bbar = bbar_buf[scan_idx];
            const float c_val = c_buf[(b * L + t) * S + s];
            const float b_val = b_buf[(b * L + t) * S + s];

            const float dh_total = gy * c_val + carry[s];
            carry[s] = dh_total * abar;
            a.out[2][scan_idx] = dh_total;
            const float d_abar = dh_total * h_prev;
            const float d_bbar = dh_total * x;
            dx += dh_total * bbar;
            d_delta += d_bbar * b_val;
            d_delta += d_abar * abar * A[d * S + s];
            a.out[3][scan_idx] = d_abar * abar * delta;
        }
        a.out[1][idx] = d_delta * sigmoid(z_delta[idx]);
        a.out[0][idx] = dx;
    }
}

// Lane (b, t, s): d_C and d_B summed over d ascending.
PULSATRIX_HOST_DEVICE inline void mamba_grad_bc(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d, S = a.s;
    const int64_t bt = lane / S, s = lane % S;
    const int64_t b = bt / L, t = bt % L;
    const float* grad_output = a.in[0];
    const float* input = a.in[1];
    const float* delta_buf = a.in[2];
    const float* states = a.in[3];
    const float* dh_total = a.in[4];
    float d_b = 0.0f;
    float d_c = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
        const int64_t idx = bt * D + d;
        const float h_t = states[((b * (L + 1) + t + 1) * D + d) * S + s];
        d_c += grad_output[idx] * h_t;
        const float d_bbar = dh_total[idx * S + s] * input[idx];
        d_b += d_bbar * delta_buf[idx];
    }
    a.out[0][lane] = d_b;
    a.out[1][lane] = d_c;
}

// Lane (b, d): MambaLRP over t descending with the per-state carry r_h_carry[b, d, :] (out[1],
// zero-filled by the caller).
PULSATRIX_HOST_DEVICE inline void mamba_lrp(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d, S = a.s;
    const int64_t b = lane / D, d = lane % D;
    const float* input = a.in[0];
    const float* output = a.in[1];
    const float* relevance_out = a.in[2];
    const float* states = a.in[3];
    const float* abar_buf = a.in[4];
    const float* bbar_buf = a.in[5];
    const float* c_buf = a.in[6];
    const float* Dv = a.in[7];
    float* carry = a.out[1] + (b * D + d) * S;
    for (int64_t t = L - 1; t >= 0; --t) {
        const int64_t idx = (b * L + t) * D + d;
        const float x = input[idx];
        const float denom_y = lrp::stabilize(output[idx], a.eps);
        const float r_y = relevance_out[idx];
        float r_x = 0.0f;
        r_x += (Dv[d] * x / denom_y) * r_y;
        for (int64_t s = 0; s < S; ++s) {
            const int64_t scan_idx = idx * S + s;
            const float h_t = states[((b * (L + 1) + t + 1) * D + d) * S + s];
            const float h_prev = states[((b * (L + 1) + t) * D + d) * S + s];
            const float abar = abar_buf[scan_idx];
            const float bbar = bbar_buf[scan_idx];
            const float c_val = c_buf[(b * L + t) * S + s];
            const float r_h = (c_val * h_t / denom_y) * r_y + carry[s];
            const float denom_h = lrp::stabilize(h_t, a.eps);
            carry[s] = (abar * h_prev / denom_h) * r_h;
            r_x += (bbar * x / denom_h) * r_h;
        }
        a.out[0][idx] = r_x;
    }
}

// ---- RWKVModule ----------------------------------------------------------------------------------

// Element (b, t, e): the three token-shift mixes; x_{-1} = 0.
PULSATRIX_HOST_DEVICE inline void rwkv_token_shift(const SsmPassArgs& a, int64_t i) {
    const int64_t L = a.l, D = a.d;
    const int64_t t = (i / D) % L, e = i % D;
    const float x_cur = a.in[0][i];
    const float x_prev = (t > 0) ? a.in[0][i - D] : 0.0f;
    a.out[0][i] = a.in[1][e] * x_cur + (1.0f - a.in[1][e]) * x_prev;
    a.out[1][i] = a.in[2][e] * x_cur + (1.0f - a.in[2][e]) * x_prev;
    a.out[2][i] = a.in[3][e] * x_cur + (1.0f - a.in[3][e]) * x_prev;
}

// Lane (b, d): the WKV recurrence over t ascending.
PULSATRIX_HOST_DEVICE inline void rwkv_forward(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d;
    const int64_t b = lane / D, d = lane % D;
    const float* z_r = a.in[0];
    const float* k_buf = a.in[1];
    const float* v_buf = a.in[2];
    const float u = a.in[3][d];
    const float w = a.in[4][d];
    float* a_states = a.out[6];
    float* b_states = a.out[7];
    for (int64_t t = 0; t < L; ++t) {
        const int64_t idx = (b * L + t) * D + d;
        const float r = sigmoid(z_r[idx]);
        const float k = k_buf[idx];
        const float v = v_buf[idx];
        a.out[0][idx] = r;

        const float a_prev = a_states[(b * (L + 1) + t) * D + d];
        const float b_prev = b_states[(b * (L + 1) + t) * D + d];
        const float e_t = expf(u + k);
        const float num = a_prev + e_t * v;
        const float den = b_prev + e_t;
        const float wkv = num / den;
        a.out[1][idx] = e_t;
        a.out[2][idx] = num;
        a.out[3][idx] = den;
        a.out[4][idx] = wkv;

        const float decay = expf(-w);
        const float kk = expf(k);
        a.out[5][idx] = kk;
        a_states[(b * (L + 1) + t + 1) * D + d] = decay * a_prev + kk * v;
        b_states[(b * (L + 1) + t + 1) * D + d] = decay * b_prev + kk;

        a.out[8][idx] = r * wkv;
    }
}

// Lane (b, d): WKV BPTT over t descending with the scalar carries da/db. u's and w's per-step
// gradient terms go to out[3]/out[4] for ReverseTimeSum.
PULSATRIX_HOST_DEVICE inline void rwkv_backward(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d;
    const int64_t b = lane / D, d = lane % D;
    const float* g_gated = a.in[0];
    const float* r_buf = a.in[1];
    const float* v_buf = a.in[2];
    const float* e_buf = a.in[3];
    const float* kk_buf = a.in[4];
    const float* num_buf = a.in[5];
    const float* den_buf = a.in[6];
    const float* a_states = a.in[7];
    const float* b_states = a.in[8];
    const float* wkv_buf = a.in[9];
    const float w = a.in[10][d];
    float da_carry = 0.0f;
    float db_carry = 0.0f;
    for (int64_t t = L - 1; t >= 0; --t) {
        const int64_t idx = (b * L + t) * D + d;
        const float r = r_buf[idx];
        const float v = v_buf[idx];
        const float e_t = e_buf[idx];
        const float kk = kk_buf[idx];
        const float num = num_buf[idx];
        const float den = den_buf[idx];
        const float a_prev = a_states[(b * (L + 1) + t) * D + d];
        const float b_prev = b_states[(b * (L + 1) + t) * D + d];
        const float decay = expf(-w);

        const float g_rwkv = g_gated[idx];
        const float dr = g_rwkv * wkv_buf[idx];
        const float dwkv = g_rwkv * r;

        const float dnum = dwkv / den;
        const float dden = -dwkv * num / (den * den);

        float da_prev = dnum;
        float db_prev = dden;
        const float de = dnum * v + dden;
        float dv_local = dnum * e_t;

        float dk_local = de * e_t;
        a.out[3][idx] = de * e_t;  // u's term

        const float ga = da_carry;
        const float gb = db_carry;
        const float d_decay_from_a = ga * a_prev;
        da_prev += ga * decay;
        float dkk = ga * v;
        dv_local += ga * kk;
        const float d_decay_from_b = gb * b_prev;
        db_prev += gb * decay;
        dkk += gb;

        a.out[4][idx] = (d_decay_from_a + d_decay_from_b) * (-decay);  // w's term

        dk_local += dkk * kk;

        da_carry = da_prev;
        db_carry = db_prev;

        a.out[1][idx] = dk_local;
        a.out[2][idx] = dv_local;
        a.out[0][idx] = dr * r * (1.0f - r);
    }
}

// Element (b, t, d): the token-shift backward. Slot t receives step t+1's x_{t-1} share first
// (the original loop ran t descending, so step t+1 added it before step t's direct share), then
// its own direct share. mu's per-step gradient terms go to out[1..3] for ReverseTimeSum.
PULSATRIX_HOST_DEVICE inline void rwkv_shift_backward(const SsmPassArgs& a, int64_t i) {
    const int64_t L = a.l, D = a.d;
    const int64_t t = (i / D) % L, d = i % D;
    const float* input = a.in[0];
    const float* g_xr = a.in[1];
    const float* g_xk = a.in[2];
    const float* g_xv = a.in[3];
    const float mu_r = a.in[4][d];
    const float mu_k = a.in[5][d];
    const float mu_v = a.in[6][d];

    const float x_cur = input[i];
    const float x_prev = (t > 0) ? input[i - D] : 0.0f;
    const float diff = x_cur - x_prev;
    a.out[1][i] = g_xr[i] * diff;
    a.out[2][i] = g_xk[i] * diff;
    a.out[3][i] = g_xv[i] * diff;

    float acc = 0.0f;
    if (t + 1 < L) {
        const int64_t n = i + D;
        acc += g_xr[n] * (1.0f - mu_r) + g_xk[n] * (1.0f - mu_k) + g_xv[n] * (1.0f - mu_v);
    }
    acc += g_xr[i] * mu_r + g_xk[i] * mu_k + g_xv[i] * mu_v;
    a.out[0][i] = acc;
}

// Lane (b, d): the detached WKV-quotient / state-carry epsilon rules over t descending, with the
// scalar carry R(A[t+1]).
PULSATRIX_HOST_DEVICE inline void rwkv_lrp(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d;
    const int64_t b = lane / D, d = lane % D;
    const float eps = a.eps;
    const float* r_gated = a.in[0];
    const float* a_states = a.in[1];
    const float* den_buf = a.in[2];
    const float* e_buf = a.in[3];
    const float* v_buf = a.in[4];
    const float* wkv_buf = a.in[5];
    const float* kk_buf = a.in[6];
    const float w = a.in[7][d];
    float r_a_next = 0.0f;
    for (int64_t t = L - 1; t >= 0; --t) {
        const int64_t idx = (b * L + t) * D + d;
        const float r_wkv = r_gated[idx];
        const float a_prev = a_states[(b * (L + 1) + t) * D + d];
        const float a_next_val = a_states[(b * (L + 1) + t + 1) * D + d];
        const float den = den_buf[idx];
        const float e_t = e_buf[idx];
        const float v_val = v_buf[idx];
        const float wkv_val = wkv_buf[idx];
        const float decay = expf(-w);
        const float kk_t = kk_buf[idx];

        const float denom_wkv = lrp::stabilize(wkv_val, eps);
        const float contrib_At_readout = (a_prev / den / denom_wkv) * r_wkv;
        const float contrib_vt_readout = (e_t * v_val / den / denom_wkv) * r_wkv;

        const float denom_a_next = lrp::stabilize(a_next_val, eps);
        const float contrib_At_state = (decay * a_prev / denom_a_next) * r_a_next;
        const float contrib_vt_state = (kk_t * v_val / denom_a_next) * r_a_next;

        r_a_next = contrib_At_readout + contrib_At_state;
        a.out[0][idx] = contrib_vt_readout + contrib_vt_state;
    }
}

// Element (b, t, e): the value token-shift epsilon rule, gathered per input slot in the original
// order (step t+1's x_{t-1} share, then step t's direct share).
PULSATRIX_HOST_DEVICE inline void rwkv_shift_lrp(const SsmPassArgs& a, int64_t i) {
    const int64_t L = a.l, D = a.d;
    const int64_t t = (i / D) % L, e = i % D;
    const float eps = a.eps;
    const float* input = a.in[0];
    const float* xv = a.in[1];
    const float* r_xv_raw = a.in[2];
    const float mu = a.in[3][e];

    float acc = 0.0f;
    if (t + 1 < L) {
        const int64_t n = i + D;
        const float xv_next = xv[n];
        const float r_xv_next = xv_next * r_xv_raw[n];
        const float denom_next = lrp::stabilize(xv_next, eps);
        acc += ((1.0f - mu) * input[i] / denom_next) * r_xv_next;
    }
    const float xv_val = xv[i];
    const float r_xv = xv_val * r_xv_raw[i];
    const float denom = lrp::stabilize(xv_val, eps);
    acc += (mu * input[i] / denom) * r_xv;
    a.out[0][i] = acc;
}

// ---- RetNetModule --------------------------------------------------------------------------------

// Lane (b, j): the retention-state recurrence over t ascending, i ascending.
PULSATRIX_HOST_DEVICE inline void retnet_forward(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d, Kd = a.s;
    const int64_t b = lane / D, j = lane % D;
    const float* q = a.in[0];
    const float* k = a.in[1];
    const float* v = a.in[2];
    float* states = a.out[0];
    for (int64_t t = 0; t < L; ++t) {
        float acc = 0.0f;
        for (int64_t i = 0; i < Kd; ++i) {
            const float s_prev = states[((b * (L + 1) + t) * Kd + i) * D + j];
            const float s_cur = a.gamma * s_prev + k[(b * L + t) * Kd + i] * v[(b * L + t) * D + j];
            states[((b * (L + 1) + t + 1) * Kd + i) * D + j] = s_cur;
            acc += q[(b * L + t) * Kd + i] * s_cur;
        }
        a.out[1][(b * L + t) * D + j] = acc;
    }
}

// Lane (b, i, j): the undecayed state gradient dS_t over t descending; the carry is dS * gamma.
PULSATRIX_HOST_DEVICE inline void retnet_state_grad(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d, Kd = a.s;
    const int64_t bi = lane / D, j = lane % D;
    const int64_t b = bi / Kd, i = bi % Kd;
    float carry = 0.0f;
    for (int64_t t = L - 1; t >= 0; --t) {
        const float ds = a.in[0][(b * L + t) * D + j] * a.in[1][(b * L + t) * Kd + i] + carry;
        a.out[0][((b * L + t) * Kd + i) * D + j] = ds;
        carry = ds * a.gamma;
    }
}

// Lane (b, t, i): dQ (readout) and dK (outer product), each summed over j ascending.
PULSATRIX_HOST_DEVICE inline void retnet_grad_qk(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d, Kd = a.s;
    const int64_t bt = lane / Kd, i = lane % Kd;
    const int64_t b = bt / L, t = bt % L;
    const float* grad_output = a.in[0];
    const float* states = a.in[1];
    const float* ds = a.in[2];
    const float* v = a.in[3];
    float dq_acc = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
        dq_acc += grad_output[bt * D + j] * states[((b * (L + 1) + t + 1) * Kd + i) * D + j];
    }
    float dk_acc = 0.0f;
    for (int64_t j = 0; j < D; ++j) {
        dk_acc += ds[lane * D + j] * v[bt * D + j];
    }
    a.out[0][lane] = dq_acc;
    a.out[1][lane] = dk_acc;
}

// Lane (b, t, j): dV summed over i ascending.
PULSATRIX_HOST_DEVICE inline void retnet_grad_v(const SsmPassArgs& a, int64_t lane) {
    const int64_t D = a.d, Kd = a.s;
    const int64_t bt = lane / D, j = lane % D;
    float acc = 0.0f;
    for (int64_t i = 0; i < Kd; ++i) {
        acc += a.in[0][(bt * Kd + i) * D + j] * a.in[1][bt * Kd + i];
    }
    a.out[0][lane] = acc;
}

// Lane (b, t, s): QK[t, s] = Q_t . K_s and G[t, s] = gamma^(t-s) * QK[t, s] for s <= t, both 0
// above the diagonal. The decay is built by the same chain of multiplications the original
// downward-s loop made (1, then *gamma once per step from t down to s), so it is the same float.
PULSATRIX_HOST_DEVICE inline void retnet_scores(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, Kd = a.s;
    const int64_t bt = lane / L, s = lane % L;
    const int64_t b = bt / L, t = bt % L;
    if (s > t) {
        a.out[0][lane] = 0.0f;
        a.out[1][lane] = 0.0f;
        return;
    }
    float dot = 0.0f;
    for (int64_t i = 0; i < Kd; ++i) {
        dot += a.in[0][bt * Kd + i] * a.in[1][(b * L + s) * Kd + i];
    }
    float decay = 1.0f;
    for (int64_t u = t; u > s; --u) {
        decay *= a.gamma;
    }
    a.out[0][lane] = dot;
    a.out[1][lane] = decay * dot;
}

// Lane (b, t, j): Y[t, j] = sum_{s <= t} G[t, s] * V[s, j], s ascending.
PULSATRIX_HOST_DEVICE inline void retnet_readout(const SsmPassArgs& a, int64_t lane) {
    const int64_t L = a.l, D = a.d;
    const int64_t bt = lane / D, j = lane % D;
    const int64_t b = bt / L, t = bt % L;
    float acc = 0.0f;
    for (int64_t s = 0; s <= t; ++s) {
        acc += a.in[0][bt * L + s] * a.in[1][(b * L + s) * D + j];
    }
    a.out[0][lane] = acc;
}

// Lane (b, t, e): the Q/K/V projections' epsilon rule, all three fanned into one input slot in
// the original order (Q then K per key index i ascending, then V per j ascending).
PULSATRIX_HOST_DEVICE inline void retnet_lrp_input(const SsmPassArgs& a, int64_t lane) {
    const int64_t D = a.d, Kd = a.s;
    const int64_t bt = lane / D, e = lane % D;
    const float eps = a.eps;
    const float* w_q = a.in[1];
    const float* w_k = a.in[2];
    const float* w_v = a.in[3];
    const float* q = a.in[4] + bt * Kd;
    const float* k = a.in[5] + bt * Kd;
    const float* v = a.in[6] + bt * D;
    const float* r_q = a.in[7] + bt * Kd;
    const float* r_k = a.in[8] + bt * Kd;
    const float* r_v = a.in[9] + bt * D;
    const float x = a.in[0][lane];
    float acc = 0.0f;
    for (int64_t i = 0; i < Kd; ++i) {
        const float denom_q = lrp::stabilize(q[i], eps);
        acc += (x * w_q[e * Kd + i] / denom_q) * r_q[i];
        const float denom_k = lrp::stabilize(k[i], eps);
        acc += (x * w_k[e * Kd + i] / denom_k) * r_k[i];
    }
    for (int64_t j = 0; j < D; ++j) {
        const float denom_v = lrp::stabilize(v[j], eps);
        acc += (x * w_v[e * D + j] / denom_v) * r_v[j];
    }
    a.out[0][lane] = acc;
}

// ---- Shared ----------------------------------------------------------------------------------------

// Lane c: out[c] = sum over t descending, then b ascending, of in[(b*L + t)*C + c] (C = a.d) -- the
// order a backward loop over t descending / b ascending added its per-step terms into a
// parameter-gradient accumulator.
PULSATRIX_HOST_DEVICE inline void reverse_time_sum(const SsmPassArgs& a, int64_t c) {
    const int64_t N = a.n, L = a.l, C = a.d;
    float acc = 0.0f;
    for (int64_t t = L - 1; t >= 0; --t) {
        for (int64_t b = 0; b < N; ++b) {
            acc += a.in[0][(b * L + t) * C + c];
        }
    }
    a.out[0][c] = acc;
}

PULSATRIX_HOST_DEVICE inline void pass(SsmPassOp op, const SsmPassArgs& a, int64_t lane) {
    switch (op) {
        case SsmPassOp::MambaForward:
            mamba_forward(a, lane);
            return;
        case SsmPassOp::MambaBackward:
            mamba_backward(a, lane);
            return;
        case SsmPassOp::MambaGradBC:
            mamba_grad_bc(a, lane);
            return;
        case SsmPassOp::MambaLrp:
            mamba_lrp(a, lane);
            return;
        case SsmPassOp::RwkvTokenShift:
            rwkv_token_shift(a, lane);
            return;
        case SsmPassOp::RwkvForward:
            rwkv_forward(a, lane);
            return;
        case SsmPassOp::RwkvBackward:
            rwkv_backward(a, lane);
            return;
        case SsmPassOp::RwkvShiftBackward:
            rwkv_shift_backward(a, lane);
            return;
        case SsmPassOp::RwkvLrp:
            rwkv_lrp(a, lane);
            return;
        case SsmPassOp::RwkvShiftLrp:
            rwkv_shift_lrp(a, lane);
            return;
        case SsmPassOp::StabilizedDiv:
            a.out[0][lane] = a.in[0][lane] / lrp::stabilize(a.in[1][lane], a.eps);
            return;
        case SsmPassOp::RetnetForward:
            retnet_forward(a, lane);
            return;
        case SsmPassOp::RetnetStateGrad:
            retnet_state_grad(a, lane);
            return;
        case SsmPassOp::RetnetGradQK:
            retnet_grad_qk(a, lane);
            return;
        case SsmPassOp::RetnetGradV:
            retnet_grad_v(a, lane);
            return;
        case SsmPassOp::RetnetScores:
            retnet_scores(a, lane);
            return;
        case SsmPassOp::RetnetReadout:
            retnet_readout(a, lane);
            return;
        case SsmPassOp::RetnetLrpInput:
            retnet_lrp_input(a, lane);
            return;
        case SsmPassOp::ReverseTimeSum:
            reverse_time_sum(a, lane);
            return;
    }
}

}  // namespace ssm
}  // namespace pulsatrix
