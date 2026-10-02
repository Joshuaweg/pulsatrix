// Per-element recurrent-cell math (RNN / LSTM / GRU) shared verbatim by CPUBackend (loops) and
// the GPU kernels (one thread per element). Each routine is the per-(n, k) body of the
// pre-campaign module host loop, unchanged in expression and order. Private to src/.
// GPU-native-kernels Mission 5.
#pragma once

#include <cstdint>

#include "lrp_math.hpp"        // lrp::stabilize
#include "pointwise_math.hpp"  // PULSATRIX_HOST_DEVICE
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace recurrent {

// Every op reads args.in[...] / writes args.out[...] at element i; the slot meaning per op is
// documented on RecurrentCellOp in device_backend.hpp.
PULSATRIX_HOST_DEVICE inline void cell(RecurrentCellOp op, const RecurrentCellArgs& a, int64_t i) {
    switch (op) {
        case RecurrentCellOp::RnnBackward: {
            const float h = a.in[2][i];
            const float dh = a.in[0][i] + a.in[1][i];
            a.out[0][i] = dh * (1.0f - h * h);
            return;
        }
        case RecurrentCellOp::LstmForward: {
            const float i_t = a.in[0][i];
            const float f_t = a.in[1][i];
            const float g_t = a.in[2][i];
            const float o_t = a.in[3][i];
            const float c_prev = a.in[4][i];
            const float c_t = f_t * c_prev + i_t * g_t;
            const float tanh_c = tanhf(c_t);
            a.out[0][i] = c_t;
            a.out[1][i] = tanh_c;
            a.out[2][i] = o_t * tanh_c;
            return;
        }
        case RecurrentCellOp::LstmBackward: {
            const float i_t = a.in[3][i];
            const float f_t = a.in[4][i];
            const float g_t = a.in[5][i];
            const float o_t = a.in[6][i];
            const float tanh_c = a.in[7][i];
            const float c_prev = a.in[8][i];
            const float dh = a.in[0][i] + a.in[1][i];
            const float dc = dh * o_t * (1.0f - tanh_c * tanh_c) + a.in[2][i];
            a.out[3][i] = dh * tanh_c * o_t * (1.0f - o_t);    // dz_o
            a.out[1][i] = dc * c_prev * f_t * (1.0f - f_t);     // dz_f
            a.out[0][i] = dc * g_t * i_t * (1.0f - i_t);        // dz_i
            a.out[2][i] = dc * i_t * (1.0f - g_t * g_t);        // dz_g
            a.out[4][i] = dc * f_t;                             // dc_prev
            return;
        }
        case RecurrentCellOp::LstmLrp: {
            const float i_t = a.in[3][i];
            const float f_t = a.in[4][i];
            const float g_t = a.in[5][i];
            const float c_prev = a.in[6][i];
            const float c_t = a.in[7][i];
            const float R_h = a.in[0][i] + a.in[1][i];
            const float R_c = R_h + a.in[2][i];
            const float cell_denom = lrp::stabilize(c_t, a.eps);
            a.out[1][i] = (f_t * c_prev / cell_denom) * R_c;  // R_c_prev
            a.out[0][i] = (i_t * g_t / cell_denom) * R_c;     // R_g
            return;
        }
        case RecurrentCellOp::GruBackward: {
            const float z_t = a.in[2][i];
            const float r_t = a.in[3][i];
            const float n_t = a.in[4][i];
            const float hn_p = a.in[5][i];
            const float h_prev = a.in[6][i];
            const float dh = a.in[0][i] + a.in[1][i];
            a.out[0][i] = dh * (1.0f - z_t);  // dh_prev (direct path)
            const float dn = dh * z_t;
            a.out[1][i] = dh * (n_t - h_prev) * z_t * (1.0f - z_t);  // dz_pre
            const float dnp = dn * (1.0f - n_t * n_t);
            a.out[2][i] = dnp;                              // dn_pre
            a.out[3][i] = dnp * hn_p * r_t * (1.0f - r_t);  // dr_pre
            a.out[4][i] = dnp * r_t;                        // dhn_prev
            return;
        }
        case RecurrentCellOp::GruLrp: {
            const float z_t = a.in[2][i];
            const float r_t = a.in[3][i];
            const float n_t = a.in[4][i];
            const float hn_p = a.in[5][i];
            const float h_prev = a.in[6][i];
            const float h_t = a.in[7][i];
            const float n_pre = a.in[8][i];
            const float R_h = a.in[0][i] + a.in[1][i];
            const float h_denom = lrp::stabilize(h_t, a.eps);
            a.out[0][i] = ((1.0f - z_t) * h_prev / h_denom) * R_h;  // R_hprev direct
            const float R_n = (z_t * n_t / h_denom) * R_h;
            a.out[1][i] = R_n;
            const float n_denom = lrp::stabilize(n_pre, a.eps);
            a.out[2][i] = (r_t * hn_p / n_denom) * R_n;  // R_term_b
            return;
        }
    }
}

// GRU's R_hprev[n][kk]: the recurrent epsilon-rule sum over k ascending, with element kk's own
// direct term inserted at k == kk -- exactly where the original loop added it.
PULSATRIX_HOST_DEVICE inline float gru_lrp_hprev(const float* h_prev_row, const float* w_hn, const float* hn_row,
                                                 const float* r_term_b_row, const float* direct_row, int64_t kk,
                                                 int64_t hidden, float eps) {
    float acc = 0.0f;
    const float hp = h_prev_row[kk];
    for (int64_t k = 0; k < hidden; ++k) {
        if (k == kk) {
            acc += direct_row[kk];
        }
        const float hn_denom = lrp::stabilize(hn_row[k], eps);
        acc += (hp * w_hn[kk * hidden + k] / hn_denom) * r_term_b_row[k];
    }
    return acc;
}

}  // namespace recurrent
}  // namespace pulsatrix
