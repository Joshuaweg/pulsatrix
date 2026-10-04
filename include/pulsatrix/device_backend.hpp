/** @file device_backend.hpp
 *  @brief Abstract interface isolating vendor-specific memory/compute operations from Tensor/ComputationGraph.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace pulsatrix {

/**
 * @brief Which physical device a Tensor's buffer resides on.
 * @note CPUBackend, CUDABackend and HIPBackend implement Cpu, Cuda and Hip respectively.
 *       Tensor::to(target, target_backend) moves a buffer between them.
 */
enum class DeviceType {
    Cpu,
    Cuda,
    Hip
};

/** @brief Direction of a DeviceBackend::copy() call. */
enum class CopyDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    HostToHost
};

/**
 * @brief Unary elementwise operations supported by DeviceBackend::elementwise().
 * @note Binary elementwise ops (add, mul) are intentionally not part of this Phase 0
 *       interface — Tensor's arithmetic operators are designed in Mission 1
 *       (Shape & Tensor Core), and a binary-op signature added speculatively now
 *       would likely need to change once that design exists.
 * @note Every op here computes the *forward* value only -- no derivative variant exists in
 *       this interface. A module that needs an activation's derivative (ReluModule,
 *       RNNModule, LSTMModule, GRUModule) computes it locally from its own cached forward
 *       output, which is why adding Tanh/Sigmoid/Silu here removes those modules' raw
 *       forward loops but not their backward/LRP derivative math.
 */
enum class ElementwiseOp {
    Relu,
    Neg,
    Tanh,     ///< tanh(x)
    Sigmoid,  ///< 1 / (1 + exp(-x))
    Silu,     ///< x * sigmoid(x) -- a.k.a. swish; the gate half of SwiGLU
    Exp       ///< exp(x) -- GPU-native-kernels Mission 1b (Reparameterize, KL divergence)
};

/** @brief Elementwise boolean gate for DeviceBackend::lrp_stabilized_divide(). */
enum class LrpGate {
    None,      ///< every element passes
    Positive,  ///< passes where gate > 0
    Negative   ///< passes where gate < 0
};

/**
 * @brief Elementwise passes of the fuzzy-logic modules, for DeviceBackend::logic_pointwise.
 * @note Paired with a norm index: 0 Product, 1 Lukasiewicz, 2 Godel -- the declaration order
 *       of ConjunctionModule::TNorm and DisjunctionModule::TConorm.
 */
enum class LogicOp {
    ConjunctionForward,
    ConjunctionBackward,
    ConjunctionLrp,
    DisjunctionForward,
    DisjunctionBackward,
    DisjunctionLrp
};

/**
 * @brief Fused per-element recurrent-cell passes, for DeviceBackend::recurrent_cell.
 *        Slots (in[] / out[]), all (rows x hidden) per timestep unless noted:
 * - RnnBackward:  in g, dh_next, h                            -> out dz
 * - LstmForward:  in i, f, g, o (activated gates), c_prev      -> out c, tanh(c), h
 * - LstmBackward: in g, dh_next, dc_next, i, f, g, o, tanh_c, c_prev
 *                 -> out dz_i, dz_f, dz_g, dz_o, dc_prev
 * - LstmLrp:      in r, R_h_next, R_c_next, i, f, g, c_prev, c  -> out R_g, R_c_prev
 * - GruBackward:  in g, dh_next, z, r, n, hn, h_prev
 *                 -> out dh_prev_direct, dz_pre, dn_pre, dr_pre, dhn_prev
 * - GruLrp:       in r, R_h_next, z, r_gate, n, hn, h_prev, h, n_pre
 *                 -> out R_hprev_direct, R_n, R_term_b
 */
enum class RecurrentCellOp { RnnBackward, LstmForward, LstmBackward, LstmLrp, GruBackward, GruLrp };

/** @brief Operand pointers for DeviceBackend::recurrent_cell (passed to kernels by value). */
struct RecurrentCellArgs {
    const float* in[10] = {};
    float* out[6] = {};
    float eps = 0.0f;  ///< LRP stabilizer (LstmLrp, GruLrp)
};

/**
 * @brief Fused passes of the state-space / linear-recurrence modules (MambaModule, RWKVModule,
 *        RetNetModule), for DeviceBackend::ssm_pass. Dims come from SsmPassArgs: n batch, l
 *        sequence length, d d_model (or C for ReverseTimeSum), s Mamba's state_size / RetNet's
 *        key_dim. Sequences are (n, l, X) row-major; "states" buffers are (n, l + 1, ...) with
 *        the zero initial state at index 0. Lanes and slots (in[] -> out[]):
 * - MambaForward  (n*d lanes, t ascending): in input, z_delta (bias added), B, C, A (d, s), D (d)
 *                 -> out delta, Abar, Bbar (n, l, d, s), states (n, l+1, d, s), output
 * - MambaBackward (n*d lanes, t descending): in grad_output, input, delta, z_delta, states, Abar,
 *                 Bbar, B, C, A, D -> out dx_direct, dz_delta, dh_total, A_terms (n, l, d, s),
 *                 carry scratch (n, d, s), zero-filled by the caller
 * - MambaGradBC   (n*l*s lanes): in grad_output, input, delta, states, dh_total -> out dB, dC
 * - MambaLrp      (n*d lanes, t descending): in input, output, relevance_out, states, Abar, Bbar,
 *                 C, D -> out relevance_in, carry scratch (n, d, s), zero-filled (uses eps)
 * - RwkvTokenShift    (n*l*d): in input, mu_r, mu_k, mu_v -> out xr, xk, xv
 * - RwkvForward       (n*d lanes, t ascending): in z_r, k, v, u, w
 *                     -> out r, e, num, den, wkv, kk, a states, b states (n, l+1, d), gated
 * - RwkvBackward      (n*d lanes, t descending): in g_gated, r, v, e, kk, num, den, a states,
 *                     b states, wkv, w -> out dz_r, dk, dv, u_terms, w_terms
 * - RwkvShiftBackward (n*l*d): in input, g_xr, g_xk, g_xv, mu_r, mu_k, mu_v
 *                     -> out grad_input, mu_r_terms, mu_k_terms, mu_v_terms
 * - RwkvLrp           (n*d lanes, t descending): in r_gated, a states, den, e, v, wkv, kk, w
 *                     -> out v_relevance (uses eps)
 * - RwkvShiftLrp      (n*l*d): in input, xv, r_xv_raw, mu_v -> out relevance_in (uses eps)
 * - StabilizedDiv     (n*l*d): in r, z -> out r / (z + eps*sign(z))
 * - RetnetForward     (n*d lanes, t ascending): in q, k, v -> out states (n, l+1, s, d), output
 *                     (uses gamma)
 * - RetnetStateGrad   (n*s*d lanes, t descending): in grad_output, q -> out dS (n, l, s, d)
 *                     (uses gamma)
 * - RetnetGradQK      (n*l*s): in grad_output, states, dS, v -> out dq, dk
 * - RetnetGradV       (n*l*d): in dS, k -> out dv
 * - RetnetScores      (n*l*l): in q, k -> out QK, G = gamma^(t-s) QK (both 0 for s > t)
 * - RetnetReadout     (n*l*d): in G, v -> out Y = G @ V over s <= t
 * - RetnetLrpInput    (n*l*d): in input, W_q, W_k, W_v, q, k, v, r_q, r_k, r_v
 *                     -> out relevance_in (uses eps)
 * - ReverseTimeSum    (d lanes): in terms (n, l, d) -> out[c] = sum over t descending, b
 *                     ascending -- a backward loop's parameter-gradient accumulation order
 */
enum class SsmPassOp {
    MambaForward,
    MambaBackward,
    MambaGradBC,
    MambaLrp,
    RwkvTokenShift,
    RwkvForward,
    RwkvBackward,
    RwkvShiftBackward,
    RwkvLrp,
    RwkvShiftLrp,
    StabilizedDiv,
    RetnetForward,
    RetnetStateGrad,
    RetnetGradQK,
    RetnetGradV,
    RetnetScores,
    RetnetReadout,
    RetnetLrpInput,
    ReverseTimeSum
};

/** @brief Operand pointers and dims for DeviceBackend::ssm_pass (passed to kernels by value). */
struct SsmPassArgs {
    const float* in[12] = {};
    float* out[10] = {};
    int64_t n = 0;
    int64_t l = 0;
    int64_t d = 0;
    int64_t s = 0;
    float eps = 0.0f;    ///< LRP stabilizer
    float gamma = 0.0f;  ///< RetNet decay
};

/**
 * @brief Window geometry for DeviceBackend::im2col / col2im_add: kernel size, stride and zero
 *        padding per axis. Defaults are stride 1 and no padding. Passed to kernels by value.
 */
struct ConvGeometry {
    size_t kh;
    size_t kw;
    size_t stride_h = 1;
    size_t stride_w = 1;
    size_t pad_h = 0;
    size_t pad_w = 0;
};

/**
 * @brief Fused per-row reinforcement-learning passes, for DeviceBackend::rl_rows. One lane per
 *        batch row (per element for PolyakBlend); rows x cols from RlRowArgs. Index slots hold
 *        validated whole-number action indices as floats. Slots (in[] -> out[]):
 * - DqnLoss:     in q (rows, cols), indices, targets -> out per-row squared TD error
 * - DqnGrad:     in q, indices, targets -> out grad (rows, cols), caller-zeroed; only the taken
 *                action's element is written (uses scale)
 * - PgLoss:      in logits, indices, returns -> out probs (rows, cols), per-row loss term
 * - PgGrad:      in probs, indices, returns -> out grad (dense; uses scale)
 * - PpoLoss:     in logits, indices, old_log_probs, advantages -> out probs, per-row loss term,
 *                ratio, mask (uses lower, upper)
 * - PpoGrad:     in probs, indices, advantages, ratios, masks -> out grad (dense; uses scale)
 * - DqnTarget:   in q_select, q_eval (both rows x cols), rewards, dones -> out target (uses gamma)
 * - PolyakBlend: in source, destination -> out tau*source + (1 - tau)*destination over rows
 *                elements (out may alias destination; uses tau)
 */
enum class RlRowOp { DqnLoss, DqnGrad, PgLoss, PgGrad, PpoLoss, PpoGrad, DqnTarget, PolyakBlend };

/** @brief Operand pointers and dims for DeviceBackend::rl_rows (passed to kernels by value). */
struct RlRowArgs {
    const float* in[5] = {};
    float* out[4] = {};
    int64_t rows = 0;
    int64_t cols = 0;
    float scale = 0.0f;  ///< gradient batch-mean scale (1/N or 2/N)
    float lower = 0.0f;  ///< PPO 1 - clip_epsilon
    float upper = 0.0f;  ///< PPO 1 + clip_epsilon
    float gamma = 0.0f;  ///< discount factor
    float tau = 0.0f;    ///< Polyak blend factor
};

/**
 * @brief Vendor-agnostic compute/memory backend. CPUBackend, CUDABackend (Phase 1.5), and
 *        HIPBackend (Phase 1.6) all implement this contract; Tensor and ComputationGraph
 *        depend only on this interface, never on a concrete backend's types.
 * @note Every concrete implementation must honor identical numeric semantics and identical
 *       error behavior (throw on failure, never return null from allocate()) — this is the
 *       Liskov Substitution contract every DeviceBackend implementation is held to.
 */
class DeviceBackend {
public:
    virtual ~DeviceBackend() = default;

    /**
     * @brief Which device this backend's buffers reside on.
     * @note Tensor's constructors that take no explicit DeviceType tag the Tensor with this,
     *       so a temporary allocated through a CUDA/HIP backend can no longer be silently
     *       labelled Cpu (the default-tag defect found by the GPU-native-kernels campaign
     *       recon: a mislabelled tensor makes Tensor pick HostToHost copies on device memory
     *       and lets host-only guards pass).
     */
    [[nodiscard]] virtual DeviceType device() const noexcept = 0;

    /**
     * @brief Allocates a buffer of the given size.
     * @param bytes Number of bytes to allocate. A request of 0 bytes returns nullptr by
     *        convention (not an error) — there is nothing to allocate.
     * @return Pointer to the allocated buffer, or nullptr iff bytes == 0.
     * @throws std::runtime_error if a nonzero-size allocation fails. Never silently
     *         returns nullptr for a nonzero request.
     */
    [[nodiscard]] virtual void* allocate(size_t bytes) = 0;

    /** @brief Frees a buffer previously returned by allocate(). Safe to call with nullptr. */
    virtual void free(void* ptr) noexcept = 0;

    /**
     * @brief Copies bytes between buffers.
     * @param dst Destination buffer, must be large enough to hold bytes.
     * @param src Source buffer.
     * @param bytes Number of bytes to copy.
     * @param dir Direction of the copy (informs device-specific implementations which
     *        memory space each pointer lives in; CPUBackend ignores it).
     */
    virtual void copy(void* dst, const void* src, size_t bytes, CopyDirection dir) = 0;

    /**
     * @brief Fills every element of a float buffer with a constant value.
     * @param ptr Buffer to fill, must hold at least n floats.
     * @param value Fill value.
     * @param n Number of elements to fill.
     */
    virtual void fill(void* ptr, float value, size_t n) = 0;

    /**
     * @brief Row-major matrix multiply: out = a * b.
     * @param a Pointer to A (m x k), row-major.
     * @param b Pointer to B (k x n), row-major.
     * @param out Pointer to the output buffer (m x n), row-major. Must be pre-allocated
     *        by the caller.
     * @param m Rows of A / rows of out.
     * @param k Columns of A / rows of B.
     * @param n Columns of B / columns of out.
     */
    virtual void gemm(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) = 0;

    /**
     * @brief Applies a unary elementwise operation to every element of a buffer.
     * @param op Operation to apply.
     * @param in Input buffer, must hold at least n floats.
     * @param out Output buffer, must hold at least n floats. May alias in for an
     *        in-place application.
     * @param n Number of elements.
     */
    virtual void elementwise(ElementwiseOp op, const float* in, float* out, size_t n) = 0;

    /**
     * @brief Elementwise binary addition: out[i] = a[i] + b[i] for i in [0, n).
     * @param a First operand, must hold at least n floats.
     * @param b Second operand, must hold at least n floats.
     * @param out Output buffer, must hold at least n floats. May alias a or b for an
     *        in-place accumulation.
     * @param n Number of elements.
     * @note A separate method from elementwise() rather than extending ElementwiseOp with
     *       a binary case -- see ElementwiseOp's own @note. Added in Mission 3 (Phase 0
     *       autograd) specifically because gradient accumulation (multiple children
     *       contributing to a shared parent's gradient) needs it; this is the resolution
     *       of the binary-op design decision deferred since Mission 0.
     */
    virtual void add(const float* a, const float* b, float* out, size_t n) = 0;

    /**
     * @brief Elementwise binary (Hadamard) multiplication: out[i] = a[i] * b[i] for i in [0, n).
     * @param a First operand, must hold at least n floats.
     * @param b Second operand, must hold at least n floats.
     * @param out Output buffer, must hold at least n floats. May alias a or b for an
     *        in-place application.
     * @param n Number of elements.
     * @note Added in Phase 6's SwiGLU mission (mission_swiglu.md) -- the third consumer
     *       needing a Hadamard product via a raw host loop (after LSTMModule's/GRUModule's
     *       gate arithmetic), the trigger point the Risk Register named for finally adding
     *       this primitive properly instead of a fourth raw loop. LSTMModule/GRUModule's
     *       existing raw loops are not retrofitted to use it -- no behavior change needed
     *       there, logged as a low-priority future cleanup only.
     */
    virtual void mul(const float* a, const float* b, float* out, size_t n) = 0;

    // ---- GPU-native-kernels Mission 1: primitives for device-resident training ----------
    // Every reduction below has a fixed summation order on every backend (no atomics), so
    // results are reproducible run to run.

    /**
     * @brief General row-major matrix multiply: out = op(A) * op(B) + beta * out.
     * @param a A, stored (m x k) row-major, or (k x m) when transpose_a.
     * @param transpose_a Use A^T.
     * @param b B, stored (k x n) row-major, or (n x k) when transpose_b.
     * @param transpose_b Use B^T.
     * @param out (m x n) row-major. Read only when beta != 0 (beta == 0 overwrites, so out may
     *        hold garbage). Must not alias a or b.
     * @param beta Scale on the existing out; 1 accumulates (gradient accumulation).
     * @note Replaces the explicit host-side transpose() copies modules built to feed gemm().
     */
    virtual void gemm_ex(const float* a, bool transpose_a, const float* b, bool transpose_b, float* out, size_t m,
                         size_t k, size_t n, float beta) = 0;

    /**
     * @brief Per-column sum of a (rows x cols) row-major matrix: out[j] = beta*out[j] + sum_i in[i][j].
     * @note Rows are summed in increasing i on every backend. out must not alias in.
     */
    virtual void column_sums(const float* in, float* out, size_t rows, size_t cols, float beta) = 0;

    /**
     * @brief Broadcast row add: out[i][j] = in[i][j] + row[j] for a (rows x cols) matrix.
     * @note out may alias in.
     */
    virtual void add_row_vector(const float* in, const float* row, float* out, size_t rows, size_t cols) = 0;

    /**
     * @brief Activation backward: grad_in[i] = grad_out[i] * f'(x[i]), f = op, x = the
     *        forward *input*.
     * @note Derivatives: Relu selects grad_out where x > 0, else 0 (0 at x == 0 and for a
     *       non-finite grad_out, matching ReluModule); Neg -1;
     *       Tanh 1 - tanh(x)^2; Sigmoid s(1 - s); Silu s + x*s*(1 - s), s = sigmoid(x);
     *       Exp exp(x).
     *       grad_in may alias grad_out or x.
     */
    virtual void elementwise_backward(ElementwiseOp op, const float* x, const float* grad_out, float* grad_in,
                                      size_t n) = 0;

    /**
     * @brief out[i] = alpha * x[i] + beta * y[i]. out may alias x or y.
     * @note beta == 0 does not read y (BLAS convention), so out = alpha * x exactly even
     *       where y holds inf/NaN -- otherwise 0 * inf would turn a scale-only call into NaN.
     */
    virtual void axpby(float alpha, const float* x, float beta, const float* y, float* out, size_t n) = 0;

    /**
     * @brief Dot product sum_i a[i]*b[i], returned to the host.
     * @note Synchronizes. Fixed-order reduction on every backend, but GPU order differs from
     *       CPU's sequential sum, so CPU and GPU agree to rounding, not bitwise.
     */
    [[nodiscard]] virtual float dot(const float* a, const float* b, size_t n) = 0;

    /** @brief Row-wise softmax of a (rows x cols) matrix, max-subtracted. out may alias in. */
    virtual void softmax_rows(const float* in, float* out, size_t rows, size_t cols) = 0;

    /**
     * @brief Softmax backward from its output y: dx[i][j] = y[i][j] * (dy[i][j] - sum_k y[i][k]*dy[i][k]).
     * @note dx must not alias y or dy.
     */
    virtual void softmax_rows_backward(const float* y, const float* dy, float* dx, size_t rows, size_t cols) = 0;

    /** @brief Per-row log-sum-exp, max-subtracted: out[i] = log sum_j exp(in[i][j]). */
    virtual void logsumexp_rows(const float* in, float* out, size_t rows, size_t cols) = 0;

    /**
     * @brief One fused Adam update over n parameters.
     * @param bias_correction1 1 - beta1^t, computed once on the host per step.
     * @param bias_correction2 1 - beta2^t, likewise.
     * @note Same per-element expression order as the pre-campaign AdamOptimizer host loop:
     *       m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g*g; p -= lr * (m/bc1) / (sqrt(v/bc2) + eps).
     */
    virtual void adam_step(float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1,
                           float beta2, float eps, float bias_correction1, float bias_correction2) = 0;

    // ---- GPU-native-kernels Mission 1b ---------------------------------------------------

    /** @brief sum_i in[i], returned to the host. Same reduction order as dot(). Synchronizes. */
    [[nodiscard]] virtual float sum(const float* in, size_t n) = 0;

    /**
     * @brief Inverted dropout with a counter-based RNG: element i is dropped iff
     *        uniform(seed, offset + i) < p; kept elements are scaled by scale.
     * @param mask Receives 1.0 (kept) or 0.0 (dropped) per element, for backward().
     * @note uniform(seed, k) is splitmix64(seed + (k + 1) * golden_gamma), top 24 bits as a
     *       float in [0, 1). Stateless and identical on every backend, so a GPU mask is
     *       bit-identical to the CPU one for the same (seed, offset) -- the property that lets
     *       Dropout be tested CPU-vs-GPU at all. A dropped element is written as 0 (a select,
     *       not in * 0), so an inf/NaN input there still yields 0.
     */
    virtual void dropout_forward(const float* in, float* out, float* mask, size_t n, float p, float scale,
                                 uint64_t seed, uint64_t offset) = 0;

    /**
     * @brief Per-element binary cross-entropy with logits:
     *        out[i] = max(x, 0) - x*y + log1p(exp(-|x|)), x = logits[i], y = target[i].
     * @note Fused rather than composed from elementwise ops: this exact evaluation order is
     *       what BCEWithLogitsLoss has always computed, and no composition reproduces its
     *       rounding.
     */
    virtual void bce_with_logits(const float* logits, const float* target, float* out, size_t n) = 0;

    /**
     * @brief BCE-with-logits gradient: grad[i] = (sigmoid(x) - y) * scale, using the
     *        overflow-free sigmoid (exp(x) / (1 + exp(x)) for x < 0).
     */
    virtual void bce_with_logits_grad(const float* logits, const float* target, float* grad, size_t n,
                                      float scale) = 0;

    // ---- GPU-native-kernels Mission 2: transformer building blocks -------------------------
    // The fused row operations below run the same per-row source on every backend (src/
    // row_math.hpp): CPU loops over rows, GPUs run one thread per row. row_std / row_rms /
    // log_prob are per-row device buffers (rows entries). Index buffers hold whole numbers as
    // floats, exact below 2^24.

    /** @brief LayerNorm forward per row: xhat = (x - mean)/sqrt(var + eps), out = gamma*xhat + beta. */
    virtual void layer_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                    float* row_std, size_t rows, size_t cols, float eps) = 0;

    /** @brief LayerNorm input gradient per row, from the cached xhat and per-row std. */
    virtual void layer_norm_backward(const float* grad_out, const float* gamma, const float* xhat,
                                     const float* row_std, float* grad_in, size_t rows, size_t cols) = 0;

    /** @brief RMSNorm forward per row: out = gamma * x / sqrt(mean(x^2) + eps). */
    virtual void rms_norm_forward(const float* in, const float* gamma, float* out, float* row_rms, size_t rows,
                                  size_t cols, float eps) = 0;

    /**
     * @brief RMSNorm input gradient per row, plus gamma_terms[r][i] = grad_out * x / rms --
     *        the per-row contributions column_sums then reduces into gamma's gradient.
     */
    virtual void rms_norm_backward(const float* grad_out, const float* gamma, const float* in, const float* row_rms,
                                   float* grad_in, float* gamma_terms, size_t rows, size_t cols) = 0;

    /**
     * @brief Rotary position embedding over (num_slices, seq_len, head_dim) data.
     * @param cos_table,sin_table (seq_len, head_dim/2) per-position rotation tables.
     * @param inverse false: forward rotation; true: its transpose (RoPE's backward).
     * @note out must not alias in.
     */
    virtual void rope_rotate(const float* in, const float* cos_table, const float* sin_table, float* out,
                             size_t num_slices, size_t seq_len, size_t head_dim, bool inverse) = 0;

    /**
     * @brief Swaps the middle two axes: in (d0, d1, d2, d3) -> out (d0, d2, d1, d3).
     * @note Attention's head split ((N, L, H, D) -> (N, H, L, D)) and merge (the reverse) are
     *       both this permutation. out must not alias in.
     */
    virtual void permute_0213(const float* in, float* out, size_t d0, size_t d1, size_t d2, size_t d3) = 0;

    /** @brief out[i][:] = table[indices[i]][:] for count rows of width dim. */
    virtual void gather_rows(const float* table, const float* indices, float* out, size_t count, size_t dim) = 0;

    /**
     * @brief table[indices[i]][:] += src[i][:] for i = 0..count-1, in increasing i.
     * @note Deterministic: the GPU kernel runs one thread per column and walks i in order --
     *       repeated indices accumulate in the same order as the CPU, no atomics.
     */
    virtual void scatter_add_rows(const float* src, const float* indices, float* table, size_t count, size_t dim) = 0;

    /**
     * @brief TanhGaussianPolicy sampling per (rows, cols) row: action = tanh(mean + exp(log_std)*eps),
     *        std_cache = exp(log_std), log_prob[r] accumulated in double.
     */
    virtual void tanh_gaussian_forward(const float* mean, const float* log_std, const float* eps, float* action,
                                       float* std_cache, float* log_prob, size_t rows, size_t cols,
                                       float stabilizer, double half_log_two_pi) = 0;

    /** @brief TanhGaussianPolicy gradients w.r.t. mean and log_std, per element. */
    virtual void tanh_gaussian_backward(const float* action, const float* std_cache, const float* eps,
                                        const float* grad_action, const float* grad_log_prob, float* grad_mean,
                                        float* grad_log_std, size_t n, float stabilizer) = 0;

    // ---- GPU-native-kernels Mission 3: LRP rules and logic modules --------------------------
    // Shared per-output source in src/lrp_math.hpp (CPU loops, GPU one thread per output).
    // Every reduction is owned by one thread and runs in the original loop order:
    // deterministic, no atomics.

    /**
     * @brief LinearModule epsilon rule: r_in[n][i] = sum_j (x[n][i] w[i][j] / stab(z[n][j])) r[n][j].
     * @param z (N, out) pre-bias outputs; w (in, out); x, r_in (N, in); r (N, out).
     */
    virtual void lrp_linear(const float* x, const float* w, const float* z, const float* r, float* r_in, size_t rows,
                            size_t in_features, size_t out_features, float eps) = 0;

    /** @brief Epsilon split of a residual sum y = a + b: r_a = (a / stab(y)) r, r_b = (b / stab(y)) r. */
    virtual void lrp_residual_split(const float* a, const float* b, const float* r, float* r_a, float* r_b, size_t n,
                                    float eps) = 0;

    /** @brief Eq. 15 for an elementwise product c = a*b: r_out = (a b / (2c + eps sign c)) r (same for both). */
    virtual void lrp_bilinear_elementwise(const float* a, const float* b, const float* r, float* r_out, size_t n,
                                          float eps) = 0;

    /**
     * @brief Eq. 15 for slices independent matmuls O = A @ B (A (M x P), B (P x Q), O and r_o (M x Q)).
     * @param b_transposed B is stored as B^T (Q x P); r_b is then written in that same layout.
     * @note r_a and r_b are overwritten (not accumulated into).
     */
    virtual void lrp_bilinear_matmul(const float* a, const float* b, const float* o, const float* r_o, float* r_a,
                                     float* r_b, size_t slices, size_t m, size_t p, size_t q, float eps,
                                     bool b_transposed) = 0;

    /** @brief SoftmaxModule rule per row: r_in = x * (r - y * sum(r)). */
    virtual void lrp_softmax_rows(const float* x, const float* y, const float* r, float* r_in, size_t rows,
                                  size_t cols) = 0;

    /** @brief RoPEModule epsilon rule over (slices, seq_len, head_dim), tables as for rope_rotate. */
    virtual void lrp_rope(const float* x, const float* y, const float* r, const float* cos_table,
                          const float* sin_table, float* r_in, size_t slices, size_t seq_len, size_t head_dim,
                          float eps) = 0;

    /**
     * @brief One elementwise pass of Conjunction/Disjunction for operands a, b.
     * @param g_or_r Upstream gradient (Backward) or relevance (Lrp); unused by Forward.
     * @param y Cached forward output (Lrp only).
     * @param out_a Forward: the output. Backward/Lrp: a's gradient/relevance.
     * @param out_b Backward/Lrp: b's gradient/relevance; unused by Forward.
     */
    virtual void logic_pointwise(LogicOp op, int norm, const float* a, const float* b, const float* g_or_r,
                                 const float* y, float* out_a, float* out_b, size_t n, float eps) = 0;

    /** @brief AggregatorModule power mean over the leading axis of an (n, cols) input, per column. */
    virtual void aggregator_forward(const float* x, float* mean_pow, float* out, size_t n, size_t cols, float p) = 0;
    /** @brief AggregatorModule input gradient, per column. */
    virtual void aggregator_backward(const float* x, const float* mean_pow, const float* grad_out, float* grad_in,
                                     size_t n, size_t cols, float p) = 0;
    /** @brief AggregatorModule epsilon rule, per column. */
    virtual void aggregator_lrp(const float* x, const float* mean_pow, const float* r_out, float* r_in, size_t n,
                                size_t cols, float p, float eps) = 0;

    // ---- GPU-native-kernels Mission 4: convolution, pooling, spatial norms -----------------
    // Shared per-output source in src/cnn_math.hpp; deterministic, no atomics.

    /**
     * @brief Unfolds (n, c, h, w) into (n, c*kh*kw, out_h*out_w) patches, with
     *        out_h = (h + 2*pad_h - kh) / stride_h + 1 (likewise out_w). Taps that fall in the
     *        zero padding read 0.
     */
    virtual void im2col(const float* in, float* col, size_t n, size_t c, size_t h, size_t w,
                        const ConvGeometry& geometry) = 0;

    /**
     * @brief Folds (n, c*kh*kw, out_h*out_w) patches back, adding into out (n, c, h, w).
     * @note A gather: one GPU thread per output pixel sums its window contributions in the same
     *       order the CPU scatter loop adds them -- deterministic, no atomics.
     */
    virtual void col2im_add(const float* col, float* out, size_t n, size_t c, size_t h, size_t w,
                            const ConvGeometry& geometry) = 0;

    /** @brief out[i][ch][k] = in[i][ch][k] + vec[ch] over (n, c, inner). out may alias in. */
    virtual void add_channel_vector(const float* in, const float* vec, float* out, size_t n, size_t c, size_t inner) =
                                    0;

    /**
     * @brief Conv2D epsilon rule in patch space: r_col (n, p, q) from the cached patches, the
     *        kernel (out_channels, p) and the pre-bias outputs (n, out_channels, q).
     */
    virtual void lrp_conv(const float* col, const float* kernel, const float* pre_bias, const float* r, float* r_col,
                          size_t n, size_t out_channels, size_t p, size_t q, float eps) = 0;

    // ---- LRP-rules campaign Mission 2: Zennit-compatible Gamma / AlphaBeta / ZBox ----------
    /**
     * @brief Gated stabilized division, the one non-gemm step of the affine LRP rules:
     *        out[i] = passes(gate[i]) ? r[i] / (denom[i] + eps sign(denom[i])) : 0, sign(0) = +1.
     * @param gate Read only when gate_mode != LrpGate::None (may then be nullptr).
     * @note out may alias r or denom.
     */
    virtual void lrp_stabilized_divide(const float* r, const float* denom, const float* gate, float* out, size_t n,
                                       float eps, LrpGate gate_mode) = 0;

    /** @brief Non-overlapping max pool over planes of (h, w); argmax = flat in-plane index (first max wins). */
    virtual void max_pool_forward(const float* in, float* out, float* argmax, size_t planes, size_t h, size_t w, size_t
                                  kh, size_t kw) = 0;

    /** @brief dst[plane][argmax] = src for every pooled element; dst must be zeroed by the caller. */
    virtual void max_unpool(const float* src, const float* argmax, float* dst, size_t planes, size_t h, size_t w, size_t
                            kh, size_t kw) = 0;

    /** @brief Non-overlapping average pool over planes of (h, w). */
    virtual void avg_pool_forward(const float* in, float* out, size_t planes, size_t h, size_t w, size_t kh, size_t kw)
                                  = 0;

    /** @brief Spreads grad_out / (kh*kw) over each window; grad_in must be zeroed by the caller. */
    virtual void avg_pool_backward(const float* grad_out, float* grad_in, size_t planes, size_t h, size_t w, size_t kh,
                                   size_t kw) = 0;

    /** @brief AvgPool2D epsilon rule; r_in must be zeroed by the caller. */
    virtual void lrp_avg_pool(const float* x, const float* r, float* r_in, size_t planes, size_t h, size_t w, size_t kh,
                              size_t kw, float eps) = 0;

    /** @brief BatchNorm over (n, spatial) per channel of (n, c, spatial) data. */
    virtual void batch_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                    float* channel_std, size_t n, size_t c, size_t spatial, float eps) = 0;

    /** @brief BatchNorm input gradient plus this call's gamma/beta gradients (overwritten, per channel). */
    virtual void batch_norm_backward(const float* grad_out, const float* gamma, const float* xhat, const float*
                                     channel_std, float* grad_in, float* gamma_grad, float* beta_grad, size_t n, size_t
                                     c, size_t spatial) = 0;

    /** @brief Folds this batch's per-channel mean and unbiased variance into the running ones
     *         (PyTorch's momentum rule; the variance is kept when a channel has one value). FND-5. */
    virtual void batch_norm_update_running(const float* in, float* running_mean, float* running_var, size_t n,
                                           size_t c, size_t spatial, float momentum) = 0;

    /** @brief Eval-mode BatchNorm from the running statistics: a per-channel affine map. FND-5. */
    virtual void batch_norm_eval_forward(const float* in, const float* gamma, const float* beta,
                                         const float* running_mean, const float* running_var, float* xhat, float* out,
                                         float* channel_std, size_t n, size_t c, size_t spatial, float eps) = 0;

    /** @brief Eval-mode BatchNorm gradient: grad_out * gamma / std, plus gamma/beta gradients
     *         (overwritten, per channel). FND-5. */
    virtual void batch_norm_eval_backward(const float* grad_out, const float* gamma, const float* xhat,
                                          const float* channel_std, float* grad_in, float* gamma_grad,
                                          float* beta_grad, size_t n, size_t c, size_t spatial) = 0;

    /** @brief GroupNorm per (example, group) of (n, c, spatial) data; group_std is (n, num_groups). */
    virtual void group_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                                    float* group_std, size_t n, size_t c, size_t spatial, size_t num_groups, float eps)
                                    = 0;

    /** @brief GroupNorm input gradient plus this call's gamma/beta gradients (overwritten, per channel). */
    virtual void group_norm_backward(const float* grad_out, const float* gamma, const float* xhat, const float*
                                     group_std, float* grad_in, float* gamma_grad, float* beta_grad, size_t n, size_t c,
                                     size_t spatial, size_t num_groups) = 0;

    // ---- GPU-native-kernels Mission 5: recurrent networks ------------------------------------

    /**
     * @brief Strided 2-D copy: rows of cols floats from src (row stride src_stride) to dst (row
     *        stride dst_stride) -- e.g. one timestep of an (N, L, D) sequence.
     */
    virtual void copy_2d(float* dst, size_t dst_stride, const float* src, size_t src_stride, size_t rows,
                         size_t cols) = 0;

    /**
     * @brief out[j] += in[i][j] for i = 0..rows-1 in order, accumulating straight into out.
     * @note Unlike column_sums(beta = 1), which adds a finished column sum to out, this adds
     *       row by row into the running value -- the association the recurrent modules' bias
     *       gradients always used.
     */
    virtual void accumulate_rows(const float* in, float* out, size_t rows, size_t cols) = 0;

    /** @brief One fused recurrent-cell pass over n elements (see RecurrentCellOp for slots). */
    virtual void recurrent_cell(RecurrentCellOp op, const RecurrentCellArgs& args, size_t n) = 0;

    /**
     * @brief GRU's R_hprev for (rows, hidden): the recurrent epsilon-rule sum through W_hn, with
     *        each element's direct term inserted where the original loop added it.
     */
    virtual void gru_lrp_hprev(const float* h_prev, const float* w_hn, const float* hn, const float* r_term_b,
                               const float* direct, float* r_hprev, size_t rows, size_t hidden, float eps) = 0;

    // ---- GPU-native-kernels Mission 6: state-space models -------------------------------------
    // Shared per-lane source in src/ssm_math.hpp; deterministic, no atomics.

    /**
     * @brief One fused Mamba / RWKV / RetNet pass (see SsmPassOp for lanes and slots).
     * @note Recurrences run one lane per independent (batch, channel[, state]) sequence, walking
     *       time in order inside the lane -- parallel over lanes, sequential over time, the CPU's
     *       order. Every reduction is owned by one lane and summed in the original loop order.
     */
    virtual void ssm_pass(SsmPassOp op, const SsmPassArgs& args) = 0;

    // ---- GPU-native-kernels Mission 7: reinforcement learning -----------------------------------
    // Shared per-row source in src/rl_math.hpp; deterministic, no atomics.

    /**
     * @brief One fused RL loss / target / Polyak pass (see RlRowOp for lanes and slots).
     * @note One lane per row: a row's softmax, argmax and gradient writes are owned by its lane and
     *       walk the columns in the original loop order. Per-row loss terms are reduced by the
     *       caller with column_sums, which adds rows in increasing order like the original loop.
     */
    virtual void rl_rows(RlRowOp op, const RlRowArgs& args) = 0;

    // ---- FND-3: selection ------------------------------------------------------------------

    /**
     * @brief Per row of a row-major (rows, cols) matrix: the k largest (or smallest) values in
     *        rank order into `values` (rows, k), and their column indices into `indices` (rows, k)
     *        as whole-number floats.
     * @note Order: NaN ranks above every number; equal values keep the lower column index first.
     *       One lane per row, pure selection: every backend's output is bit-identical.
     * @note Preconditions, validated by the caller (top_k()): 1 <= k <= cols, cols <= 2^24.
     */
    virtual void top_k_rows(const float* in, float* values, float* indices, size_t rows, size_t cols, size_t k,
                            bool largest) = 0;
};

}  // namespace pulsatrix
