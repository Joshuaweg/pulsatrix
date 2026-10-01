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
};

}  // namespace pulsatrix
