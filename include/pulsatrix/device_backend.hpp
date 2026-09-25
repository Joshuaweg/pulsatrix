/** @file device_backend.hpp
 *  @brief Abstract interface isolating vendor-specific memory/compute operations from Tensor/ComputationGraph.
 */
#pragma once

#include <cstddef>

namespace pulsatrix {

/**
 * @brief Which physical device a Tensor's buffer resides on.
 * @note Only Cpu has a DeviceBackend implementation as of Phase 0. Cuda/Hip exist here now
 *       so Tensor::to() has a stable enum to target starting Phase 1.5/1.6, without needing
 *       to change Tensor's public signature when those backends are added.
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
    Silu      ///< x * sigmoid(x) -- a.k.a. swish; the gate half of SwiGLU
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
};

}  // namespace pulsatrix
