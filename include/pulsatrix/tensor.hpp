/** @file tensor.hpp
 *  @brief N-dimensional tensor -- owns a buffer via DeviceBackend*, RAII (Rule of Five).
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/shape.hpp"

namespace pulsatrix {

/**
 * @brief N-dimensional tensor. Owns its data buffer exclusively; a DeviceBackend* is
 *        injected (not owned) -- the backend must outlive every Tensor constructed
 *        against it, per cpp_style_guide/context_style_project_conventions.md's
 *        ownership table.
 * @note Invariant: data() == nullptr iff numel() == 0. A zero-element Tensor never
 *       allocates (matches DeviceBackend::allocate()'s zero-byte-returns-nullptr
 *       convention) -- this is not an error state, it's the expected representation
 *       of an empty tensor.
 */
class Tensor {
public:
    /**
     * @brief Constructs a zero-initialized tensor.
     * @param shape Tensor shape.
     * @param backend Backend to allocate/fill through. Not owned; must outlive this Tensor.
     * @param device Which device this tensor's buffer conceptually resides on. Defaults to
     *        Cpu -- only Cpu has a DeviceBackend implementation as of Phase 0.
     */
    explicit Tensor(Shape shape, DeviceBackend* backend, DeviceType device = DeviceType::Cpu);

    /**
     * @brief Constructs a tensor from explicit values.
     * @param shape Tensor shape. values.size() must equal shape.numel().
     * @param backend Backend to allocate/copy through.
     * @param values Initial values, in row-major order.
     * @param device Which device this tensor's buffer conceptually resides on.
     * @note Copies via CopyDirection::HostToHost if device is Cpu, else HostToDevice --
     *       values.begin() is always a genuine host pointer (std::initializer_list lives on
     *       the host) regardless of the destination.
     */
    Tensor(Shape shape, DeviceBackend* backend, std::initializer_list<float> values,
           DeviceType device = DeviceType::Cpu);

    /**
     * @brief Constructs a tensor from explicit values, runtime-sized source.
     * @param shape Tensor shape. values.size() must equal shape.numel().
     * @param backend Backend to allocate/copy through.
     * @param values Initial values, in row-major order.
     * @param device Which device this tensor's buffer conceptually resides on.
     * @note Same semantics as the std::initializer_list overload above -- exists because
     *       std::initializer_list has no portable public constructor from a runtime-sized
     *       buffer (pointer + size), so any caller with data whose size isn't known at the
     *       call site (loading weights from a file, marshalling a numpy array across the
     *       Phase 5 Python bindings) cannot use the initializer_list overload at all, not
     *       just less conveniently.
     */
    Tensor(Shape shape, DeviceBackend* backend, const std::vector<float>& values,
           DeviceType device = DeviceType::Cpu);

    ~Tensor();

    /**
     * @brief Deep-copies another tensor's buffer.
     * @note Copies via CopyDirection::HostToHost if device() is Cpu, else DeviceToDevice --
     *       both this tensor's and other's buffers live on the same device, since both are
     *       allocated by the same backend_.
     */
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    /** @brief This tensor's shape. */
    [[nodiscard]] const Shape& shape() const { return shape_; }

    /** @brief Total element count -- shape().numel(). */
    [[nodiscard]] int64_t numel() const { return shape_.numel(); }

    /** @brief Number of dimensions -- shape().rank(). */
    [[nodiscard]] int64_t rank() const { return shape_.rank(); }

    /** @brief Which device this tensor's buffer conceptually resides on. */
    [[nodiscard]] DeviceType device() const { return device_; }

    /** @brief Raw buffer access. nullptr iff numel() == 0. */
    [[nodiscard]] const float* data() const { return data_; }

    /** @brief Raw buffer access (mutable). nullptr iff numel() == 0. */
    [[nodiscard]] float* data() { return data_; }

    /**
     * @brief Element access by multi-dimensional index (row-major).
     * @param index One index per dimension; index.size() must equal rank().
     * @return Reference to the element.
     * @note Bounds and rank are checked via PULSATRIX_ASSERT (programmer-error contract, not a
     *       condition a well-formed caller can legitimately trigger) -- see
     *       cpp_style_guide/context_style_project_conventions.md's assert-vs-throw table.
     *       Assumes a host-addressable (Cpu) backend; revisit when a GPU-resident Tensor's
     *       at() needs a host round-trip (Phase 1.5+).
     */
    [[nodiscard]] float& at(std::initializer_list<int64_t> index);

    /** @brief Const overload of at(). */
    [[nodiscard]] const float& at(std::initializer_list<int64_t> index) const;

    /**
     * @brief Flat (rank-agnostic) element access by linear offset into the row-major buffer.
     * @note PULSATRIX_ASSERT-gated bounds check, not throw -- internal invariant per
     *       campaign_exai_dl_library_adversarial_hardening.md's Mission 0 classification:
     *       this is a hot path called internally (CPUBackend loops, module forward/backward)
     *       with an already-computed, already-valid index, never directly from unvalidated
     *       external input.
     */
    [[nodiscard]] float& operator[](int64_t flat_index) {
        PULSATRIX_ASSERT(flat_index >= 0 && flat_index < numel());
        return data_[flat_index];
    }

    /** @brief Const overload of operator[]. */
    [[nodiscard]] const float& operator[](int64_t flat_index) const {
        PULSATRIX_ASSERT(flat_index >= 0 && flat_index < numel());
        return data_[flat_index];
    }

    /**
     * @brief Sets every element to value. Safe no-op on a zero-element tensor.
     * @param value Fill value.
     * @return *this, for chaining (e.g. t.fill(0.0f).fill_diagonal(1.0f)).
     */
    Tensor& fill(float value);

    /**
     * @brief In-place elementwise accumulation: this[i] += other[i] for every element.
     * @param other Tensor to add into this one. Must have the same shape.
     * @return *this, for chaining (e.g. grad.accumulate(a).accumulate(b)).
     * @note Distinct from a general-purpose arithmetic operator+ -- that remains
     *       deferred (see Mission 0/1 AAR) until Tensor's broader math API is designed
     *       in Phase 1. This method exists specifically for gradient accumulation
     *       (Mission 3 autograd), where the in-place, same-shape-only semantics are
     *       exactly what's needed and nothing more.
     */
    Tensor& accumulate(const Tensor& other);

    /**
     * @brief Reinterprets this tensor's dimensions in place -- same buffer, new shape.
     * @param new_shape Target shape. Must have the same numel() as the current shape.
     * @return *this, for chaining.
     * @throws std::invalid_argument if new_shape.numel() != numel().
     */
    Tensor& reshape(Shape new_shape);

    /**
     * @brief Transfers this tensor to a different device.
     * @param target Target device.
     * @return *this, for chaining.
     * @note A no-op when target == device() (charter's Phase 0 exit gate requires this
     *       specific path). Any genuine cross-device transfer throws until a DeviceBackend
     *       for that device exists (Phase 1.5/1.6) -- there is currently nothing to
     *       transfer to or from.
     * @throws std::runtime_error if target != device().
     */
    Tensor& to(DeviceType target);

private:
    [[nodiscard]] int64_t flat_index_of(std::initializer_list<int64_t> index) const;

    float* data_;
    Shape shape_;
    DeviceBackend* backend_;
    DeviceType device_;
};

}  // namespace pulsatrix
