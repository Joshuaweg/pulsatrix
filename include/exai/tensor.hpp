/** @file tensor.hpp
 *  @brief N-dimensional tensor -- owns a buffer via DeviceBackend*, RAII (Rule of Five).
 */
#pragma once

#include <initializer_list>

#include "exai/device_backend.hpp"
#include "exai/shape.hpp"

namespace exai {

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
     */
    Tensor(Shape shape, DeviceBackend* backend, std::initializer_list<float> values,
           DeviceType device = DeviceType::Cpu);

    ~Tensor();

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

private:
    float* data_;
    Shape shape_;
    DeviceBackend* backend_;
    DeviceType device_;
};

}  // namespace exai
