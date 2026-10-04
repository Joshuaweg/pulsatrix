/** @file tensor.hpp
 *  @brief N-dimensional tensor -- owns a buffer via DeviceBackend*, RAII (Rule of Five).
 *  @ingroup dl_modules
 */
#pragma once

#include <string>
#include <stdexcept>
#include <initializer_list>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/host_guard.hpp"
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
     * @brief Constructs a zero-initialized tensor on backend's own device (backend->device()).
     * @param shape Tensor shape.
     * @param backend Backend to allocate/fill through. Not owned; must outlive this Tensor.
     */
    Tensor(Shape shape, DeviceBackend* backend);

    /**
     * @brief Constructs a zero-initialized tensor with an explicit device tag.
     * @param shape Tensor shape.
     * @param backend Backend to allocate/fill through. Not owned; must outlive this Tensor.
     * @param device Which device this tensor's buffer conceptually resides on. Should equal
     *        backend->device(); deliberately not enforced, because tests tag a host-backed
     *        buffer Cuda/Hip to exercise device guards without GPU hardware.
     */
    Tensor(Shape shape, DeviceBackend* backend, DeviceType device);

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
    Tensor(Shape shape, DeviceBackend* backend, std::initializer_list<float> values, DeviceType device);

    /** @brief As above, tagged with backend->device(). */
    Tensor(Shape shape, DeviceBackend* backend, std::initializer_list<float> values);

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
    Tensor(Shape shape, DeviceBackend* backend, const std::vector<float>& values, DeviceType device);

    /** @brief As above, tagged with backend->device(). */
    Tensor(Shape shape, DeviceBackend* backend, const std::vector<float>& values);

    ~Tensor();

    /**
     * @brief Concatenates N tensors along their leading dimension into one batch Tensor --
     *        pulsatrix's collate-time primitive (campaign_exai_dl_library_data_pipeline,
     *        Mission 0). E.g. stacking three (1, 28, 28) MNIST-style per-sample images
     *        produces one (3, 28, 28) batch.
     * @param tensors Non-empty list of tensors, each rank >= 1, each on the same device,
     *        all identical in every dimension except the leading one.
     * @param backend Backend to allocate the output buffer through. Not owned.
     * @return A new Tensor whose leading dimension is the sum of every input tensor's
     *         leading dimension, and whose remaining dimensions match the inputs'.
     * @throws std::invalid_argument if tensors is empty, any tensor has rank 0, ranks
     *         differ across tensors, non-leading dimensions differ across tensors, or
     *         devices differ across tensors -- external boundary: the list of tensors to
     *         stack is assembled by a DataLoader/collate function from independently
     *         constructed Dataset samples, not a compile-time-known invariant.
     */
    [[nodiscard]] static Tensor Stack(const std::vector<Tensor>& tensors, DeviceBackend* backend);

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

    /**
     * @brief The backend that owns this tensor's buffer. Not owned by the Tensor.
     * @note Lets device-generic code (e.g. an optimizer stepping a module's parameters)
     *       compute through the buffer's own backend instead of carrying a second pointer
     *       that could name a different device.
     */
    [[nodiscard]] DeviceBackend* backend() const { return backend_; }

    /**
     * @brief Whether a module's backward() should accumulate a gradient for this tensor when
     *        it is a parameter, and whether optimizers should update it. Defaults to true.
     * @note Meaningful only for parameters (roadmap FND-2); ignored everywhere else.
     * @note The flag belongs to the object, not its values: copy and move construction carry
     *       it over, but copy and move assignment keep the destination's own flag. Loading new
     *       weights into a frozen parameter (`weight_ = loaded;`) therefore leaves it frozen,
     *       matching PyTorch's `param.data = x`.
     */
    [[nodiscard]] bool requires_grad() const { return requires_grad_; }

    /** @brief Sets requires_grad(); see there. */
    void set_requires_grad(bool requires_grad) { requires_grad_ = requires_grad; }

    /**
     * @brief Reads one element to the host through the owning backend, on any device.
     * @note One synchronous device-to-host copy -- for scalars (a loss value, a picked
     *       logit), never for loops. Bounds are PULSATRIX_ASSERT-checked like operator[].
     */
    [[nodiscard]] float read_element(int64_t flat_index) const;

    /**
     * @brief Copies the whole buffer to a host vector through the owning backend, on any device.
     * @return numel() floats in row-major order (empty for a zero-element tensor).
     * @note One synchronous copy (HostToHost on Cpu, DeviceToHost otherwise) -- the explicit
     *       transfer a deliberate host boundary (an environment, a replay buffer, an agent's
     *       action selection; GPU-native-kernels Mission 7) uses to read a possibly-device
     *       tensor once, instead of dereferencing data() in a host loop.
     */
    [[nodiscard]] std::vector<float> to_host_vector() const;

    /** @brief Writes one element from the host through the owning backend, on any device. */
    void write_element(int64_t flat_index, float value);

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
     *       Assumes a host-addressable (Cpu) backend: on a Cuda/Hip Tensor, move it with
     *       to(DeviceType::Cpu, cpu_backend) first.
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
     * @brief Same-device no-op form of to().
     * @param target Target device. Must equal device().
     * @return *this, for chaining.
     * @note A Tensor holds exactly one non-owned DeviceBackend*, and there is no global
     *       backend registry, so a cross-device move cannot know which backend should own
     *       the new buffer -- use to(target, target_backend) for that. This overload exists
     *       for the charter's Phase 0 exit gate (to(device()) is a no-op).
     * @throws std::invalid_argument if target != device().
     */
    Tensor& to(DeviceType target);

    /**
     * @brief Moves this tensor's buffer to another device, owned by target_backend.
     * @param target Device the new buffer resides on. Must be the device target_backend
     *        allocates on (Cpu for CPUBackend, Cuda for CUDABackend, Hip for HIPBackend) --
     *        not checkable here, since DeviceBackend does not report its own device.
     * @param target_backend Backend to allocate the new buffer through. Not owned; must
     *        outlive this Tensor, exactly as the constructor's backend must.
     * @return *this, for chaining. Afterwards device() == target and every subsequent
     *         allocate/copy/free goes through target_backend.
     * @note The copy is issued by whichever backend owns the device-side pointer:
     *       Cpu -> device uses target_backend (HostToDevice); device -> Cpu uses the current
     *       backend (DeviceToHost); between two different GPU device types (Cuda <-> Hip) the
     *       data is staged through a host buffer, since neither vendor's runtime can address
     *       the other's memory. Same device type through a different backend instance copies
     *       directly (HostToHost / DeviceToDevice).
     * @note Strong exception guarantee: if allocation or the copy throws, this Tensor is left
     *       unchanged (same buffer, backend and device) and the new buffer is released.
     * @note A no-op when target == device() and target_backend is the current backend.
     * @throws std::invalid_argument if target_backend is nullptr.
     */
    Tensor& to(DeviceType target, DeviceBackend* target_backend);

private:
    [[nodiscard]] int64_t flat_index_of(std::initializer_list<int64_t> index) const;

    float* data_;
    Shape shape_;
    DeviceBackend* backend_;
    DeviceType device_;
    bool requires_grad_ = true;
};

/**
 * @brief Throws unless `t` lives on `expected` -- the check every module and loss runs on the
 *        tensors handed to it (roadmap FND-8, gpu_review #1).
 * @param where Names the entry point in the error message, e.g. "LinearModule::backward".
 * @throws std::invalid_argument on a mismatch. Without it, a CPU tensor reaching a GPU kernel
 *         is an uncatchable memory fault (HSA abort on gfx1151, a sticky CUDA error), and a GPU
 *         tensor reaching the CPU backend segfaults.
 */
inline void require_device(const Tensor& t, DeviceType expected, const char* where) {
    if (t.device() != expected) {
        auto name = [](DeviceType d) {
            switch (d) {
                case DeviceType::Cpu:
                    return "Cpu";
                case DeviceType::Cuda:
                    return "Cuda";
                case DeviceType::Hip:
                    return "Hip";
            }
            return "unknown";
        };
        throw std::invalid_argument(std::string(where) + ": tensor is on " + name(t.device()) +
                                    " but this computes on " + name(expected) + "; move it first with Tensor::to()");
    }
}

}  // namespace pulsatrix
