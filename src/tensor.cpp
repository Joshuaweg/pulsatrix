#include "exai/tensor.hpp"

#include <stdexcept>

#include "exai/assert.hpp"

namespace exai {

namespace {
[[nodiscard]] float* allocate_buffer(DeviceBackend* backend, int64_t numel) {
    return static_cast<float*>(backend->allocate(static_cast<size_t>(numel) * sizeof(float)));
}
}  // namespace

Tensor::Tensor(Shape shape, DeviceBackend* backend, DeviceType device)
    : data_(nullptr), shape_(std::move(shape)), backend_(backend), device_(device) {
    data_ = allocate_buffer(backend_, shape_.numel());
    if (data_ != nullptr) {
        backend_->fill(data_, 0.0f, static_cast<size_t>(shape_.numel()));
    }
}

Tensor::Tensor(Shape shape, DeviceBackend* backend, std::initializer_list<float> values, DeviceType device)
    : data_(nullptr), shape_(std::move(shape)), backend_(backend), device_(device) {
    EXAI_ASSERT(static_cast<int64_t>(values.size()) == shape_.numel());
    data_ = allocate_buffer(backend_, shape_.numel());
    if (data_ != nullptr) {
        // values.begin() is always a genuine host pointer (std::initializer_list lives on
        // the host); data_ may not be, depending on device_.
        CopyDirection dir = (device_ == DeviceType::Cpu) ? CopyDirection::HostToHost : CopyDirection::HostToDevice;
        backend_->copy(data_, values.begin(), static_cast<size_t>(shape_.numel()) * sizeof(float), dir);
    }
}

Tensor::~Tensor() {
    backend_->free(data_);
}

Tensor::Tensor(const Tensor& other)
    : data_(nullptr), shape_(other.shape_), backend_(other.backend_), device_(other.device_) {
    data_ = allocate_buffer(backend_, shape_.numel());
    if (data_ != nullptr) {
        // Both data_ and other.data_ live on the SAME device (both allocated by backend_).
        CopyDirection dir =
            (device_ == DeviceType::Cpu) ? CopyDirection::HostToHost : CopyDirection::DeviceToDevice;
        backend_->copy(data_, other.data_, static_cast<size_t>(shape_.numel()) * sizeof(float), dir);
    }
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) {
        return *this;
    }
    Tensor tmp(other);
    *this = std::move(tmp);
    return *this;
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)), backend_(other.backend_), device_(other.device_) {
    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    backend_->free(data_);
    data_ = other.data_;
    shape_ = std::move(other.shape_);
    backend_ = other.backend_;
    device_ = other.device_;
    other.data_ = nullptr;
    return *this;
}

int64_t Tensor::flat_index_of(std::initializer_list<int64_t> index) const {
    EXAI_ASSERT(static_cast<int64_t>(index.size()) == rank());

    int64_t flat = 0;
    int64_t stride = 1;
    // Walk dimensions right-to-left accumulating a row-major stride, matching index
    // right-to-left in lockstep (both are the same length, checked above).
    for (int64_t dim_pos = rank() - 1; dim_pos >= 0; --dim_pos) {
        int64_t idx_at_dim = *(index.begin() + dim_pos);
        EXAI_ASSERT(idx_at_dim >= 0 && idx_at_dim < shape_.dim(static_cast<size_t>(dim_pos)));
        flat += idx_at_dim * stride;
        stride *= shape_.dim(static_cast<size_t>(dim_pos));
    }
    return flat;
}

float& Tensor::at(std::initializer_list<int64_t> index) {
    return data_[flat_index_of(index)];
}

const float& Tensor::at(std::initializer_list<int64_t> index) const {
    return data_[flat_index_of(index)];
}

Tensor& Tensor::fill(float value) {
    if (data_ != nullptr) {
        backend_->fill(data_, value, static_cast<size_t>(numel()));
    }
    return *this;
}

Tensor& Tensor::accumulate(const Tensor& other) {
    EXAI_ASSERT(shape_ == other.shape_);
    if (data_ != nullptr) {
        backend_->add(data_, other.data_, data_, static_cast<size_t>(numel()));
    }
    return *this;
}

Tensor& Tensor::reshape(Shape new_shape) {
    if (!shape_.is_reshape_compatible(new_shape)) {
        throw std::invalid_argument("Tensor::reshape: element count mismatch");
    }
    shape_ = std::move(new_shape);
    return *this;
}

Tensor& Tensor::to(DeviceType target) {
    if (target == device_) {
        return *this;
    }
    throw std::runtime_error(
        "Tensor::to: no DeviceBackend exists yet for the requested device (Phase 1.5/1.6)");
}

}  // namespace exai
