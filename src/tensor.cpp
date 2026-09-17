#include "exai/tensor.hpp"

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
        backend_->copy(data_, values.begin(), static_cast<size_t>(shape_.numel()) * sizeof(float),
                        CopyDirection::HostToHost);
    }
}

Tensor::~Tensor() {
    backend_->free(data_);
}

Tensor::Tensor(const Tensor& other)
    : data_(nullptr), shape_(other.shape_), backend_(other.backend_), device_(other.device_) {
    data_ = allocate_buffer(backend_, shape_.numel());
    if (data_ != nullptr) {
        backend_->copy(data_, other.data_, static_cast<size_t>(shape_.numel()) * sizeof(float),
                        CopyDirection::HostToHost);
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

}  // namespace exai
