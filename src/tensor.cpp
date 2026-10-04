#include "pulsatrix/tensor.hpp"

#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

namespace {
[[nodiscard]] float* allocate_buffer(DeviceBackend* backend, int64_t numel) {
    return static_cast<float*>(backend->allocate(static_cast<size_t>(numel) * sizeof(float)));
}
}  // namespace

Tensor::Tensor(Shape shape, DeviceBackend* backend) : Tensor(std::move(shape), backend, backend->device()) {}

Tensor::Tensor(Shape shape, DeviceBackend* backend, std::initializer_list<float> values)
    : Tensor(std::move(shape), backend, values, backend->device()) {}

Tensor::Tensor(Shape shape, DeviceBackend* backend, const std::vector<float>& values)
    : Tensor(std::move(shape), backend, values, backend->device()) {}

Tensor::Tensor(Shape shape, DeviceBackend* backend, DeviceType device)
    : data_(nullptr), shape_(std::move(shape)), backend_(backend), device_(device) {
    data_ = allocate_buffer(backend_, shape_.numel());
    if (data_ != nullptr) {
        backend_->fill(data_, 0.0f, static_cast<size_t>(shape_.numel()));
    }
}

Tensor::Tensor(Shape shape, DeviceBackend* backend, std::initializer_list<float> values, DeviceType device)
    : data_(nullptr), shape_(std::move(shape)), backend_(backend), device_(device) {
    // External boundary (Mission 0 classification): values-constructors are called
    // directly with externally-supplied data via bindings/pulsatrix_py.cpp's Tensor::from_values.
    if (static_cast<int64_t>(values.size()) != shape_.numel()) {
        throw std::invalid_argument("Tensor: values.size() does not match shape's element count");
    }
    data_ = allocate_buffer(backend_, shape_.numel());
    if (data_ != nullptr) {
        // values.begin() is always a genuine host pointer (std::initializer_list lives on
        // the host); data_ may not be, depending on device_.
        CopyDirection dir = (device_ == DeviceType::Cpu) ? CopyDirection::HostToHost : CopyDirection::HostToDevice;
        backend_->copy(data_, values.begin(), static_cast<size_t>(shape_.numel()) * sizeof(float), dir);
    }
}

Tensor::Tensor(Shape shape, DeviceBackend* backend, const std::vector<float>& values, DeviceType device)
    : data_(nullptr), shape_(std::move(shape)), backend_(backend), device_(device) {
    if (static_cast<int64_t>(values.size()) != shape_.numel()) {
        throw std::invalid_argument("Tensor: values.size() does not match shape's element count");
    }
    data_ = allocate_buffer(backend_, shape_.numel());
    if (data_ != nullptr) {
        CopyDirection dir = (device_ == DeviceType::Cpu) ? CopyDirection::HostToHost : CopyDirection::HostToDevice;
        backend_->copy(data_, values.data(), static_cast<size_t>(shape_.numel()) * sizeof(float), dir);
    }
}

Tensor::~Tensor() {
    backend_->free(data_);
}

Tensor Tensor::Stack(const std::vector<Tensor>& tensors, DeviceBackend* backend) {
    // External boundary (data-pipeline campaign, Mission 0 classification): the list of
    // tensors to stack is assembled by a DataLoader/collate function from independently
    // constructed Dataset samples, not a compile-time-known invariant.
    if (tensors.empty()) {
        throw std::invalid_argument("Tensor::Stack: tensors must not be empty");
    }

    const Shape& first_shape = tensors[0].shape();
    int64_t rank = first_shape.rank();
    if (rank < 1) {
        throw std::invalid_argument("Tensor::Stack: tensors must have rank >= 1 (a leading batch dimension)");
    }
    DeviceType device = tensors[0].device();

    int64_t total_leading = 0;
    for (const Tensor& t : tensors) {
        if (t.rank() != rank) {
            throw std::invalid_argument("Tensor::Stack: all tensors must have the same rank");
        }
        if (t.device() != device) {
            throw std::invalid_argument("Tensor::Stack: all tensors must be on the same device");
        }
        for (int64_t d = 1; d < rank; ++d) {
            if (t.shape().dim(static_cast<size_t>(d)) != first_shape.dim(static_cast<size_t>(d))) {
                throw std::invalid_argument("Tensor::Stack: all tensors must match on every non-leading dimension");
            }
        }
        total_leading += t.shape().dim(0);
    }

    std::vector<int64_t> out_dims;
    out_dims.reserve(static_cast<size_t>(rank));
    out_dims.push_back(total_leading);
    for (int64_t d = 1; d < rank; ++d) {
        out_dims.push_back(first_shape.dim(static_cast<size_t>(d)));
    }

    // The result lives where `backend` allocates (FND-8, gpu_review #2): tag it with that device,
    // not the sources', and copy accordingly. Reads from a GPU into host memory go through the
    // source's own backend, the only one that can read that memory; everything else through the
    // destination's.
    const DeviceType target = backend->device();
    if (device != DeviceType::Cpu && target != DeviceType::Cpu && device != target) {
        throw std::invalid_argument("Tensor::Stack: cannot copy directly between two different GPU types");
    }
    Tensor result(Shape(out_dims), backend, target);
    if (result.data() == nullptr) {
        return result;
    }

    float* dst = result.data();
    for (const Tensor& t : tensors) {
        int64_t chunk_numel = t.numel();
        if (chunk_numel > 0) {
            const size_t bytes = static_cast<size_t>(chunk_numel) * sizeof(float);
            if (target == DeviceType::Cpu) {
                t.backend()->copy(dst, t.data(), bytes,
                                  device == DeviceType::Cpu ? CopyDirection::HostToHost : CopyDirection::DeviceToHost);
            } else {
                backend->copy(dst, t.data(), bytes,
                              device == DeviceType::Cpu ? CopyDirection::HostToDevice : CopyDirection::DeviceToDevice);
            }
            dst += chunk_numel;
        }
    }
    return result;
}

Tensor::Tensor(const Tensor& other)
    : data_(nullptr),
      shape_(other.shape_),
      backend_(other.backend_),
      device_(other.device_),
      requires_grad_(other.requires_grad_) {
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
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      backend_(other.backend_),
      device_(other.device_),
      requires_grad_(other.requires_grad_) {
    other.data_ = nullptr;
    // Restore the class's own documented invariant ("data() == nullptr iff numel() == 0")
    // for the moved-from object. std::move on shape_ alone leaves an unspecified-but-valid
    // Shape (typically empty dims -- rank 0, numel() == 1 by the scalar convention),
    // inconsistent with data_ == nullptr: a subsequent at()/operator[] on the moved-from
    // Tensor would compute a valid-looking index into a null buffer. See
    // campaign_exai_dl_library_adversarial_hardening.md's Mission 0, finding 9.
    other.shape_ = Shape({0});
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
    // requires_grad_ deliberately not taken from other -- see requires_grad()'s note.
    other.data_ = nullptr;
    other.shape_ = Shape({0});  // see move ctor's note
    return *this;
}

int64_t Tensor::flat_index_of(std::initializer_list<int64_t> index) const {
    PULSATRIX_ASSERT(static_cast<int64_t>(index.size()) == rank());

    int64_t flat = 0;
    int64_t stride = 1;
    // Walk dimensions right-to-left accumulating a row-major stride, matching index
    // right-to-left in lockstep (both are the same length, checked above).
    for (int64_t dim_pos = rank() - 1; dim_pos >= 0; --dim_pos) {
        int64_t idx_at_dim = *(index.begin() + dim_pos);
        PULSATRIX_ASSERT(idx_at_dim >= 0 && idx_at_dim < shape_.dim(static_cast<size_t>(dim_pos)));
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

float Tensor::read_element(int64_t flat_index) const {
    PULSATRIX_ASSERT(flat_index >= 0 && flat_index < numel());
    float value = 0.0f;
    const CopyDirection dir = (device_ == DeviceType::Cpu) ? CopyDirection::HostToHost : CopyDirection::DeviceToHost;
    backend_->copy(&value, data_ + flat_index, sizeof(float), dir);
    return value;
}

std::vector<float> Tensor::to_host_vector() const {
    std::vector<float> values(static_cast<size_t>(numel()));
    if (data_ != nullptr) {
        const CopyDirection dir =
            (device_ == DeviceType::Cpu) ? CopyDirection::HostToHost : CopyDirection::DeviceToHost;
        backend_->copy(values.data(), data_, values.size() * sizeof(float), dir);
    }
    return values;
}

void Tensor::write_element(int64_t flat_index, float value) {
    PULSATRIX_ASSERT(flat_index >= 0 && flat_index < numel());
    const CopyDirection dir = (device_ == DeviceType::Cpu) ? CopyDirection::HostToHost : CopyDirection::HostToDevice;
    backend_->copy(data_ + flat_index, &value, sizeof(float), dir);
}

Tensor& Tensor::fill(float value) {
    if (data_ != nullptr) {
        backend_->fill(data_, value, static_cast<size_t>(numel()));
    }
    return *this;
}

Tensor& Tensor::accumulate(const Tensor& other) {
    // External boundary (Mission 0 classification): gradient-accumulation shapes trace
    // back to module construction parameters, which can originate from external
    // configuration -- escalated from PULSATRIX_ASSERT-only to a real throw.
    if (!(shape_ == other.shape_)) {
        throw std::invalid_argument("Tensor::accumulate: shape mismatch");
    }
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
    throw std::invalid_argument(
        "Tensor::to: a cross-device move needs the target DeviceBackend -- use to(target, target_backend)");
}

Tensor& Tensor::to(DeviceType target, DeviceBackend* target_backend) {
    // External boundary: target_backend is caller-supplied, so a null one is a reachable
    // misuse rather than an internal invariant.
    if (target_backend == nullptr) {
        throw std::invalid_argument("Tensor::to: target_backend must not be null");
    }
    if (target == device_ && target_backend == backend_) {
        return *this;
    }

    const size_t bytes = static_cast<size_t>(numel()) * sizeof(float);
    float* new_data = allocate_buffer(target_backend, numel());
    if (new_data != nullptr) {
        try {
            const bool src_is_host = device_ == DeviceType::Cpu;
            const bool dst_is_host = target == DeviceType::Cpu;
            if (src_is_host && dst_is_host) {
                target_backend->copy(new_data, data_, bytes, CopyDirection::HostToHost);
            } else if (src_is_host) {
                target_backend->copy(new_data, data_, bytes, CopyDirection::HostToDevice);
            } else if (dst_is_host) {
                backend_->copy(new_data, data_, bytes, CopyDirection::DeviceToHost);
            } else if (target == device_) {
                target_backend->copy(new_data, data_, bytes, CopyDirection::DeviceToDevice);
            } else {
                // Cuda <-> Hip: neither runtime can address the other's memory.
                std::vector<float> staging(static_cast<size_t>(numel()));
                backend_->copy(staging.data(), data_, bytes, CopyDirection::DeviceToHost);
                target_backend->copy(new_data, staging.data(), bytes, CopyDirection::HostToDevice);
            }
        } catch (...) {
            target_backend->free(new_data);
            throw;
        }
    }

    backend_->free(data_);
    data_ = new_data;
    backend_ = target_backend;
    device_ = target;
    return *this;
}

}  // namespace pulsatrix
