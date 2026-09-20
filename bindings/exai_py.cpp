// Phase 5 Mission 0: pybind11 module exposing Tensor and Module/concrete subclasses.
// Non-negotiable #2: this file #includes core headers, never the reverse -- no core
// header may ever include a pybind11 header. See
// campaign_exai_dl_library_phase5_bindings.md and this mission's own Recon for the two
// binding-layer-specific design decisions this file embodies (default backend ownership,
// buffer-protocol device guard) that are not changes to the core itself.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>

#include "exai/assert.hpp"
#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/flatten_module.hpp"
#include "exai/linear_module.hpp"
#include "exai/module.hpp"
#include "exai/relu_module.hpp"
#include "exai/shape.hpp"
#include "exai/tensor.hpp"

namespace py = pybind11;

namespace {

// Every Tensor requires a non-owned DeviceBackend* (Tensor's constructor signature is
// unchanged by this file -- see Recon). Python callers have no C++ backend object to
// supply, so the bindings translation unit owns a single default CPUBackend instance,
// exactly the same "caller owns a backend, passes a pointer in" pattern every C++ test
// and example already uses -- this is a binding-layer concern only.
exai::CPUBackend& default_backend() {
    static exai::CPUBackend backend;
    return backend;
}

exai::Shape shape_from_list(const std::vector<int64_t>& dims) {
    switch (dims.size()) {
        case 0:
            return exai::Shape({});
        case 1:
            return exai::Shape({dims[0]});
        case 2:
            return exai::Shape({dims[0], dims[1]});
        case 3:
            return exai::Shape({dims[0], dims[1], dims[2]});
        case 4:
            return exai::Shape({dims[0], dims[1], dims[2], dims[3]});
        default:
            throw std::invalid_argument("exai_py: shapes above rank 4 are not supported by this binding");
    }
}

std::vector<int64_t> dims_of(const exai::Tensor& t) {
    std::vector<int64_t> dims;
    for (int64_t i = 0; i < t.rank(); ++i) {
        dims.push_back(t.shape().dim(static_cast<size_t>(i)));
    }
    return dims;
}

// Row-major flat offset from a multi-dimensional index -- reimplemented here (rather
// than calling Tensor::at(std::initializer_list<int64_t>)) because std::initializer_list
// has no portable public constructor from a runtime-sized buffer, so a Python-supplied,
// dynamically-sized index list can't be forwarded to it. Same row-major convention
// Tensor::at() itself uses internally.
int64_t flat_index_of(const exai::Tensor& t, const std::vector<int64_t>& index) {
    EXAI_ASSERT(static_cast<int64_t>(index.size()) == t.rank());
    int64_t flat = 0;
    for (int64_t i = 0; i < t.rank(); ++i) {
        flat = flat * t.shape().dim(static_cast<size_t>(i)) + index[static_cast<size_t>(i)];
    }
    return flat;
}

}  // namespace

PYBIND11_MODULE(exai_py, m) {
    m.doc() = "exai_dl_library Python bindings (Phase 5) -- optional, non-load-bearing per charter non-negotiable #2";

    py::class_<exai::Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init([](const std::vector<int64_t>& dims) {
                 return exai::Tensor(shape_from_list(dims), &default_backend());
             }),
             py::arg("shape"), "Zero-initialized tensor of the given shape.")
        .def_static(
            "zeros",
            [](const std::vector<int64_t>& dims) { return exai::Tensor(shape_from_list(dims), &default_backend()); },
            py::arg("shape"))
        .def_static(
            "from_values",
            [](const std::vector<int64_t>& dims, const std::vector<float>& values) {
                return exai::Tensor(shape_from_list(dims), &default_backend(), values);
            },
            py::arg("shape"), py::arg("values"))
        .def("shape", &dims_of)
        .def("numel", &exai::Tensor::numel)
        .def("at", [](exai::Tensor& t, const std::vector<int64_t>& index) { return t[flat_index_of(t, index)]; })
        .def("set_at",
             [](exai::Tensor& t, const std::vector<int64_t>& index, float value) {
                 t[flat_index_of(t, index)] = value;
             })
        .def_buffer([](exai::Tensor& t) -> py::buffer_info {
            // Phase 1.5 Mission 2's guard discipline, applied at this new boundary:
            // Tensor::data() is only a genuine host pointer when CPU-backed. A CUDA-backed
            // Tensor reaching the buffer protocol unguarded would be silent UB.
            EXAI_ASSERT(t.device() == exai::DeviceType::Cpu);

            int64_t rank = t.rank();
            std::vector<py::ssize_t> shape(static_cast<size_t>(rank));
            std::vector<py::ssize_t> strides(static_cast<size_t>(rank));
            py::ssize_t running_stride = sizeof(float);
            for (int64_t i = rank - 1; i >= 0; --i) {
                shape[static_cast<size_t>(i)] = t.shape().dim(static_cast<size_t>(i));
                strides[static_cast<size_t>(i)] = running_stride;
                running_stride *= t.shape().dim(static_cast<size_t>(i));
            }
            return py::buffer_info(t.data(), sizeof(float), py::format_descriptor<float>::format(), rank, shape,
                                    strides);
        });

    py::class_<exai::Module>(m, "Module").def("forward", &exai::Module::forward, py::arg("input"));

    py::class_<exai::LinearModule, exai::Module>(m, "LinearModule")
        .def(py::init([](int64_t in_features, int64_t out_features) {
                 return new exai::LinearModule(in_features, out_features, &default_backend());
             }),
             py::arg("in_features"), py::arg("out_features"))
        .def("set_weight",
             static_cast<void (exai::LinearModule::*)(const std::vector<float>&)>(&exai::LinearModule::set_weight))
        .def("set_bias",
             static_cast<void (exai::LinearModule::*)(const std::vector<float>&)>(&exai::LinearModule::set_bias));

    py::class_<exai::ReluModule, exai::Module>(m, "ReluModule").def(
        py::init([]() { return new exai::ReluModule(&default_backend()); }));

    py::class_<exai::FlattenModule, exai::Module>(m, "FlattenModule")
        .def(py::init([]() { return new exai::FlattenModule(&default_backend()); }));

    py::class_<exai::Conv2DModule, exai::Module>(m, "Conv2DModule")
        .def(py::init([](int64_t in_channels, int64_t out_channels, int64_t kernel_h, int64_t kernel_w) {
                 return new exai::Conv2DModule(in_channels, out_channels, kernel_h, kernel_w, &default_backend());
             }),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_h"), py::arg("kernel_w"))
        .def("set_kernel",
             static_cast<void (exai::Conv2DModule::*)(const std::vector<float>&)>(&exai::Conv2DModule::set_kernel))
        .def("set_bias",
             static_cast<void (exai::Conv2DModule::*)(const std::vector<float>&)>(&exai::Conv2DModule::set_bias));
}
