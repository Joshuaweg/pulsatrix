// Phase 5 Mission 0: pybind11 module exposing Tensor and Module/concrete subclasses.
// Non-negotiable #2: this file #includes core headers, never the reverse -- no core
// header may ever include a pybind11 header. See
// campaign_exai_dl_library_phase5_bindings.md and this mission's own Recon for the two
// binding-layer-specific design decisions this file embodies (default backend ownership,
// buffer-protocol device guard) that are not changes to the core itself.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

#include "exai/assert.hpp"
#include "exai/attribution.hpp"
#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/flatten_module.hpp"
#include "exai/grad_cam.hpp"
#include "exai/integrated_gradients.hpp"
#include "exai/kernel_shap.hpp"
#include "exai/lime.hpp"
#include "exai/linear_module.hpp"
#include "exai/metrics_sink.hpp"
#include "exai/module.hpp"
#include "exai/pdp.hpp"
#include "exai/relu_module.hpp"
#include "exai/saliency.hpp"
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
//
// Genuine external boundary (campaign_exai_dl_library_adversarial_hardening.md's Mission
// 0 classification, applied to the binding layer): index comes directly from Python, so
// this throws on rank mismatch or an out-of-range index rather than EXAI_ASSERT-ing --
// the binding-layer echo of Tensor::operator[]'s own internal-invariant/EXAI_ASSERT
// choice, since here the caller genuinely isn't validated by construction.
int64_t flat_index_of(const exai::Tensor& t, const std::vector<int64_t>& index) {
    if (static_cast<int64_t>(index.size()) != t.rank()) {
        throw std::invalid_argument("exai_py: index rank does not match tensor rank");
    }
    int64_t flat = 0;
    for (int64_t i = 0; i < t.rank(); ++i) {
        int64_t dim = t.shape().dim(static_cast<size_t>(i));
        int64_t idx = index[static_cast<size_t>(i)];
        if (idx < 0 || idx >= dim) {
            throw std::out_of_range("exai_py: index out of range for tensor shape");
        }
        flat = flat * dim + idx;
    }
    return flat;
}

// Phase 5 Mission 1, Objective 4. MetricsSink is a genuine pure-virtual interface with an
// explicit charter intent for Python-side concrete writers (TensorBoard/W&B/...) -- the
// one class in this mission that actually needs a trampoline (context_pybind11_ownership_
// gil.md's pattern). PYBIND11_OVERRIDE_PURE acquires the GIL internally before calling
// into the Python override; no manual py::gil_scoped_acquire needed here.
class PyMetricsSink : public exai::MetricsSink {
public:
    using exai::MetricsSink::MetricsSink;

    void log_scalar(const std::string& tag, double value, int step) override {
        PYBIND11_OVERRIDE_PURE(void, exai::MetricsSink, log_scalar, tag, value, step);
    }

    void log_histogram(const std::string& tag, const exai::Tensor& values, int step) override {
        PYBIND11_OVERRIDE_PURE(void, exai::MetricsSink, log_histogram, tag, values, step);
    }
};

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

    // Phase 5 Mission 1, Objective 1. Attribution has no default constructor (mirrors
    // Tensor's own no-default-ctor design) -- bind a 3-arg constructor mirroring the
    // aggregate's field order, and expose the fields read-only. Not how explainers build
    // one internally (aggregate-init, per every explainer header's own note), but this is
    // the Python-side entry point once Objective 2/3's explainers start returning them.
    py::class_<exai::Attribution>(m, "Attribution")
        .def(py::init([](std::string method, exai::Tensor values, std::unordered_map<std::string, std::string> metadata) {
                 return exai::Attribution{std::move(method), std::move(values), std::move(metadata)};
             }),
             py::arg("method"), py::arg("values"), py::arg("metadata"))
        .def_readonly("method", &exai::Attribution::method)
        .def_readonly("values", &exai::Attribution::values)
        .def_readonly("metadata", &exai::Attribution::metadata);

    // Construction-only surface (Recon: forward_pass/backward_pass/graph()/activation()/
    // gradient() stay C++-internal plumbing the explainers call themselves in Objective 2 --
    // nothing in this mission's scope needs them exposed to Python directly). The
    // std::invalid_argument the ctor throws on an empty or null-containing module vector
    // (adversarial hardening, Mission 2, finding 5) maps to Python's ValueError
    // automatically via pybind11's built-in std::invalid_argument translation.
    py::class_<exai::ExplainerContext>(m, "ExplainerContext")
        .def(py::init<std::vector<exai::Module*>>(), py::arg("modules"));

    // Phase 5 Mission 1, Objective 2: graph-native explainers. Each C++ explain() takes
    // a DeviceBackend* -- narrowed here to always pass default_backend(), the same
    // binding-layer-only simplification Mission 0 already applied to every Module
    // subclass constructor (Python callers have no CUDABackend concept exposed at all,
    // so this parameter has exactly one reachable value from Python; not a core change,
    // the C++ signature itself is untouched).
    py::class_<exai::Saliency>(m, "Saliency")
        .def(py::init<>())
        .def(
            "explain",
            [](const exai::Saliency& self, exai::ExplainerContext& ctx, const exai::Tensor& input,
               int64_t target_index) { return self.explain(ctx, input, target_index, &default_backend()); },
            py::arg("ctx"), py::arg("input"), py::arg("target_index"));

    py::class_<exai::IntegratedGradients>(m, "IntegratedGradients")
        .def(py::init<>())
        .def(
            "explain",
            [](const exai::IntegratedGradients& self, exai::ExplainerContext& ctx, const exai::Tensor& input,
               const exai::Tensor& baseline, int64_t target_index, int64_t steps) {
                return self.explain(ctx, input, baseline, target_index, steps, &default_backend());
            },
            py::arg("ctx"), py::arg("input"), py::arg("baseline"), py::arg("target_index"), py::arg("steps"));

    py::class_<exai::GradCAM>(m, "GradCAM")
        .def(py::init<>())
        .def(
            "explain",
            [](const exai::GradCAM& self, exai::ExplainerContext& ctx, const exai::Tensor& input,
               int64_t target_index) { return self.explain(ctx, input, target_index, &default_backend()); },
            py::arg("ctx"), py::arg("input"), py::arg("target_index"));

    // Phase 5 Mission 1, Objective 3: surrogate explainers. Graph-free -- take a Python
    // callable (input Tensor -> output Tensor) directly via pybind11/functional.h's
    // built-in std::function<Tensor(const Tensor&)> caster, no ExplainerContext involved
    // at all. Same default_backend() narrowing as Objective 2's graph-native explainers.
    py::class_<exai::LIME>(m, "LIME")
        .def(py::init<>())
        .def(
            "explain",
            [](const exai::LIME& self, const std::function<exai::Tensor(const exai::Tensor&)>& predict,
               const exai::Tensor& input, int64_t target_index, int64_t num_samples, float sigma, float l2_lambda,
               unsigned seed) {
                return self.explain(predict, input, target_index, num_samples, sigma, l2_lambda, seed,
                                     &default_backend());
            },
            py::arg("predict"), py::arg("input"), py::arg("target_index"), py::arg("num_samples"),
            py::arg("sigma"), py::arg("l2_lambda"), py::arg("seed"));

    py::class_<exai::KernelSHAP>(m, "KernelSHAP")
        .def(py::init<>())
        .def(
            "explain",
            [](const exai::KernelSHAP& self, const std::function<exai::Tensor(const exai::Tensor&)>& predict,
               const exai::Tensor& input, const exai::Tensor& baseline, int64_t target_index) {
                return self.explain(predict, input, baseline, target_index, &default_backend());
            },
            py::arg("predict"), py::arg("input"), py::arg("baseline"), py::arg("target_index"));

    py::class_<exai::PDP>(m, "PDP")
        .def(py::init<>())
        .def(
            "explain",
            [](const exai::PDP& self, const std::function<exai::Tensor(const exai::Tensor&)>& predict,
               const std::vector<exai::Tensor>& background, int64_t feature_index, int64_t target_index,
               float grid_min, float grid_max, int64_t grid_size) {
                return self.explain(predict, background, feature_index, target_index, grid_min, grid_max, grid_size,
                                     &default_backend());
            },
            py::arg("predict"), py::arg("background"), py::arg("feature_index"), py::arg("target_index"),
            py::arg("grid_min"), py::arg("grid_max"), py::arg("grid_size"));

    // Phase 5 Mission 1, Objective 4. PyMetricsSink is the trampoline; NoOpMetricsSink is
    // the one concrete C++ writer the charter ships (real writers are explicitly
    // out-of-scope for the core project, Python-side is where they belong).
    py::class_<exai::MetricsSink, PyMetricsSink>(m, "MetricsSink").def(py::init<>());

    py::class_<exai::NoOpMetricsSink, exai::MetricsSink>(m, "NoOpMetricsSink").def(py::init<>());

    // Binding-layer-only test helper: calls both MetricsSink methods through a base
    // MetricsSink& the same way metrics_sink_test.cpp's own CallableThroughBasePointer
    // test does in C++, proving PyMetricsSink's overrides are actually reachable from a
    // real (non-Python) call site -- not a new production API.
    m.def("_drive_metrics_sink_for_test", [](exai::MetricsSink& sink) {
        exai::Tensor values(exai::Shape({2}), &default_backend(), {1.0f, 2.0f});
        sink.log_scalar("loss", 0.5, 3);
        sink.log_histogram("weights", values, 3);
    });
}
