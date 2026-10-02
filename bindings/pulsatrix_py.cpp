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
#include <string>
#include <vector>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/attribution.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/grad_cam.hpp"
#include "pulsatrix/integrated_gradients.hpp"
#include "pulsatrix/kernel_shap.hpp"
#include "pulsatrix/lime.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/pdp.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/saliency.hpp"
#include "pulsatrix/shape.hpp"
#include "pulsatrix/tensor.hpp"

#ifdef PULSATRIX_PY_WITH_CUDA
#include "pulsatrix/cuda_backend.hpp"
#endif
#ifdef PULSATRIX_PY_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace py = pybind11;

namespace {

// Every Tensor requires a non-owned DeviceBackend* (Tensor's constructor signature is
// unchanged by this file -- see Recon). Python callers have no C++ backend object to
// supply, so the bindings translation unit owns a single default CPUBackend instance,
// exactly the same "caller owns a backend, passes a pointer in" pattern every C++ test
// and example already uses -- this is a binding-layer concern only.
pulsatrix::CPUBackend& default_backend() {
    static pulsatrix::CPUBackend backend;
    return backend;
}

// One binding-owned backend per device, same ownership pattern as default_backend(). GPU
// backends are created on first use rather than at import, so importing the module on a
// machine without the GPU still works -- only asking for that device fails, and it fails as
// a Python exception (the backend constructor throws) rather than at load time.
pulsatrix::DeviceBackend& backend_for(pulsatrix::DeviceType device) {
    switch (device) {
        case pulsatrix::DeviceType::Cpu:
            return default_backend();
        case pulsatrix::DeviceType::Cuda: {
#ifdef PULSATRIX_PY_WITH_CUDA
            static pulsatrix::CUDABackend backend;
            return backend;
#else
            throw std::invalid_argument("pulsatrix_py: built without CUDA support (PULSATRIX_ENABLE_CUDA=OFF)");
#endif
        }
        case pulsatrix::DeviceType::Hip: {
#ifdef PULSATRIX_PY_WITH_HIP
            static pulsatrix::HIPBackend backend;
            return backend;
#else
            throw std::invalid_argument("pulsatrix_py: built without HIP support (PULSATRIX_ENABLE_HIP=OFF)");
#endif
        }
    }
    throw std::invalid_argument("pulsatrix_py: unknown device");
}

std::vector<pulsatrix::DeviceType> compiled_devices() {
    std::vector<pulsatrix::DeviceType> devices{pulsatrix::DeviceType::Cpu};
#ifdef PULSATRIX_PY_WITH_CUDA
    devices.push_back(pulsatrix::DeviceType::Cuda);
#endif
#ifdef PULSATRIX_PY_WITH_HIP
    devices.push_back(pulsatrix::DeviceType::Hip);
#endif
    return devices;
}

// Element access and the buffer protocol dereference Tensor::data() on the host, which is
// only valid for a Cpu tensor. These are external boundaries (Python), so a device tensor
// gets a real exception -- a PULSATRIX_ASSERT compiles out under NDEBUG and would leave UB.
void require_host(const pulsatrix::Tensor& t, const char* what) {
    if (t.device() != pulsatrix::DeviceType::Cpu) {
        throw std::invalid_argument(std::string("pulsatrix_py: ") + what +
                                    " needs a Cpu tensor -- call .to(DeviceType.Cpu) first");
    }
}

pulsatrix::Shape shape_from_list(const std::vector<int64_t>& dims) {
    switch (dims.size()) {
        case 0:
            return pulsatrix::Shape({});
        case 1:
            return pulsatrix::Shape({dims[0]});
        case 2:
            return pulsatrix::Shape({dims[0], dims[1]});
        case 3:
            return pulsatrix::Shape({dims[0], dims[1], dims[2]});
        case 4:
            return pulsatrix::Shape({dims[0], dims[1], dims[2], dims[3]});
        default:
            throw std::invalid_argument("pulsatrix_py: shapes above rank 4 are not supported by this binding");
    }
}

std::vector<int64_t> dims_of(const pulsatrix::Tensor& t) {
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
// this throws on rank mismatch or an out-of-range index rather than PULSATRIX_ASSERT-ing --
// the binding-layer echo of Tensor::operator[]'s own internal-invariant/PULSATRIX_ASSERT
// choice, since here the caller genuinely isn't validated by construction.
int64_t flat_index_of(const pulsatrix::Tensor& t, const std::vector<int64_t>& index) {
    if (static_cast<int64_t>(index.size()) != t.rank()) {
        throw std::invalid_argument("pulsatrix_py: index rank does not match tensor rank");
    }
    int64_t flat = 0;
    for (int64_t i = 0; i < t.rank(); ++i) {
        int64_t dim = t.shape().dim(static_cast<size_t>(i));
        int64_t idx = index[static_cast<size_t>(i)];
        if (idx < 0 || idx >= dim) {
            throw std::out_of_range("pulsatrix_py: index out of range for tensor shape");
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
class PyMetricsSink : public pulsatrix::MetricsSink {
public:
    using pulsatrix::MetricsSink::MetricsSink;

    void log_scalar(const std::string& tag, double value, int step) override {
        PYBIND11_OVERRIDE_PURE(void, pulsatrix::MetricsSink, log_scalar, tag, value, step);
    }

    void log_histogram(const std::string& tag, const pulsatrix::Tensor& values, int step) override {
        PYBIND11_OVERRIDE_PURE(void, pulsatrix::MetricsSink, log_histogram, tag, values, step);
    }
};

}  // namespace

PYBIND11_MODULE(pulsatrix_py, m) {
    m.doc() = "pulsatrix Python bindings (Phase 5) -- optional, non-load-bearing per charter non-negotiable #2";

    py::enum_<pulsatrix::DeviceType>(m, "DeviceType")
        .value("Cpu", pulsatrix::DeviceType::Cpu)
        .value("Cuda", pulsatrix::DeviceType::Cuda)
        .value("Hip", pulsatrix::DeviceType::Hip);

    m.def("compiled_devices", &compiled_devices,
          "Devices this build has a backend for. Cpu is always present; a GPU device listed here "
          "can still fail on first use if no matching GPU is visible at runtime.");

    py::class_<pulsatrix::Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init([](const std::vector<int64_t>& dims) {
                 return pulsatrix::Tensor(shape_from_list(dims), &default_backend());
             }),
             py::arg("shape"), "Zero-initialized tensor of the given shape.")
        .def_static(
            "zeros",
            [](const std::vector<int64_t>& dims) { return pulsatrix::Tensor(shape_from_list(dims), &default_backend()); },
            py::arg("shape"))
        .def_static(
            "from_values",
            [](const std::vector<int64_t>& dims, const std::vector<float>& values) {
                return pulsatrix::Tensor(shape_from_list(dims), &default_backend(), values);
            },
            py::arg("shape"), py::arg("values"))
        .def("shape", &dims_of)
        .def("numel", &pulsatrix::Tensor::numel)
        .def("device", &pulsatrix::Tensor::device)
        .def(
            "to",
            [](pulsatrix::Tensor& t, pulsatrix::DeviceType device) -> pulsatrix::Tensor& {
                return t.to(device, &backend_for(device));
            },
            py::arg("device"), py::return_value_policy::reference_internal,
            "Moves this tensor's buffer to device in place, through the binding-owned backend "
            "for that device. Returns self.")
        .def("at",
             [](pulsatrix::Tensor& t, const std::vector<int64_t>& index) {
                 require_host(t, "at()");
                 return t[flat_index_of(t, index)];
             })
        .def("set_at",
             [](pulsatrix::Tensor& t, const std::vector<int64_t>& index, float value) {
                 require_host(t, "set_at()");
                 t[flat_index_of(t, index)] = value;
             })
        .def_buffer([](pulsatrix::Tensor& t) -> py::buffer_info {
            // Phase 1.5 Mission 2's guard discipline, applied at this new boundary:
            // Tensor::data() is only a genuine host pointer when CPU-backed.
            require_host(t, "the buffer protocol (numpy)");

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

    py::class_<pulsatrix::Module>(m, "Module").def("forward", &pulsatrix::Module::forward, py::arg("input"));

    py::class_<pulsatrix::LinearModule, pulsatrix::Module>(m, "LinearModule")
        .def(py::init([](int64_t in_features, int64_t out_features, pulsatrix::DeviceType device) {
                 return new pulsatrix::LinearModule(in_features, out_features, &backend_for(device), device);
             }),
             py::arg("in_features"), py::arg("out_features"), py::arg("device") = pulsatrix::DeviceType::Cpu)
        .def("set_weight",
             static_cast<void (pulsatrix::LinearModule::*)(const std::vector<float>&)>(&pulsatrix::LinearModule::set_weight))
        .def("set_bias",
             static_cast<void (pulsatrix::LinearModule::*)(const std::vector<float>&)>(&pulsatrix::LinearModule::set_bias));

    py::class_<pulsatrix::ReluModule, pulsatrix::Module>(m, "ReluModule")
        .def(py::init([](pulsatrix::DeviceType device) {
                 return new pulsatrix::ReluModule(&backend_for(device), device);
             }),
             py::arg("device") = pulsatrix::DeviceType::Cpu);

    py::class_<pulsatrix::FlattenModule, pulsatrix::Module>(m, "FlattenModule")
        .def(py::init([]() { return new pulsatrix::FlattenModule(&default_backend()); }));

    py::class_<pulsatrix::Conv2DModule, pulsatrix::Module>(m, "Conv2DModule")
        .def(py::init([](int64_t in_channels, int64_t out_channels, int64_t kernel_h, int64_t kernel_w) {
                 return new pulsatrix::Conv2DModule(in_channels, out_channels, kernel_h, kernel_w, &default_backend());
             }),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_h"), py::arg("kernel_w"))
        .def("set_kernel",
             static_cast<void (pulsatrix::Conv2DModule::*)(const std::vector<float>&)>(&pulsatrix::Conv2DModule::set_kernel))
        .def("set_bias",
             static_cast<void (pulsatrix::Conv2DModule::*)(const std::vector<float>&)>(&pulsatrix::Conv2DModule::set_bias));

    // Phase 5 Mission 1, Objective 1. Attribution has no default constructor (mirrors
    // Tensor's own no-default-ctor design) -- bind a 3-arg constructor mirroring the
    // aggregate's field order, and expose the fields read-only. Not how explainers build
    // one internally (aggregate-init, per every explainer header's own note), but this is
    // the Python-side entry point once Objective 2/3's explainers start returning them.
    py::class_<pulsatrix::Attribution>(m, "Attribution")
        .def(py::init([](std::string method, pulsatrix::Tensor values, std::unordered_map<std::string, std::string> metadata) {
                 return pulsatrix::Attribution{std::move(method), std::move(values), std::move(metadata)};
             }),
             py::arg("method"), py::arg("values"), py::arg("metadata"))
        .def_readonly("method", &pulsatrix::Attribution::method)
        .def_readonly("values", &pulsatrix::Attribution::values)
        .def_readonly("metadata", &pulsatrix::Attribution::metadata);

    // Construction-only surface (Recon: forward_pass/backward_pass/graph()/activation()/
    // gradient() stay C++-internal plumbing the explainers call themselves in Objective 2 --
    // nothing in this mission's scope needs them exposed to Python directly). The
    // std::invalid_argument the ctor throws on an empty or null-containing module vector
    // (adversarial hardening, Mission 2, finding 5) maps to Python's ValueError
    // automatically via pybind11's built-in std::invalid_argument translation.
    py::class_<pulsatrix::ExplainerContext>(m, "ExplainerContext")
        .def(py::init<std::vector<pulsatrix::Module*>>(), py::arg("modules"));

    // Phase 5 Mission 1, Objective 2: graph-native explainers. Each C++ explain() takes
    // a DeviceBackend* -- narrowed here to always pass default_backend(), the same
    // binding-layer-only simplification Mission 0 already applied to every Module
    // subclass constructor (Python callers have no CUDABackend concept exposed at all,
    // so this parameter has exactly one reachable value from Python; not a core change,
    // the C++ signature itself is untouched).
    // LRP campaign Mission 1: whole-model LRP. Seeds are "output_value" (relevance starts as the
    // explained logit) or "one_hot" (unit relevance -- Zennit/LXT's convention).
    py::enum_<pulsatrix::LRPSeed>(m, "LRPSeed")
        .value("OutputValue", pulsatrix::LRPSeed::OutputValue)
        .value("OneHot", pulsatrix::LRPSeed::OneHot);

    // LRP-rules Mission 2: rule choice (uniform, via keyword arguments) and Zennit's composite
    // presets (per-layer rules, as static factories). Invalid alpha/beta and unsupported rules
    // raise ValueError (std::invalid_argument) at explain().
    py::enum_<pulsatrix::LRPRule>(m, "LRPRule")
        .value("Epsilon", pulsatrix::LRPRule::Epsilon)
        .value("Gamma", pulsatrix::LRPRule::Gamma)
        .value("AlphaBeta", pulsatrix::LRPRule::AlphaBeta)
        .value("ZBox", pulsatrix::LRPRule::ZBox);

    const pulsatrix::LRPRuleConfig lrp_defaults{};
    py::class_<pulsatrix::LRP>(m, "LRP")
        .def(py::init([](float epsilon, pulsatrix::LRPRule rule, float gamma, float alpha, float beta, float low,
                         float high, bool epsilon_bias_in_denominator) {
                 pulsatrix::LRPRuleConfig config{epsilon};
                 config.rule = rule;
                 config.gamma = gamma;
                 config.alpha = alpha;
                 config.beta = beta;
                 config.low = low;
                 config.high = high;
                 config.epsilon_bias_in_denominator = epsilon_bias_in_denominator;
                 return pulsatrix::LRP(config);
             }),
             py::arg("epsilon") = lrp_defaults.epsilon, py::kw_only(), py::arg("rule") = lrp_defaults.rule,
             py::arg("gamma") = lrp_defaults.gamma, py::arg("alpha") = lrp_defaults.alpha,
             py::arg("beta") = lrp_defaults.beta, py::arg("low") = lrp_defaults.low,
             py::arg("high") = lrp_defaults.high,
             py::arg("epsilon_bias_in_denominator") = lrp_defaults.epsilon_bias_in_denominator)
        .def_static("epsilon_plus", &pulsatrix::LRP::epsilon_plus, py::arg("epsilon") = 1e-6f,
                    "Zennit EpsilonPlus: Epsilon for Linear, ZPlus for Conv2D.")
        .def_static("epsilon_alpha2_beta1", &pulsatrix::LRP::epsilon_alpha2_beta1, py::arg("epsilon") = 1e-6f,
                    "Zennit EpsilonAlpha2Beta1: Epsilon for Linear, AlphaBeta(2, 1) for Conv2D.")
        .def_static("epsilon_gamma_box", &pulsatrix::LRP::epsilon_gamma_box, py::arg("low"), py::arg("high"),
                    py::arg("gamma") = 0.25f, py::arg("epsilon") = 1e-6f,
                    "Zennit EpsilonGammaBox: ZBox on the first Conv2D, Gamma on other Conv2D, Epsilon on Linear.")
        .def(
            "explain",
            [](const pulsatrix::LRP& self, pulsatrix::ExplainerContext& ctx, const pulsatrix::Tensor& input,
               std::vector<int64_t> targets, std::vector<int64_t> contrasts, pulsatrix::LRPSeed seed) {
                pulsatrix::LRPTarget target{std::move(targets), std::move(contrasts), seed};
                return self.explain(ctx, input, target, input.backend());
            },
            py::arg("ctx"), py::arg("input"), py::arg("targets"), py::arg("contrasts") = std::vector<int64_t>{},
            py::arg("seed") = pulsatrix::LRPSeed::OutputValue,
            "targets / contrasts: one index for every row, or one per row of the output.");

    py::class_<pulsatrix::Saliency>(m, "Saliency")
        .def(py::init<>())
        .def(
            "explain",
            [](const pulsatrix::Saliency& self, pulsatrix::ExplainerContext& ctx, const pulsatrix::Tensor& input,
               int64_t target_index) { return self.explain(ctx, input, target_index, &default_backend()); },
            py::arg("ctx"), py::arg("input"), py::arg("target_index"));

    py::class_<pulsatrix::IntegratedGradients>(m, "IntegratedGradients")
        .def(py::init<>())
        .def(
            "explain",
            [](const pulsatrix::IntegratedGradients& self, pulsatrix::ExplainerContext& ctx, const pulsatrix::Tensor& input,
               const pulsatrix::Tensor& baseline, int64_t target_index, int64_t steps) {
                return self.explain(ctx, input, baseline, target_index, steps, &default_backend());
            },
            py::arg("ctx"), py::arg("input"), py::arg("baseline"), py::arg("target_index"), py::arg("steps"));

    py::class_<pulsatrix::GradCAM>(m, "GradCAM")
        .def(py::init<>())
        .def(
            "explain",
            [](const pulsatrix::GradCAM& self, pulsatrix::ExplainerContext& ctx, const pulsatrix::Tensor& input,
               int64_t target_index) { return self.explain(ctx, input, target_index, &default_backend()); },
            py::arg("ctx"), py::arg("input"), py::arg("target_index"));

    // Phase 5 Mission 1, Objective 3: surrogate explainers. Graph-free -- take a Python
    // callable (input Tensor -> output Tensor) directly via pybind11/functional.h's
    // built-in std::function<Tensor(const Tensor&)> caster, no ExplainerContext involved
    // at all. Same default_backend() narrowing as Objective 2's graph-native explainers.
    py::class_<pulsatrix::LIME>(m, "LIME")
        .def(py::init<>())
        .def(
            "explain",
            [](const pulsatrix::LIME& self, const std::function<pulsatrix::Tensor(const pulsatrix::Tensor&)>& predict,
               const pulsatrix::Tensor& input, int64_t target_index, int64_t num_samples, float sigma, float l2_lambda,
               unsigned seed) {
                return self.explain(predict, input, target_index, num_samples, sigma, l2_lambda, seed,
                                     &default_backend());
            },
            py::arg("predict"), py::arg("input"), py::arg("target_index"), py::arg("num_samples"),
            py::arg("sigma"), py::arg("l2_lambda"), py::arg("seed"));

    py::class_<pulsatrix::KernelSHAP>(m, "KernelSHAP")
        .def(py::init<>())
        .def(
            "explain",
            [](const pulsatrix::KernelSHAP& self, const std::function<pulsatrix::Tensor(const pulsatrix::Tensor&)>& predict,
               const pulsatrix::Tensor& input, const pulsatrix::Tensor& baseline, int64_t target_index) {
                return self.explain(predict, input, baseline, target_index, &default_backend());
            },
            py::arg("predict"), py::arg("input"), py::arg("baseline"), py::arg("target_index"));

    py::class_<pulsatrix::PDP>(m, "PDP")
        .def(py::init<>())
        .def(
            "explain",
            [](const pulsatrix::PDP& self, const std::function<pulsatrix::Tensor(const pulsatrix::Tensor&)>& predict,
               const std::vector<pulsatrix::Tensor>& background, int64_t feature_index, int64_t target_index,
               float grid_min, float grid_max, int64_t grid_size) {
                return self.explain(predict, background, feature_index, target_index, grid_min, grid_max, grid_size,
                                     &default_backend());
            },
            py::arg("predict"), py::arg("background"), py::arg("feature_index"), py::arg("target_index"),
            py::arg("grid_min"), py::arg("grid_max"), py::arg("grid_size"));

    // Phase 5 Mission 1, Objective 4. PyMetricsSink is the trampoline; NoOpMetricsSink is
    // the one concrete C++ writer the charter ships (real writers are explicitly
    // out-of-scope for the core project, Python-side is where they belong).
    py::class_<pulsatrix::MetricsSink, PyMetricsSink>(m, "MetricsSink").def(py::init<>());

    py::class_<pulsatrix::NoOpMetricsSink, pulsatrix::MetricsSink>(m, "NoOpMetricsSink").def(py::init<>());

    // Binding-layer-only test helper: calls both MetricsSink methods through a base
    // MetricsSink& the same way metrics_sink_test.cpp's own CallableThroughBasePointer
    // test does in C++, proving PyMetricsSink's overrides are actually reachable from a
    // real (non-Python) call site -- not a new production API.
    m.def("_drive_metrics_sink_for_test", [](pulsatrix::MetricsSink& sink) {
        pulsatrix::Tensor values(pulsatrix::Shape({2}), &default_backend(), {1.0f, 2.0f});
        sink.log_scalar("loss", 0.5, 3);
        sink.log_histogram("weights", values, 3);
    });
}
