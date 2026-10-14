// DistilBERT's student (KD-4): some of a teacher's layers, a tap for a hidden-state loss, and the cosine loss.

#include "pulsatrix/layer_student.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

std::vector<int64_t> EveryOtherLayer(int64_t num_layers) {
    if (num_layers < 1) throw std::invalid_argument("EveryOtherLayer: num_layers must be positive");
    std::vector<int64_t> out;
    for (int64_t i = 0; i < num_layers; i += 2) out.push_back(i);
    return out;
}

namespace {

// "layers.<j>.rest" -> "layers.<layers[j]>.rest"; other names are the same in both models.
std::string TeacherName(const std::string& name, const std::vector<int64_t>& layers) {
    const std::string prefix = "layers.";
    if (name.compare(0, prefix.size(), prefix) != 0) return name;
    const size_t dot = name.find('.', prefix.size());
    const auto j = static_cast<size_t>(std::stoll(name.substr(prefix.size(), dot - prefix.size())));
    return prefix + std::to_string(layers.at(j)) + name.substr(dot);
}

template <typename Ref>
void CopyByName(std::vector<Ref> student, std::vector<Ref> teacher, const std::vector<int64_t>& layers, DeviceBackend* backend,
                const char* what) {
    std::map<std::string, Tensor*> by_name;
    for (Ref& r : teacher) {
        if (r.name.compare(0, 14, "interventions.") == 0) continue;
        by_name[r.name] = r.ref.value;
    }
    for (Ref& r : student) {
        const std::string name = TeacherName(r.name, layers);
        const auto found = by_name.find(name);
        if (found == by_name.end() || found->second->shape() != r.ref.value->shape()) {
            throw std::invalid_argument(std::string("MakeStudentFromLayers: the teacher has no ") + what + " " + name +
                                        " of the student's shape");
        }
        *r.ref.value = *found->second;  // takes the teacher's device, so move it back
        r.ref.value->to(backend->device(), backend);
    }
}

}  // namespace

std::unique_ptr<CausalLM> MakeStudentFromLayers(CausalLM& teacher, const std::vector<int64_t>& layers, DeviceBackend* backend) {
    if (layers.empty()) throw std::invalid_argument("MakeStudentFromLayers: no layers");
    if (std::set<int64_t>(layers.begin(), layers.end()).size() != layers.size()) {
        throw std::invalid_argument("MakeStudentFromLayers: a layer is chosen twice");
    }
    HfModelConfig config = teacher.config();
    for (int64_t i : layers) {
        if (i < 0 || i >= teacher.num_layers()) {
            throw std::invalid_argument("MakeStudentFromLayers: the teacher has no layer " + std::to_string(i));
        }
    }
    if (!config.layer_types.empty()) {
        std::vector<std::string> kept;
        for (int64_t i : layers) kept.push_back(config.layer_types.at(static_cast<size_t>(i)));
        config.layer_types = std::move(kept);
    }
    config.num_hidden_layers = static_cast<int64_t>(layers.size());
    // The teacher exists, so whatever its config asks for, it already runs.
    auto student = std::make_unique<CausalLM>(config, backend, /*allow_unsupported=*/true);
    // A teacher with LoRA adapters has parameters the student lacks; count them as a mismatch.
    const auto student_params = student->named_parameters();
    size_t teacher_params = 0;
    for (const NamedParamRef& p : teacher.named_parameters()) {
        if (p.name.compare(0, 14, "interventions.") == 0) continue;
        const std::string prefix = "layers.";
        if (p.name.compare(0, prefix.size(), prefix) == 0) {
            const int64_t i = std::stoll(p.name.substr(prefix.size(), p.name.find('.', prefix.size()) - prefix.size()));
            if (std::find(layers.begin(), layers.end(), i) == layers.end()) continue;
        }
        ++teacher_params;
    }
    if (teacher_params != student_params.size()) {
        throw std::invalid_argument("MakeStudentFromLayers: the teacher's chosen layers have " + std::to_string(teacher_params) +
                                    " parameters and the student " + std::to_string(student_params.size()) +
                                    "; adapters or other changes to the teacher aren't copied");
    }
    struct P {
        std::string name;
        ParamRef ref;
    };
    std::vector<P> s, t;
    for (const NamedParamRef& p : student_params) s.push_back({p.name, p.ref});
    for (const NamedParamRef& p : teacher.named_parameters()) t.push_back({p.name, p.ref});
    CopyByName(s, t, layers, backend, "parameter");
    std::vector<P> sb, tb;
    for (const NamedBufferRef& b : student->named_buffers()) sb.push_back({b.name, ParamRef{b.value, nullptr}});
    for (const NamedBufferRef& b : teacher.named_buffers()) tb.push_back({b.name, ParamRef{b.value, nullptr}});
    CopyByName(sb, tb, layers, backend, "buffer");
    return student;
}

HiddenStateTap::HiddenStateTap(DeviceBackend* backend)
    : backend_(backend), hidden_(Shape({1}), backend), extra_(Shape({1}), backend) {
    if (!backend) throw std::invalid_argument("HiddenStateTap: backend is null");
}

Tensor HiddenStateTap::forward_at(const Tensor& hidden, int64_t /*start*/, int64_t /*prompt_length*/) {
    hidden_ = hidden;
    has_hidden_ = true;
    has_extra_ = false;
    return hidden;
}

Tensor HiddenStateTap::backward(const Tensor& grad_output) {
    if (!has_extra_) return grad_output;
    if (grad_output.shape() != extra_.shape()) {
        throw std::invalid_argument("HiddenStateTap::backward: the gradient's shape isn't the hidden states'");
    }
    require_device(grad_output, backend_->device(), "HiddenStateTap::backward");
    Tensor out(grad_output.shape(), backend_, backend_->device());
    backend_->axpby(1.0f, grad_output.data(), 1.0f, extra_.data(), out.data(), static_cast<size_t>(out.numel()));
    has_extra_ = false;
    return out;
}

Tensor HiddenStateTap::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& /*config*/) { return relevance_out; }

void HiddenStateTap::release_activations() {
    hidden_ = Tensor(Shape({1}), backend_);
    has_hidden_ = false;
}

const Tensor& HiddenStateTap::hidden() const {
    if (!has_hidden_) throw std::logic_error("HiddenStateTap::hidden: no forward() yet");
    return hidden_;
}

void HiddenStateTap::add_gradient(const Tensor& gradient) {
    if (!has_hidden_) throw std::logic_error("HiddenStateTap::add_gradient: no forward() yet");
    if (gradient.shape() != hidden_.shape()) {
        throw std::invalid_argument("HiddenStateTap::add_gradient: the gradient's shape isn't the hidden states'");
    }
    require_device(gradient, backend_->device(), "HiddenStateTap::add_gradient");
    if (has_extra_) {
        backend_->axpby(1.0f, extra_.data(), 1.0f, gradient.data(), extra_.data(), static_cast<size_t>(extra_.numel()));
    } else {
        extra_ = gradient;
        has_extra_ = true;
    }
}

HiddenStateCosineLoss::HiddenStateCosineLoss(DeviceBackend* backend) : backend_(backend) {
    if (!backend) throw std::invalid_argument("HiddenStateCosineLoss: backend is null");
}

float HiddenStateCosineLoss::forward(const Tensor& student, const Tensor& teacher, const std::vector<float>& mask) {
    if (student.rank() < 2) throw std::invalid_argument("HiddenStateCosineLoss::forward: the rank must be at least 2");
    if (student.shape() != teacher.shape()) throw std::invalid_argument("HiddenStateCosineLoss::forward: the shapes differ");
    if (student.device() != teacher.device()) throw std::invalid_argument("HiddenStateCosineLoss::forward: the tensors are on different devices");
    const int64_t width = student.shape().dim(student.rank() - 1);
    const int64_t rows = student.numel() / width;
    if (!mask.empty() && static_cast<int64_t>(mask.size()) != rows) {
        throw std::invalid_argument("HiddenStateCosineLoss::forward: the mask needs one value per row, " + std::to_string(rows));
    }
    int64_t kept = 0;
    for (int64_t r = 0; r < rows; ++r) kept += mask.empty() || mask[static_cast<size_t>(r)] != 0.0f ? 1 : 0;
    if (kept == 0) throw std::invalid_argument("HiddenStateCosineLoss::forward: the mask keeps no row");
    const std::vector<float> s = student.to_host_vector();
    const std::vector<float> t = teacher.to_host_vector();
    grad_.assign(s.size(), 0.0);
    double total = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        if (!mask.empty() && mask[static_cast<size_t>(r)] == 0.0f) continue;
        const size_t o = static_cast<size_t>(r * width);
        double st = 0.0, ss = 0.0, tt = 0.0;
        for (int64_t k = 0; k < width; ++k) {
            const double a = s[o + static_cast<size_t>(k)], b = t[o + static_cast<size_t>(k)];
            st += a * b;
            ss += a * a;
            tt += b * b;
        }
        if (ss == 0.0 || tt == 0.0) {
            total += 1.0;
            continue;
        }
        const double ns = std::sqrt(ss), nt = std::sqrt(tt), cos = st / (ns * nt);
        total += 1.0 - cos;
        // d(1 - cos)/ds = -(t / (|s||t|) - cos s / |s|²), over the kept rows.
        for (int64_t k = 0; k < width; ++k) {
            const size_t i = o + static_cast<size_t>(k);
            grad_[i] = -(t[i] / (ns * nt) - cos * s[i] / ss) / static_cast<double>(kept);
        }
    }
    shape_ = student.shape();
    device_ = student.device();
    has_forwarded_ = true;
    return static_cast<float>(total / static_cast<double>(kept));
}

Tensor HiddenStateCosineLoss::backward() const {
    if (!has_forwarded_) throw std::logic_error("HiddenStateCosineLoss::backward: no forward() yet");
    return Tensor(shape_, backend_, std::vector<float>(grad_.begin(), grad_.end()), device_);
}

}  // namespace pulsatrix
