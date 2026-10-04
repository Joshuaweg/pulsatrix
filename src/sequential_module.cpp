#include "pulsatrix/sequential_module.hpp"

#include <stdexcept>
#include <utility>

namespace pulsatrix {

SequentialModule::SequentialModule(std::vector<Module*> layers) : layers_(std::move(layers)) {
    // External boundary (constructor arguments can originate from Phase 5's Python
    // bindings with no upstream validation). An empty container would make the
    // before-forward guard below vacuous -- rejected explicitly rather than silently
    // behaving as a no-op identity.
    if (layers_.empty()) {
        throw std::invalid_argument("SequentialModule: layers must not be empty");
    }
    for (Module* layer : layers_) {
        if (layer == nullptr) {
            throw std::invalid_argument("SequentialModule: layers must not contain a null entry");
        }
    }
}

Tensor SequentialModule::forward_impl(const Tensor& input) {
    Tensor current(input);
    for (Module* layer : layers_) {
        current = layer->forward(current);
    }
    has_forwarded_ = true;
    return current;
}

Tensor SequentialModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("SequentialModule::backward: called before any forward()");
    }
    Tensor grad(grad_output);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
    return grad;
}

Tensor SequentialModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("SequentialModule::propagate_relevance: called before any forward()");
    }
    Tensor relevance(relevance_out);
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        relevance = (*it)->propagate_relevance(relevance, config);
    }
    return relevance;
}

std::vector<NamedBufferRef> SequentialModule::named_buffers() {
    std::vector<NamedBufferRef> result;
    for (size_t i = 0; i < layers_.size(); ++i) {
        append_named_buffers(result, std::to_string(i), *layers_[i]);
    }
    return result;
}

std::vector<NamedParamRef> SequentialModule::named_parameters() {
    std::vector<NamedParamRef> result;
    for (size_t i = 0; i < layers_.size(); ++i) {
        append_named_parameters(result, std::to_string(i), *layers_[i]);
    }
    return result;
}

void SequentialModule::set_training(bool training) {
    Module::set_training(training);
    for (Module* layer : layers_) {
        layer->set_training(training);
    }
}

}  // namespace pulsatrix
