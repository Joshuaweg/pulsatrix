#include "pulsatrix/grad_clipping.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace pulsatrix {

float ClipGradNorm(Module& module, float max_norm, bool error_if_nonfinite) {
    if (!(max_norm > 0.0f) || !std::isfinite(max_norm)) {
        throw std::invalid_argument("ClipGradNorm: max_norm must be positive and finite");
    }
    std::vector<ParamRef> trainable;
    for (ParamRef p : module.parameters()) {
        if (p.value->requires_grad()) {
            trainable.push_back(p);
        }
    }
    // Each parameter's squared norm on its own device, summed on the host in double, in
    // parameters() order. HIP-6: when they share a backend, the squared norms are written to one
    // device buffer and read back together, one wait instead of one per parameter.
    double sum_sq = 0.0;
    DeviceBackend* shared = trainable.empty() ? nullptr : trainable.front().grad->backend();
    for (ParamRef p : trainable) {
        if (p.grad->backend() != shared) shared = nullptr;
    }
    if (shared != nullptr) {
        Tensor squares(Shape({static_cast<int64_t>(trainable.size())}), shared, trainable.front().grad->device());
        for (size_t i = 0; i < trainable.size(); ++i) {
            const Tensor& g = *trainable[i].grad;
            shared->dot_into(g.data(), g.data(), static_cast<size_t>(g.numel()), squares.data() + i);
        }
        for (const float sq : squares.to_host_vector()) sum_sq += static_cast<double>(sq);
    } else {
        for (ParamRef p : trainable) {
            const auto n = static_cast<size_t>(p.grad->numel());
            sum_sq += static_cast<double>(p.grad->backend()->dot(p.grad->data(), p.grad->data(), n));
        }
    }
    const auto norm = static_cast<float>(std::sqrt(sum_sq));
    if (!std::isfinite(norm)) {
        if (error_if_nonfinite) {
            throw std::runtime_error("ClipGradNorm: the gradient norm is not finite");
        }
        return norm;
    }
    const float coef = max_norm / (norm + 1e-6f);
    if (coef < 1.0f) {
        for (ParamRef p : trainable) {
            const auto n = static_cast<size_t>(p.grad->numel());
            p.grad->backend()->axpby(coef, p.grad->data(), 0.0f, p.grad->data(), p.grad->data(), n);
        }
    }
    return norm;
}

}  // namespace pulsatrix
