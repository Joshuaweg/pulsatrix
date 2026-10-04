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
    // parameters() order.
    double sum_sq = 0.0;
    for (ParamRef p : trainable) {
        const auto n = static_cast<size_t>(p.grad->numel());
        sum_sq += static_cast<double>(p.grad->backend()->dot(p.grad->data(), p.grad->data(), n));
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
