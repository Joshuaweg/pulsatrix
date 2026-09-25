#include "pulsatrix/hypergrid_env.hpp"

#include <stdexcept>

namespace pulsatrix {

HyperGridEnv::HyperGridEnv(DeviceBackend* backend, int64_t ndim, int64_t side_length, float r0, float r1, float r2,
                            int64_t max_steps)
    : backend_(backend), ndim_(ndim), side_length_(side_length), r0_(r0), r1_(r1), r2_(r2), max_steps_(max_steps) {
    if (ndim < 1) {
        throw std::invalid_argument("HyperGridEnv: ndim must be >= 1");
    }
    if (side_length < 4) {
        throw std::invalid_argument("HyperGridEnv: side_length must be >= 4");
    }
    if (r0 < 0.0f || r1 < 0.0f || r2 < 0.0f) {
        throw std::invalid_argument("HyperGridEnv: reward constants must be >= 0");
    }
    if (max_steps < 1) {
        throw std::invalid_argument("HyperGridEnv: max_steps must be >= 1");
    }
    coord_.assign(static_cast<size_t>(ndim), 0);
}

Tensor HyperGridEnv::observation() const {
    throw std::logic_error("HyperGridEnv::observation not yet implemented");
}

void HyperGridEnv::decode_state(const Tensor&, std::vector<int64_t>&) const {
    throw std::logic_error("HyperGridEnv::decode_state not yet implemented");
}

Tensor HyperGridEnv::reset() {
    throw std::logic_error("HyperGridEnv::reset not yet implemented");
}

Tensor HyperGridEnv::reset(const Tensor&) {
    throw std::logic_error("HyperGridEnv::reset(initial_state) not yet implemented");
}

float HyperGridEnv::reward(const Tensor&) const {
    throw std::logic_error("HyperGridEnv::reward not yet implemented");
}

StepResult HyperGridEnv::step(const Tensor&) {
    throw std::logic_error("HyperGridEnv::step not yet implemented");
}

}  // namespace pulsatrix
