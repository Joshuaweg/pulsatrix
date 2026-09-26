#include "pulsatrix/learnable_scalar.hpp"

#include <stdexcept>

namespace pulsatrix {

LearnableScalar::LearnableScalar(float initial_value) : value_(initial_value) {}

void LearnableScalar::accumulate_grad(float grad) {
    throw std::logic_error("LearnableScalar::accumulate_grad not yet implemented");
}

void LearnableScalar::zero_grad() {
    throw std::logic_error("LearnableScalar::zero_grad not yet implemented");
}

void LearnableScalar::step(float learning_rate) {
    throw std::logic_error("LearnableScalar::step not yet implemented");
}

}  // namespace pulsatrix
