#include "pulsatrix/learnable_scalar.hpp"

namespace pulsatrix {

LearnableScalar::LearnableScalar(float initial_value) : value_(initial_value) {}

void LearnableScalar::accumulate_grad(float grad) { grad_ += grad; }

void LearnableScalar::zero_grad() { grad_ = 0.0f; }

void LearnableScalar::step(float learning_rate) { value_ -= learning_rate * grad_; }

}  // namespace pulsatrix
