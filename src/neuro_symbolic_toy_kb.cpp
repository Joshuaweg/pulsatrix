#include "pulsatrix/neuro_symbolic_toy_kb.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

// Sigmoid squash + its elementary backward -- file-local glue, not a new library Module.
// See neuro_symbolic_toy_kb.hpp's class note ("Sigmoid squashing is file-local glue").
// Raw host loop; PULSATRIX_ASSERT-guarded against a non-Cpu tensor, same convention as
// every other raw-host-loop Module in this codebase (NegationModule, LinearModule).
Tensor sigmoid(const Tensor& z, DeviceBackend* backend) {
    PULSATRIX_REQUIRE_HOST(z);
    Tensor y(z.shape(), backend, z.device());
    const int64_t n = z.numel();
    for (int64_t i = 0; i < n; ++i) {
        y.data()[i] = 1.0f / (1.0f + std::exp(-z.data()[i]));
    }
    return y;
}

// dL/dz = dL/dy * y*(1-y), given the cached forward-pass output y.
Tensor sigmoid_backward(const Tensor& grad_y, const Tensor& y, DeviceBackend* backend) {
    PULSATRIX_REQUIRE_HOST(grad_y);
    Tensor grad_z(y.shape(), backend, y.device());
    const int64_t n = y.numel();
    for (int64_t i = 0; i < n; ++i) {
        const float yv = y.data()[i];
        grad_z.data()[i] = grad_y.data()[i] * yv * (1.0f - yv);
    }
    return grad_z;
}

// Splits a DisjunctionModule::backward()-shaped stacked gradient (leading dim 2) back into
// its two operand-shaped halves -- same "small, per-file helper" convention as
// disjunction_module.cpp's own split_operands/combine_operands (mission_0's Stage 3 note).
std::pair<Tensor, Tensor> split_stacked_grad(const Tensor& stacked, DeviceBackend* backend) {
    std::vector<int64_t> operand_dims;
    for (int64_t i = 1; i < stacked.rank(); ++i) {
        operand_dims.push_back(stacked.shape().dim(static_cast<size_t>(i)));
    }
    Shape operand_shape(operand_dims);
    const int64_t half = operand_shape.numel();

    // Dereferences Tensor::data() directly in a raw host loop -- not yet backend-generic
    // (GPU-native-kernels campaign, Mission 0 O4).
    PULSATRIX_REQUIRE_HOST(stacked);
    Tensor a(operand_shape, backend, stacked.device());
    Tensor b(operand_shape, backend, stacked.device());
    PULSATRIX_REQUIRE_HOST(a);
    PULSATRIX_REQUIRE_HOST(b);
    for (int64_t i = 0; i < half; ++i) {
        a.data()[i] = stacked.data()[i];
        b.data()[i] = stacked.data()[half + i];
    }
    return {std::move(a), std::move(b)};
}

}  // namespace

ToyKnowledgeBase::ToyKnowledgeBase(DeviceBackend* backend, float p)
    : backend_(backend),
      linear_a_(1, 1, backend),
      linear_b_(1, 1, backend),
      neg_a_(backend),
      neg_b_(backend),
      disj_(backend, DisjunctionModule::TConorm::Product),
      sat_loss_(backend, p),
      last_a_(Shape({0}), backend),
      last_b_(Shape({0}), backend),
      last_rule_(Shape({0}), backend) {
    // Mission file Stage 3's own hand-picked, small, sign-varied initial weights --
    // mirrors XorNetwork's documented non-zero-initialization rationale.
    linear_a_.set_weight({0.6f});
    linear_a_.set_bias({0.0f});
    linear_b_.set_weight({-0.4f});
    linear_b_.set_bias({0.0f});
}

float ToyKnowledgeBase::forward(const Tensor& x) {
    if (x.rank() != 2 || x.shape().dim(1) != 1 || x.shape().dim(0) <= 0) {
        throw std::invalid_argument("ToyKnowledgeBase::forward: x must have shape (N, 1) with N > 0");
    }

    Tensor z_a = linear_a_.forward(x);  // (N, 1)
    Tensor a = sigmoid(z_a, backend_);  // (N, 1), predicate A's truth degree
    Tensor z_b = linear_b_.forward(x);  // (N, 1)
    Tensor b = sigmoid(z_b, backend_);  // (N, 1), predicate B's truth degree

    Tensor not_a = neg_a_.forward(a);           // (N, 1)
    Tensor not_b = neg_b_.forward(b);           // (N, 1)
    Tensor rule2d = disj_.forward(not_a, not_b);  // (N, 1): not(A) or not(B), Product

    Tensor rule1d(rule2d);
    rule1d.reshape(Shape({rule2d.numel()}));  // (N,) -- SatisfactionLoss's own rank-1 scope

    last_a_ = std::move(a);
    last_b_ = std::move(b);
    last_rule_ = rule1d;
    has_forwarded_ = true;

    return sat_loss_.forward(rule1d);
}

void ToyKnowledgeBase::backward() {
    if (!has_forwarded_) {
        throw std::logic_error("ToyKnowledgeBase::backward: forward() has not been called yet");
    }

    Tensor grad_rule1d = sat_loss_.backward();  // (N,)
    Tensor grad_rule2d(grad_rule1d);
    grad_rule2d.reshape(Shape({grad_rule1d.numel(), 1}));  // (N, 1)

    Tensor grad_stack = disj_.backward(grad_rule2d);  // (2, N, 1)
    auto [grad_not_a, grad_not_b] = split_stacked_grad(grad_stack, backend_);

    Tensor grad_a = neg_a_.backward(grad_not_a);  // (N, 1)
    Tensor grad_b = neg_b_.backward(grad_not_b);  // (N, 1)

    Tensor grad_z_a = sigmoid_backward(grad_a, last_a_, backend_);
    Tensor grad_z_b = sigmoid_backward(grad_b, last_b_, backend_);

    (void)linear_a_.backward(grad_z_a);  // accumulates weight_A/bias_A grad; grad-wrt-x unused
    (void)linear_b_.backward(grad_z_b);  // accumulates weight_B/bias_B grad; grad-wrt-x unused
}

float ToyKnowledgeBase::train_step(const Tensor& x, SGDOptimizer& optimizer) {
    optimizer.zero_grad(linear_a_);
    optimizer.zero_grad(linear_b_);
    const float loss = forward(x);
    backward();
    optimizer.step(linear_a_);
    optimizer.step(linear_b_);
    return loss;
}

}  // namespace pulsatrix
