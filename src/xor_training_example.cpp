#include "exai/xor_training_example.hpp"

namespace exai {

XorNetwork::XorNetwork(DeviceBackend* backend)
    : backend_(backend), linear1_(2, 4, backend), relu_(backend), linear2_(4, 1, backend), loss_(backend) {
    // Fixed, hand-picked, non-zero, deliberately asymmetric initial weights -- see the
    // class-level @note for why zero-init doesn't work here. An earlier sign-mirrored
    // choice (row1 ~= -row0) was tried and rejected during this mission: it made the
    // hidden layer's activation for input (1,1) an exact scalar multiple of its
    // activation for (0,0) at initialization, a structural degeneracy that persisted
    // through training and prevented the network from ever distinguishing those two
    // inputs. This initialization was checked to have no such proportionality between any
    // pair of the four XOR inputs' hidden activations before being used here.
    linear1_.set_weight({0.6f, -0.3f, 0.4f, -0.7f, 0.2f, 0.5f, -0.6f, 0.1f});
    linear1_.set_bias({0.05f, -0.05f, 0.02f, -0.08f});
    linear2_.set_weight({0.4f, -0.6f, 0.5f, -0.3f});
    linear2_.set_bias({0.1f});
}

Tensor XorNetwork::forward(const Tensor& input) {
    Tensor h1 = linear1_.forward(input);
    Tensor h2 = relu_.forward(h1);
    return linear2_.forward(h2);
}

float XorNetwork::train_step(const Tensor& input, const Tensor& target, AdamOptimizer& optimizer, MetricsSink& sink,
                              int step) {
    // Gradients accumulate (Tensor::accumulate) across backward() calls by design (Mission
    // 3 -- needed for fan-out accumulation in general). A single train_step must zero them
    // first, or every step's gradient silently adds onto every previous step's, growing
    // unboundedly over a long training run. Found during this mission's own convergence
    // testing: training was diverging with a suspiciously constant per-step update
    // magnitude, traced to exactly this missing zero_grad() call.
    optimizer.zero_grad(linear1_);
    optimizer.zero_grad(linear2_);

    Tensor prediction = forward(input);
    float loss_value = loss_.forward(prediction, target);

    Tensor grad_pred = loss_.backward();
    Tensor grad_relu_out = linear2_.backward(grad_pred);
    Tensor grad_linear1_out = relu_.backward(grad_relu_out);
    (void)linear1_.backward(grad_linear1_out);  // input has no further upstream to receive this

    optimizer.step(linear1_);
    optimizer.step(linear2_);
    // relu_ has no parameters (Module::parameters() default) -- optimizer.step(relu_)
    // would be a safe no-op too, but calling it would imply there's something to update.

    sink.log_scalar("loss", static_cast<double>(loss_value), step);
    return loss_value;
}

}  // namespace exai
