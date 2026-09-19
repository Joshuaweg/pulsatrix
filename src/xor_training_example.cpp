#include "exai/xor_training_example.hpp"

namespace exai {

XorNetwork::XorNetwork(DeviceBackend* backend)
    : backend_(backend), linear1_(2, 4, backend), relu_(backend), linear2_(4, 1, backend), loss_(backend) {
    // Fixed, hand-picked, non-zero, sign-varied initial weights -- see the class-level
    // @note for why zero-init doesn't work here.
    linear1_.set_weight({0.5f, -0.5f, 0.3f, -0.3f, -0.4f, 0.4f, -0.2f, 0.2f});
    linear1_.set_bias({0.1f, -0.1f, 0.1f, -0.1f});
    linear2_.set_weight({0.5f, -0.5f, 0.3f, -0.3f});
    linear2_.set_bias({0.0f});
}

Tensor XorNetwork::forward(const Tensor& input) {
    Tensor h1 = linear1_.forward(input);
    Tensor h2 = relu_.forward(h1);
    return linear2_.forward(h2);
}

float XorNetwork::train_step(const Tensor& input, const Tensor& target, AdamOptimizer& optimizer, MetricsSink& sink,
                              int step) {
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
