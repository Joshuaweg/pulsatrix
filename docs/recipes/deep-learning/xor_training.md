# Recipe: XOR Training Walkthrough

**What you'll build:** a tiny `Linear(2,4) -> ReLU -> Linear(4,1)` network trained on XOR. XOR is
the classic problem a single linear layer can't solve, but a small MLP can represent exactly.

CMake target: `xor_training_recipe` (`examples/recipes/xor_training.cpp`).

Run it: `./build/xor_training_recipe` (Windows: `build\Release\xor_training_recipe.exe`).

## Code

```cpp
CPUBackend backend;
XorNetwork net(&backend);
AdamOptimizer optimizer(0.01f, &backend);
NoOpMetricsSink sink;

std::vector<std::pair<Tensor, Tensor>> dataset;
dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {0.0f, 0.0f}), Tensor(Shape({1, 1}), &backend, {0.0f}));
dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {0.0f, 1.0f}), Tensor(Shape({1, 1}), &backend, {1.0f}));
dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {1.0f, 0.0f}), Tensor(Shape({1, 1}), &backend, {1.0f}));
dataset.emplace_back(Tensor(Shape({1, 2}), &backend, {1.0f, 1.0f}), Tensor(Shape({1, 1}), &backend, {0.0f}));

int step = 0;
for (int epoch = 1; epoch <= 300; ++epoch) {
    for (auto& [input, target] : dataset) {
        (void)net.train_step(input, target, optimizer, sink, step++);
    }
}
```

Full source: [`examples/recipes/xor_training.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/xor_training.cpp).

## Expected output

```
XOR training recipe -- Linear(2,4)->ReLU->Linear(4,1), Adam(lr=0.01)
epoch     0 | mean loss 0.370519
epoch    50 | mean loss 0.099930
epoch   100 | mean loss 0.015606
epoch   150 | mean loss 0.000720
epoch   200 | mean loss 0.000022
epoch   250 | mean loss 0.000000
epoch   300 | mean loss 0.000000

Final predictions:
  (0, 0) -> raw 0.0001, rounded 0 (target 0)
  (0, 1) -> raw 1.0000, rounded 1 (target 1)
  (1, 0) -> raw 1.0000, rounded 1 (target 1)
  (1, 1) -> raw 0.0000, rounded 0 (target 0)
```

## What's happening

`XorNetwork` chains its `LinearModule` and `ReluModule` layers directly. Each `train_step()` call:

1. zeroes the gradients,
2. runs the forward pass and computes the MSE loss,
3. calls each layer's `backward()` in reverse order to fill its parameter gradients,
4. applies one `AdamOptimizer` step per `LinearModule`, and
5. logs the loss to the metrics sink.

The loss drops to (numerically) zero well before 300 epochs. This network can represent XOR
exactly, so there's no approximation error left to converge past.

See also: [Deep Learning Modules and Layers](../../deep-learning/index.md#building-and-training-a-small-network).
