# Recipe: RNN vs. LSTM vs. GRU on a Parity Task

**What you'll build:** three recurrent modules (`RNNModule`, `LSTMModule`, `GRUModule`) trained
side by side on the same synthetic running-parity (cumulative XOR) task. The comparison shows
why gated recurrent units exist.

CMake target: `sequence_models_recipe`
(`examples/recipes/sequence_models_rnn_lstm_gru.cpp`).

Run it: `./build/sequence_models_recipe` (Windows: `build\Release\sequence_models_recipe.exe`).

## Code

```cpp
RNNModule rnn(1, 1, &backend);
// ... seeded weight init ...

LSTMModule lstm(1, 1, &backend);
// ... seeded weight init ...

GRUModule gru(1, 1, &backend);
// ... seeded weight init ...

// train_and_report(): the same loop runs for rnn, lstm and gru
AdamOptimizer optimizer(0.05f, &backend);
MSELoss loss(&backend);
float final_loss = 0.0f;
for (int epoch = 0; epoch <= 2000; ++epoch) {
    optimizer.zero_grad(module);
    Tensor prediction = module.forward(input);   // input: (6, 10, 1) binary sequences
    final_loss = loss.forward(prediction, target);
    if (epoch == 2000) break;                    // last pass only measures the loss
    Tensor grad = loss.backward();
    (void)module.backward(grad);
    optimizer.step(module);
}
```

Full source: [`examples/recipes/sequence_models_rnn_lstm_gru.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/sequence_models_rnn_lstm_gru.cpp).

## Expected output

```
Sequence-model recipe -- running parity of 6 seeded sequences (length 10)

RNNModule  | final loss 0.241627 | timestep accuracy  32/60 (53.3%)
LSTMModule | final loss 0.013926 | timestep accuracy  60/60 (100.0%)
GRUModule  | final loss 0.000020 | timestep accuracy  60/60 (100.0%)
...
```

## What's happening

The task is `h_t = XOR(h_{t-1}, x_t)`. A vanilla Elman cell computes
`tanh(w_x*x_t + w_h*h_{t-1} + b)`, which is monotone in each argument. A monotone function cannot
represent XOR (see `examples/sequence_model_demo.cpp` for the full argument). So `RNNModule`
settles on the best monotone fit: predict the current bit and ignore history. It lands on loss
0.2416 regardless of learning rate or seed.

`LSTMModule` and `GRUModule` solve the task exactly. Their *multiplicative* gates let the carried
state flip sign depending on the current input, which a plain tanh unit can't do. The RNN's
failure is a limit of what it can represent, not an optimization failure.

See also: [Deep Learning Modules and Layers](../../deep-learning/index.md).
