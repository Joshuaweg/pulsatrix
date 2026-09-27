# Recipe: CSV + DataLoader training

**What you'll build:** a `LinearModule(2,1)` regressor trained to recover `y = 2*x1 - 3*x2 + 1`
from a synthetic CSV file, pulling shuffled batches through `CsvDataset` + `DataLoader`
instead of hand-building `Tensor`s in code (contrast with the
[XOR recipe](../deep-learning/xor_training.md)'s hardcoded four-example dataset).

CMake target: `csv_dataloader_training_recipe` (`examples/recipes/csv_dataloader_training.cpp`).

## Code

```cpp
auto dataset = std::make_shared<CsvDataset>(csv_path, std::vector<std::string>{"x1", "x2"}, "y", &backend);

DataLoaderOptions options;
options.batch_size = 4;
options.shuffle = true;
options.shuffle_seed = 42;

LinearModule model(/*in_features=*/2, /*out_features=*/1, &backend);
AdamOptimizer optimizer(0.05f, &backend);
MSELoss loss_fn(&backend);

for (int epoch = 1; epoch <= 50; ++epoch) {
    DataLoader loader(dataset, &backend, options);  // fresh Sampler -> reshuffled order each epoch
    while (auto batch = loader.next_batch()) {
        optimizer.zero_grad(model);
        Tensor prediction = model.forward(batch->fields[0]);
        float loss_value = loss_fn.forward(prediction, batch->fields[1]);
        Tensor grad_prediction = loss_fn.backward();
        (void)model.backward(grad_prediction);
        optimizer.step(model);
    }
}
```

Full source: [`examples/recipes/csv_dataloader_training.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/csv_dataloader_training.cpp).

## Expected output

```
CSV + DataLoader training recipe -- Linear(2,1) regressing y = 2*x1 - 3*x2 + 1
20 rows, batch_size=4, shuffled each epoch

epoch  1 | mean batch MSE 53.0817
epoch 10 | mean batch MSE 13.8806
epoch 20 | mean batch MSE 3.6161
epoch 30 | mean batch MSE 0.6891
epoch 40 | mean batch MSE 0.1259
epoch 50 | mean batch MSE 0.0374
```

## What's happening

The recipe writes its own 20-row CSV fixture at startup (recipes are self-contained — no
external file to fetch), then loads it through `CsvDataset`, which tokenizes each row into a
`(1, 2)` feature `Tensor` and a `(1,)` label `Tensor`. A fresh `DataLoader` is constructed each
epoch — `ShuffleSampler` reseeds and reshuffles on every `DataLoader` construction, so each
epoch sees a different batch order, the same regularization effect any framework's per-epoch
reshuffling gives you. `LinearModule` starts zero-initialized (no random init needed here —
unlike a multi-layer network with `ReLU` in between, a single linear layer's gradient doesn't
vanish at zero weights, so training starts from an honest "knows nothing" state) and converges
to the true `y = 2*x1 - 3*x2 + 1` relationship purely from the batches `DataLoader` produces.

See also: [Data Loading, Transformation & Validation](../../data-pipeline/index.md#loading-a-csv-file-through-dataloader).
