# Recipe: CSV + DataLoader Training

**What you'll build:** a `LinearModule(2,1)` regressor trained to recover `y = 2*x1 - 3*x2 + 1`
from a synthetic CSV file. It pulls shuffled batches through `CsvDataset` + `DataLoader` instead
of building `Tensor`s by hand (compare the [XOR recipe](../deep-learning/xor_training.md)'s
hardcoded four-example dataset).

CMake target: `csv_dataloader_training_recipe` (`examples/recipes/csv_dataloader_training.cpp`).

Run it: `./build/csv_dataloader_training_recipe` (Windows:
`build\Release\csv_dataloader_training_recipe.exe`).

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

At startup the recipe writes a 20-row CSV file, `csv_dataloader_training_recipe_data.csv`, into
the current working directory. It deletes the file on exit, so there's nothing to download.

`CsvDataset` loads the file and turns each row into a `(1, 2)` feature `Tensor` and a `(1,)`
label `Tensor`. A new `DataLoader` is built each epoch. Its `ShuffleSampler` reshuffles on every
construction, so each epoch sees the batches in a different order.

`LinearModule` starts with all weights at zero. That's fine for a single linear layer, whose
gradient doesn't vanish at zero weights (unlike a multi-layer network with `ReLU` in between).
Training converges toward the true `y = 2*x1 - 3*x2 + 1` using only the batches `DataLoader`
produces.

See also: [Data Loading, Transformation & Validation](../../data-pipeline/index.md#loading-a-csv-file-through-dataloader).
