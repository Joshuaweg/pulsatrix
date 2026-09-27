# Data Loading, Transformation & Validation

Every layer, optimizer, and explainer in [Deep Learning Modules and Layers](../deep-learning/index.md)
consumes a `Tensor` — but nothing produces one from a real file until this layer. `Dataset`/
`IterableDataset`/`DataLoader`/`Transform`/`Compose`/`CollateFn` are the generic core;
everything else in this section is a concrete implementation built on top of it, one per
modality (tabular, image, text, audio, and a reduced-scope video stub), plus generic dataset
validation. The core abstractions needed **zero interface changes** across all five modality
implementations — every new capability was purely additive.

## What's inside

- **Core**: `Sample`, `Dataset`, `IterableDataset`, `Sampler` (`SequentialSampler`/
  `ShuffleSampler`), `Transform`/`Compose`/`TransformDataset`, `Batch`/`CollateFn`/
  `DefaultCollate`, `DataLoader`/`DataLoaderOptions`, `BoundedQueue`/`DataThreadPool` (built,
  not yet wired into `DataLoader` — see below)
- **Tabular**: `CsvReader`/`CsvTable`, `CsvDataset`
- **Image**: `ImageDecoder` (stb_image-backed), `ResizeTransform`/`CenterCropTransform`/
  `NormalizeTransform`/`HorizontalFlipTransform`, `ImageFolderDataset`
- **Text**: `Tokenizer`, `Vocabulary`/`BuildVocabulary`, `TextDataset`, `PadCollate`
- **Audio**: `WavReader`/`WavData`, `AudioFolderDataset`, `ResampleTransform`,
  `AudioPadCollate`
- **Video** (reduced-scope stub — pre-extracted frames only, no container/codec decode):
  `VideoFrameDirectoryDataset`, `UniformFrameSampleTransform`
- **Validation**: `FieldStatistics`/`DatasetStatistics`/`ValidationIssue`,
  `DatasetValidator::ComputeStatistics`/`DetectIssues` (descriptive statistics, schema
  consistency, missingness/outlier detection — distributional drift detection and
  bias/fairness metrics are explicitly out of scope; see Notes)
- **Retrofit**: `MnistDatasetAdapter` adapts the original `MnistIdxLoader` (see
  [Deep Learning Modules and Layers](../deep-learning/index.md)) onto the `Dataset` interface

Full API reference: [Doxygen: Data Loading, Transformation & Validation](../api/group__data__pipeline.html)

## How to implement

### Loading a CSV file through `DataLoader`

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/csv_dataset.hpp"
#include "pulsatrix/data_loader.hpp"

using namespace pulsatrix;

CPUBackend backend;
auto dataset = std::make_shared<CsvDataset>("data.csv", std::vector<std::string>{"x1", "x2"},
                                             /*label_column=*/"y", &backend);

DataLoaderOptions options;
options.batch_size = 4;
options.shuffle = true;

DataLoader loader(dataset, &backend, options);
while (auto batch = loader.next_batch()) {
    // batch->fields[0]: (N, 2) features, batch->fields[1]: (N,) labels
}
```

**What's happening:** `CsvDataset::get(i)` tokenizes row `i` into a `(1, num_features)`
feature `Tensor` and a `(1,)` label `Tensor` — the same batch-of-one-per-sample convention
`MnistDatasetAdapter` already uses. `DataLoader` drives a `Sampler` (shuffled or sequential)
over the dataset's indices and calls `Tensor::Stack` (via `DefaultCollate`) to concatenate
each field across the batch. Everything here is synchronous — `num_workers` defaults to `0`,
so no threads are spawned; `BoundedQueue`/`DataThreadPool` exist and are unit-tested but
aren't wired into a parallel fetch path yet (a named follow-up).

Recipe: [CSV + DataLoader training](../recipes/data-pipeline/csv_dataloader_training.md).

### Handling variable-length data with a custom `CollateFn`

```cpp
#include "pulsatrix/text_collate.hpp"
#include "pulsatrix/text_dataset.hpp"

using namespace pulsatrix;

Vocabulary vocab = BuildVocabulary(/* tokenized corpus */ {});
auto dataset = std::make_shared<TextDataset>("corpus.txt", &vocab, &backend);

DataLoaderOptions options;
options.batch_size = 8;
options.collate_fn = PadCollate();  // default DefaultCollate can't handle ragged sequences

DataLoader loader(dataset, &backend, options);
// batch->fields[0]: (N, max_len) right-padded token indices
// batch->fields[1]: (N,) real (pre-padding) lengths
```

**What's happening:** `TextDataset::get(i)` produces a `(1, seq_len)` token-index `Tensor`
per line, with `seq_len` varying per sample — `DefaultCollate`'s `Tensor::Stack` can't
concatenate mismatched shapes. Overriding `DataLoaderOptions::collate_fn` with `PadCollate()`
is the extension point: it pads every sample to the batch's own max length and returns a
lengths field alongside the padded data, with no changes needed to `Dataset`/`DataLoader`/
`Batch` themselves. `AudioPadCollate` (audio) is the same pattern applied to a second ragged
modality.

## Notes

- **Token IDs are stored as float32 `Tensor` values**, not a separate integer type —
  `Tensor` is float32-only throughout this library, and vocab sizes are always far below
  float32's exact-integer range (2^24). This is the same integer-as-float32 convention
  `CsvDataset`'s label column and `MnistDatasetAdapter`'s class label already use.
- **Video is a reduced-scope stub**: `VideoFrameDirectoryDataset` reads directories of
  already-extracted frame images (via `ImageDecoder`) — there is no real container/codec
  decode (no FFmpeg dependency, no licensing exposure). Frame-sampled clips
  (`UniformFrameSampleTransform`'s output) are `(N, C, H, W)`, identical in shape convention
  to an image batch.
- **Data validation is descriptive-stats-only** — `DatasetValidator` computes per-field
  statistics and flags missing (NaN) values and z-score outliers, generically against any
  `Dataset`. Distributional drift detection and bias/fairness metrics are explicitly out of
  scope, named follow-ups for a future effort.

## Recipes

- [CSV + DataLoader training](../recipes/data-pipeline/csv_dataloader_training.md)

See also [`examples/mnist_dataloader_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/mnist_dataloader_demo.cpp)
for a full demo training `MnistConvNet` on real MNIST data pulled through `DataLoader` +
`MnistDatasetAdapter` instead of iterating raw vectors directly.
