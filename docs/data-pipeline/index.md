# Data Loading, Transformation & Validation

This section covers how to get data from files on disk into batched `Tensor`s for training or
explanation. Use it to read CSV tables, image folders, text corpora, WAV audio or extracted
video frames, transform them, and check them for data-quality problems.

The generic core is `Dataset`/`IterableDataset`, `DataLoader`, `Transform`/`Compose` and
`CollateFn`. Everything else is a concrete implementation built on that core, one per modality:
tabular, image, text, audio and video (frames only). Generic dataset validation sits alongside.

## What's inside

- **Core**: `Sample`, `Dataset`, `IterableDataset` (streaming), `Sampler`
  (`SequentialSampler`/`ShuffleSampler`), `Transform`/`Compose`/`TransformDataset`,
  `Batch`/`CollateFn`/`DefaultCollate`, `DataLoader`/`DataLoaderOptions`.
  `BoundedQueue`/`DataThreadPool` are built but not yet wired into `DataLoader` (see below).
- **Tabular**: `CsvReader`/`CsvTable`, `CsvDataset`
- **Image**: `ImageDecoder` (backed by stb_image), `ResizeTransform`/`CenterCropTransform`/
  `NormalizeTransform`/`HorizontalFlipTransform`, `ImageFolderDataset`
- **Text**: `Tokenizer`, `Vocabulary`/`BuildVocabulary`, `TextDataset`, `PadCollate`
- **Tokenizers**: `TextTokenizer` (ids, token strings, byte offsets and a special-token mask),
  `LoadTokenizerJson` for Hugging Face byte-level BPE tokenizers (SmolLM2, Qwen, Llama 3,
  gpt-oss, ...), and word-level, byte and character tokenizers (see below)
- **Audio**: `WavReader`/`WavData`, `AudioFolderDataset`, `ResampleTransform`,
  `AudioPadCollate`
- **Video** (pre-extracted frames only; see Notes): `VideoFrameDirectoryDataset`,
  `UniformFrameSampleTransform`
- **Validation**: `FieldStatistics`/`DatasetStatistics`/`ValidationIssue`,
  `DatasetValidator::ComputeStatistics`/`DetectIssues`. These compute descriptive statistics
  and flag missing values and outliers (see Notes).
- **MNIST**: `MnistDatasetAdapter` wraps the MNIST IDX loader (`MnistIdxLoader`,
  `mnist_loader.hpp`) as a `Dataset`.

`DataLoaderOptions` fields: `batch_size` (default 1), `shuffle` and `shuffle_seed` (unset: drawn
from the global seed stream; see [Reproducibility](../deep-learning/index.md#reproducibility)),
`drop_last`, `collate_fn` (default `DefaultCollate`), `num_workers` and `prefetch_batches`.
`DataLoader` also accepts an `IterableDataset`; with one, `num_workers` must be 0 or 1.

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

**What's happening:** `CsvDataset::get(i)` parses row `i` into a `(1, num_features)` feature
`Tensor` and a `(1,)` label `Tensor`. `MnistDatasetAdapter` uses the same one-row-per-sample
convention. `DataLoader` walks the dataset's indices with a `Sampler`, shuffled or sequential.
`DefaultCollate` then calls `Tensor::Stack` to stack each field along dimension 0 into a batch.

Loading is synchronous: `num_workers` defaults to `0`, so no threads are spawned.
`BoundedQueue` and `DataThreadPool` exist and are unit-tested, but parallel fetching through
them is not wired in yet.

Recipe: [CSV + DataLoader training](../recipes/data-pipeline/csv_dataloader_training.md).

### Transforming images

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/image_folder_dataset.hpp"
#include "pulsatrix/image_transforms.hpp"
#include "pulsatrix/transform.hpp"

using namespace pulsatrix;

CPUBackend backend;
// root/<class_name>/<image_file>; labels are the sorted class-folder indices
auto images = std::make_shared<ImageFolderDataset>("data/images", &backend);

auto pipeline = std::make_shared<Compose>(std::vector<std::shared_ptr<Transform>>{
    std::make_shared<ResizeTransform>(32, 32, &backend),
    std::make_shared<NormalizeTransform>(std::vector<float>{0.5f, 0.5f, 0.5f},    // mean, one per channel
                                         std::vector<float>{0.5f, 0.5f, 0.5f}),  // std, one per channel
});
auto dataset = std::make_shared<TransformDataset>(images, pipeline);

DataLoaderOptions options;
options.batch_size = 16;
DataLoader loader(dataset, &backend, options);
// batch->fields[0]: (N, C, 32, 32) images, batch->fields[1]: (N,) labels
```

**What's happening:** `TransformDataset` applies the `Compose`d transforms, in order, to each
sample as it is fetched. Images are decoded lazily, one per `get()` call. `NormalizeTransform`
needs one mean and one std per channel; it throws if the counts don't match the image.

### Tokenizing for a Hugging Face model

`LoadTokenizerJson` reads a model's `tokenizer.json` and runs the same pipeline Hugging Face
`tokenizers` does: added tokens first, then the normalizer (NFC), the pre-tokenizer (regex,
digit and byte-level splitting), BPE, and the post-processor (BOS and EOS). There's no Python
involved.

```cpp
#include "pulsatrix/tokenizer_json.hpp"

using namespace pulsatrix;

TextTokenizer tok = LoadTokenizerJson("models/Llama-3.2-1B/tokenizer.json");
Encoding e = tok.encode("Grüße, world!");
// e.ids:     [128000, 6600, 2448, 24352, 11, 1917, 0]  (<|begin_of_text|> first)
// e.tokens:  the token strings
// e.offsets: UTF-8 byte ranges of the input: text.substr(o.begin, o.end - o.begin)
// e.special_tokens_mask: 1 for <|begin_of_text|> and other special tokens
std::string back = tok.decode(e.ids, /*skip_special_tokens=*/true);  // "Grüße, world!"
```

Offsets map every token back to the input, which explanations need to merge subword scores
into words. They are byte offsets, where Hugging Face uses characters. When NFC composes
characters, a composed character's offset covers all of its source characters; Hugging Face
gives it only the first.

**Checked against Hugging Face:** on a 10,000-line multilingual corpus (the Universal
Declaration of Human Rights in about 530 languages, plus code, numbers, whitespace and emoji
stress lines), the ids and the decoded text match `tokenizers` 0.23 exactly for SmolLM2-135M,
Qwen2.5, Qwen3, Llama 3.2 and gpt-oss (`tools/tokenizers/`, `pulsatrix_tokenizer_parity`).
Offsets match on every line that round-trips exactly. A component the loader doesn't support,
such as SentencePiece byte fallback (Gemma, TOK-3), is refused by name rather than approximated.

### Handling variable-length data with a custom `CollateFn`

`DefaultCollate` can only stack samples of the same shape. For ragged data such as text,
swap in a padding collate function:

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/text_collate.hpp"
#include "pulsatrix/text_dataset.hpp"
#include "pulsatrix/tokenizer.hpp"
#include "pulsatrix/vocabulary.hpp"

#include <fstream>

using namespace pulsatrix;

CPUBackend backend;

// Build the vocabulary from the same corpus, one token list per line.
std::vector<std::vector<std::string>> tokenized;
std::ifstream file("corpus.txt");
for (std::string line; std::getline(file, line);) {
    tokenized.push_back(Tokenizer::Tokenize(line));
}
Vocabulary vocab = BuildVocabulary(tokenized);

auto dataset = std::make_shared<TextDataset>("corpus.txt", &vocab, &backend);

DataLoaderOptions options;
options.batch_size = 8;
options.collate_fn = PadCollate();  // pads each batch to its longest sequence

DataLoader loader(dataset, &backend, options);
while (auto batch = loader.next_batch()) {
    // batch->fields[0]: (N, max_len) right-padded token indices
    // batch->fields[1]: (N,) lengths before padding
}
```

**What's happening:** `TextDataset::get(i)` turns line `i` into a `(1, seq_len)` tensor of
token indices, and `seq_len` varies per line. `PadCollate()` pads every sample to the batch's
longest length and adds a lengths field. `Dataset`, `DataLoader` and `Batch` need no changes.
`AudioPadCollate()` applies the same pattern to variable-length audio.

The default pad value, 0, is also the index of the unknown token `<unk>`. Use the lengths
field, not the token value, to find padding.

### Validating a dataset

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/csv_dataset.hpp"
#include "pulsatrix/dataset_validator.hpp"

using namespace pulsatrix;

CPUBackend backend;
CsvDataset dataset("data.csv", {"x1", "x2"}, /*label_column=*/"y", &backend);  // any Dataset works

DatasetStatistics stats = DatasetValidator::ComputeStatistics(dataset);
std::vector<ValidationIssue> issues =
    DatasetValidator::DetectIssues(dataset, stats, /*z_score_threshold=*/3.0f);
for (const ValidationIssue& issue : issues) {
    // issue.sample_index, issue.field_index, issue.description
}
```

**What's happening:** `ComputeStatistics` scans every sample and computes per-field
statistics. It throws if samples have different numbers of fields. `DetectIssues` then flags
missing (NaN) values and elements more than `z_score_threshold` standard deviations from their
field's mean. Both work on any `Dataset`.

## Notes

- **Token IDs are stored as float32 `Tensor` values**, not a separate integer type. `Tensor`
  is float32-only throughout this library, and float32 represents every integer up to 2^24
  exactly, far above any vocabulary size. `CsvDataset`'s label column and
  `MnistDatasetAdapter`'s class label use the same convention.
- **Video support reads pre-extracted frames only.** `VideoFrameDirectoryDataset` reads
  directories of frame images through `ImageDecoder`. It does not decode video containers or
  codecs, so there is no FFmpeg dependency. Clips sampled by `UniformFrameSampleTransform` are
  `(N, C, H, W)`, the same shape convention as an image batch.
- **Validation is descriptive only.** `DatasetValidator` computes per-field statistics and
  flags missing values and z-score outliers. Distribution drift detection and bias/fairness
  metrics are out of scope for now.

## Recipes

- [CSV + DataLoader training](../recipes/data-pipeline/csv_dataloader_training.md)

See also [`examples/mnist_dataloader_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/mnist_dataloader_demo.cpp)
(CMake target `mnist_dataloader_demo`). It trains `MnistConvNet` on MNIST, loading the data
through `DataLoader` and `MnistDatasetAdapter`.
