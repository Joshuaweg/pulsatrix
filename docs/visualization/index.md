# Visualization

Use this section to see your model's explanations and training progress as charts instead of
numbers. It provides native C++ charts, dashboards, and live views built on Dear ImGui and
ImPlot (immediate-mode GUI and plotting libraries).

The explainers, LRP rules, and interpretability tools elsewhere in pulsatrix return plain data
(`Attribution`, `CircuitGraph`, `ActivationSnapshot`) and never draw anything. This layer is
optional and sits on top of that data. `pulsatrix_core` never depends on it, the same way core
does not depend on the Python bindings (`pulsatrix_py`).

## What's inside

**Data transforms** (`viz/plot_data.hpp`, `viz/colormap.hpp`). These are plain functions with
no ImGui or ImPlot code. They are always compiled into `pulsatrix_core` and are unit-tested:

- `ToFeatureImportanceBars` / `ToWaterfallSteps` / `ToWaterfallBars`: turn an `Attribution`
  into sorted bars, or into a waterfall from the baseline to the prediction.
- `CircuitNodeDisplayLabel`: a readable label for a `CircuitGraph` node.
- `ToSaliencyHeatmap` / `ComputeHeatmapColorScale`: turn an `Attribution` into a 2D grid and
  choose its color scale. The scale is symmetric around zero when any value is negative, and
  `[0, max]` otherwise.
- `ToBeeswarmPoints`: spread one feature's values across many `Attribution`s into
  non-overlapping points. The jitter is deterministic.
- `ToFieldHistogramBins`: bin a `Dataset` field's values into an equal-width histogram.
- `ToRgbImageBuffer`: convert an image `Tensor` into bytes ready for a GPU texture.
- `ViridisColormap` / `DivergingColormap`: colorblind-safe colormaps for unsigned and signed
  values.

**Metrics sink** (`viz/implot_metrics_sink.hpp`). `ImPlotMetricsSink` is a `MetricsSink` that
stores every logged scalar and histogram as a per-tag series. Its logging methods
(`log_scalar()`, `log_histogram()`) are compiled into `pulsatrix_core` and need no flag. Only
`Draw()` needs `pulsatrix_viz`. To write your own sink, see
[Adding a new metrics sink](../customization/index.md#adding-a-new-metrics-sink).

**Chart widgets** (only with `PULSATRIX_ENABLE_VIZ=ON`). Each one is a thin ImGui/ImPlot
drawing call over the data transforms:

| Widget | Shows |
|---|---|
| `AttributionBarChart` | Top features of one attribution, as bars |
| `AttributionWaterfallChart` | How features move the output from the baseline to the prediction |
| `AttributionBeeswarmView` | One feature's attribution across many inputs |
| `SaliencyHeatmapView` | An attribution laid over an image |
| `CircuitGraphView` | A `CircuitGraph`'s per-node ablation effect |
| `ConfidenceMeter` | A confidence value as a labeled bar |
| `ExplanationScoreCard` | One panel: prediction, top features, confidence, and trust checks (`ComputeConservation`, `ComputeAttributionStability`) |
| `DatasetStatisticsView` | Per-field histograms of a dataset |
| `ImageGridView` | A grid of image thumbnails with captions |
| `TrainingDashboard`, `ImPlotMetricsSink::Draw()` | Live training curves |

**JSON documents** (`viz/document.hpp`, in `pulsatrix_core`). Every view's data can be saved as
a versioned JSON file and read back, so a figure can be redrawn later, by another program or by
a renderer outside pulsatrix. See [JSON documents](#json-documents).

**SVG figures** (`viz/svg.hpp`, in `pulsatrix_core`). Bar, waterfall, heatmap, token strip and
beeswarm charts as standalone SVG files, with no GPU or display. See [SVG figures](#svg-figures).

**Infrastructure.** `VizWindow` (`viz/window.hpp`) owns the window, the OpenGL context, and the
ImGui/ImPlot setup. You write a per-frame draw callback and call `window.run(...)`.
`TextureCache` uploads images to the GPU once and reuses them across frames.

Full API reference: [Doxygen: Visualization](../api/group__visualization.html)

## JSON documents

Interactive viewers go stale faster than file formats do, so every view reads from a versioned
JSON document. The ImGui widgets, the SVG renderer (VIZ-2) and any tool outside pulsatrix all
read the same data. The documents are part of `pulsatrix_core` and need no flag.

```cpp
#include <fstream>
#include <sstream>

#include "pulsatrix/viz/attribution_bar_chart.hpp"  // the widget needs PULSATRIX_ENABLE_VIZ
#include "pulsatrix/viz/document.hpp"

// Save an explanation...
AttributionDocument doc = ToAttributionDocument(attribution);
std::ofstream("explanation.json") << ToJson(doc);

// ...and draw it later, possibly in another program.
std::stringstream text;
text << std::ifstream("explanation.json").rdbuf();
if (VizDocumentKind(text.str()) == "attribution") {
    AttributionBarChart::Draw("Top features", ParseAttributionDocument(text.str()));
}
```

There are six kinds. Each has a struct, a writer (`ToJson`) and a reader (`Parse<Kind>Document`):

| Schema | Struct | Holds | Converts from and to |
|---|---|---|---|
| `pulsatrix.attribution.v1` | `AttributionDocument` | `method`, `shape`, row-major `values`, `metadata` | `Attribution` |
| `pulsatrix.heatmap.v1` | `HeatmapDocument` | `title`, `rows`, `cols`, `values`, optional `row_labels` and `col_labels` | `HeatmapGrid` |
| `pulsatrix.token_relevance.v1` | `TokenRelevanceDocument` | `method`, `tokens`, one `relevance` per token, `target` | — |
| `pulsatrix.circuit_graph.v1` | `CircuitGraphDocument` | `nodes` (`id`, `op_type`, `label`, `ablation_effect`) and `edges` (`from`, `to`, `weight`) | `CircuitGraph` |
| `pulsatrix.training_log.v1` | `TrainingLogDocument` | `scalars` (`tag`, `steps`, `values`) and the latest `histograms` per tag | `ImPlotMetricsSink`; `ReplayTrainingLog` logs a saved run to any `MetricsSink` |
| `pulsatrix.feature_dashboard.v1` | `FeatureDashboardDocument` | One feature's `source`, `feature_index`, `activation_density`, `max_activation`, activation histogram and `top_examples` | — |

The widgets read documents directly: `AttributionBarChart`, `AttributionWaterfallChart`,
`SaliencyHeatmapView` and `CircuitGraphView` each have a `Draw` overload for their document. For
the training dashboard, replay the log into an `ImPlotMetricsSink` once and draw that.

An attribution document looks like this:

```json
{
  "schema": "pulsatrix.attribution.v1",
  "method": "integrated_gradients",
  "shape": [2, 3],
  "values": [0.5, -1.25, 0, 3, 0.1, -0.001],
  "metadata": {
    "baseline": "zero",
    "steps": "50"
  }
}
```

Examples of every kind are in
[`tests/fixtures/viz/`](https://github.com/Joshuaweg/pulsatrix/tree/master/tests/fixtures/viz).
Tests check that the writer reproduces each one byte for byte.

### Format rules

- **Schema and version.** Every document is an object whose `"schema"` is
  `pulsatrix.<kind>.v<N>`. `VizDocumentKind()` returns the kind, so a viewer can pick the
  reader. A reader accepts only its own major version.
- **Compatibility within a version.** Version 1 only ever gains fields, and readers ignore
  fields they don't know, so an older reader can read a newer v1 file. Removing, renaming or
  changing the meaning of a field means v2.
- **NaN and infinity.** JSON has no literal for them. A non-finite number is written as `null`,
  and the top-level `"nonfinite"` object maps its
  [JSON Pointer](https://www.rfc-editor.org/rfc/rfc6901) to `"nan"`, `"inf"` or `"-inf"`:

    ```json
    "values": [1, null],
    "nonfinite": {
      "/values/1": "nan"
    }
    ```

    The member appears only when the document has a non-finite number. A reader rejects a
    `null` number with no entry, and an entry that doesn't point at a `null` number, so no value
    is ever silently lost.
- **Numbers round-trip exactly.** Each number is written with the shortest text that reads back
  as the same `float` (or `double`, for training-log values), formatted the way JavaScript
  prints numbers. Reading the file gives back the same bits.
- **Deterministic output.** The same document always gives the same bytes on every platform:
  fixed member order, metadata sorted by key, two-space indent, and number arrays on one line.
- **Strict reading.** The parser follows RFC 8259 strictly. It rejects comments, trailing
  commas, duplicate keys, invalid UTF-8 and nesting deeper than 256 levels. Every reader error
  names the JSON Pointer of the problem, for example
  `pulsatrix.heatmap.v1 document: /row_labels: needs one label per row, or none`.
- **Text is UTF-8.** Token strings must be valid UTF-8, so decode byte-level BPE tokens before
  writing them.

The parser is fuzzed (`tools/fuzz/viz_document_fuzz.cpp`). `pulsatrix/json.hpp` exposes the
underlying `JsonValue`, `ParseJson` and `WriteJson` for other uses.

## SVG figures

For papers, reports and CI, `viz/svg.hpp` draws documents as standalone SVG files. It needs no
GPU, display, font library or ImGui, and it is part of `pulsatrix_core`. Colors come from
`colormap.hpp`, the same as the ImGui widgets: blue-white-red for signed values and Viridis for
magnitudes.

| Function | Draws | From |
|---|---|---|
| `RenderBarChartSvg(doc, top_k)` | The top features by \|attribution\|, on an axis through zero | `AttributionDocument` |
| `RenderWaterfallSvg(doc, baseline)` | Each feature moving the output from the baseline to the prediction | `AttributionDocument` |
| `RenderHeatmapSvg(doc)` | The grid, with a color bar and any row and column labels | `HeatmapDocument` |
| `RenderTokenStripSvg(doc)` | The text's tokens, wrapped, each on a background colored by its relevance | `TokenRelevanceDocument` |
| `RenderBeeswarmSvg(docs, features)` | One row per feature, one point per input | several `AttributionDocument`s |

```cpp
#include <fstream>

#include "pulsatrix/viz/svg.hpp"

SvgOptions options;
options.title = "Why the model said yes";
std::ofstream("bars.svg") << RenderBarChartSvg(ToAttributionDocument(attribution), 10, options);
```

![A bar chart of six features' attributions](figures/bar_chart.svg)

![A waterfall from a baseline of 0.25 to a prediction of 0.188](figures/waterfall.svg)

`SvgOptions` sets the width (default 640 pixels), a title and the font size. The height follows
from the content. Every figure has a title element for screen readers, and every bar, cell and
token has a tooltip with its exact value.

**From the command line.** The `pulsatrix_svg` tool, built and installed with the library,
turns documents into figures:

```bash
pulsatrix_svg explanation.json -o bars.svg                 # attribution: bar chart
pulsatrix_svg explanation.json --waterfall 0.12 -o wf.svg  # attribution: waterfall
pulsatrix_svg run*.json --beeswarm 0,3,7 -o swarm.svg      # many attributions: beeswarm
pulsatrix_svg saliency.json -o map.svg                     # heatmap
pulsatrix_svg tokens.json -o text.svg                      # token relevance
```

Other options: `--top-k N`, `--width W`, `--font-size N`, `--title TEXT`.

**Details:**

- **Deterministic.** The same input gives the same bytes on every platform. Tests compare each
  chart with a stored file.
- **Large heatmaps.** Grids up to 4,096 cells are one vector rectangle per cell. Larger grids,
  such as a 224 x 224 saliency map, are embedded as a lossless PNG with sharp pixel edges, which
  keeps the file small (about 45 KB for 224 x 224).
- **Non-finite values.** A heatmap cell or token whose value is NaN or infinite is drawn gray and
  left out of the color scale. The bar, waterfall and beeswarm charts refuse non-finite values,
  because there's no honest bar length for them.
- **Text widths are estimated** at 0.6 em per character, since there is no font library. Token
  strips use a monospace font, so their widths are exact. Long labels are shortened with an
  ellipsis.

## Building with visualization enabled

### Prerequisites

The default build has no GUI dependencies. `-DPULSATRIX_ENABLE_VIZ=ON` adds these:

- **Network access at configure time.** CMake downloads GLFW 3.4, Dear ImGui v1.90.9, and
  ImPlot v0.16 with `FetchContent`.
- **OpenGL development files.** CMake runs `find_package(OpenGL REQUIRED)`. On Debian/Ubuntu,
  install `libgl1-mesa-dev`.
- **GLFW's build dependencies on Linux.** GLFW 3.4 builds X11 and Wayland support by default.
  On Debian/Ubuntu, install `libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev`
  for X11, and `libwayland-dev libxkbcommon-dev wayland-protocols` for Wayland. To skip
  Wayland, add `-DGLFW_BUILD_WAYLAND=OFF`.
- **A display and a GPU driver** to run anything that links `pulsatrix_viz`. That is why
  the option defaults to `OFF` and CI builds without it.

### Build

```bash
cmake -S . -B build -DPULSATRIX_ENABLE_VIZ=ON
cmake --build build --target training_dashboard_demo --config Release
```

Link your own executable against `pulsatrix_viz`. It brings in `pulsatrix_core`, `imgui`, and
`implot`, including their include directories:

```cmake
add_executable(my_viz_app my_viz_app.cpp)
target_link_libraries(my_viz_app PRIVATE pulsatrix_viz)
```

The data transforms and `ImPlotMetricsSink`'s logging methods need no flag. They are always
part of `pulsatrix_core`.

### Demo apps

| Target | What it shows |
|---|---|
| `training_dashboard_demo` | Live loss curves while an XOR network trains |
| `explanation_dashboard_demo` | Bar chart, waterfall, saliency heatmap, circuit graph, and score card for a small XOR network |
| `dataset_preview_demo` | `DatasetStatisticsView` and `ImageGridView` on small synthetic datasets |
| `live_inference_demo` | An `ExplanationScoreCard` that updates as it cycles through XOR inputs |
| `mnist_viz_gallery` | Every widget on MNIST digits (see [below](#mnist-gallery)) |

## How to implement

### A live training dashboard

```cpp
#include <imgui.h>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/training_dashboard.hpp"
#include "pulsatrix/viz/window.hpp"
#include "pulsatrix/xor_training_example.hpp"

using namespace pulsatrix;

CPUBackend backend;
XorNetwork net(&backend);
AdamOptimizer optimizer(0.05f, &backend);
ImPlotMetricsSink sink;  // a MetricsSink that buffers logged values for plotting

// The four XOR examples, as (1, 2) inputs and (1, 1) targets.
const float xs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
const float ys[4] = {0, 1, 1, 0};
int step = 0;

VizWindow window("Training Dashboard", 1000, 700);
window.run([&]() {
    for (int i = 0; i < 4 && step < 4000; ++i, ++step) {   // a few training steps per frame
        Tensor input(Shape({1, 2}), &backend, {xs[step % 4][0], xs[step % 4][1]});
        Tensor target(Shape({1, 1}), &backend, {ys[step % 4]});
        net.train_step(input, target, optimizer, sink, step);  // logs the loss to sink
    }
    ImGui::Begin("Training Dashboard");
    TrainingDashboard::Draw(sink);
    ImGui::End();
});
```

Full source: [`examples/viz/training_dashboard_demo.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/viz/training_dashboard_demo.cpp).

**What's happening:** `XorNetwork::train_step()` and `MnistConvNet::train_step()` take a
`MetricsSink&`. Passing an `ImPlotMetricsSink` adds plotting with no change to the training
code. Each frame, `TrainingDashboard::Draw()` reads the sink's per-tag series. It draws a
one-line summary per tag (last, min, max, current step) and a small line chart for each tag.

Logging and drawing are separate methods. The logging half works, and is unit-tested, even
with `PULSATRIX_ENABLE_VIZ=OFF`. Only `Draw()` needs the GUI stack.

Recipe: [Training dashboard](../recipes/visualization/training_dashboard.md).

## MNIST gallery

`mnist_viz_gallery` runs every widget on this page against MNIST test digits:

```bash
python3 tools/fetch_mnist.py              # once: data/MNIST/raw/ (gitignored)
cmake -S . -B build-viz -DPULSATRIX_ENABLE_VIZ=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-viz --target mnist_viz_gallery -j
./build-viz/mnist_viz_gallery                          # interactive
./build-viz/mnist_viz_gallery --screenshot out/        # one PNG per page, then exit
```

Run it from the repository root, or pass `--data DIR` (default `data/MNIST/raw`).

| Option | Meaning | Default |
|---|---|---|
| `--epochs N` | Training epochs | 1 |
| `--train N` | Training digits (about 25 s per epoch on one CPU core at the default) | 60000 |
| `--images N` | Test digits explained with every method. If none is misclassified, the first misclassified test digit is added. | 4 |
| `--eval N` | Held-out digits for the accuracy curve | 2000 |
| `--beeswarm N` | Digits behind each beeswarm | 200 |
| `--screenshot DIR` | Save one PNG per page to `DIR`, then exit | off |

| Page | Widgets | What it shows |
|---|---|---|
| Training | `ImPlotMetricsSink`, `TrainingDashboard` | `MnistConvNet` (Conv2D(1,8,5x5) -> ReLU -> Flatten -> Linear) training live: 100-step mean loss, held-out accuracy, final conv-kernel and classifier-weight histograms |
| Dataset | `DatasetStatisticsView`, `ImageGridView`, `TextureCache` | Pixel and label histograms of the first 1000 test digits; 32 test digits captioned with true and predicted label |
| Prediction | `ConfidenceMeter` | 10-class softmax of the selected digit |
| Heatmaps | `SaliencyHeatmapView` | Saliency, IG, Grad-CAM, LRP epsilon / EpsilonPlus / EpsilonAlpha2Beta1 / EpsilonGammaBox(0,1), LIME, KernelSHAP side by side |
| Bar + waterfall | `AttributionBarChart`, `AttributionWaterfallChart` | Top-12 features of the selected method, starting from its reference value (black-image logit for IG/SHAP, 0 otherwise) |
| LRP score cards | `ExplanationScoreCard` | One card per LRP rule set, on a shared scale: confidence, top relevance, conservation delta, stability over repeated runs |
| Beeswarm | `AttributionBeeswarmView` | The 4 pixels with the highest mean \|attribution\| across 200 test digits, per gradient/LRP method |
| Circuit graph | `CircuitGraphView` | `ExplainerContext::build_circuit_graph` on the CNN: per-layer zero-ablation effect |

For what the LRP rule names mean, see [LRP](../interpretability/lrp.md).

### Explainer settings

| Method | Setting |
|---|---|
| Integrated Gradients | 128 steps from a black baseline |
| Grad-CAM | The 24x24 map is zero-padded by 2 px onto the 28x28 grid, so each cell sits at the center of its 5x5 receptive field |
| KernelSHAP | 16 superpixels (a 4x4 grid of 7x7 patches); all 2^16 coalitions enumerated exactly; an absent patch is black |
| LIME | The same 16 superpixels; 2000 samples; sigma 0.5 |

!!! note "Why LIME uses superpixels"
    This library's LIME sets its locality kernel width equal to the perturbation sigma. On all
    784 pixels, every sample weight underflows to zero, so pixel-level LIME is not meaningful
    here.

The gallery prints the test accuracy to stdout. For each explainer it also prints the
attribution sum, the max |attribution|, and the IG-completeness, SHAP-efficiency, or
LRP-conservation check.

Signed attributions use the blue-white-red diverging map, centered on zero. Unsigned ones
(Grad-CAM) use Viridis. `SaliencyHeatmapView` picks the map via `ComputeHeatmapColorScale`.

### Screenshots

Captured with `--screenshot` on a model trained for one epoch (96.78% test accuracy), test
digit #0 (a 7):

| | |
|---|---|
| ![Training dashboard](../assets/visualization/mnist_gallery/01_training.png) | ![Dataset statistics and image grid](../assets/visualization/mnist_gallery/02_dataset.png) |
| Training: live loss, held-out accuracy, weight histograms | Dataset: pixel/label histograms, captioned test digits (X = misclassified) |
| ![Heatmaps for every explainer](../assets/visualization/mnist_gallery/04_heatmaps.png) | ![LRP epsilon bar and waterfall charts](../assets/visualization/mnist_gallery/11_charts_lrp_epsilon.png) |
| Heatmaps: all nine explainers side by side | LRP epsilon: heatmap, top-12 bars, waterfall to the 2.436 logit |
| ![LRP score cards](../assets/visualization/mnist_gallery/17_lrp_score_cards.png) | ![Beeswarm for LRP epsilon](../assets/visualization/mnist_gallery/21_beeswarm_lrp_epsilon.png) |
| Score cards: one per LRP rule set, conservation and stability | Beeswarm: top-4 pixels across 200 test digits |
| ![Circuit graph](../assets/visualization/mnist_gallery/25_circuit.png) | |
| Circuit graph: per-layer zero-ablation effect (the output node is the reference, not ablated) | |

## Recipes

- [Training dashboard](../recipes/visualization/training_dashboard.md)
