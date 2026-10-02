# Visualization

Every explainer, LRP rule, and mechanistic-interpretability tool elsewhere in pulsatrix
(`Attribution`, `CircuitGraph`, `ActivationSnapshot`) deliberately ships raw data only — no
plotting, no rendering, no dependency on a GUI stack. Visualization is a separate, opt-in
layer on top of that data: a native C++ module built on Dear ImGui + ImPlot that turns
`Attribution`/`CircuitGraph`/training-metric data into charts, dashboards, and live views.
It never becomes a dependency of `pulsatrix_core` — the relationship is one-directional, the
same plugin-boundary discipline as the Python bindings (`pulsatrix_py`).

## What's inside

**Data transforms** (`plot_data.hpp`, `colormap.hpp`) — pure functions with no ImGui/ImPlot
include, always built as part of `pulsatrix_core`, unit-tested directly:

- `ToFeatureImportanceBars` / `ToWaterfallSteps` / `ToWaterfallBars` — an `Attribution`'s
  values into sorted bars, or a baseline-to-prediction cascade and its floating bars
- `CircuitNodeDisplayLabel` — a readable label for a `CircuitGraph` node
- `ToSaliencyHeatmap` / `ComputeHeatmapColorScale` — an `Attribution`'s values into a 2D grid,
  and the colour scale it calls for (symmetric diverging range when any value is negative,
  `[0, max]` sequential otherwise)
- `ToBeeswarmPoints` — many `Attribution`s' values for one feature into deterministic,
  collision-avoiding jittered points
- `ToFieldHistogramBins` — a `Dataset` field's values into equal-width histogram bins
- `ToRgbImageBuffer` — a decoded image `Tensor` into a texture-upload-ready byte buffer
- `ViridisColormap` / `DivergingColormap` — perceptually uniform, colorblind-safe colormaps
  for unsigned magnitude and signed (directional) values respectively
- `ImPlotMetricsSink` — a concrete `MetricsSink` (see [Customization](../customization/index.md))
  that buffers every logged scalar/histogram into per-tag series; `log_scalar()`/
  `log_histogram()` are unconditional and unit-tested even without `PULSATRIX_ENABLE_VIZ`

**Chart widgets** (built only under `PULSATRIX_ENABLE_VIZ`, each a thin ImGui/ImPlot draw
call over the data-transform layer above): `AttributionBarChart`, `AttributionWaterfallChart`,
`AttributionBeeswarmView`, `SaliencyHeatmapView`, `CircuitGraphView`, `ConfidenceMeter`,
`ExplanationScoreCard` (a combined "what/why/how confident/how trustworthy" panel, pulling in
`ComputeConservation`/`ComputeAttributionStability`), `DatasetStatisticsView`, `ImageGridView`,
`TextureCache`, and `ImPlotMetricsSink::Draw()`/`TrainingDashboard` for live training curves.

`VizWindow` (`window.hpp`) owns the GLFW window, OpenGL3 context, and ImGui/ImPlot lifecycle
every demo app runs inside — write a per-frame draw callback and call `window.run(...)`.

Full API reference: [Doxygen: Visualization](../api/group__visualization.html)

## Building with visualization enabled

The chart widgets and `VizWindow` are gated behind the `PULSATRIX_ENABLE_VIZ` CMake option
(default `OFF`) — a display and GPU driver are required to run anything that links against
`pulsatrix_viz`, so it's opt-in rather than part of the default headless build CI runs:

```bash
cmake -S . -B build -DPULSATRIX_ENABLE_VIZ=ON
cmake --build build --target training_dashboard_demo --config Release
```

Link against `pulsatrix_viz` (which itself links `pulsatrix_core`, `imgui`, and `implot`) for
any executable that calls into ImGui/ImPlot directly:

```cmake
add_executable(my_viz_app my_viz_app.cpp)
target_link_libraries(my_viz_app PRIVATE pulsatrix_viz)
```

The data-transform layer (`plot_data.hpp`/`colormap.hpp`/`ImPlotMetricsSink`'s logging methods)
needs no flag — it's always available from `pulsatrix_core`.

## How to implement

### A live training dashboard

```cpp
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/training_dashboard.hpp"
#include "pulsatrix/viz/window.hpp"
#include "pulsatrix/xor_training_example.hpp"

using namespace pulsatrix;

CPUBackend backend;
XorNetwork net(&backend);
AdamOptimizer optimizer(0.05f, &backend);
ImPlotMetricsSink sink;  // MetricsSink implementation -- buffers scalars/histograms for plotting

VizWindow window("Training Dashboard", 1000, 700);
window.run([&]() {
    for (int i = 0; i < 4; ++i) {
        // net.train_step(input, target, optimizer, sink, step) logs loss via sink each call
    }
    ImGui::Begin("Training Dashboard");
    TrainingDashboard::Draw(sink);  // per-tag summary + ImPlotMetricsSink's line charts
    ImGui::End();
});
```

**What's happening:** `ImPlotMetricsSink` is a drop-in `MetricsSink` — every training loop in
this codebase already threads a `MetricsSink&` through `train_step()`, so plugging in a
plotting sink needs zero changes to the training code itself. `TrainingDashboard::Draw()` reads
the sink's buffered per-tag series and draws a data-ink-minimal summary line (last/min/max,
current step) followed by one small-multiple line chart per tag. Because logging
(`log_scalar()`/`log_histogram()`) and drawing (`Draw()`) are split into separate methods, the
logging half works and is unit-tested even in a build with `PULSATRIX_ENABLE_VIZ=OFF` — only
the actual `Draw()` call needs the GUI stack.

Recipe: [Training dashboard](../recipes/visualization/training_dashboard.md).

## MNIST gallery

`mnist_viz_gallery` runs every widget in the pack against real MNIST test digits, end to end:

```bash
python3 tools/fetch_mnist.py              # once: data/MNIST/raw/ (gitignored)
cmake -S . -B build-viz -DPULSATRIX_ENABLE_VIZ=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-viz --target mnist_viz_gallery -j
./build-viz/mnist_viz_gallery                          # interactive
./build-viz/mnist_viz_gallery --screenshot out/        # one PNG per page, then exit
```

Run it from the repository root (or pass `--data DIR`). Options: `--epochs N` (default 1),
`--train N` training digits (default 60000, ~25 s per epoch on one CPU core), `--images N`
test digits explained with every method (default 4; the first misclassified test digit is
added when none of those is wrong), `--eval N` held-out digits for the accuracy curve,
`--beeswarm N` digits behind each beeswarm (default 200).

| Page | Widgets | What it shows |
|---|---|---|
| Training | `ImPlotMetricsSink`, `TrainingDashboard` | `MnistConvNet` (Conv2D(1,8,5x5) -> ReLU -> Flatten -> Linear) trained live: 100-step mean loss, held-out accuracy, final conv-kernel and classifier-weight histograms |
| Dataset | `DatasetStatisticsView`, `ImageGridView`, `TextureCache` | pixel/label histograms of the first 1000 test digits; 32 test digits captioned with true/predicted label |
| Prediction | `ConfidenceMeter` | 10-class softmax of the selected digit |
| Heatmaps | `SaliencyHeatmapView` | Saliency, IG, Grad-CAM, LRP epsilon / EpsilonPlus / EpsilonAlpha2Beta1 / EpsilonGammaBox(0,1), LIME, KernelSHAP side by side |
| Bar + waterfall | `AttributionBarChart`, `AttributionWaterfallChart` | top-12 features of the selected method, cascading from its reference value (black-image logit for IG/SHAP, 0 otherwise) |
| LRP score cards | `ExplanationScoreCard` | one card per LRP rule set, shared scale: confidence, top relevance, conservation delta, stability over repeated runs |
| Beeswarm | `AttributionBeeswarmView` | the 4 highest-mean-\|attribution\| pixels across 200 test digits, per gradient/LRP method |
| Circuit graph | `CircuitGraphView` | `ExplainerContext::build_circuit_graph` on the CNN: per-layer zero-ablation effect |

Explainer budgets: IG uses 128 steps from a black baseline. Grad-CAM's 24x24 map is
zero-padded 2 px onto the 28x28 grid (each cell sits on its 5x5 receptive-field centre).
LIME and KernelSHAP run on 16 superpixels (a 4x4 grid of 7x7 patches): KernelSHAP enumerates
all 2^16 coalitions exactly (absent patch = black), LIME uses 2000 samples with sigma 0.5.
Pixel-level LIME is not meaningful with this library's LIME: its locality kernel width equals
the perturbation sigma, so on 784 inputs every sample weight underflows to zero. The gallery
prints test accuracy and, per explainer, the attribution sum, max |attribution|, and the
IG-completeness / SHAP-efficiency / LRP-conservation checks to stdout.

Signed attributions are drawn with the blue-white-red diverging map centred on zero, unsigned
ones (Grad-CAM) with Viridis -- `SaliencyHeatmapView` picks the map via
`ComputeHeatmapColorScale`.

## Recipes

- [Training dashboard](../recipes/visualization/training_dashboard.md)
