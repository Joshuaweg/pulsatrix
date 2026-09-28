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

- `ToFeatureImportanceBars` / `ToWaterfallSteps` — an `Attribution`'s values into sorted bars
  or a baseline-to-prediction cascade
- `ToSaliencyHeatmap` — an `Attribution`'s values into a 2D grid
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

## Recipes

- [Training dashboard](../recipes/visualization/training_dashboard.md)
