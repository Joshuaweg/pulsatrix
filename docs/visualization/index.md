# Visualization

Use this section to see your model's explanations and training progress as figures instead of
numbers: standalone SVG files, interactive HTML pages, Jupyter notebooks and reports, and native
desktop windows. Everything except the desktop windows is part of `pulsatrix_core` and needs no
GUI, display or extra dependency.

The explainers, LRP rules and interpretability tools elsewhere in pulsatrix return plain data
(`Attribution`, `CircuitGraph`, `ActivationSnapshot`) and never draw anything. This section turns
that data into figures. Only the desktop windows add dependencies (GLFW, Dear ImGui, ImPlot), in
the separate `pulsatrix_viz` library.

## Which output do you want?

| You want | Use | Page |
|---|---|---|
| A figure for a paper or a slide | SVG files (`RenderBarChartSvg`, `RenderHeatmapSvg`, ...) or the `pulsatrix_svg` tool | [SVG Figures](svg.md) |
| A page to explore, with hover, zoom and export | Interactive HTML pages (Vega-Lite) | [Interactive HTML](html.md) |
| A notebook or a shareable report with text, code and figures | `Report`, written as `.ipynb` or HTML | [Notebooks and Reports](notebooks.md) |
| Text colored by relevance, from a language model | Token and word relevance documents, `pulsatrix_explain_text` | [JSON Documents](documents.md#token-relevance-for-text) |
| A language model's attribution graph in Neuronpedia or circuit-tracer | `ToNeuronpediaJson` | [Attribution Graph Viewers](attribution-graphs.md) |
| Live training curves or an interactive explanation dashboard | Dear ImGui windows (`-DPULSATRIX_ENABLE_VIZ=ON`) | [Desktop Windows](desktop.md) |
| To save an explanation and draw it later, or elsewhere | Versioned JSON documents | [JSON Documents](documents.md) |

Every option draws from the same versioned [JSON documents](documents.md), so one explanation can
be saved once and shown in any of these forms.

## Data transforms

The data transforms (`viz/plot_data.hpp`, `viz/colormap.hpp`) turn explanations into what a
figure plots. They are plain, unit-tested functions in `pulsatrix_core`, with no ImGui or ImPlot
code:

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

The desktop chart widgets are thin ImGui/ImPlot drawing calls over these transforms; see
[Desktop Windows](desktop.md).

Full API reference: [Doxygen: Visualization](../api/group__visualization.html)

## Recipes

- [A notebook report from C++](../recipes/visualization/notebook_report.md)
- [Training dashboard](../recipes/visualization/training_dashboard.md)
