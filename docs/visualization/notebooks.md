# Notebooks and Reports

Show pulsatrix results in Jupyter notebooks, and write reports as `.ipynb` notebooks or HTML
pages directly from C++, with no Python and no kernel.

- **[Reports](#reports)** (`viz/report.hpp`): markdown, code and
  figures written to a notebook that GitHub, VS Code, JupyterLab and Quarto display as it is, or
  to one HTML page. Start here; the
  [notebook report recipe](../recipes/visualization/notebook_report.md) builds one end to end.
- **[Rich display](#rich-display)** (`viz/mime_bundle.hpp`): the formats each
  pulsatrix type is shown in, which reports use and a C++ notebook kernel can display.

## Reports

`Report` (`viz/report.hpp`) writes results as a Jupyter notebook (an `.ipynb` file, nbformat 4.5)
or as one HTML page, from C++, with no Python and no kernel:

```cpp
#include "pulsatrix/viz/report.hpp"

using namespace pulsatrix;

Attribution a = LRP::epsilon_plus().explain(ctx, x, target, &backend);  // compute as usual

Report report("LRP on ResNet18");
report.markdown("We explain the top class with **EpsilonPlus**.")
      .code("Attribution a = LRP::epsilon_plus().explain(ctx, x, target, &backend);")  // shown, not run
      .show(a);           // anything with a mime_bundle_repr(): here, the heatmap
report.text("top class: 24 (great grey owl)");
report.save_ipynb("resnet18.ipynb");
report.save_html("resnet18.html");
```

- **Cells.** The title becomes a heading at the top. `markdown()` adds a markdown cell and
  `code()` a code cell. The code is only shown: you compute the values yourself, as above.
  `show()` and `output()` add rich outputs (see [Rich display](#rich-display) for each type), and `text()` adds plain text.
  Outputs attach to the code cell before them; one with no code cell before it gets its own cell,
  with the empty input hidden.
- **Opening the notebook.** It keeps its outputs, so it shows as it is on GitHub, in VS Code,
  JupyterLab and Quarto. Its kernel is set to xeus-cpp's C++17 kernel; JupyterLab without that
  kernel asks for one, and choosing "No Kernel" still shows every output. The same report always
  writes the same bytes, and CI checks a sample one with `nbformat.validate`.
- **Opening the HTML page.** Any browser. Each output shows in its richest format: charts, then
  SVG, PNG, HTML and text. Charts need the Vega libraries, which the page loads from a CDN by
  default (small, but needs a network connection) or contains itself (about 830 KB more, works
  offline and as an email attachment):

  ```cpp
  HtmlOptions options;
  options.scripts = HtmlScripts::Inline;
  options.script_dir = "vega";  // tools/render/fetch_vega.sh downloads the three files
  report.save_html("resnet18.html", options);
  ```

  `Report` uses only these two options. A report without charts has no scripts at all.
- **Markdown in the HTML page** covers headings, paragraphs, lists, fenced code, pipe tables, bold,
  italic, code spans and links. Raw HTML in the markdown is escaped, and links go only to http,
  https, mailto or relative URLs, so text from elsewhere can't add scripts to the page. The
  notebook keeps the markdown as written, for Jupyter to render in full.

## Rich display

A *MIME bundle* is one output in several formats (an interactive chart, an SVG image, plain
text), so that each front end shows the best one it supports. `mime_bundle_repr()`
(`viz/mime_bundle.hpp`) makes one for each pulsatrix type below. Reports (above) are
built from them, and a C++ notebook kernel can display them.

```cpp
#include "pulsatrix/viz/mime_bundle.hpp"

using namespace pulsatrix;

MimeBundle b = mime_bundle_repr(attribution);
// b.entries: {"application/vnd.vegalite.v5+json", spec}, {"image/svg+xml", svg}, {"text/plain", text}
```

| Value | Shown as |
|---|---|
| `Tensor` | A summary as text and an HTML table: shape, device, min, max, mean, standard deviation, non-finite count and the leading values |
| `Attribution` | Rank 3 or more: a heatmap of the first example, channels summed. Up to 64 × 64 cells it is Vega-Lite plus SVG; larger maps are a PNG, one pixel per cell. Rank 1 or 2: a bar chart of the first example's largest features |
| `CircuitGraph`, `CircuitGraphDocument` | A Vega-Lite node-link view: columns by depth, nodes colored by ablation effect, edges as thick as their weight |
| Documents with a chart (attribution, heatmap, partial dependence, sensitivity, counterfactual, Morris, Sobol) | Vega-Lite and the SVG figure |
| `TrainingLogDocument` | Vega-Lite |
| `TokenRelevanceDocument`, the protein views | The SVG figure |

- **Attribution shapes.** The first dimension is the batch: rank 3 is read as (N, H, W) and rank
  4 as (N, C, H, W). Give a single image's attribution a batch dimension of 1 before displaying
  it, or a (3, H, W) map is drawn as three examples.
- **Formats.** Every bundle ends in `text/plain`, and its richest format comes first. The charts
  are Vega-Lite specifications sent as version 5 (`application/vnd.vegalite.v5+json`), which
  JupyterLab, Notebook 7 and VS Code draw without anything installed. They are checked against
  Vega-Lite 5's schema. The SVG is the fallback where Vega doesn't run, such as GitHub's notebook
  preview. Circuit graphs and training logs have no SVG, so GitHub shows only their text. The
  HTML uses inline styles only, so nothing leaks into the rest of the notebook.
- **Other front ends.** `MimeBundle::to_json_value()` returns the bundle as one JSON object,
  `{"text/plain": ..., ...}`.

!!! warning "C++ notebooks are experimental"
    [xeus-cpp](https://github.com/compiler-research/xeus-cpp) is a Jupyter kernel that runs C++
    cells through Clang's interpreter (`conda install -c conda-forge xeus-cpp`). Its
    `xcpp::display(x)` finds pulsatrix's `mime_bundle_repr` automatically, and pulsatrix doesn't
    depend on xeus. The tests check that lookup with a copy of xeus-cpp 0.10's display code, but
    pulsatrix hasn't been run inside a live kernel yet: loading the library into one is a planned
    roadmap item. Until then, write notebooks with `Report` below.

    ```cpp
    // in a xeus-cpp cell, once the library is loaded
    #include "pulsatrix/viz/mime_bundle.hpp"
    xcpp::display(lrp.explain(ctx, x, target, &backend));  // a heatmap
    ```
