# Interactive HTML

## Interactive HTML

`viz/html.hpp` writes the same views as self-contained HTML pages, drawn in the browser by
[Vega-Lite](https://vega.github.io/vega-lite/). Hover any bar, point or cell to see its exact
value. Zoom continuous charts with the mouse wheel and pan by dragging. Save a PNG or SVG from
the `…` menu. Pages open in any browser and display in a Jupyter cell (`IPython.display.HTML`).
There are two views the SVG doesn't have: the training log, with one zoomable chart per scalar
and a histogram per weight tag, and the feature dashboard, with statistics, the activation
histogram and top examples.

| Function | Draws |
|---|---|
| `RenderBarChartHtml`, `RenderWaterfallHtml`, `RenderHeatmapHtml`, `RenderBeeswarmHtml` | The attribution and heatmap views, as above |
| `RenderPartialDependenceHtml`, `RenderTornadoHtml`, `RenderCounterfactualHtml`, `RenderCounterfactualSetHtml`, `RenderMorrisHtml`, `RenderSobolHtml` | The CFS views, as above |
| `RenderTrainingLogHtml(doc)` | A line chart per scalar tag and a histogram per histogram tag (HTML only) |
| `RenderFeatureDashboardHtml(doc)` | A feature's statistics, activation histogram and top examples (HTML only) |
| `RenderTokenRelevanceHtml(doc)` | The text, each piece on a background colored by its relevance |

Each chart is also available as a bare Vega-Lite spec (`ToVegaLiteBarChart`, `ToVegaLiteHeatmap`,
and so on) that you can embed in your own page, or wrap with `VegaLitePage(spec, heading)`.

```cpp
#include <fstream>

#include "pulsatrix/viz/html.hpp"

HtmlOptions options;  // width, title, and where the scripts come from
std::ofstream("bars.html") << RenderBarChartHtml(ToAttributionDocument(attribution), 10, options);
```

**Where the JavaScript comes from.** By default a page loads Vega, Vega-Lite and vega-embed
(BSD-3-Clause) from the jsDelivr CDN. The versions are pinned, and Subresource Integrity hashes
make the browser refuse any changed file. The page is a few kilobytes, but viewing it needs a
network connection. For a page that works offline, or that you can mail as one file, copy the
scripts in (about 830 KB more):

```bash
tools/render/fetch_vega.sh ~/vega                     # downloads and checks the three files once
pulsatrix_svg x.json -o x.html --inline-js ~/vega     # or set PULSATRIX_VEGA_DIR=~/vega
```

In C++, set `options.scripts = HtmlScripts::Inline` and `options.script_dir`. `--offline` finds
the scripts in `$PULSATRIX_VEGA_DIR`, a `vega/` directory next to `pulsatrix_svg`, or
`share/pulsatrix/vega` beside its `bin/`.

**From the command line.** `pulsatrix_svg` writes HTML when the output ends in `.html` (or with
`--html`), and takes the same options as for SVG:

```bash
pulsatrix_svg explanation.json -o bars.html               # hover, zoom, export
pulsatrix_svg pd.json --ice centered -o ice.html
pulsatrix_svg run.json -o run.html                        # training log
pulsatrix_svg feature.json -o feature.html                # feature dashboard
pulsatrix_explain_text MODEL_DIR "some text" -o why.html  # token relevance, straight from a model
```

**Text in every language.** The token relevance page is plain HTML with no scripts. The browser
lays out the text, so right-to-left scripts, joined Arabic letters, Indic conjuncts and emoji
sequences all display correctly, and long text wraps to the window. Hover a piece to see its
score.

**Safety.** Labels are escaped: a feature called `</script>` stays text. The chart's data is
embedded as JSON, not as code.
