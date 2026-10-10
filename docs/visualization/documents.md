# JSON Documents

Every view in pulsatrix reads from a versioned JSON document, so an explanation can be saved,
redrawn later, or drawn by a program outside pulsatrix. The documents are part of
`pulsatrix_core` and need no build flag.

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
| `pulsatrix.morris.v1` | `MorrisDocument` | `target`, `num_trajectories`, and per feature `name`, `mu`, `mu_star`, `sigma`, `mu_star_conf` | `MorrisResult` |
| `pulsatrix.sobol.v1` | `SobolDocument` | `target`, `num_samples`, and per feature `name`, `first_order`, `total_order` and their `_conf` half-widths | `SobolResult` |
| `pulsatrix.counterfactual.v1` | `CounterfactualDocument` | `target`, `valid`, `output_before`, `output_after`, and per feature its `name`, `original` and `counterfactual` value and distance `scale` | `CounterfactualResult` |
| `pulsatrix.sensitivity.v1` | `SensitivityDocument` | `target`, the unchanged `output`, and per feature its `name`, `value`, `low`, `high`, `output_low` and `output_high` | `LocalSensitivityResult` |
| `pulsatrix.partial_dependence.v1` | `PartialDependenceDocument` | `method` (`"partial_dependence"` or `"ale"`), `feature`, `target`, `grid`, `partial_dependence`, and optionally `num_instances` ICE curves (`ice`) and the instances' `feature_values` | `IceResult`, `AleResult` |
| `pulsatrix.mutation_map.v1` | `MutationMapDocument` | `title`, `sequence`, `first_position`, `method`, `alphabet`, and row-major `values` (residues × alphabet) | `VariantScorer::single_mutant_scan` |
| `pulsatrix.sequence_logo.v1` | `SequenceLogoDocument` | `title`, `sequence`, `first_position`, `method`, `alphabet`, and row-major `probabilities` (positions × alphabet, each row a distribution) | `ResidueLogProbs` |
| `pulsatrix.contact_map.v1` | `ContactMapDocument` | `title`, `sequence`, `first_position`, `method`, `predicted` (L × L) and `truth` (L × L of 0, 1 and NaN, or empty) | `ContactMap` |
| `pulsatrix.residue_tracks.v1` | `ResidueTracksDocument` | `title`, `sequence`, `first_position`, `tracks` (`name`, one value per residue, `signed`), `features` (`name`, `start`, `end`, `category`) and `residue_ids` | — |

### Token relevance for text

`viz/text_relevance.hpp` builds a `token_relevance.v1` document from a real explanation, so its
pieces read as the original text:

- **`MakeTokenRelevanceDocument(text, encoding, scores, method, target)`** shows each token as the
  text its offset covers, so byte-level BPE tokens appear decoded. Tokens that split one character
  (an emoji's bytes) become one piece with their scores summed. Text no token covers becomes
  unscored context. BOS and other inserted tokens appear as their token strings, or go to
  `unassigned` when left out.
- **`MakeWordRelevanceDocument(text, word_scores, method, target)`** takes the output of
  `AggregateToWords` (TOK-4): each word is scored, and the spaces between words are unscored
  context.

The document grows three optional fields, all backward compatible, each written only when it
says something:
- `granularity` (`"token"` or `"word"`)
- `scored` (one flag per piece; unscored pieces are context)
- `unassigned` (relevance that belongs to no piece shown)

The SVG token strip draws context as plain text, and so does the `TokenRelevanceView` ImGui
widget (`viz/token_relevance_view.hpp`).

`pulsatrix_explain_text` does all of this for a Hugging Face model in one step. It tokenizes,
runs AttnLRP from the most likely next token, and writes the figure or document:

```sh
pulsatrix_explain_text models/SmolLM2-135M "The Eiffel Tower is located in the city of" -o paris.svg
pulsatrix_explain_text models/SmolLM2-135M "The Eiffel Tower is located in the city of" --words -o paris.json
token_relevance_demo paris.json   # the ImGui widget (PULSATRIX_ENABLE_VIZ)
```

For " Paris", the word view gives "Eiffel" +4.09, and every other word stays under 1.6 in
magnitude ("The" is −1.60). The token view shows the relevance sits on "iffel"; "E" gets little.

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
