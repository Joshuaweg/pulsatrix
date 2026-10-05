/** @file document.hpp
 *  @brief Versioned JSON documents (`pulsatrix.<kind>.v1`) that every renderer reads.
 *  @ingroup visualization
 *
 *  Each kind is a plain struct with a writer (ToJson) and a reader (Parse<Kind>Document). The
 *  structs hold data only, never colors or layout, so the ImGui widgets, the SVG renderer and
 *  anything outside pulsatrix draw the same numbers.
 *
 *  **Format rules** (docs/visualization/index.md has the full schema):
 *  - Every document is a JSON object whose `"schema"` member is `"pulsatrix.<kind>.v<N>"`.
 *  - **Versioning.** A reader accepts its own major version only. Within a version, fields are
 *    only ever added, and readers ignore fields they don't know, so an older reader can read a
 *    newer v1 file. Anything else is a new version.
 *  - **NaN and infinity.** JSON has no literal for them. A non-finite number is written as
 *    `null`, and the document's top-level `"nonfinite"` object maps that number's JSON Pointer
 *    (RFC 6901, for example `"/values/3"`) to `"nan"`, `"inf"` or `"-inf"`. The member is
 *    present only when the document has a non-finite number. A `null` where a number belongs and
 *    no `"nonfinite"` entry for it is an error, so data is never silently lost.
 *  - Readers throw std::invalid_argument naming the JSON Pointer of the first problem.
 */
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/circuit_graph.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/ice.hpp"
#include "pulsatrix/sensitivity.hpp"
#include "pulsatrix/metrics_sink.hpp"
#include "pulsatrix/viz/implot_metrics_sink.hpp"
#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {

/** @brief The document format version this build writes and reads. */
inline constexpr int kVizDocumentVersion = 1;

/**
 * @brief Reads only a document's `"schema"` member and returns its kind (for example
 *        `"attribution"`), so a viewer can pick the matching reader.
 * @throws std::invalid_argument if @p json isn't a JSON object with a `"schema"` of the form
 *         `pulsatrix.<kind>.v<N>`, or if N isn't kVizDocumentVersion.
 */
[[nodiscard]] std::string VizDocumentKind(std::string_view json);

// ---- attribution ---------------------------------------------------------------------------

/** @brief `pulsatrix.attribution.v1`: an Attribution's values, shape, method and metadata. */
struct AttributionDocument {
    std::string method;
    /** @brief The values' tensor shape; values has the product of these many elements. */
    std::vector<int64_t> shape;
    /** @brief Row-major values. */
    std::vector<float> values;
    /** @brief Written with keys sorted, so a document doesn't depend on hash-map order. */
    std::map<std::string, std::string> metadata;
};

/** @brief Copies @p attr into a document, reading its values to the host. */
[[nodiscard]] AttributionDocument ToAttributionDocument(const Attribution& attr);
/** @brief Rebuilds an Attribution whose values live on @p backend. */
[[nodiscard]] Attribution ToAttribution(const AttributionDocument& doc, DeviceBackend* backend);
/** @throws std::invalid_argument if values.size() isn't the product of shape. */
[[nodiscard]] std::string ToJson(const AttributionDocument& doc);
[[nodiscard]] AttributionDocument ParseAttributionDocument(std::string_view json);

// ---- heatmap -------------------------------------------------------------------------------

/** @brief `pulsatrix.heatmap.v1`: a row-major grid of values with optional axis labels. */
struct HeatmapDocument {
    std::string title;
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<float> values;
    /** @brief Empty, or one label per row. */
    std::vector<std::string> row_labels;
    /** @brief Empty, or one label per column. */
    std::vector<std::string> col_labels;
};

[[nodiscard]] HeatmapDocument ToHeatmapDocument(const HeatmapGrid& grid, std::string title = "");
[[nodiscard]] HeatmapGrid ToHeatmapGrid(const HeatmapDocument& doc);
/** @throws std::invalid_argument if values.size() != rows * cols or a label list has the wrong
 *          length. */
[[nodiscard]] std::string ToJson(const HeatmapDocument& doc);
[[nodiscard]] HeatmapDocument ParseHeatmapDocument(std::string_view json);

// ---- token relevance -----------------------------------------------------------------------

/** @brief `pulsatrix.token_relevance.v1`: one relevance score per token of a text. */
struct TokenRelevanceDocument {
    std::string method;
    /** @brief The tokens as text, in order. Must be valid UTF-8: decode byte-level tokens first. */
    std::vector<std::string> tokens;
    /** @brief One score per token. */
    std::vector<float> relevance;
    /** @brief What was explained, for example the predicted next token. Optional. */
    std::string target;
};

/** @throws std::invalid_argument if tokens and relevance differ in length. */
[[nodiscard]] std::string ToJson(const TokenRelevanceDocument& doc);
[[nodiscard]] TokenRelevanceDocument ParseTokenRelevanceDocument(std::string_view json);

// ---- circuit graph -------------------------------------------------------------------------

/** @brief `pulsatrix.circuit_graph.v1`: a CircuitGraph's scored nodes and weighted edges. */
struct CircuitGraphDocument {
    struct Node {
        int64_t id = 0;
        /** @brief The OpType's name, for example `"Linear"`. */
        std::string op_type;
        std::optional<std::string> label;
        float ablation_effect = 0.0f;
    };
    struct Edge {
        int64_t from = 0;
        int64_t to = 0;
        float weight = 0.0f;
    };
    std::vector<Node> nodes;
    std::vector<Edge> edges;
};

[[nodiscard]] CircuitGraphDocument ToCircuitGraphDocument(const CircuitGraph& graph);
/** @throws std::invalid_argument if a node's op_type isn't an OpType name. */
[[nodiscard]] CircuitGraph ToCircuitGraph(const CircuitGraphDocument& doc);
/** @throws std::invalid_argument if node ids repeat or are negative, or an edge names a node
 *          that isn't in the list. */
[[nodiscard]] std::string ToJson(const CircuitGraphDocument& doc);
[[nodiscard]] CircuitGraphDocument ParseCircuitGraphDocument(std::string_view json);

/** @brief The name ToCircuitGraphDocument writes for @p op_type, for example `"Linear"`. */
[[nodiscard]] const char* OpTypeName(OpType op_type);
/** @throws std::invalid_argument if @p name isn't an OpType name. */
[[nodiscard]] OpType OpTypeFromName(std::string_view name);

// ---- training log --------------------------------------------------------------------------

/** @brief `pulsatrix.training_log.v1`: scalar series and the latest histogram per tag. */
struct TrainingLogDocument {
    struct Series {
        std::string tag;
        std::vector<int64_t> steps;
        std::vector<double> values;
    };
    struct Histogram {
        std::string tag;
        std::vector<float> values;
    };
    /** @brief Sorted by tag. */
    std::vector<Series> scalars;
    /** @brief Sorted by tag. */
    std::vector<Histogram> histograms;
};

/** @brief Copies everything @p sink has logged, sorted by tag. */
[[nodiscard]] TrainingLogDocument ToTrainingLogDocument(const ImPlotMetricsSink& sink);
/** @brief Logs every scalar in @p doc to @p sink in the order recorded, then every histogram (at
 *         step 0, since the log keeps only the latest one), so any sink can show a saved run: an
 *         ImPlotMetricsSink for the training dashboard, for example.
 *  @throws std::invalid_argument for the same problems ToJson rejects. */
void ReplayTrainingLog(const TrainingLogDocument& doc, MetricsSink& sink);
/** @throws std::invalid_argument if a series' steps and values differ in length, or a tag
 *          repeats. */
[[nodiscard]] std::string ToJson(const TrainingLogDocument& doc);
[[nodiscard]] TrainingLogDocument ParseTrainingLogDocument(std::string_view json);

// ---- feature dashboard ---------------------------------------------------------------------

/**
 * @brief `pulsatrix.feature_dashboard.v1`: one learned feature (for example a sparse
 *        autoencoder unit) with its activation statistics and top-activating examples.
 * @note Follows the core of SAEDashboard and Neuronpedia's feature pages. Logit effects and
 *       other fields can be added later within v1.
 */
struct FeatureDashboardDocument {
    /** @brief Where the feature lives, for example `"sae.blocks.3"`. */
    std::string source;
    int64_t feature_index = 0;
    /** @brief The fraction of inputs on which the feature is active, in [0, 1]. */
    float activation_density = 0.0f;
    float max_activation = 0.0f;
    /** @brief Activation histogram: bin edges (one more than counts), or both empty. */
    std::vector<float> histogram_edges;
    std::vector<int64_t> histogram_counts;
    struct Example {
        /** @brief Where the example came from, for example a dataset index. */
        std::string label;
        /** @brief Token texts for a text example; empty for any other input. */
        std::vector<std::string> tokens;
        /** @brief The feature's activation per token, or a single value for a non-text input. */
        std::vector<float> activations;
    };
    /** @brief Strongest first. */
    std::vector<Example> top_examples;
};

/** @throws std::invalid_argument if the histogram's lengths don't match, a count is negative,
 *          or an example has tokens and a different number of activations. */
[[nodiscard]] std::string ToJson(const FeatureDashboardDocument& doc);
[[nodiscard]] FeatureDashboardDocument ParseFeatureDashboardDocument(std::string_view json);

// ---- partial dependence --------------------------------------------------------------------

/**
 * @brief `pulsatrix.partial_dependence.v1`: a partial dependence curve over one feature, with
 *        the individual (ICE) curves it averages when they were computed (CFS-1).
 * @note Holds the raw curves only. Centered and derivative ICE are views of them, computed by
 *       whoever draws them (IceResult::centered, IceResult::derivative), so every renderer agrees.
 */
struct PartialDependenceDocument {
    /** @brief Name of the varied feature. Optional. */
    std::string feature;
    /** @brief What the curve predicts, for example a class name. Optional. */
    std::string target;
    /** @brief Strictly increasing feature values. */
    std::vector<float> grid;
    /** @brief One value per grid point. */
    std::vector<float> partial_dependence;
    /** @brief Number of ICE curves; 0 when the document holds the average only. */
    int64_t num_instances = 0;
    /** @brief Row-major (num_instances, grid.size()), or empty. */
    std::vector<float> ice;
    /** @brief Each instance's own value of the feature (num_instances of them), or empty. */
    std::vector<float> feature_values;
};

/** @brief Copies @p result, its partial dependence curve included. */
[[nodiscard]] PartialDependenceDocument ToPartialDependenceDocument(const IceResult& result, std::string feature = "",
                                                                    std::string target = "");
/** @brief Rebuilds the ICE curves for their centered and derivative views.
 *  @throws std::invalid_argument if the document holds no ICE curves. */
[[nodiscard]] IceResult ToIceResult(const PartialDependenceDocument& doc);
/** @brief A two-feature partial dependence as a heatmap: rows follow grid_y, columns grid_x, and
 *         the labels are the grid values. */
[[nodiscard]] HeatmapDocument ToHeatmapDocument(const PartialDependence2D& pd, std::string title = "");
/** @throws std::invalid_argument if the grid is empty or not strictly increasing, or a length
 *          doesn't match it. */
[[nodiscard]] std::string ToJson(const PartialDependenceDocument& doc);
[[nodiscard]] PartialDependenceDocument ParsePartialDependenceDocument(std::string_view json);

// ---- sensitivity ---------------------------------------------------------------------------

/**
 * @brief `pulsatrix.sensitivity.v1`: how one output moves when each input feature, alone, goes to
 *        a low and a high value (CFS-3). The data behind a tornado chart.
 */
struct SensitivityDocument {
    /** @brief What the output is, for example a class name. Optional. */
    std::string target;
    /** @brief The output at the unchanged input. */
    float output = 0.0f;
    struct Feature {
        std::string name;
        float value = 0.0f;  ///< The input's own value
        float low = 0.0f;
        float high = 0.0f;
        float output_low = 0.0f;
        float output_high = 0.0f;
    };
    /** @brief In any order; views sort them by swing. */
    std::vector<Feature> features;
};

/** @brief Copies @p result. @p names, if not empty, has one name per input feature (flat index);
 *         features without a name are called `feature_<index>`. */
[[nodiscard]] SensitivityDocument ToSensitivityDocument(const LocalSensitivityResult& result,
                                                        const std::vector<std::string>& names = {},
                                                        std::string target = "");
[[nodiscard]] std::string ToJson(const SensitivityDocument& doc);
[[nodiscard]] SensitivityDocument ParseSensitivityDocument(std::string_view json);

}  // namespace pulsatrix
