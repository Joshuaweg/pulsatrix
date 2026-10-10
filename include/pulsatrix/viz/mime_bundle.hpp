/** @file mime_bundle.hpp
 *  @brief Rich notebook display: `mime_bundle_repr()` for pulsatrix's types (roadmap NB-1).
 *  @ingroup visualization
 */
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/protein_documents.hpp"

namespace pulsatrix {

class Tensor;
class CircuitGraph;
struct Attribution;

/** @brief The MIME type Jupyter front ends (JupyterLab, Notebook 7, VS Code) render Vega-Lite
 *         from. pulsatrix's specs use only Vega-Lite 5 features, so they're sent as version 5. */
inline constexpr const char* kVegaLiteMimeType = "application/vnd.vegalite.v5+json";

/**
 * @brief One displayable value in several formats: MIME type to data, richest first. Text
 *        formats (`text/plain`, `text/html`, `image/svg+xml`, base64 `image/png`) are strings;
 *        `+json` formats are JSON objects, as Jupyter's display protocol and nbformat store them.
 */
struct MimeBundle {
    std::vector<std::pair<std::string, JsonValue>> entries;

    /** @throws std::invalid_argument if @p mime is already present. */
    void add(std::string mime, JsonValue data);
    /** @brief The data for @p mime, or nullptr. */
    [[nodiscard]] const JsonValue* find(std::string_view mime) const;
    /** @brief The bundle as one JSON object, `{"text/plain": ..., ...}`. */
    [[nodiscard]] JsonValue to_json_value() const;
};

/** @name mime_bundle_repr
 *  @brief What a notebook shows for each type. Every bundle has `text/plain`.
 *
 *  xeus-cpp's `xcpp::display(x)` (and a cell ending in `x`) calls `mime_bundle_repr(x)`
 *  unqualified, so argument-dependent lookup finds these overloads, and the result converts to
 *  xeus's `nlohmann::json` through to_json() below; pulsatrix itself doesn't depend on xeus or on
 *  nlohmann. NB-2's notebook writer and NB-3's Python `_repr_mimebundle_` use the same bundles.
 *
 *  - Tensor: a summary (shape, device, statistics, leading values) as text and an HTML table.
 *  - Attribution: an image-shaped one (rank 2 to 4, channels summed) as a heatmap, PNG when it
 *    has more than 64 x 64 cells and SVG plus Vega-Lite otherwise; any other shape as a bar chart
 *    of its largest features. A batch shows its first example.
 *  - Documents with a Vega-Lite view get Vega-Lite and the SVG figure; token relevance, protein
 *    views and circuit graphs get their SVG (circuit graphs: Vega-Lite only).
 */
///@{
[[nodiscard]] MimeBundle mime_bundle_repr(const Tensor& tensor);
[[nodiscard]] MimeBundle mime_bundle_repr(const Attribution& attribution);
[[nodiscard]] MimeBundle mime_bundle_repr(const CircuitGraph& graph);
[[nodiscard]] MimeBundle mime_bundle_repr(const AttributionDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const HeatmapDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const TokenRelevanceDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const CircuitGraphDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const TrainingLogDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const PartialDependenceDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const SensitivityDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const CounterfactualDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const MorrisDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const SobolDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const MutationMapDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const SequenceLogoDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const ContactMapDocument& doc);
[[nodiscard]] MimeBundle mime_bundle_repr(const ResidueTracksDocument& doc);
///@}

/**
 * @brief A Vega-Lite node-link view of a circuit graph: nodes in columns by depth (longest path
 *        from a source), colored by ablation effect, edges as lines weighted by |weight|.
 * @throws std::invalid_argument for an empty graph or an edge naming a missing node.
 */
[[nodiscard]] JsonValue ToVegaLiteCircuitGraph(const CircuitGraphDocument& doc);

/** @brief Standard base64 (RFC 4648, with padding), for `image/png` data. */
[[nodiscard]] std::string Base64Encode(const std::vector<uint8_t>& bytes);

/**
 * @brief Converts a JsonValue to another JSON library's value type by its common interface
 *        (assignment from null, bool, number and string; `object()`, `array()`, `operator[]`,
 *        `push_back`). nlohmann::json finds it by argument-dependent lookup, so
 *        `nlohmann::json j = value;` works without pulsatrix including nlohmann.
 * @note Integers (number text without '.', 'e' or 'E' that fits int64_t) stay integers.
 */
template <class Json>
void to_json(Json& j, const JsonValue& value) {
    switch (value.type()) {
        case JsonValue::Type::Null:
            j = nullptr;
            return;
        case JsonValue::Type::Bool:
            j = value.as_bool();
            return;
        case JsonValue::Type::Number: {
            const std::string& text = value.number_text();
            if (text.find_first_of(".eE") == std::string::npos) {
                try {
                    j = static_cast<std::int64_t>(value.as_int64());
                    return;
                } catch (const std::invalid_argument&) {
                }
            }
            j = value.as_double();
            return;
        }
        case JsonValue::Type::String:
            j = value.as_string();
            return;
        case JsonValue::Type::Array: {
            j = Json::array();
            for (const JsonValue& element : value.as_array()) {
                Json converted;
                to_json(converted, element);
                j.push_back(std::move(converted));
            }
            return;
        }
        case JsonValue::Type::Object: {
            j = Json::object();
            for (const auto& [key, member] : value.as_object()) {
                Json converted;
                to_json(converted, member);
                j[key] = std::move(converted);
            }
            return;
        }
    }
}

/** @brief The bundle as a JSON object of MIME type to data (see to_json(Json&, const JsonValue&)). */
template <class Json>
void to_json(Json& j, const MimeBundle& bundle) {
    to_json(j, bundle.to_json_value());
}

}  // namespace pulsatrix
