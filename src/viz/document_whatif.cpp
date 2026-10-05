// Documents for the counterfactual and sensitivity tools (CFS epic).

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "pulsatrix/json.hpp"
#include "pulsatrix/viz/document.hpp"
#include "document_io.hpp"

namespace pulsatrix {

using namespace document_io;

// ---- partial dependence --------------------------------------------------------------------

namespace {

std::string PartialDependenceProblem(const PartialDependenceDocument& doc) {
    if (doc.grid.empty()) {
        return "/grid: is empty";
    }
    for (size_t k = 1; k < doc.grid.size(); ++k) {
        if (!(doc.grid[k] > doc.grid[k - 1])) {
            return "/grid: must be strictly increasing (at " + std::to_string(k) + ")";
        }
    }
    if (doc.partial_dependence.size() != doc.grid.size()) {
        return "/partial_dependence: needs one value per grid point";
    }
    if (doc.num_instances < 0) {
        return "/num_instances: is negative";
    }
    const auto n = static_cast<size_t>(doc.num_instances);
    if (doc.ice.size() != n * doc.grid.size()) {
        return "/ice: needs num_instances * grid values";
    }
    if (!doc.feature_values.empty() && doc.feature_values.size() != n) {
        return "/feature_values: needs one value per instance, or none";
    }
    return "";
}

std::string Label(float v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4g", static_cast<double>(v));
    return buf;
}

}  // namespace

PartialDependenceDocument ToPartialDependenceDocument(const IceResult& result, std::string feature,
                                                      std::string target) {
    PartialDependenceDocument doc;
    doc.feature = std::move(feature);
    doc.target = std::move(target);
    doc.grid = result.grid;
    doc.partial_dependence = result.partial_dependence();
    doc.num_instances = result.num_instances;
    doc.ice = result.curves;
    doc.feature_values = result.feature_values;
    return doc;
}

IceResult ToIceResult(const PartialDependenceDocument& doc) {
    if (doc.num_instances == 0) {
        throw std::invalid_argument("ToIceResult: the document holds no ICE curves");
    }
    IceResult r;
    r.grid = doc.grid;
    r.num_instances = doc.num_instances;
    r.curves = doc.ice;
    r.feature_values = doc.feature_values;
    return r;
}

HeatmapDocument ToHeatmapDocument(const PartialDependence2D& pd, std::string title) {
    HeatmapDocument doc;
    doc.title = std::move(title);
    doc.rows = static_cast<int64_t>(pd.grid_y.size());
    doc.cols = static_cast<int64_t>(pd.grid_x.size());
    doc.values = pd.values;
    for (float v : pd.grid_y) {
        doc.row_labels.push_back(Label(v));
    }
    for (float v : pd.grid_x) {
        doc.col_labels.push_back(Label(v));
    }
    return doc;
}

std::string ToJson(const PartialDependenceDocument& doc) {
    if (std::string problem = PartialDependenceProblem(doc); !problem.empty()) {
        Invalid("partial_dependence", problem);
    }
    DocWriter w("partial_dependence");
    w.root().add("feature", doc.feature);
    w.root().add("target", doc.target);
    w.root().add("grid", w.floats(doc.grid, "/grid"));
    w.root().add("partial_dependence", w.floats(doc.partial_dependence, "/partial_dependence"));
    w.root().add("num_instances", JsonValue(doc.num_instances));
    w.root().add("ice", w.floats(doc.ice, "/ice"));
    w.root().add("feature_values", w.floats(doc.feature_values, "/feature_values"));
    return w.finish();
}

PartialDependenceDocument ParsePartialDependenceDocument(std::string_view json) {
    DocReader r(json, "partial_dependence");
    const JsonValue& root = r.root();
    PartialDependenceDocument doc;
    doc.feature = r.string(r.member(root, "", "feature"), "/feature");
    doc.target = r.string(r.member(root, "", "target"), "/target");
    doc.grid = r.floats(r.member(root, "", "grid"), "/grid");
    doc.partial_dependence = r.floats(r.member(root, "", "partial_dependence"), "/partial_dependence");
    doc.num_instances = r.integer(r.member(root, "", "num_instances"), "/num_instances", 0, kMaxIndex);
    doc.ice = r.floats(r.member(root, "", "ice"), "/ice");
    doc.feature_values = r.floats(r.member(root, "", "feature_values"), "/feature_values");
    if (std::string problem = PartialDependenceProblem(doc); !problem.empty()) {
        size_t colon = problem.find(':');
        r.fail(problem.substr(0, colon), problem.substr(colon + 2));
    }
    r.finish();
    return doc;
}

// ---- sensitivity ---------------------------------------------------------------------------

SensitivityDocument ToSensitivityDocument(const LocalSensitivityResult& result, const std::vector<std::string>& names,
                                          std::string target) {
    SensitivityDocument doc;
    doc.target = std::move(target);
    doc.output = result.output;
    for (const FeatureSensitivity& f : result.features) {
        const auto i = static_cast<size_t>(f.feature_index);
        std::string name = i < names.size() ? names[i] : "feature_" + std::to_string(f.feature_index);
        doc.features.push_back({std::move(name), f.value, f.low, f.high, f.output_low, f.output_high});
    }
    return doc;
}

std::string ToJson(const SensitivityDocument& doc) {
    DocWriter w("sensitivity");
    w.root().add("target", doc.target);
    w.root().add("output", w.num(doc.output, "/output"));
    JsonValue features{JsonValue::Array{}};
    for (size_t i = 0; i < doc.features.size(); ++i) {
        const auto& f = doc.features[i];
        const std::string p = Index("/features", i);
        JsonValue o{JsonValue::Object{}};
        o.add("name", f.name);
        o.add("value", w.num(f.value, p + "/value"));
        o.add("low", w.num(f.low, p + "/low"));
        o.add("high", w.num(f.high, p + "/high"));
        o.add("output_low", w.num(f.output_low, p + "/output_low"));
        o.add("output_high", w.num(f.output_high, p + "/output_high"));
        features.push_back(std::move(o));
    }
    w.root().add("features", std::move(features));
    return w.finish();
}

SensitivityDocument ParseSensitivityDocument(std::string_view json) {
    DocReader r(json, "sensitivity");
    const JsonValue& root = r.root();
    SensitivityDocument doc;
    doc.target = r.string(r.member(root, "", "target"), "/target");
    doc.output = r.number(r.member(root, "", "output"), "/output");
    const JsonValue::Array& features = r.array(r.member(root, "", "features"), "/features");
    for (size_t i = 0; i < features.size(); ++i) {
        const std::string p = Index("/features", i);
        r.object(features[i], p);
        SensitivityDocument::Feature f;
        f.name = r.string(r.member(features[i], p, "name"), p + "/name");
        f.value = r.number(r.member(features[i], p, "value"), p + "/value");
        f.low = r.number(r.member(features[i], p, "low"), p + "/low");
        f.high = r.number(r.member(features[i], p, "high"), p + "/high");
        f.output_low = r.number(r.member(features[i], p, "output_low"), p + "/output_low");
        f.output_high = r.number(r.member(features[i], p, "output_high"), p + "/output_high");
        doc.features.push_back(std::move(f));
    }
    r.finish();
    return doc;
}

}  // namespace pulsatrix
