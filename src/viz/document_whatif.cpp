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
    if (doc.method != "partial_dependence" && doc.method != "ale") {
        return "/method: must be \"partial_dependence\" or \"ale\"";
    }
    if (doc.method == "ale" && doc.num_instances != 0) {
        return "/num_instances: an ALE document has no ICE curves";
    }
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
    if (doc.method == "partial_dependence" && !doc.feature_values.empty() && doc.feature_values.size() != n) {
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

PartialDependenceDocument ToPartialDependenceDocument(const AleResult& result, std::string feature,
                                                      std::string target) {
    PartialDependenceDocument doc;
    doc.method = "ale";
    doc.feature = std::move(feature);
    doc.target = std::move(target);
    doc.grid = result.edges;
    doc.partial_dependence = result.effects;
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
    w.root().add("method", doc.method);
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
    if (const JsonValue* method = root.find("method")) {
        doc.method = r.string(*method, "/method");
    }
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

// ---- counterfactual ------------------------------------------------------------------------

CounterfactualDocument ToCounterfactualDocument(const Tensor& input, const CounterfactualResult& result,
                                                const std::vector<std::string>& names,
                                                const std::vector<float>& scale, std::string target) {
    const std::vector<float> x = input.to_host_vector();
    const std::vector<float> cf = result.counterfactual.to_host_vector();
    if (cf.size() != x.size()) {
        throw std::invalid_argument("ToCounterfactualDocument: the counterfactual and the input differ in size");
    }
    if ((!names.empty() && names.size() != x.size()) || (!scale.empty() && scale.size() != x.size())) {
        throw std::invalid_argument("ToCounterfactualDocument: names and scale need one entry per feature");
    }
    CounterfactualDocument doc;
    doc.target = std::move(target);
    doc.valid = result.valid;
    doc.output_before = result.output_before;
    doc.output_after = result.output_after;
    for (size_t j = 0; j < x.size(); ++j) {
        doc.features.push_back({names.empty() ? "feature_" + std::to_string(j) : names[j], x[j], cf[j],
                                scale.empty() ? 1.0f : scale[j]});
    }
    return doc;
}

std::string ToJson(const CounterfactualDocument& doc) {
    DocWriter w("counterfactual");
    w.root().add("target", doc.target);
    w.root().add("valid", JsonValue(doc.valid));
    w.root().add("output_before", w.num(doc.output_before, "/output_before"));
    w.root().add("output_after", w.num(doc.output_after, "/output_after"));
    JsonValue features{JsonValue::Array{}};
    for (size_t i = 0; i < doc.features.size(); ++i) {
        const auto& f = doc.features[i];
        const std::string p = Index("/features", i);
        if (!(f.scale > 0.0f)) {
            Invalid("counterfactual", p + "/scale: must be positive");
        }
        JsonValue o{JsonValue::Object{}};
        o.add("name", f.name);
        o.add("original", w.num(f.original, p + "/original"));
        o.add("counterfactual", w.num(f.counterfactual, p + "/counterfactual"));
        o.add("scale", w.num(f.scale, p + "/scale"));
        features.push_back(std::move(o));
    }
    w.root().add("features", std::move(features));
    return w.finish();
}

CounterfactualDocument ParseCounterfactualDocument(std::string_view json) {
    DocReader r(json, "counterfactual");
    const JsonValue& root = r.root();
    CounterfactualDocument doc;
    doc.target = r.string(r.member(root, "", "target"), "/target");
    doc.valid = r.boolean(r.member(root, "", "valid"), "/valid");
    doc.output_before = r.number(r.member(root, "", "output_before"), "/output_before");
    doc.output_after = r.number(r.member(root, "", "output_after"), "/output_after");
    const JsonValue::Array& features = r.array(r.member(root, "", "features"), "/features");
    for (size_t i = 0; i < features.size(); ++i) {
        const std::string p = Index("/features", i);
        r.object(features[i], p);
        CounterfactualDocument::Feature f;
        f.name = r.string(r.member(features[i], p, "name"), p + "/name");
        f.original = r.number(r.member(features[i], p, "original"), p + "/original");
        f.counterfactual = r.number(r.member(features[i], p, "counterfactual"), p + "/counterfactual");
        f.scale = r.number(r.member(features[i], p, "scale"), p + "/scale");
        if (!(f.scale > 0.0f)) {
            r.fail(p + "/scale", "must be positive");
        }
        doc.features.push_back(std::move(f));
    }
    r.finish();
    return doc;
}

// ---- global sensitivity --------------------------------------------------------------------

namespace {

std::string FeatureName(const std::vector<std::string>& names, size_t i, int64_t index, const char* where) {
    if (!names.empty() && i >= names.size()) {
        throw std::invalid_argument(std::string(where) + ": needs one name per varied feature");
    }
    return names.empty() ? "feature_" + std::to_string(index) : names[i];
}

}  // namespace

MorrisDocument ToMorrisDocument(const MorrisResult& result, const std::vector<std::string>& names, std::string target) {
    if (!names.empty() && names.size() != result.features.size()) {
        throw std::invalid_argument("ToMorrisDocument: needs one name per varied feature");
    }
    MorrisDocument doc;
    doc.target = std::move(target);
    doc.num_trajectories = result.num_trajectories;
    for (size_t i = 0; i < result.features.size(); ++i) {
        doc.features.push_back({FeatureName(names, i, result.features[i], "ToMorrisDocument"), result.mu[i],
                                result.mu_star[i], result.sigma[i], result.mu_star_conf[i]});
    }
    return doc;
}

SobolDocument ToSobolDocument(const SobolResult& result, const std::vector<std::string>& names, std::string target) {
    if (!names.empty() && names.size() != result.features.size()) {
        throw std::invalid_argument("ToSobolDocument: needs one name per varied feature");
    }
    SobolDocument doc;
    doc.target = std::move(target);
    doc.num_samples = result.num_samples;
    for (size_t i = 0; i < result.features.size(); ++i) {
        doc.features.push_back({FeatureName(names, i, result.features[i], "ToSobolDocument"), result.first_order[i],
                                result.total_order[i], result.first_order_conf[i], result.total_order_conf[i]});
    }
    return doc;
}

std::string ToJson(const MorrisDocument& doc) {
    DocWriter w("morris");
    w.root().add("target", doc.target);
    w.root().add("num_trajectories", JsonValue(doc.num_trajectories));
    JsonValue features{JsonValue::Array{}};
    for (size_t i = 0; i < doc.features.size(); ++i) {
        const auto& f = doc.features[i];
        const std::string p = Index("/features", i);
        JsonValue o{JsonValue::Object{}};
        o.add("name", f.name);
        o.add("mu", w.num(f.mu, p + "/mu"));
        o.add("mu_star", w.num(f.mu_star, p + "/mu_star"));
        o.add("sigma", w.num(f.sigma, p + "/sigma"));
        o.add("mu_star_conf", w.num(f.mu_star_conf, p + "/mu_star_conf"));
        features.push_back(std::move(o));
    }
    w.root().add("features", std::move(features));
    return w.finish();
}

MorrisDocument ParseMorrisDocument(std::string_view json) {
    DocReader r(json, "morris");
    const JsonValue& root = r.root();
    MorrisDocument doc;
    doc.target = r.string(r.member(root, "", "target"), "/target");
    doc.num_trajectories = r.integer(r.member(root, "", "num_trajectories"), "/num_trajectories", 0, kMaxIndex);
    const JsonValue::Array& features = r.array(r.member(root, "", "features"), "/features");
    for (size_t i = 0; i < features.size(); ++i) {
        const std::string p = Index("/features", i);
        r.object(features[i], p);
        MorrisDocument::Feature f;
        f.name = r.string(r.member(features[i], p, "name"), p + "/name");
        f.mu = r.number(r.member(features[i], p, "mu"), p + "/mu");
        f.mu_star = r.number(r.member(features[i], p, "mu_star"), p + "/mu_star");
        f.sigma = r.number(r.member(features[i], p, "sigma"), p + "/sigma");
        f.mu_star_conf = r.number(r.member(features[i], p, "mu_star_conf"), p + "/mu_star_conf");
        doc.features.push_back(std::move(f));
    }
    r.finish();
    return doc;
}

std::string ToJson(const SobolDocument& doc) {
    DocWriter w("sobol");
    w.root().add("target", doc.target);
    w.root().add("num_samples", JsonValue(doc.num_samples));
    JsonValue features{JsonValue::Array{}};
    for (size_t i = 0; i < doc.features.size(); ++i) {
        const auto& f = doc.features[i];
        const std::string p = Index("/features", i);
        JsonValue o{JsonValue::Object{}};
        o.add("name", f.name);
        o.add("first_order", w.num(f.first_order, p + "/first_order"));
        o.add("total_order", w.num(f.total_order, p + "/total_order"));
        o.add("first_order_conf", w.num(f.first_order_conf, p + "/first_order_conf"));
        o.add("total_order_conf", w.num(f.total_order_conf, p + "/total_order_conf"));
        features.push_back(std::move(o));
    }
    w.root().add("features", std::move(features));
    return w.finish();
}

SobolDocument ParseSobolDocument(std::string_view json) {
    DocReader r(json, "sobol");
    const JsonValue& root = r.root();
    SobolDocument doc;
    doc.target = r.string(r.member(root, "", "target"), "/target");
    doc.num_samples = r.integer(r.member(root, "", "num_samples"), "/num_samples", 0, kMaxIndex);
    const JsonValue::Array& features = r.array(r.member(root, "", "features"), "/features");
    for (size_t i = 0; i < features.size(); ++i) {
        const std::string p = Index("/features", i);
        r.object(features[i], p);
        SobolDocument::Feature f;
        f.name = r.string(r.member(features[i], p, "name"), p + "/name");
        f.first_order = r.number(r.member(features[i], p, "first_order"), p + "/first_order");
        f.total_order = r.number(r.member(features[i], p, "total_order"), p + "/total_order");
        f.first_order_conf = r.number(r.member(features[i], p, "first_order_conf"), p + "/first_order_conf");
        f.total_order_conf = r.number(r.member(features[i], p, "total_order_conf"), p + "/total_order_conf");
        doc.features.push_back(std::move(f));
    }
    r.finish();
    return doc;
}

}  // namespace pulsatrix
