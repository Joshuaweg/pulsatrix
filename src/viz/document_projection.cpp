// The projection document (INT-5): inputs placed in a few dimensions, for a scatter plot.

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

#include "pulsatrix/json.hpp"
#include "pulsatrix/viz/document.hpp"
#include "document_io.hpp"

namespace pulsatrix {

using namespace document_io;

namespace {

// The first problem with a projection document as "<pointer>: <what>", or "" if there is none.
std::string ProjectionProblem(const ProjectionDocument& doc) {
    if (doc.axes.size() < 2) return "/axes: needs at least two";
    if (doc.coordinates.size() % doc.axes.size() != 0) return "/coordinates: must be whole rows of one value per axis";
    for (size_t i = 0; i < doc.coordinates.size(); ++i) {
        if (!std::isfinite(doc.coordinates[i])) return Index("/coordinates", i) + ": must be finite";
    }
    if (!doc.explained_variance_ratio.empty() && doc.explained_variance_ratio.size() != doc.axes.size()) {
        return "/explained_variance_ratio: needs one value per axis, or none";
    }
    const auto n = static_cast<size_t>(doc.num_points());
    if (!doc.labels.empty() && doc.labels.size() != n) return "/labels: needs one per point, or none";
    if (!doc.groups.empty() && doc.groups.size() != n) return "/groups: needs one per point, or none";
    if (!doc.values.empty() && doc.values.size() != n) return "/values: needs one per point, or none";
    return "";
}

}  // namespace

ProjectionDocument MakeProjectionDocument(const Pca& pca, const Tensor& scores, std::vector<std::string> labels,
                                          std::vector<std::string> groups, std::vector<float> values, std::string title) {
    const int64_t k = pca.num_components();
    if (k < 2) throw std::invalid_argument("MakeProjectionDocument: a scatter plot needs at least two components");
    if (scores.rank() != 2 || scores.shape().dim(1) != k) {
        throw std::invalid_argument("MakeProjectionDocument: scores must be (n, " + std::to_string(k) + "), from PcaProject");
    }
    ProjectionDocument doc;
    doc.title = std::move(title);
    doc.method = "pca";
    for (int64_t c = 0; c < k; ++c) {
        doc.axes.push_back("PC" + std::to_string(c + 1));
        doc.explained_variance_ratio.push_back(static_cast<float>(pca.explained_variance_ratio[static_cast<size_t>(c)]));
    }
    doc.coordinates = scores.to_host_vector();
    doc.labels = std::move(labels);
    doc.groups = std::move(groups);
    doc.values = std::move(values);
    const std::string problem = ProjectionProblem(doc);
    if (!problem.empty()) throw std::invalid_argument("MakeProjectionDocument: " + problem);
    return doc;
}

std::string ToJson(const ProjectionDocument& doc) {
    const std::string problem = ProjectionProblem(doc);
    if (!problem.empty()) throw std::invalid_argument("ToJson(ProjectionDocument): " + problem);
    DocWriter w("projection");
    w.root().add("title", doc.title);
    w.root().add("method", doc.method);
    w.root().add("axes", Strings(doc.axes));
    w.root().add("explained_variance_ratio", w.floats(doc.explained_variance_ratio, "/explained_variance_ratio"));
    w.root().add("coordinates", w.floats(doc.coordinates, "/coordinates"));
    if (!doc.labels.empty()) w.root().add("labels", Strings(doc.labels));
    if (!doc.groups.empty()) w.root().add("groups", Strings(doc.groups));
    if (!doc.values.empty()) w.root().add("values", w.floats(doc.values, "/values"));
    return w.finish();
}

ProjectionDocument ParseProjectionDocument(std::string_view json) {
    DocReader r(json, "projection");
    const JsonValue& root = r.root();
    ProjectionDocument doc;
    doc.title = r.string(r.member(root, "", "title"), "/title");
    doc.method = r.string(r.member(root, "", "method"), "/method");
    doc.axes = r.strings(r.member(root, "", "axes"), "/axes");
    doc.explained_variance_ratio = r.floats(r.member(root, "", "explained_variance_ratio"), "/explained_variance_ratio");
    doc.coordinates = r.floats(r.member(root, "", "coordinates"), "/coordinates");
    if (const JsonValue* v = root.find("labels")) doc.labels = r.strings(*v, "/labels");
    if (const JsonValue* v = root.find("groups")) doc.groups = r.strings(*v, "/groups");
    if (const JsonValue* v = root.find("values")) doc.values = r.floats(*v, "/values");
    r.finish();
    const std::string problem = ProjectionProblem(doc);
    if (!problem.empty()) {
        const size_t colon = problem.find(": ");
        r.fail(problem.substr(0, colon), problem.substr(colon + 2));
    }
    return doc;
}

}  // namespace pulsatrix
