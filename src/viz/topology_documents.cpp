// The persistence diagram document (TDA-9).

#include "pulsatrix/viz/topology_documents.hpp"

#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "pulsatrix/json.hpp"
#include "document_io.hpp"

namespace pulsatrix {

using namespace document_io;

namespace {

// The first problem with the document as "<pointer>: <what>", or "" if there is none.
std::string PersistenceProblem(const PersistenceDiagramDocument& doc) {
    if (doc.num_points < 0) return "/num_points: must not be negative";
    if (!std::isfinite(doc.threshold) || doc.threshold < 0) return "/threshold: must be finite and not negative";
    if (doc.diagrams.empty()) return "/diagrams: needs at least one dimension";
    std::set<int> seen;
    for (size_t d = 0; d < doc.diagrams.size(); ++d) {
        const PersistenceDiagramEntry& e = doc.diagrams[d];
        const std::string p = Index("/diagrams", d);
        if (e.dimension < 0) return p + "/dimension: must not be negative";
        if (!seen.insert(e.dimension).second) return p + "/dimension: H" + std::to_string(e.dimension) + " is given twice";
        if (e.deaths.size() != e.births.size()) return p + "/deaths: needs one per birth";
        for (size_t i = 0; i < e.births.size(); ++i) {
            if (!std::isfinite(e.births[i])) return Index(p + "/births", i) + ": must be finite";
            const double death = e.deaths[i];
            if (std::isnan(death) || death == -std::numeric_limits<double>::infinity()) {
                return Index(p + "/deaths", i) + ": must be a number or +infinity";
            }
            if (death < e.births[i]) return Index(p + "/deaths", i) + ": is before its birth";
        }
        if (!std::isfinite(e.error_bound) || e.error_bound < 0) return p + "/error_bound: must be finite and not negative";
        if (!e.p_values.empty() && e.p_values.size() != e.births.size()) return p + "/p_values: needs one per feature, or none";
        for (size_t i = 0; i < e.p_values.size(); ++i) {
            const double v = e.p_values[i];
            if (!std::isnan(v) && !(v >= 0 && v <= 1)) return Index(p + "/p_values", i) + ": must be in [0, 1], or NaN";
        }
        if (!(e.significance_threshold >= 0 && e.significance_threshold <= 1)) {
            return p + "/significance_threshold: must be in [0, 1]";
        }
        if (!std::isfinite(e.bootstrap_band) || e.bootstrap_band < 0) return p + "/bootstrap_band: must be finite and not negative";
    }
    return "";
}

void Check(const PersistenceDiagramDocument& doc, const char* who) {
    const std::string problem = PersistenceProblem(doc);
    if (!problem.empty()) throw std::invalid_argument(std::string(who) + ": " + problem);
}

PersistenceDiagramEntry& Entry(PersistenceDiagramDocument& doc, int dimension, const char* who) {
    PersistenceDiagramEntry* e = doc.find(dimension);
    if (e == nullptr) throw std::invalid_argument(std::string(who) + ": the document has no H" + std::to_string(dimension));
    return *e;
}

}  // namespace

const PersistenceDiagramEntry* PersistenceDiagramDocument::find(int dimension) const {
    for (const PersistenceDiagramEntry& e : diagrams) {
        if (e.dimension == dimension) return &e;
    }
    return nullptr;
}

PersistenceDiagramEntry* PersistenceDiagramDocument::find(int dimension) {
    return const_cast<PersistenceDiagramEntry*>(static_cast<const PersistenceDiagramDocument&>(*this).find(dimension));
}

PersistenceDiagramDocument ToPersistenceDiagramDocument(const std::vector<PersistenceDiagram>& diagrams, std::string title) {
    PersistenceDiagramDocument doc;
    doc.title = std::move(title);
    for (const PersistenceDiagram& d : diagrams) {
        PersistenceDiagramEntry e;
        e.dimension = d.dimension;
        e.error_bound = d.error_bound;
        for (const PersistencePair& p : d.pairs) {
            e.births.push_back(p.birth);
            e.deaths.push_back(p.death);
        }
        doc.diagrams.push_back(std::move(e));
    }
    Check(doc, "ToPersistenceDiagramDocument");
    return doc;
}

PersistenceDiagramDocument ToPersistenceDiagramDocument(const RipsResult& result, std::string title) {
    PersistenceDiagramDocument doc = ToPersistenceDiagramDocument(result.diagrams, std::move(title));
    doc.filtration = "rips";
    doc.threshold = std::isfinite(result.threshold) ? result.threshold : 0.0;
    Check(doc, "ToPersistenceDiagramDocument");
    return doc;
}

void AttachSignificance(PersistenceDiagramDocument& doc, int dimension, const UniversalNullResult& test) {
    PersistenceDiagramEntry& e = Entry(doc, dimension, "AttachSignificance");
    std::vector<double> p(e.size(), std::numeric_limits<double>::quiet_NaN());
    for (const FeatureSignificance& f : test.features) {
        if (f.index < 0 || static_cast<size_t>(f.index) >= p.size()) {
            throw std::invalid_argument("AttachSignificance: feature " + std::to_string(f.index) + " is not in H" +
                                        std::to_string(dimension) + ", which has " + std::to_string(p.size()) +
                                        "; was the test run on this diagram?");
        }
        // The test reports each feature's death / birth; a test of another diagram won't match.
        const size_t i = static_cast<size_t>(f.index);
        const double ratio = e.deaths[i] / e.births[i];
        if (!(std::abs(ratio - f.ratio) <= 1e-9 * std::abs(f.ratio))) {
            throw std::invalid_argument("AttachSignificance: feature " + std::to_string(f.index) + " of H" +
                                        std::to_string(dimension) + " has death / birth " + std::to_string(ratio) +
                                        ", but the test saw " + std::to_string(f.ratio) +
                                        "; was the test run on this diagram?");
        }
        p[i] = f.p_value;
    }
    e.p_values = std::move(p);
    e.significance_threshold = test.threshold;
}

void AttachBootstrapBand(PersistenceDiagramDocument& doc, int dimension, const BootstrapBandResult& band) {
    Entry(doc, dimension, "AttachBootstrapBand").bootstrap_band = band.band;
}

bool IsSignificant(const PersistenceDiagramEntry& entry, size_t i) {
    if (!entry.p_values.empty()) {
        const double p = entry.p_values.at(i);
        return !std::isnan(p) && p < entry.significance_threshold;
    }
    if (entry.bootstrap_band > 0) return entry.deaths.at(i) - entry.births.at(i) > 2 * entry.bootstrap_band;
    return true;
}

std::string ToJson(const PersistenceDiagramDocument& doc) {
    Check(doc, "ToJson(PersistenceDiagramDocument)");
    DocWriter w("persistence_diagram");
    w.root().add("title", doc.title);
    w.root().add("filtration", doc.filtration);
    w.root().add("source", doc.source);
    w.root().add("num_points", JsonValue(doc.num_points));
    w.root().add("threshold", JsonValue(doc.threshold));
    JsonValue diagrams{JsonValue::Array{}};
    for (size_t d = 0; d < doc.diagrams.size(); ++d) {
        const PersistenceDiagramEntry& e = doc.diagrams[d];
        const std::string p = Index("/diagrams", d);
        auto doubles = [&](const std::vector<double>& values, const std::string& pointer) {
            JsonValue a{JsonValue::Array{}};
            for (size_t i = 0; i < values.size(); ++i) a.push_back(w.num_double(values[i], Index(pointer, i)));
            return a;
        };
        JsonValue entry{JsonValue::Object{}};
        entry.add("dimension", JsonValue(e.dimension));
        entry.add("births", doubles(e.births, p + "/births"));
        entry.add("deaths", doubles(e.deaths, p + "/deaths"));
        entry.add("error_bound", JsonValue(e.error_bound));
        if (!e.p_values.empty()) {
            entry.add("p_values", doubles(e.p_values, p + "/p_values"));
            entry.add("significance_threshold", JsonValue(e.significance_threshold));
        }
        if (e.bootstrap_band > 0) entry.add("bootstrap_band", JsonValue(e.bootstrap_band));
        diagrams.push_back(std::move(entry));
    }
    w.root().add("diagrams", std::move(diagrams));
    return w.finish();
}

PersistenceDiagramDocument ParsePersistenceDiagramDocument(std::string_view json) {
    DocReader r(json, "persistence_diagram");
    const JsonValue& root = r.root();
    PersistenceDiagramDocument doc;
    doc.title = r.string(r.member(root, "", "title"), "/title");
    doc.filtration = r.string(r.member(root, "", "filtration"), "/filtration");
    doc.source = r.string(r.member(root, "", "source"), "/source");
    doc.num_points = r.integer(r.member(root, "", "num_points"), "/num_points", 0);
    doc.threshold = r.number_double(r.member(root, "", "threshold"), "/threshold");
    const JsonValue::Array& diagrams = r.array(r.member(root, "", "diagrams"), "/diagrams");
    for (size_t d = 0; d < diagrams.size(); ++d) {
        const std::string p = Index("/diagrams", d);
        (void)r.object(diagrams[d], p);
        auto doubles = [&](const JsonValue& v, const std::string& pointer) {
            const JsonValue::Array& a = r.array(v, pointer);
            std::vector<double> out;
            for (size_t i = 0; i < a.size(); ++i) out.push_back(r.number_double(a[i], Index(pointer, i)));
            return out;
        };
        PersistenceDiagramEntry e;
        e.dimension = static_cast<int>(r.integer(r.member(diagrams[d], p, "dimension"), p + "/dimension", 0, 1000));
        e.births = doubles(r.member(diagrams[d], p, "births"), p + "/births");
        e.deaths = doubles(r.member(diagrams[d], p, "deaths"), p + "/deaths");
        e.error_bound = r.number_double(r.member(diagrams[d], p, "error_bound"), p + "/error_bound");
        if (const JsonValue* v = diagrams[d].find("p_values")) {
            e.p_values = doubles(*v, p + "/p_values");
            e.significance_threshold =
                r.number_double(r.member(diagrams[d], p, "significance_threshold"), p + "/significance_threshold");
        }
        if (const JsonValue* v = diagrams[d].find("bootstrap_band")) e.bootstrap_band = r.number_double(*v, p + "/bootstrap_band");
        doc.diagrams.push_back(std::move(e));
    }
    r.finish();
    const std::string problem = PersistenceProblem(doc);
    if (!problem.empty()) {
        const size_t colon = problem.find(": ");
        r.fail(problem.substr(0, colon), problem.substr(colon + 2));
    }
    return doc;
}

}  // namespace pulsatrix
