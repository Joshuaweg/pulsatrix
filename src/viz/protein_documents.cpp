#include "pulsatrix/viz/protein_documents.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "document_io.hpp"
#include "protein_detail.hpp"

namespace pulsatrix {

using namespace document_io;
using namespace protein_detail;

namespace {

/** @brief "pointer: problem" for an alphabet, or empty. */
std::string AlphabetProblem(const std::string& alphabet) {
    if (alphabet.empty()) return "/alphabet: is empty";
    for (size_t i = 0; i < alphabet.size(); ++i) {
        if (alphabet.find(alphabet[i], i + 1) != std::string::npos) return "/alphabet: repeats '" + std::string(1, alphabet[i]) + "'";
    }
    return "";
}

/** @brief Residue numbers stay well inside int64_t, so position arithmetic can't overflow. */
constexpr int64_t kMaxFirstPosition = int64_t{1} << 40;

std::string NumberingProblem(int64_t first_position) {
    if (first_position < -kMaxFirstPosition || first_position > kMaxFirstPosition) return "/first_position: is out of range";
    return "";
}

/** @brief Throws a reader's error for a problem string "pointer: what", or nothing if it's empty. */
void FailOn(const DocReader& r, const std::string& problem) {
    if (problem.empty()) return;
    const size_t colon = problem.find(':');
    r.fail(problem.substr(0, colon), problem.substr(colon + 2));
}

void CheckOnWrite(std::string_view kind, const std::string& problem) {
    if (!problem.empty()) Invalid(kind, problem);
}

/** @brief The members every protein document starts with. */
void WriteHeader(DocWriter& w, const std::string& title, const std::string& sequence, int64_t first_position) {
    w.root().add("title", title);
    w.root().add("sequence", sequence);
    w.root().add("first_position", JsonValue(first_position));
}

void ReadHeader(DocReader& r, std::string& title, std::string& sequence, int64_t& first_position) {
    const JsonValue& root = r.root();
    title = r.string(r.member(root, "", "title"), "/title");
    sequence = r.string(r.member(root, "", "sequence"), "/sequence");
    first_position = r.integer(r.member(root, "", "first_position"), "/first_position");
}

}  // namespace

// ---- mutation map --------------------------------------------------------------------------

std::string protein_detail::MutationMapProblem(const MutationMapDocument& doc) {
    if (std::string p = NumberingProblem(doc.first_position); !p.empty()) return p;
    if (doc.sequence.empty()) return "/sequence: is empty";
    if (std::string p = AlphabetProblem(doc.alphabet); !p.empty()) return p;
    if (doc.values.size() != doc.sequence.size() * doc.alphabet.size()) return "/values: needs one value per residue and letter";
    return "";
}

MutationMapDocument MakeMutationMapDocument(std::string_view sequence, const std::vector<float>& scan, std::string method,
                                            int64_t first_position) {
    const size_t letters = std::string_view(kAminoAcids).size();
    if (scan.size() != sequence.size() * letters) {
        throw std::invalid_argument("MakeMutationMapDocument: the scan needs 20 values per residue");
    }
    MutationMapDocument doc;
    doc.method = std::move(method);
    doc.sequence = std::string(sequence);
    doc.first_position = first_position;
    doc.values = scan;
    return doc;
}

std::string ToJson(const MutationMapDocument& doc) {
    CheckOnWrite("mutation_map", MutationMapProblem(doc));
    DocWriter w("mutation_map");
    WriteHeader(w, doc.title, doc.sequence, doc.first_position);
    w.root().add("method", doc.method);
    w.root().add("alphabet", doc.alphabet);
    w.root().add("values", w.floats(doc.values, "/values"));
    return w.finish();
}

MutationMapDocument ParseMutationMapDocument(std::string_view json) {
    DocReader r(json, "mutation_map");
    MutationMapDocument doc;
    ReadHeader(r, doc.title, doc.sequence, doc.first_position);
    doc.method = r.string(r.member(r.root(), "", "method"), "/method");
    doc.alphabet = r.string(r.member(r.root(), "", "alphabet"), "/alphabet");
    doc.values = r.floats(r.member(r.root(), "", "values"), "/values");
    FailOn(r, MutationMapProblem(doc));
    r.finish();
    return doc;
}

// ---- sequence logo -------------------------------------------------------------------------

int64_t SequenceLogoDocument::positions() const {
    return alphabet.empty() ? 0 : static_cast<int64_t>(probabilities.size() / alphabet.size());
}

std::string protein_detail::SequenceLogoProblem(const SequenceLogoDocument& doc) {
    if (std::string p = NumberingProblem(doc.first_position); !p.empty()) return p;
    if (std::string p = AlphabetProblem(doc.alphabet); !p.empty()) return p;
    const size_t A = doc.alphabet.size();
    if (doc.probabilities.empty() || doc.probabilities.size() % A != 0) {
        return "/probabilities: needs one row of alphabet-many values per position";
    }
    const size_t P = doc.probabilities.size() / A;
    if (!doc.sequence.empty() && doc.sequence.size() != P) return "/sequence: needs one letter per position";
    for (size_t i = 0; i < P; ++i) {
        double sum = 0;
        for (size_t a = 0; a < A; ++a) {
            const float p = doc.probabilities[i * A + a];
            if (!(p >= 0.0f && p <= 1.001f)) return "/probabilities/" + std::to_string(i * A + a) + ": isn't a probability";
            sum += p;
        }
        if (std::abs(sum - 1.0) > 1e-3) return "/probabilities: position " + std::to_string(i) + " sums to " + std::to_string(sum) + ", not 1";
    }
    return "";
}

SequenceLogoDocument MakeSequenceLogoDocument(const ResidueLogProbs& log_probs, const TextTokenizer& tokenizer, std::string_view sequence,
                                              std::string method, int64_t first_position) {
    if (log_probs.length != static_cast<int64_t>(sequence.size())) {
        throw std::invalid_argument("MakeSequenceLogoDocument: needs one row of log-probabilities per residue");
    }
    SequenceLogoDocument doc;
    doc.method = std::move(method);
    doc.sequence = std::string(sequence);
    doc.first_position = first_position;
    std::vector<int64_t> ids;
    for (char a : doc.alphabet) {
        const auto id = tokenizer.token_to_id(std::string(1, a));
        if (!id || *id >= log_probs.vocab) throw std::invalid_argument(std::string("MakeSequenceLogoDocument: no token for ") + a);
        ids.push_back(*id);
    }
    for (int64_t i = 0; i < log_probs.length; ++i) {
        double total = 0;
        std::vector<double> p;
        for (int64_t id : ids) {
            p.push_back(std::exp(static_cast<double>(log_probs.at(i, id))));
            total += p.back();
        }
        for (double v : p) doc.probabilities.push_back(static_cast<float>(v / total));
    }
    return doc;
}

std::vector<float> InformationContent(const SequenceLogoDocument& doc) {
    const size_t A = doc.alphabet.size();
    const int64_t P = doc.positions();
    std::vector<float> out(static_cast<size_t>(P));
    for (int64_t i = 0; i < P; ++i) {
        double entropy = 0;
        for (size_t a = 0; a < A; ++a) {
            const double p = doc.probabilities[static_cast<size_t>(i) * A + a];
            if (p > 0) entropy -= p * std::log2(p);
        }
        out[static_cast<size_t>(i)] = static_cast<float>(std::max(0.0, std::log2(static_cast<double>(A)) - entropy));
    }
    return out;
}

std::string ToJson(const SequenceLogoDocument& doc) {
    CheckOnWrite("sequence_logo", SequenceLogoProblem(doc));
    DocWriter w("sequence_logo");
    WriteHeader(w, doc.title, doc.sequence, doc.first_position);
    w.root().add("method", doc.method);
    w.root().add("alphabet", doc.alphabet);
    w.root().add("probabilities", w.floats(doc.probabilities, "/probabilities"));
    return w.finish();
}

SequenceLogoDocument ParseSequenceLogoDocument(std::string_view json) {
    DocReader r(json, "sequence_logo");
    SequenceLogoDocument doc;
    ReadHeader(r, doc.title, doc.sequence, doc.first_position);
    doc.method = r.string(r.member(r.root(), "", "method"), "/method");
    doc.alphabet = r.string(r.member(r.root(), "", "alphabet"), "/alphabet");
    doc.probabilities = r.floats(r.member(r.root(), "", "probabilities"), "/probabilities");
    FailOn(r, SequenceLogoProblem(doc));
    r.finish();
    return doc;
}

// ---- contact map ---------------------------------------------------------------------------

std::string protein_detail::ContactMapProblem(const ContactMapDocument& doc) {
    if (std::string p = NumberingProblem(doc.first_position); !p.empty()) return p;
    if (doc.sequence.empty()) return "/sequence: is empty";
    const size_t cells = doc.sequence.size() * doc.sequence.size();
    if (doc.predicted.size() != cells) return "/predicted: needs L x L values for the sequence's L residues";
    if (!doc.truth.empty()) {
        if (doc.truth.size() != cells) return "/truth: needs L x L values, or none";
        for (size_t k = 0; k < cells; ++k) {
            const float t = doc.truth[k];
            if (!std::isnan(t) && t != 0.0f && t != 1.0f) return "/truth/" + std::to_string(k) + ": must be 0, 1 or NaN";
        }
    }
    return "";
}

ContactMapDocument MakeContactMapDocument(std::string_view sequence, const ContactMap& predicted, const ContactMap* truth,
                                          std::string method, int64_t first_position) {
    const auto L = static_cast<int64_t>(sequence.size());
    if (predicted.length != L || predicted.values.size() != static_cast<size_t>(L * L) ||
        (truth != nullptr && (truth->length != L || truth->values.size() != static_cast<size_t>(L * L)))) {
        throw std::invalid_argument("MakeContactMapDocument: the maps must be L x L for the sequence's L residues");
    }
    ContactMapDocument doc;
    doc.method = std::move(method);
    doc.sequence = std::string(sequence);
    doc.first_position = first_position;
    doc.predicted = predicted.values;
    if (truth != nullptr) doc.truth = truth->values;
    return doc;
}

std::string ToJson(const ContactMapDocument& doc) {
    CheckOnWrite("contact_map", ContactMapProblem(doc));
    DocWriter w("contact_map");
    WriteHeader(w, doc.title, doc.sequence, doc.first_position);
    w.root().add("method", doc.method);
    w.root().add("predicted", w.floats(doc.predicted, "/predicted"));
    w.root().add("truth", w.floats(doc.truth, "/truth"));
    return w.finish();
}

ContactMapDocument ParseContactMapDocument(std::string_view json) {
    DocReader r(json, "contact_map");
    ContactMapDocument doc;
    ReadHeader(r, doc.title, doc.sequence, doc.first_position);
    doc.method = r.string(r.member(r.root(), "", "method"), "/method");
    doc.predicted = r.floats(r.member(r.root(), "", "predicted"), "/predicted");
    doc.truth = r.floats(r.member(r.root(), "", "truth"), "/truth");
    FailOn(r, ContactMapProblem(doc));
    r.finish();
    return doc;
}

// ---- residue tracks ------------------------------------------------------------------------

std::string protein_detail::ResidueTracksProblem(const ResidueTracksDocument& doc) {
    if (std::string p = NumberingProblem(doc.first_position); !p.empty()) return p;
    if (doc.sequence.empty()) return "/sequence: is empty";
    const auto L = static_cast<int64_t>(doc.sequence.size());
    for (size_t t = 0; t < doc.tracks.size(); ++t) {
        const std::string at = "/tracks/" + std::to_string(t);
        if (doc.tracks[t].name.empty()) return at + "/name: is empty";
        if (static_cast<int64_t>(doc.tracks[t].values.size()) != L) return at + "/values: needs one value per residue";
    }
    for (size_t f = 0; f < doc.features.size(); ++f) {
        const auto& x = doc.features[f];
        const std::string at = "/features/" + std::to_string(f);
        if (x.name.empty()) return at + "/name: is empty";
        if (x.start > x.end) return at + ": starts after it ends";
        if (x.start < doc.first_position || x.end >= doc.first_position + L) return at + ": is outside the sequence";
    }
    if (!doc.residue_ids.empty() && static_cast<int64_t>(doc.residue_ids.size()) != L) return "/residue_ids: needs one id per residue, or none";
    return "";
}

std::vector<float> MeanSubstitutionScore(const MutationMapDocument& doc) {
    (void)ToJson(doc);  // validates
    const size_t A = doc.alphabet.size();
    std::vector<float> out(doc.sequence.size(), std::numeric_limits<float>::quiet_NaN());
    for (size_t i = 0; i < doc.sequence.size(); ++i) {
        double sum = 0;
        int n = 0;
        for (size_t a = 0; a < A; ++a) {
            const float v = doc.values[i * A + a];
            if (doc.alphabet[a] == doc.sequence[i] || !std::isfinite(v)) continue;
            sum += v;
            ++n;
        }
        if (n > 0) out[i] = static_cast<float>(sum / n);
    }
    return out;
}

std::vector<std::string> ResidueIds(const StructureChain& chain) {
    std::vector<std::string> ids;
    ids.reserve(chain.residues.size());
    for (const StructureResidue& r : chain.residues) {
        ids.push_back(std::to_string(r.number) + (r.insertion_code == ' ' ? std::string() : std::string(1, r.insertion_code)));
    }
    return ids;
}

std::string ToJson(const ResidueTracksDocument& doc) {
    CheckOnWrite("residue_tracks", ResidueTracksProblem(doc));
    DocWriter w("residue_tracks");
    WriteHeader(w, doc.title, doc.sequence, doc.first_position);
    JsonValue tracks{JsonValue::Array{}};
    for (size_t t = 0; t < doc.tracks.size(); ++t) {
        JsonValue o{JsonValue::Object{}};
        o.add("name", doc.tracks[t].name);
        o.add("values", w.floats(doc.tracks[t].values, "/tracks/" + std::to_string(t) + "/values"));
        o.add("signed", JsonValue(doc.tracks[t].is_signed));
        tracks.push_back(std::move(o));
    }
    w.root().add("tracks", std::move(tracks));
    JsonValue features{JsonValue::Array{}};
    for (const auto& f : doc.features) {
        JsonValue o{JsonValue::Object{}};
        o.add("name", f.name);
        o.add("start", JsonValue(f.start));
        o.add("end", JsonValue(f.end));
        o.add("category", f.category);
        features.push_back(std::move(o));
    }
    w.root().add("features", std::move(features));
    w.root().add("residue_ids", Strings(doc.residue_ids));
    return w.finish();
}

ResidueTracksDocument ParseResidueTracksDocument(std::string_view json) {
    DocReader r(json, "residue_tracks");
    ResidueTracksDocument doc;
    ReadHeader(r, doc.title, doc.sequence, doc.first_position);
    const JsonValue::Array& tracks = r.array(r.member(r.root(), "", "tracks"), "/tracks");
    for (size_t t = 0; t < tracks.size(); ++t) {
        const std::string at = "/tracks/" + std::to_string(t);
        (void)r.object(tracks[t], at);
        ResidueTracksDocument::Track track;
        track.name = r.string(r.member(tracks[t], at, "name"), at + "/name");
        track.values = r.floats(r.member(tracks[t], at, "values"), at + "/values");
        if (const JsonValue* s = tracks[t].find("signed")) track.is_signed = r.boolean(*s, at + "/signed");
        doc.tracks.push_back(std::move(track));
    }
    const JsonValue::Array& features = r.array(r.member(r.root(), "", "features"), "/features");
    for (size_t f = 0; f < features.size(); ++f) {
        const std::string at = "/features/" + std::to_string(f);
        (void)r.object(features[f], at);
        ResidueTracksDocument::Feature x;
        x.name = r.string(r.member(features[f], at, "name"), at + "/name");
        x.start = r.integer(r.member(features[f], at, "start"), at + "/start");
        x.end = r.integer(r.member(features[f], at, "end"), at + "/end");
        x.category = r.string(r.member(features[f], at, "category"), at + "/category");
        doc.features.push_back(std::move(x));
    }
    if (const JsonValue* ids = r.root().find("residue_ids")) doc.residue_ids = r.strings(*ids, "/residue_ids");
    FailOn(r, ResidueTracksProblem(doc));
    r.finish();
    return doc;
}

}  // namespace pulsatrix
