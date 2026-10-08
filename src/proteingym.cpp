#include "pulsatrix/proteingym.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <stdexcept>

#include "pulsatrix/csv_reader.hpp"

namespace pulsatrix {
namespace {

size_t Column(const CsvTable& t, const std::string& name, const std::string& path) {
    const auto it = std::find(t.header.begin(), t.header.end(), name);
    if (it == t.header.end()) throw std::invalid_argument(path + " has no \"" + name + "\" column");
    return static_cast<size_t>(it - t.header.begin());
}

std::string Upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

}  // namespace

std::vector<ProteinGymAssay> ReadProteinGymReference(const std::string& path) {
    const CsvTable t = CsvReader::Load(path);
    const size_t id = Column(t, "DMS_id", path), file = Column(t, "DMS_filename", path), seq = Column(t, "target_seq", path);
    const auto start = std::find(t.header.begin(), t.header.end(), "start_idx");
    std::vector<ProteinGymAssay> out;
    for (const auto& row : t.rows) {
        ProteinGymAssay a{row[id], row[file], Upper(row[seq]), 1};
        if (start != t.header.end() && !row[static_cast<size_t>(start - t.header.begin())].empty()) {
            a.offset = std::stoll(row[static_cast<size_t>(start - t.header.begin())]);
        }
        out.push_back(std::move(a));
    }
    return out;
}

DmsVariants ReadDmsVariants(const std::string& path) {
    const CsvTable t = CsvReader::Load(path);
    const size_t m = Column(t, "mutant", path), s = Column(t, "DMS_score", path), b = Column(t, "DMS_score_bin", path);
    DmsVariants v;
    for (const auto& row : t.rows) {
        v.mutants.push_back(row[m]);
        v.scores.push_back(std::stod(row[s]));
        v.bins.push_back(static_cast<int>(std::stod(row[b])));
    }
    return v;
}

AssayResult EvaluateAssay(VariantScorer& scorer, const ProteinGymAssay& assay, const DmsVariants& variants, VariantStrategy strategy) {
    const auto start = std::chrono::steady_clock::now();
    const ResidueLogProbs marginals =
        strategy == VariantStrategy::MaskedMarginals ? scorer.masked_marginals(assay.target_seq) : scorer.wild_type_marginals(assay.target_seq);
    AssayResult r;
    r.id = assay.id;
    r.length = static_cast<int64_t>(assay.target_seq.size());
    r.scores.reserve(variants.mutants.size());
    for (const std::string& m : variants.mutants) r.scores.push_back(scorer.score(marginals, assay.target_seq, ParseMutations(m), assay.offset));
    r.metrics = EvaluateFitness(variants.scores, variants.bins, r.scores);
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return r;
}

}  // namespace pulsatrix
