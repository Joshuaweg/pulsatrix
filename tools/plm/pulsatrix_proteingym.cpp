// pulsatrix_proteingym: scores ProteinGym deep-mutational-scanning assays with an ESM-2 model and
// reports ProteinGym's metrics (roadmap PLM-3), optionally beside ProteinGym's published numbers.
//
//   pulsatrix_proteingym esm2_t6_8M_UR50D --reference DMS_substitutions.csv --dms-dir DMS_ProteinGym_substitutions \
//       --assays BLAT_ECOLX_Stiffler_2015,RL40A_YEAST_Roscoe_2013 \
//       --published spearman_dms.csv --published-column "ESM2 (8M)" \
//       --published-scores zero_shot_substitutions_scores --published-score-column ESM2_8M
//
// Data: https://proteingym.org (reference file DMS_substitutions.csv, the substitution assays, the
// per-assay Spearman table, and the per-variant zero-shot scores).
// Options: --assays A,B,... (default: every assay found in --dms-dir); --strategy masked-marginals
// (default, ProteinGym's choice for ESM-2) or wt-marginals; --batch N; --max-pass-gb G (the memory
// one forward pass may use, default 4: long windows run fewer per batch); --scores-out DIR (one CSV
// of variant scores per assay); --out results.csv; --device cpu|hip.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/csv_reader.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/proteingym.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_proteingym: " << problem << "\n"
              << "usage: pulsatrix_proteingym MODEL_DIR --reference DMS_substitutions.csv --dms-dir DIR [--assays A,B,...]\n"
              << "       [--strategy masked-marginals|wt-marginals] [--batch N] [--max-pass-gb G] [--published SPEARMAN.csv --published-column NAME]\n"
              << "       [--published-scores DIR --published-score-column NAME] [--scores-out DIR] [--out results.csv] [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string model, reference, dms_dir, published, published_column, published_scores, published_score_column, scores_out, out;
    std::string device = "cpu";
    std::vector<std::string> assays;
    pulsatrix::VariantStrategy strategy = pulsatrix::VariantStrategy::MaskedMarginals;
    int64_t batch = 8;
    double max_pass_gb = 4;
};

std::vector<std::string> Split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, sep);) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

/** @brief A column of a CSV keyed by another column. */
std::map<std::string, std::string> Lookup(const std::string& path, const std::string& key, const std::string& value) {
    const pulsatrix::CsvTable t = pulsatrix::CsvReader::Load(path);
    const auto k = std::find(t.header.begin(), t.header.end(), key), v = std::find(t.header.begin(), t.header.end(), value);
    if (k == t.header.end() || v == t.header.end()) throw std::invalid_argument(path + " lacks \"" + key + "\" or \"" + value + "\"");
    std::map<std::string, std::string> out;
    for (const auto& row : t.rows) out[row[static_cast<size_t>(k - t.header.begin())]] = row[static_cast<size_t>(v - t.header.begin())];
    return out;
}

int Run(pulsatrix::DeviceBackend* backend, const Options& o) {
    using namespace pulsatrix;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model, backend);
    const TextTokenizer tok = LoadEsmTokenizer(o.model + "/vocab.txt");
    VariantScoringOptions so;
    so.batch_size = o.batch;
    so.max_pass_bytes = static_cast<int64_t>(o.max_pass_gb * double(int64_t{1} << 30));
    VariantScorer scorer(*model, tok, backend, so);
    std::map<std::string, std::string> published;
    if (!o.published.empty()) published = Lookup(o.published, "DMS ID", o.published_column);

    std::vector<ProteinGymAssay> assays;
    for (const ProteinGymAssay& a : ReadProteinGymReference(o.reference)) {
        const bool wanted = o.assays.empty() ? std::filesystem::exists(o.dms_dir + "/" + a.filename)
                                             : std::find(o.assays.begin(), o.assays.end(), a.id) != o.assays.end();
        if (wanted) assays.push_back(a);
    }
    if (assays.empty()) throw std::invalid_argument("no assays to run");
    std::ofstream results;
    if (!o.out.empty()) {
        results.open(o.out);
        results << "DMS_id,length,variants,spearman,published_spearman,auc,mcc,ndcg,top_recall,max_abs_diff_published_scores,seconds\n";
    }
    std::printf("%-36s %6s %8s %9s %9s %7s %7s %7s %7s %10s %8s\n", "assay", "length", "variants", "Spearman", "published", "AUC", "MCC", "NDCG",
                "recall", "max|diff|", "seconds");
    double sum_ours = 0, sum_published = 0;
    int n_published = 0;
    for (const ProteinGymAssay& a : assays) {
        const DmsVariants v = ReadDmsVariants(o.dms_dir + "/" + a.filename);
        const AssayResult r = EvaluateAssay(scorer, a, v, o.strategy);
        double max_diff = std::nan("");
        if (!o.published_scores.empty()) {
            // Per-variant published scores, matched by mutant name.
            const auto ref = Lookup(o.published_scores + "/" + a.id + ".csv", "mutant", o.published_score_column);
            max_diff = 0;
            for (size_t i = 0; i < v.mutants.size(); ++i) {
                const auto it = ref.find(v.mutants[i]);
                if (it == ref.end()) throw std::invalid_argument("published scores for " + a.id + " lack " + v.mutants[i]);
                max_diff = std::max(max_diff, std::abs(r.scores[i] - std::stod(it->second)));
            }
        }
        const auto p = published.find(a.id);
        const double pub = p == published.end() ? std::nan("") : std::stod(p->second);
        if (!std::isnan(pub)) {
            sum_ours += r.metrics.spearman;
            sum_published += pub;
            ++n_published;
        }
        std::printf("%-36s %6ld %8zu %9.4f %9.3f %7.3f %7.3f %7.3f %7.3f %10.2e %8.1f\n", a.id.c_str(), static_cast<long>(r.length), v.mutants.size(),
                    r.metrics.spearman, pub, r.metrics.auc, r.metrics.mcc, r.metrics.ndcg, r.metrics.top_recall, max_diff, r.seconds);
        std::fflush(stdout);
        if (results.is_open()) {
            results << a.id << ',' << r.length << ',' << v.mutants.size() << ',' << r.metrics.spearman << ',' << pub << ',' << r.metrics.auc << ','
                    << r.metrics.mcc << ',' << r.metrics.ndcg << ',' << r.metrics.top_recall << ',' << max_diff << ',' << r.seconds << '\n';
        }
        if (!o.scores_out.empty()) {
            std::filesystem::create_directories(o.scores_out);
            std::ofstream s(o.scores_out + "/" + a.id + ".csv");
            s << "mutant,DMS_score,score\n";
            char buf[64];
            for (size_t i = 0; i < v.mutants.size(); ++i) {
                std::snprintf(buf, sizeof(buf), "%.9g", r.scores[i]);
                s << v.mutants[i] << ',' << v.scores[i] << ',' << buf << '\n';
            }
        }
    }
    if (n_published > 0) {
        std::printf("mean Spearman over %d assays: %.4f (published %.4f)\n", n_published, sum_ours / n_published, sum_published / n_published);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) Usage("needs a model directory");
    Options o;
    o.model = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--reference") o.reference = value();
        else if (a == "--dms-dir") o.dms_dir = value();
        else if (a == "--assays") o.assays = Split(value(), ',');
        else if (a == "--strategy") {
            const std::string s = value();
            if (s == "masked-marginals") o.strategy = pulsatrix::VariantStrategy::MaskedMarginals;
            else if (s == "wt-marginals") o.strategy = pulsatrix::VariantStrategy::WildTypeMarginals;
            else Usage("--strategy takes masked-marginals or wt-marginals");
        } else if (a == "--batch") o.batch = std::atoll(value().c_str());
        else if (a == "--max-pass-gb") o.max_pass_gb = std::atof(value().c_str());
        else if (a == "--published") o.published = value();
        else if (a == "--published-column") o.published_column = value();
        else if (a == "--published-scores") o.published_scores = value();
        else if (a == "--published-score-column") o.published_score_column = value();
        else if (a == "--scores-out") o.scores_out = value();
        else if (a == "--out") o.out = value();
        else if (a == "--device") o.device = value();
        else Usage("unknown option " + a);
    }
    if (o.reference.empty() || o.dms_dir.empty()) Usage("needs --reference and --dms-dir");
    if (!o.published.empty() && o.published_column.empty()) Usage("--published needs --published-column");
    if (!o.published_scores.empty() && o.published_score_column.empty()) Usage("--published-scores needs --published-score-column");
    try {
        if (o.device == "cpu") {
            pulsatrix::CPUBackend cpu;
            return Run(&cpu, o);
        }
#ifdef PULSATRIX_GOLDEN_WITH_HIP
        if (o.device == "hip") {
            pulsatrix::HIPBackend hip;
            return Run(&hip, o);
        }
#endif
        Usage("device \"" + o.device + "\" is not in this build");
    } catch (const std::exception& e) {
        std::cerr << "pulsatrix_proteingym: " << e.what() << "\n";
        return 1;
    }
}
