// pulsatrix_explain_protein: explains an ESM-2 model residue by residue with AttnLRP and checks the
// explanations (roadmap PLM-6): randomization, deletion, and agreement with a deep mutational scan
// and with conservation in an alignment.
//
//   pulsatrix_explain_protein esm2_t33_650M_UR50D --assay BLAT_ECOLX_Stiffler_2015 \
//       --reference DMS_substitutions.csv --dms-dir DMS_ProteinGym_substitutions \
//       --msa BLAT_ECOLX_full_11-26-2021_b02.a2m --profile --out tem1/ --device hip
//   pulsatrix_explain_protein esm2_t6_8M_UR50D --sequence MKTAYIAKQR... --mutation K2R
//
// The explained target is --mutation (default: the assay's most damaging single mutant, or, for a
// bare sequence, the first residue's wild type). For it: the top residues, the deletion check
// against random orders, and the randomization check. --profile also explains every residue's
// masked wild-type logit (RelevanceProfile) and compares the profile with the assay's per-position
// sensitivity and the alignment's conservation; the model's own mean substitution score is shown
// beside it for scale. --out DIR writes residue_tracks.{v1.json,svg,html} with every signal, and
// graph.json, the target's attribution graph for Neuronpedia's viewer (sequences up to 254
// residues). Options: --device cpu|hip; --steps N and --random-orders N (deletion, default 20, 5);
// --no-randomization; --top-fraction F (agreement, default 0.1).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_explanations.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/proteingym.hpp"
#include "pulsatrix/viz/attribution_graph.hpp"
#include "pulsatrix/viz/protein_views.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_explain_protein: " << problem << "\n"
              << "usage: pulsatrix_explain_protein MODEL_DIR (--sequence SEQ | --assay ID --reference CSV --dms-dir DIR)\n"
              << "       [--mutation A23G] [--msa FILE.a2m] [--profile] [--out DIR] [--steps N] [--random-orders N]\n"
              << "       [--no-randomization] [--top-fraction F] [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string model, sequence, assay, reference, dms_dir, mutation, msa, out;
    std::string device = "cpu";
    bool profile = false, randomization = true;
    int64_t steps = 20, random_orders = 5;
    double top_fraction = 0.1;
};

void Agreement(const char* name, const std::vector<float>& explanation, const std::vector<float>& signal, double top_fraction) {
    const pulsatrix::ResidueAgreement a = pulsatrix::CompareResidueSignals(explanation, signal, top_fraction);
    std::printf("  %-34s Spearman %+.3f   top %ld overlap %.2f (chance %.2f) over %ld residues\n", name, a.spearman,
                static_cast<long>(a.top), a.top_overlap, a.chance_overlap, static_cast<long>(a.compared));
}

int Run(pulsatrix::DeviceBackend* backend, const Options& o) {
    using namespace pulsatrix;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model, backend);
    const TextTokenizer tok = LoadEsmTokenizer(o.model + "/vocab.txt");
    EncoderExplainer explainer(*model, tok, backend);

    std::string seq = o.sequence;
    int64_t offset = 1;
    DmsVariants dms;
    if (!o.assay.empty()) {
        for (const ProteinGymAssay& a : ReadProteinGymReference(o.reference)) {
            if (a.id != o.assay) continue;
            seq = a.target_seq;
            offset = a.offset;
            dms = ReadDmsVariants(o.dms_dir + "/" + a.filename);
        }
        if (seq.empty()) throw std::invalid_argument("no assay " + o.assay + " in " + o.reference);
    }
    std::printf("%s: %zu residues\n", o.assay.empty() ? "sequence" : o.assay.c_str(), seq.size());

    // The target: the given mutation, the assay's most damaging single mutant, or residue 1's wild type.
    EncoderTarget target = EncoderTarget::MaskedToken(0, seq[0]);
    if (!o.mutation.empty()) {
        target = EncoderTarget::ForMutation(ParseMutations(o.mutation).at(0), offset);
    } else if (!dms.mutants.empty()) {
        size_t worst = dms.mutants.size();
        for (size_t v = 0; v < dms.mutants.size(); ++v) {
            if (ParseMutations(dms.mutants[v]).size() == 1 && (worst == dms.mutants.size() || dms.scores[v] < dms.scores[worst])) worst = v;
        }
        if (worst < dms.mutants.size()) target = EncoderTarget::ForMutation(ParseMutations(dms.mutants[worst])[0], offset);
    }
    const ResidueRelevance r = explainer.explain(seq, target);
    std::printf("explaining %s = %.4g; relevance %.4g in total (residues, <cls> %.3g, <eos> %.3g)\n", r.target.c_str(), r.value, r.total(), r.cls,
                r.eos);
    std::vector<size_t> order(seq.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return std::abs(r.residues[a]) > std::abs(r.residues[b]); });
    std::printf("  most relevant:");
    for (size_t k = 0; k < 10 && k < order.size(); ++k) {
        std::printf(" %c%ld %+.3g", seq[order[k]], static_cast<long>(order[k] + offset), r.residues[order[k]]);
    }
    std::printf("\n");

    const ResidueDeletionCurve d = DeletionCheck(explainer, seq, target, r.residues, o.steps, o.random_orders, 1);
    std::printf("deletion (most relevant first): AUC %.4g against %.4g for random orders: %s\n", d.auc, d.random_auc,
                d.auc < d.random_auc ? "faithful (falls faster)" : "NOT faster than random");
    if (o.randomization) {
        const RandomizationCheckResult z = RandomizationCheck(explainer, seq, target, 1);
        std::printf("randomization (Spearman with the original after randomizing from the top):\n ");
        for (size_t i = 0; i < z.layers.size(); ++i) std::printf(" %s %.2f", z.layers[i].c_str(), z.similarity[i]);
        std::printf("\n");
    }

    ResidueTracksDocument tracks;
    tracks.title = std::filesystem::path(o.model).filename().string() + ": " + r.target;
    tracks.sequence = seq;
    tracks.first_position = offset;
    tracks.tracks.push_back({"relevance: " + r.target, r.residues, true});
    std::vector<float> sensitivity, conservation;
    if (!dms.mutants.empty()) sensitivity = DmsPositionSensitivity(dms, seq, offset);
    if (!o.msa.empty()) {
        const std::vector<FastaRecord> alignment = ReadFasta(o.msa);
        if (AlignmentQuery(alignment) != seq) throw std::invalid_argument(o.msa + "'s query isn't the sequence");
        conservation = AlignmentConservation(alignment);
    }
    if (o.profile) {
        std::fprintf(stderr, "relevance profile: %zu explanations\n", seq.size());
        const std::vector<float> profile = RelevanceProfile(explainer, seq, [](int64_t done, int64_t total) {
            if (done % 25 == 0 || done == total) std::fprintf(stderr, "  %ld/%ld\n", static_cast<long>(done), static_cast<long>(total));
        });
        tracks.tracks.push_back({"relevance profile", profile, false});
        std::printf("agreement of the relevance profile (top %.0f%%):\n", 100 * o.top_fraction);
        if (!sensitivity.empty()) Agreement("with DMS sensitivity", profile, sensitivity, o.top_fraction);
        if (!conservation.empty()) Agreement("with conservation", profile, conservation, o.top_fraction);
        if (!sensitivity.empty() && !conservation.empty()) Agreement("(conservation with DMS sensitivity)", conservation, sensitivity, o.top_fraction);
    }
    if (!sensitivity.empty()) tracks.tracks.push_back({"DMS sensitivity", sensitivity, false});
    if (!conservation.empty()) tracks.tracks.push_back({"conservation (bits)", conservation, false});

    if (!o.out.empty()) {
        std::filesystem::create_directories(o.out);
        std::ofstream(o.out + "/residue_tracks.v1.json") << ToJson(tracks);
        SvgOptions svg;
        svg.width = 900;
        std::ofstream(o.out + "/residue_tracks.svg") << RenderResidueTracksSvg(tracks, svg);
        HtmlOptions html;
        html.width = 900;
        std::ofstream(o.out + "/residue_tracks.html") << RenderResidueTracksHtml(tracks, html);
        if (seq.size() + 2 <= 256) {
            RelevanceGraphOptions g;
            g.slug = "pulsatrix-residues";
            g.scan = std::filesystem::path(o.model).filename().string();
            g.prompt = seq;
            g.prompt_tokens.push_back("<cls>");
            for (char c : seq) g.prompt_tokens.emplace_back(1, c);
            g.prompt_tokens.push_back("<eos>");
            std::ofstream(o.out + "/graph.json") << ToNeuronpediaJson(explainer.relevance_graph(seq, target, g));
        }
        std::printf("wrote %s/residue_tracks.{v1.json,svg,html}%s\n", o.out.c_str(), seq.size() + 2 <= 256 ? " and graph.json" : "");
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
        if (a == "--sequence") o.sequence = value();
        else if (a == "--assay") o.assay = value();
        else if (a == "--reference") o.reference = value();
        else if (a == "--dms-dir") o.dms_dir = value();
        else if (a == "--mutation") o.mutation = value();
        else if (a == "--msa") o.msa = value();
        else if (a == "--profile") o.profile = true;
        else if (a == "--no-randomization") o.randomization = false;
        else if (a == "--out") o.out = value();
        else if (a == "--steps") o.steps = std::atoll(value().c_str());
        else if (a == "--random-orders") o.random_orders = std::atoll(value().c_str());
        else if (a == "--top-fraction") o.top_fraction = std::atof(value().c_str());
        else if (a == "--device") o.device = value();
        else Usage("unknown option " + a);
    }
    if (o.sequence.empty() == o.assay.empty()) Usage("needs --sequence or --assay, not both");
    if (!o.assay.empty() && (o.reference.empty() || o.dms_dir.empty())) Usage("--assay needs --reference and --dms-dir");
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
        std::cerr << "pulsatrix_explain_protein: " << e.what() << "\n";
        return 1;
    }
}
