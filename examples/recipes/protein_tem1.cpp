// Recipe: one protein end to end (roadmap PLM-9). TEM-1 β-lactamase with ESM-2: score every
// mutation of a deep mutational scan zero-shot, draw the model's mutation map beside the measured
// one, predict contacts and check them against the crystal structure, then explain the most
// damaging mutation and show the explanation on the 3D structure.
//
//   protein_tem1_recipe MODEL_DIR PROTEINGYM_DIR STRUCTURE.cif OUT_DIR [--device hip]
//
// MODEL_DIR: an ESM-2 checkpoint (esm2_t6_8M_UR50D runs in a minute on a CPU; esm2_t33_650M_UR50D
// is the one to use, on a GPU). PROTEINGYM_DIR: DMS_substitutions.csv and
// DMS_ProteinGym_substitutions/. STRUCTURE.cif: 1BTL. See docs/recipes/protein-models/protein_tem1.md.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/fitness_metrics.hpp"
#include "pulsatrix/protein_contacts.hpp"
#include "pulsatrix/protein_explanations.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_structure.hpp"
#include "pulsatrix/proteingym.hpp"
#include "pulsatrix/variant_scoring.hpp"
#include "pulsatrix/viz/protein_views.hpp"
#ifdef PULSATRIX_DEMO_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

constexpr const char* kAssay = "BLAT_ECOLX_Stiffler_2015";

void Save(const std::string& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
    std::printf("  wrote %s\n", path.c_str());
}

int Run(DeviceBackend* backend, const std::string& model_dir, const std::string& gym, const std::string& structure_path, const std::string& out) {
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(model_dir, backend);
    const TextTokenizer tok = LoadEsmTokenizer(model_dir + "/vocab.txt");
    const std::string name = std::filesystem::path(model_dir).filename().string();
    std::filesystem::create_directories(out);

    // The assay: TEM-1's 286-residue precursor and every single mutant's measured fitness.
    ProteinGymAssay assay;
    for (const ProteinGymAssay& a : ReadProteinGymReference(gym + "/DMS_substitutions.csv")) {
        if (a.id == kAssay) assay = a;
    }
    const DmsVariants dms = ReadDmsVariants(gym + "/DMS_ProteinGym_substitutions/" + assay.filename);
    const std::string& seq = assay.target_seq;
    std::printf("%s: %zu residues, %zu measured variants; model %s\n\n", kAssay, seq.size(), dms.mutants.size(), name.c_str());

    // 1. Score every mutation: one masked pass per residue (masked marginals).
    std::printf("1. Scoring every mutation\n");
    VariantScorer scorer(*model, tok, backend);
    const ResidueLogProbs marginals = scorer.masked_marginals(seq);
    std::vector<double> predicted, measured;
    MutationMapDocument measured_map = MakeMutationMapDocument(seq, std::vector<float>(seq.size() * 20, std::nanf("")), "DMS (Stiffler 2015)");
    for (size_t v = 0; v < dms.mutants.size(); ++v) {
        const std::vector<Mutation> m = ParseMutations(dms.mutants[v]);
        predicted.push_back(scorer.score(marginals, seq, m, assay.offset));
        measured.push_back(dms.scores[v]);
        const size_t i = static_cast<size_t>(m[0].position - assay.offset);
        measured_map.values[i * 20 + std::string(kAminoAcids).find(m[0].mutant)] = static_cast<float>(dms.scores[v]);
    }
    for (size_t i = 0; i < seq.size(); ++i) measured_map.values[i * 20 + std::string(kAminoAcids).find(seq[i])] = 0.0f;
    std::printf("  Spearman with the measured fitness: %.4f over %zu variants\n", SpearmanCorrelation(predicted, measured), predicted.size());
    MutationMapDocument model_map = MakeMutationMapDocument(seq, scorer.single_mutant_scan(marginals, seq), "masked_marginals");
    model_map.title = name + ": every substitution, zero-shot";
    measured_map.title = "Measured: ampicillin resistance (Stiffler 2015)";
    SvgOptions wide;
    wide.width = 900;
    Save(out + "/mutation_map_model.svg", RenderMutationMapSvg(model_map, wide));
    Save(out + "/mutation_map_measured.svg", RenderMutationMapSvg(measured_map, wide));

    // 2. Contacts against the crystal structure. The structure starts at the mature protein
    // (Ambler residue 26); find its residues in the precursor.
    std::printf("\n2. Contacts against %s\n", std::filesystem::path(structure_path).filename().c_str());
    const std::string structure_text = [&] {
        std::ifstream in(structure_path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    }();
    const StructureChain chain = ReadStructure(structure_path).chain("A");
    const std::string mature = chain.sequence();
    // The crystallized protein needn't be the assay's exactly: find the ungapped placement with
    // the most identical residues, and list the differences.
    size_t start = 0, best = 0;
    for (size_t at = 0; at + mature.size() <= seq.size(); ++at) {
        size_t same = 0;
        for (size_t i = 0; i < mature.size(); ++i) same += seq[at + i] == mature[i] ? 1 : 0;
        if (same > best) {
            best = same;
            start = at;
        }
    }
    if (best < mature.size() * 95 / 100) throw std::runtime_error("the structure's sequence doesn't match the assay's");
    std::printf("  chain A: %zu residues, precursor positions %zu to %zu;", mature.size(), start + 1, start + mature.size());
    for (size_t i = 0; i < mature.size(); ++i) {
        if (seq[start + i] != mature[i]) std::printf(" %c%ld%c", seq[start + i], static_cast<long>(chain.residues[i].number), mature[i]);
    }
    std::printf(" in the crystal\n");
    ContactPredictor predictor(*model, tok, backend);
    const ContactMap contacts = predictor.predict(mature, LoadEsmContactHead(model_dir, model->config()));
    const ContactMap truth = TrueContacts(chain);
    const ContactEvaluation e = EvaluateContacts(contacts, truth);
    std::printf("  long-range precision: P@L %.3f, P@L/2 %.3f, P@L/5 %.3f\n", e.long_range.at_l, e.long_range.at_l2, e.long_range.at_l5);
    ContactMapDocument contact_doc = MakeContactMapDocument(mature, contacts, &truth, "contact head", chain.residues.front().number);
    contact_doc.title = name + ": predicted contacts against 1BTL";
    Save(out + "/contact_map.svg", RenderContactMapSvg(contact_doc, {}, wide));

    // 3. Explain the most damaging single mutant, check the explanation, and show it in 3D.
    size_t worst = 0;
    for (size_t v = 1; v < dms.mutants.size(); ++v) {
        if (dms.scores[v] < dms.scores[worst]) worst = v;
    }
    const Mutation m = ParseMutations(dms.mutants[worst])[0];
    const std::vector<std::string> ids = ResidueIds(chain);
    auto ambler = [&](size_t precursor_index) {  // the structure's numbering, where it has the residue
        return precursor_index >= start && precursor_index < start + mature.size() ? ids[precursor_index - start] : std::string("-");
    };
    std::printf("\n3. Explaining %s (Ambler %c%s%c), the most damaging mutation measured\n", dms.mutants[worst].c_str(), m.wild_type,
                ambler(static_cast<size_t>(m.position - assay.offset)).c_str(), m.mutant);
    EncoderExplainer explainer(*model, tok, backend);
    const EncoderTarget target = EncoderTarget::ForMutation(m, assay.offset);
    const ResidueRelevance r = explainer.explain(seq, target);
    std::vector<size_t> order(seq.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return std::abs(r.residues[a]) > std::abs(r.residues[b]); });
    std::printf("  log-odds %.3f; most relevant residues (Ambler):", r.value);
    for (size_t k = 0; k < 8; ++k) std::printf(" %c%s %+.3f", seq[order[k]], ambler(order[k]).c_str(), r.residues[order[k]]);
    std::printf("\n");
    const ResidueDeletionCurve d = DeletionCheck(explainer, seq, target, r.residues, 10, 3, 1);
    std::printf("  deletion check: area %.3f, against %.3f for random orders (%s)\n", d.auc, d.random_auc,
                d.auc < d.random_auc ? "faithful" : "not faithful");

    ResidueTracksDocument tracks;
    tracks.title = name + ": " + r.target + " on TEM-1";
    tracks.sequence = mature;
    tracks.first_position = static_cast<int64_t>(start) + 1;
    tracks.residue_ids = ids;
    auto slice = [&](const std::vector<float>& v) { return std::vector<float>(v.begin() + static_cast<std::ptrdiff_t>(start),
                                                                               v.begin() + static_cast<std::ptrdiff_t>(start + mature.size())); };
    tracks.tracks.push_back({"relevance: " + r.target, slice(r.residues), true});
    tracks.tracks.push_back({"DMS sensitivity", slice(DmsPositionSensitivity(dms, seq, assay.offset)), false});
    tracks.tracks.push_back({"model: mean substitution score", slice(MeanSubstitutionScore(model_map)), true});
    Save(out + "/residue_tracks.svg", RenderResidueTracksSvg(tracks, wide));
    HtmlOptions page;
    page.width = 900;
    Save(out + "/structure.html", RenderStructureHtml(structure_text, "A", tracks, page));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: protein_tem1_recipe MODEL_DIR PROTEINGYM_DIR STRUCTURE.cif OUT_DIR [--device hip]\n");
        return 2;
    }
    const bool hip = argc > 6 && std::string(argv[5]) == "--device" && std::string(argv[6]) == "hip";
    try {
        if (hip) {
#ifdef PULSATRIX_DEMO_WITH_HIP
            HIPBackend gpu;
            return Run(&gpu, argv[1], argv[2], argv[3], argv[4]);
#else
            std::fprintf(stderr, "this build has no HIP backend\n");
            return 2;
#endif
        }
        CPUBackend cpu;
        return Run(&cpu, argv[1], argv[2], argv[3], argv[4]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "protein_tem1_recipe: %s\n", e.what());
        return 1;
    }
}
