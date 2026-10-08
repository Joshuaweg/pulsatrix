// pulsatrix_protein_views: draws what an ESM-2 model says about one protein (roadmap PLM-5): its
// mutation map, sequence logo, contact map and residue tracks, as JSON documents, SVG figures and
// HTML pages, and, given a structure, the 3D structure colored by the tracks.
//
//   pulsatrix_protein_views esm2_t33_650M_UR50D --structure 1BTL.cif --chain A --out views/
//   pulsatrix_protein_views esm2_t6_8M_UR50D --sequence MKTAYIAKQR... --out views/
//
// Writes into --out: mutation_map, sequence_logo, contact_map and residue_tracks, each as
// .v1.json, .svg and .html, and structure.html with --structure. With a structure, the model
// sees the chain's observed sequence and the contact map shows the true contacts (Cβ under 8 Å).
// Options: --first-position N (numbering, default 1 or the chain's first residue number);
// --device cpu|hip; --max-pass-gb G (default 4); --width W; --inline-js DIR (3Dmol-min.js, for an
// offline structure page; tools/render/fetch_vega.sh downloads it).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_contacts.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_structure.hpp"
#include "pulsatrix/variant_scoring.hpp"
#include "pulsatrix/viz/protein_views.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_protein_views: " << problem << "\n"
              << "usage: pulsatrix_protein_views MODEL_DIR (--sequence SEQ | --structure FILE --chain C) --out DIR\n"
              << "       [--first-position N] [--device cpu|hip] [--max-pass-gb G] [--width W] [--inline-js DIR]\n";
    std::exit(2);
}

struct Options {
    std::string model, sequence, structure, chain, out, inline_js;
    std::string device = "cpu";
    int64_t first_position = 0;  // 0: 1, or the chain's first residue number
    double max_pass_gb = 4;
    int width = 900;
};

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("can't read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void Write(const std::string& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
    std::printf("wrote %s\n", path.c_str());
}

int Run(pulsatrix::DeviceBackend* backend, const Options& o) {
    using namespace pulsatrix;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model, backend);
    const TextTokenizer tok = LoadEsmTokenizer(o.model + "/vocab.txt");
    const std::string model_name = std::filesystem::path(o.model).filename().string();

    std::string sequence = o.sequence, structure_text;
    std::unique_ptr<StructureChain> chain;
    if (!o.structure.empty()) {
        structure_text = ReadFile(o.structure);
        chain = std::make_unique<StructureChain>(ReadStructure(o.structure).chain(o.chain));
        sequence = chain->sequence();
    }
    const int64_t first = o.first_position != 0 ? o.first_position : (chain ? chain->residues.front().number : 1);

    VariantScoringOptions vo;
    vo.max_pass_bytes = static_cast<int64_t>(o.max_pass_gb * double(int64_t{1} << 30));
    VariantScorer scorer(*model, tok, backend, vo);
    const ResidueLogProbs marginals = scorer.masked_marginals(sequence);

    MutationMapDocument mutations = MakeMutationMapDocument(sequence, scorer.single_mutant_scan(marginals, sequence), "masked_marginals", first);
    mutations.title = model_name + ": every substitution";
    SequenceLogoDocument logo = MakeSequenceLogoDocument(marginals, tok, sequence, "masked_marginals", first);
    logo.title = model_name + ": what the model expects at each residue";

    ContactOptions co;
    co.max_pass_bytes = vo.max_pass_bytes;
    ContactPredictor predictor(*model, tok, backend, co);
    const ContactMap predicted = predictor.predict(sequence, LoadEsmContactHead(o.model, model->config()));
    std::unique_ptr<ContactMap> truth;
    if (chain) truth = std::make_unique<ContactMap>(TrueContacts(*chain));
    ContactMapDocument contacts = MakeContactMapDocument(sequence, predicted, truth.get(), "contact head", first);
    contacts.title = model_name + ": predicted contacts" + (chain ? " against " + std::filesystem::path(o.structure).filename().string() : "");

    ResidueTracksDocument tracks;
    tracks.title = model_name + ": per-residue signals";
    tracks.sequence = sequence;
    tracks.first_position = first;
    tracks.tracks.push_back({"mean substitution score", MeanSubstitutionScore(mutations), true});
    tracks.tracks.push_back({"information (bits)", InformationContent(logo), false});
    std::vector<float> wild(sequence.size());
    for (size_t i = 0; i < sequence.size(); ++i) wild[i] = std::exp(marginals.at(static_cast<int64_t>(i), scorer.token_of(sequence[i])));
    tracks.tracks.push_back({"wild-type probability", wild, false});
    if (chain) tracks.residue_ids = ResidueIds(*chain);

    std::filesystem::create_directories(o.out);
    HtmlOptions html;
    html.width = o.width;
    if (!o.inline_js.empty()) {
        html.scripts = HtmlScripts::Inline;
        html.script_dir = o.inline_js;
    }
    SvgOptions svg;
    svg.width = o.width;
    const std::string d = o.out + "/";
    Write(d + "mutation_map.v1.json", ToJson(mutations));
    Write(d + "mutation_map.svg", RenderMutationMapSvg(mutations, svg));
    Write(d + "mutation_map.html", RenderMutationMapHtml(mutations, html));
    Write(d + "sequence_logo.v1.json", ToJson(logo));
    Write(d + "sequence_logo.svg", RenderSequenceLogoSvg(logo, svg));
    Write(d + "sequence_logo.html", RenderSequenceLogoHtml(logo, html));
    Write(d + "contact_map.v1.json", ToJson(contacts));
    Write(d + "contact_map.svg", RenderContactMapSvg(contacts, {}, svg));
    Write(d + "contact_map.html", RenderContactMapHtml(contacts, {}, html));
    Write(d + "residue_tracks.v1.json", ToJson(tracks));
    Write(d + "residue_tracks.svg", RenderResidueTracksSvg(tracks, svg));
    Write(d + "residue_tracks.html", RenderResidueTracksHtml(tracks, html));
    if (chain) Write(d + "structure.html", RenderStructureHtml(structure_text, o.chain, tracks, html));
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
        else if (a == "--structure") o.structure = value();
        else if (a == "--chain") o.chain = value();
        else if (a == "--out") o.out = value();
        else if (a == "--first-position") o.first_position = std::atoll(value().c_str());
        else if (a == "--device") o.device = value();
        else if (a == "--max-pass-gb") o.max_pass_gb = std::atof(value().c_str());
        else if (a == "--width") o.width = std::atoi(value().c_str());
        else if (a == "--inline-js") o.inline_js = value();
        else Usage("unknown option " + a);
    }
    if (o.out.empty()) Usage("needs --out");
    if (o.sequence.empty() == o.structure.empty()) Usage("needs --sequence or --structure, not both");
    if (!o.structure.empty() && o.chain.empty()) Usage("--structure needs --chain");
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
        std::cerr << "pulsatrix_protein_views: " << e.what() << "\n";
        return 1;
    }
}
