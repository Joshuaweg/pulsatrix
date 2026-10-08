// pulsatrix_contacts: predicts residue contacts with an ESM-2 model and scores them against
// experimental structures (roadmap PLM-4): ESM's contact head, and optionally the average of the
// top-K heads chosen on other proteins.
//
//   pulsatrix_contacts esm2_t33_650M_UR50D --structures DIR --proteins 1UBQ:A,1BTL:A,2LZM:A \
//       --choose-heads 1PGA:A,5P21:A --top-k 10
//
// Structures: DIR/<entry>.cif (or .pdb), for example from https://files.rcsb.org/download/<entry>.cif;
// each protein is <entry>:<chain>, using the author's chain id. The model sees the chain's observed
// sequence. Contacts are Cβ (Cα for glycine) under 8 Å; precision is at L, L/2 and L/5 in CASP's
// ranges (short 6–11, medium 12–23, long 24+).
// Options: --choose-heads P,... and --top-k K (default 10): rank every head by long-range P@L on
// those proteins and also report the mean of the best K heads' maps; --atom beta|alpha|virtual;
// --golden (compare predictions with DIR's contacts_golden.safetensors from
// tools/golden/make_contact_golden.py); --max-pass-gb G (default 4); --out results.csv;
// --device cpu|hip.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_contacts.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_structure.hpp"
#include "pulsatrix/safetensors.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_contacts: " << problem << "\n"
              << "usage: pulsatrix_contacts MODEL_DIR --structures DIR --proteins ENTRY:CHAIN,...\n"
              << "       [--choose-heads ENTRY:CHAIN,... [--top-k K]] [--atom beta|alpha|virtual] [--golden]\n"
              << "       [--max-pass-gb G] [--out results.csv] [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string model, structures, out;
    std::string device = "cpu";
    std::vector<std::string> proteins, choose;
    int64_t top_k = 10;
    pulsatrix::ContactAtom atom = pulsatrix::ContactAtom::Beta;
    bool golden = false;
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

/** @brief ENTRY:CHAIN's chain, from DIR/ENTRY.cif or DIR/ENTRY.pdb. */
pulsatrix::StructureChain Chain(const std::string& dir, const std::string& protein) {
    const size_t colon = protein.find(':');
    if (colon == std::string::npos) throw std::invalid_argument("\"" + protein + "\" isn't ENTRY:CHAIN");
    const std::string entry = protein.substr(0, colon);
    const std::string cif = dir + "/" + entry + ".cif";
    const std::string path = std::filesystem::exists(cif) ? cif : dir + "/" + entry + ".pdb";
    return pulsatrix::ReadStructure(path).chain(protein.substr(colon + 1));
}

int Run(pulsatrix::DeviceBackend* backend, const Options& o) {
    using namespace pulsatrix;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model, backend);
    const TextTokenizer tok = LoadEsmTokenizer(o.model + "/vocab.txt");
    const EsmContactHead head = LoadEsmContactHead(o.model, model->config());
    ContactOptions co;
    co.max_pass_bytes = static_cast<int64_t>(o.max_pass_gb * double(int64_t{1} << 30));
    ContactPredictor predictor(*model, tok, backend, co);

    std::vector<AttentionHead> chosen;
    if (!o.choose.empty()) {
        std::vector<LabeledProtein> labeled;
        for (const std::string& p : o.choose) {
            if (std::find(o.proteins.begin(), o.proteins.end(), p) != o.proteins.end()) {
                std::fprintf(stderr, "pulsatrix_contacts: warning: %s both chooses heads and is evaluated\n", p.c_str());
            }
            const StructureChain c = Chain(o.structures, p);
            labeled.push_back({c.sequence(), TrueContacts(c, 8.0, o.atom)});
        }
        const std::vector<HeadPrecision> ranked = RankContactHeads(predictor, labeled);
        std::printf("top %ld heads by long-range P@L on %zu proteins:", static_cast<long>(o.top_k), labeled.size());
        for (int64_t k = 0; k < o.top_k && k < static_cast<int64_t>(ranked.size()); ++k) {
            chosen.push_back(ranked[static_cast<size_t>(k)].head);
            std::printf(" %ld.%ld (%.3f)", static_cast<long>(ranked[static_cast<size_t>(k)].head.layer),
                        static_cast<long>(ranked[static_cast<size_t>(k)].head.head), ranked[static_cast<size_t>(k)].precision);
        }
        std::printf("\n");
    }
    std::unique_ptr<SafetensorsFile> golden;
    if (o.golden) golden = std::make_unique<SafetensorsFile>(SafetensorsFile::Map(o.model + "/contacts_golden.safetensors"));
    CPUBackend host;

    std::ofstream results;
    if (!o.out.empty()) {
        results.open(o.out);
        results << "protein,length,short_pl,medium_pl,long_pl,long_pl2,long_pl5,topk_long_pl,topk_long_pl5,max_abs_diff_golden\n";
    }
    std::printf("%-8s %6s %8s %8s %8s %8s %8s %9s %9s %10s\n", "protein", "length", "short", "medium", "long", "long L/2", "long L/5",
                "top-K", "top-K L/5", "max|diff|");
    double sum[7] = {};
    for (const std::string& p : o.proteins) {
        const StructureChain c = Chain(o.structures, p);
        const ContactMap truth = TrueContacts(c, 8.0, o.atom);
        const ContactMap predicted = predictor.predict(c.sequence(), head);
        const ContactEvaluation e = EvaluateContacts(predicted, truth);
        ContactPrecisions topk{std::nan(""), std::nan(""), std::nan("")};
        if (!chosen.empty()) topk = PrecisionAtL(predictor.average_heads(c.sequence(), chosen), truth, kLongRange);
        double max_diff = std::nan("");
        if (golden) {
            std::string key = p;
            key[key.find(':')] = '.';
            const std::vector<float> want = golden->tensor("contacts." + key, &host).to_host_vector();
            if (want.size() != predicted.values.size()) throw std::invalid_argument("contacts_golden's " + key + " is another size");
            max_diff = 0;
            for (size_t k = 0; k < want.size(); ++k) max_diff = std::max(max_diff, std::abs(double(predicted.values[k]) - want[k]));
        }
        const double row[7] = {e.short_range.at_l, e.medium_range.at_l, e.long_range.at_l, e.long_range.at_l2, e.long_range.at_l5, topk.at_l, topk.at_l5};
        for (int k = 0; k < 7; ++k) sum[k] += row[k];
        std::printf("%-8s %6ld %8.3f %8.3f %8.3f %8.3f %8.3f %9.3f %9.3f %10.2e\n", p.c_str(), static_cast<long>(predicted.length), row[0], row[1],
                    row[2], row[3], row[4], row[5], row[6], max_diff);
        std::fflush(stdout);
        if (results.is_open()) {
            results << p << ',' << predicted.length;
            for (double v : row) results << ',' << v;
            results << ',' << max_diff << '\n';
        }
    }
    const double n = static_cast<double>(o.proteins.size());
    std::printf("%-8s %6s %8.3f %8.3f %8.3f %8.3f %8.3f %9.3f %9.3f\n", "mean", "", sum[0] / n, sum[1] / n, sum[2] / n, sum[3] / n, sum[4] / n,
                sum[5] / n, sum[6] / n);
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
        if (a == "--structures") o.structures = value();
        else if (a == "--proteins") o.proteins = Split(value(), ',');
        else if (a == "--choose-heads") o.choose = Split(value(), ',');
        else if (a == "--top-k") o.top_k = std::atoll(value().c_str());
        else if (a == "--atom") {
            const std::string s = value();
            if (s == "beta") o.atom = pulsatrix::ContactAtom::Beta;
            else if (s == "alpha") o.atom = pulsatrix::ContactAtom::Alpha;
            else if (s == "virtual") o.atom = pulsatrix::ContactAtom::VirtualBeta;
            else Usage("--atom takes beta, alpha or virtual");
        } else if (a == "--golden") o.golden = true;
        else if (a == "--max-pass-gb") o.max_pass_gb = std::atof(value().c_str());
        else if (a == "--out") o.out = value();
        else if (a == "--device") o.device = value();
        else Usage("unknown option " + a);
    }
    if (o.structures.empty() || o.proteins.empty()) Usage("needs --structures and --proteins");
    if (o.top_k < 1) Usage("--top-k must be at least 1");
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
        std::cerr << "pulsatrix_contacts: " << e.what() << "\n";
        return 1;
    }
}
