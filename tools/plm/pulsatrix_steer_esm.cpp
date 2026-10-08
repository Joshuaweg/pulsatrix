// pulsatrix_steer_esm: steering ESM-2 toward transmembrane residues, and how reliably it works
// (FEAT-7).
//
//   pulsatrix_steer_esm esm2_t6_8M_UR50D --annotations annotated.jsonl --out steer/ [--layer 4]
//       [--proteins 1000] [--test-proteins 100] [--k 16] [--features 2560] [--sae-epochs 5]
//       [--max-coefficient C] [--device cpu|hip] [--seed S]
//
// annotated.jsonl comes from tools/plm/fetch_swissprot_annotations.py. Of --proteins proteins,
// four in five fit the directions at layer --layer:
// - difference of means: the mean representation of transmembrane residues minus the others';
// - an SAE feature: a TopK SAE (k --k, --features latents, standardized inputs) is trained, and
//   the feature whose activations match transmembrane residues best (F1) gives the direction,
//   scaled to the difference of means' norm.
// The behavior steering should raise: at masked positions (15%, as in training), the
// log-probability of a hydrophobic residue (AVILMFWC) minus a polar or charged one
// (DEKRNQSTH), averaged per protein. MeasureSteering adds each direction to every position at
// five coefficients from -C to C (--max-coefficient, default 0.5: the slopes assume a linear
// response, and larger steps disrupt the model in any direction), beside three random
// directions of the same norm, on up to
// --test-proteins held-out soluble proteins and as many held-out transmembrane ones. Writes
// steering.csv (the per-coefficient means) and steerability.csv (each protein's slope).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_concepts.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_training.hpp"
#include "pulsatrix/steering.hpp"
#include "pulsatrix/topk_sparse_autoencoder.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

using namespace pulsatrix;

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_steer_esm: " << problem << "\n"
              << "usage: pulsatrix_steer_esm MODEL_DIR --annotations FILE.jsonl --out DIR [--layer L] [--proteins N]\n"
              << "       [--test-proteins N] [--k K] [--features N] [--sae-epochs N] [--max-coefficient C] [--seed S]\n"
              << "       [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string model, annotations, out, device = "cpu";
    int64_t layer = 4, proteins = 1000, test_proteins = 100, k = 16, features = 2560, sae_epochs = 5;
    double max_coefficient = 0.5;
    uint64_t seed = 0;
};

bool IsTransmembrane(const AnnotatedProtein& p) {
    const std::vector<int> l = ConceptLabels(p, "Transmembrane");
    return std::find(l.begin(), l.end(), 1) != l.end();
}

/** @brief The SAE feature that best matches transmembrane residues, as a raw-space direction. */
std::vector<float> SaeDirection(DeviceBackend* backend, const ResidueEmbeddings& e, const std::vector<int>& labels, const Options& o,
                                double* f1_out) {
    const int64_t h = e.hidden, N = e.residues;
    const std::vector<float>& rows = e.values[0];
    std::vector<double> mean(static_cast<size_t>(h), 0.0), sd(static_cast<size_t>(h), 0.0);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < h; ++j) mean[static_cast<size_t>(j)] += rows[static_cast<size_t>(r * h + j)] / static_cast<double>(N);
    }
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < h; ++j) {
            const double v = rows[static_cast<size_t>(r * h + j)] - mean[static_cast<size_t>(j)];
            sd[static_cast<size_t>(j)] += v * v / static_cast<double>(N);
        }
    }
    for (double& v : sd) v = std::sqrt(v) + 1e-6;
    std::vector<float> x(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) x[i] = static_cast<float>((rows[i] - mean[i % static_cast<size_t>(h)]) / sd[i % static_cast<size_t>(h)]);
    TopKSaeOptions t;
    t.k = o.k;
    t.dead_after = N;
    t.seed = o.seed + 11;
    TopKSparseAutoencoder sae(h, o.features, backend, t);
    AdamOptimizer opt(1e-3f, backend);
    std::vector<int64_t> order(static_cast<size_t>(N));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::mt19937_64 rng(o.seed + 5);
    const int64_t B = 512;
    std::vector<float> xb(static_cast<size_t>(B * h));
    for (int64_t epoch = 1; epoch <= o.sae_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        FeaturizerLoss last;
        for (int64_t from = 0; from + B <= N; from += B) {
            for (int64_t b = 0; b < B; ++b) std::copy_n(x.begin() + order[static_cast<size_t>(from + b)] * h, h, xb.begin() + b * h);
            last = TrainFeaturizer(sae, Tensor(Shape({B, h}), backend, xb, backend->device()), opt);
        }
        std::fprintf(stderr, "  SAE epoch %ld: reconstruction %.4f\n", static_cast<long>(epoch), last.reconstruction);
    }
    const ConceptMatch m = MatchConcept(EncodeSparse(sae, x, backend), labels, "Transmembrane");
    *f1_out = m.f1;
    std::fprintf(stderr, "  best transmembrane feature %ld, F1 %.3f\n", static_cast<long>(m.feature), m.f1);
    // A step d in standardized space is d * sd in the model's.
    std::vector<float> d = FeaturizerDirection(sae, m.feature);
    for (int64_t j = 0; j < h; ++j) d[static_cast<size_t>(j)] = static_cast<float>(d[static_cast<size_t>(j)] * sd[static_cast<size_t>(j)]);
    return d;
}

double Norm(const std::vector<float>& v) {
    double s = 0;
    for (float x : v) s += static_cast<double>(x) * x;
    return std::sqrt(s);
}

int Run(DeviceBackend* backend, const Options& o) {
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model, backend);
    const TextTokenizer tok = LoadEsmTokenizer(o.model + "/vocab.txt");
    std::vector<AnnotatedProtein> all = ReadAnnotatedProteins(o.annotations);
    all.erase(std::remove_if(all.begin(), all.end(), [](const AnnotatedProtein& p) { return p.sequence.size() > 1000; }), all.end());
    std::shuffle(all.begin(), all.end(), std::mt19937_64(o.seed));
    if (static_cast<int64_t>(all.size()) > o.proteins) all.resize(static_cast<size_t>(o.proteins));
    std::vector<AnnotatedProtein> fit, soluble, membrane;
    for (size_t i = 0; i < all.size(); ++i) {
        if (i % 5 != 4) {
            fit.push_back(all[i]);
        } else if (IsTransmembrane(all[i])) {
            if (static_cast<int64_t>(membrane.size()) < o.test_proteins) membrane.push_back(all[i]);
        } else if (static_cast<int64_t>(soluble.size()) < o.test_proteins) {
            soluble.push_back(all[i]);
        }
    }
    std::filesystem::create_directories(o.out);

    // Directions, from the fitting proteins' residues at the layer.
    std::vector<std::string> seqs;
    std::vector<int> labels;
    for (const AnnotatedProtein& p : fit) {
        seqs.push_back(p.sequence);
        const std::vector<int> l = ConceptLabels(p, "Transmembrane");
        labels.insert(labels.end(), l.begin(), l.end());
    }
    std::fprintf(stderr, "embedding %zu proteins at layer %ld\n", seqs.size(), static_cast<long>(o.layer));
    const ResidueEmbeddings e = EmbedResidues(*model, tok, backend, seqs, {o.layer});
    const int64_t h = e.hidden;
    std::vector<float> pos, neg;
    double residual_norm = 0;
    for (int64_t r = 0; r < e.residues; ++r) {
        const float* row = e.row(0, r);
        (labels[static_cast<size_t>(r)] ? pos : neg).insert((labels[static_cast<size_t>(r)] ? pos : neg).end(), row, row + h);
        residual_norm += Norm(std::vector<float>(row, row + h)) / static_cast<double>(e.residues);
    }
    if (pos.empty()) throw std::runtime_error("no transmembrane residues among the fitting proteins");
    const std::vector<float> dom = DifferenceOfMeans(pos, neg, h);
    double f1 = 0;
    std::vector<float> sae = SaeDirection(backend, e, labels, o, &f1);
    const double scale = Norm(dom) / Norm(sae);
    for (float& v : sae) v = static_cast<float>(v * scale);
    std::printf("layer %ld: mean residual norm %.2f; difference of means norm %.2f (%zu transmembrane residues of %ld); SAE feature F1 %.3f\n\n",
                static_cast<long>(o.layer), residual_norm, Norm(dom), pos.size() / static_cast<size_t>(h), static_cast<long>(e.residues), f1);

    // The behavior: hydrophobic minus polar log-probability at masked positions.
    const std::string hydrophobic = "AVILMFWC", polar = "DEKRNQSTH";
    auto ids_of = [&](const std::string& letters) {
        std::vector<int64_t> ids;
        for (char c : letters) ids.push_back(*tok.token_to_id(std::string(1, c)));
        return ids;
    };
    const std::vector<int64_t> hyd = ids_of(hydrophobic), pol = ids_of(polar);
    const int64_t V = model->config().vocab_size;
    model->set_keep_activations(false);
    std::ofstream means(o.out + "/steering.csv"), slopes(o.out + "/steerability.csv");
    means << "proteins,direction,coefficient,mean_behavior,random_mean_behavior\n";
    slopes << "proteins,direction,protein,steerability\n";
    for (const auto& [set_name, set] : {std::pair<const char*, const std::vector<AnnotatedProtein>*>{"soluble", &soluble}, {"transmembrane", &membrane}}) {
        std::vector<MaskedLMBatch> batches;
        for (size_t i = 0; i < set->size(); ++i) {
            MaskedLMCollator collator(tok, backend, {}, o.seed + 100 + i);
            batches.push_back(collator.collate({(*set)[i].sequence}));
        }
        auto behavior = [&](int64_t i, const HiddenStateHook& hook) {
            const MaskedLMBatch& b = batches[static_cast<size_t>(i)];
            model->set_hidden_state_hook(hook);
            const std::vector<float> logits = model->forward(b.ids).to_host_vector();
            model->set_hidden_state_hook({});
            const std::vector<float> targets = b.targets.to_host_vector();
            double total = 0;
            int64_t count = 0;
            for (size_t t = 0; t < targets.size(); ++t) {
                if (targets[t] < 0) continue;
                const float* l = logits.data() + t * static_cast<size_t>(V);
                const float mx = *std::max_element(l, l + V);
                auto logsum = [&](const std::vector<int64_t>& ids) {
                    double s = 0;
                    for (int64_t id : ids) s += std::exp(static_cast<double>(l[id]) - mx);
                    return std::log(s);
                };
                total += logsum(hyd) - logsum(pol);
                ++count;
            }
            return count > 0 ? total / static_cast<double>(count) : 0.0;
        };
        std::printf("%s held-out proteins (%zu):\n", set_name, set->size());
        std::printf("  %-22s %10s %10s %13s %12s %12s\n", "direction", "steer.", "sd", "anti-steer.", "random max", "over random");
        for (const auto& [dir_name, dir] : {std::pair<const char*, const std::vector<float>*>{"difference of means", &dom}, {"SAE feature", &sae}}) {
            SteeringOptions so;
            so.seed = o.seed + 3;
            const double c = o.max_coefficient;
            so.coefficients = {-c, -c / 2, 0.0, c / 2, c};
            const SteeringReport r = MeasureSteering(behavior, static_cast<int64_t>(set->size()), o.layer, *dir, so);
            std::printf("  %-22s %10.3f %10.3f %12.1f%% %12.3f %12.2f\n", dir_name, r.mean_steerability, r.steerability_sd,
                        100 * r.anti_steerable_fraction, r.random_max_abs_steerability, r.over_random);
            for (size_t c = 0; c < r.coefficients.size(); ++c) {
                means << set_name << ',' << dir_name << ',' << r.coefficients[c] << ',' << r.mean_behavior[c] << ',' << r.random_mean_behavior[c] << '\n';
            }
            for (size_t i = 0; i < r.steerability.size(); ++i) slopes << set_name << ',' << dir_name << ',' << (*set)[i].accession << ',' << r.steerability[i] << '\n';
        }
        std::printf("\n");
    }
    std::printf("wrote %s/steering.csv and %s/steerability.csv\n", o.out.c_str(), o.out.c_str());
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
        if (a == "--annotations") o.annotations = value();
        else if (a == "--out") o.out = value();
        else if (a == "--layer") o.layer = std::atoll(value().c_str());
        else if (a == "--proteins") o.proteins = std::atoll(value().c_str());
        else if (a == "--test-proteins") o.test_proteins = std::atoll(value().c_str());
        else if (a == "--k") o.k = std::atoll(value().c_str());
        else if (a == "--features") o.features = std::atoll(value().c_str());
        else if (a == "--sae-epochs") o.sae_epochs = std::atoll(value().c_str());
        else if (a == "--max-coefficient") o.max_coefficient = std::strtod(value().c_str(), nullptr);
        else if (a == "--seed") o.seed = std::strtoull(value().c_str(), nullptr, 10);
        else if (a == "--device") o.device = value();
        else Usage("unknown option " + a);
    }
    if (o.annotations.empty() || o.out.empty()) Usage("needs --annotations and --out");
    try {
        if (o.device == "cpu") {
            CPUBackend cpu;
            return Run(&cpu, o);
        }
#ifdef PULSATRIX_GOLDEN_WITH_HIP
        if (o.device == "hip") {
            HIPBackend hip;
            return Run(&hip, o);
        }
#endif
        Usage("device \"" + o.device + "\" is not in this build");
    } catch (const std::exception& e) {
        std::cerr << "pulsatrix_steer_esm: " << e.what() << "\n";
        return 1;
    }
}
