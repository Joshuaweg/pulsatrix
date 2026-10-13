// Recipe: How sure is the model? (KS-3)
//
// ESM-2 has read millions of proteins but never a measurement of TEM-1 β-lactamase. A small head on
// its representations learns, from a deep mutational scan (Stiffler et al. 2015), which single
// mutants still give ampicillin resistance. Then the recipe asks how far to trust each prediction:
//
//   1. Is it calibrated? A reliability diagram of one network, Monte Carlo dropout and an ensemble.
//   2. How sure is it, and why not? The predictive entropy split into the noise the data has
//      (aleatoric) and what the model doesn't know (epistemic), on mutations at positions it was
//      trained on and at a stretch of positions it never saw.
//   3. What can it promise? Conformal prediction sets that hold the truth 90% of the time when the
//      new mutation is like the calibration ones, and the coverage of each slice of the protein.
//   4. Where on the protein is it unsure? The uncertainty residue by residue, on the structure.
//
// The split: the measured positions are cut into five contiguous blocks, and every mutation in the
// middle block is held out ("unseen positions"). The rest are shuffled into training (50%),
// calibration (25%) and test (25%) sets: mutations at positions the head has seen other mutations
// at ("seen positions").
//
// The predictions, fixed before the run:
//   1. The ensemble is better calibrated than one network on the test set: a lower ECE and log
//      loss. Monte Carlo dropout lies between them.
//   2. Epistemic uncertainty is higher at unseen positions than at seen ones, by a one-sided
//      permutation test (p < 0.05). Aleatoric uncertainty rises less.
//   3. Uncertainty ranks the errors: the most certain half of the test predictions is more accurate
//      than all of them.
//   4. The conformal sets cover at least 90% of the test mutations, within sampling error (the
//      guarantee). At unseen positions, which aren't exchangeable with the calibration set, they
//      cover less.
//   5. No prediction for burial: the test says whether buried, intermediate or exposed residues
//      are undercovered.
//
// Usage: how_sure_is_the_model_recipe MODEL_DIR PROTEINGYM_DIR STRUCTURE.cif OUT_DIR
//            [--device cpu|hip] [--members N] [--smoke]
//   MODEL_DIR: an ESM-2 checkpoint, such as ~/.cache/pulsatrix/golden/esm2_t33_650M_UR50D.
//   PROTEINGYM_DIR: holds DMS_substitutions.csv and DMS_ProteinGym_substitutions/. STRUCTURE.cif:
//   1BTL. --smoke runs the tiny test model (tests/fixtures/hf_tiny/esm) on crambin (1CRN) with a
//   made-up assay, to check the plumbing; its numbers mean nothing, and PROTEINGYM_DIR is ignored.
//
// Docs: docs/recipes/interpretability/how_sure_is_the_model.md
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/conformal.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/fitness_metrics.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/performance.hpp"
#include "pulsatrix/protein_concepts.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_structure.hpp"
#include "pulsatrix/proteingym.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/uncertainty.hpp"
#include "pulsatrix/variant_scoring.hpp"
#include "pulsatrix/viz/performance_views.hpp"
#include "pulsatrix/viz/protein_views.hpp"
#ifdef PULSATRIX_DEMO_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

constexpr const char* kAssay = "BLAT_ECOLX_Stiffler_2015";

struct Options {
    std::string model_dir, gym_dir, structure, out;
    bool hip = false;
    bool smoke = false;
    int members = 5;
};

void Save(const std::string& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
    std::printf("  wrote %s\n", path.c_str());
}

// A seeded generator that gives the same numbers on every platform, unlike std::shuffle.
struct SplitMix {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    template <typename T>
    void shuffle(std::vector<T>& v) {
        for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[next() % i]);
    }
};

// The assay: the protein, and each measured single mutant's position (0-based in the sequence),
// new residue, fitness and whether it still works.
struct Assay {
    std::string sequence;
    int64_t offset = 1;
    std::vector<std::string> names;
    std::vector<size_t> positions;
    std::vector<char> mutants;
    std::vector<double> scores;
    std::vector<int64_t> labels;
};

Assay ReadAssay(const std::string& gym) {
    ProteinGymAssay info;
    for (const ProteinGymAssay& a : ReadProteinGymReference(gym + "/DMS_substitutions.csv")) {
        if (a.id == kAssay) info = a;
    }
    if (info.id.empty()) throw std::runtime_error(std::string("no ") + kAssay + " in " + gym);
    const DmsVariants dms = ReadDmsVariants(gym + "/DMS_ProteinGym_substitutions/" + info.filename);
    Assay a;
    a.sequence = info.target_seq;
    a.offset = info.offset;
    for (size_t v = 0; v < dms.mutants.size(); ++v) {
        const std::vector<Mutation> m = ParseMutations(dms.mutants[v]);
        if (m.size() != 1) continue;
        a.names.push_back(dms.mutants[v]);
        a.positions.push_back(static_cast<size_t>(m[0].position - a.offset));
        a.mutants.push_back(m[0].mutant);
        a.scores.push_back(dms.scores[v]);
        a.labels.push_back(dms.bins[v]);
    }
    return a;
}

// --smoke: every single mutant of the structure's sequence, with a made-up fitness that falls with
// the change in hydrophobicity, more steeply in the middle of the chain.
Assay SmokeAssay(const std::string& sequence) {
    const std::string hydrophobic = "AVILMFWC";
    Assay a;
    a.sequence = sequence;
    for (size_t i = 0; i < sequence.size(); ++i) {
        for (const char* m = kAminoAcids; *m != '\0'; ++m) {
            if (*m == sequence[i]) continue;
            const double change = (hydrophobic.find(sequence[i]) != std::string::npos) != (hydrophobic.find(*m) != std::string::npos) ? 1 : 0;
            const double weight = 1.0 - std::abs(static_cast<double>(i) / static_cast<double>(sequence.size()) - 0.5);
            a.names.push_back(sequence[i] + std::to_string(i + 1) + *m);
            a.positions.push_back(i);
            a.mutants.push_back(*m);
            a.scores.push_back(-change * weight + 0.01 * static_cast<double>((i * 7 + static_cast<size_t>(*m)) % 13));
        }
    }
    std::vector<double> sorted = a.scores;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    for (double s : a.scores) a.labels.push_back(s >= median ? 1 : 0);
    return a;
}

// How buried each residue of the sequence is: the number of other residues' Cβ (Cα for glycine)
// within 10 Å of its own, from the structure. -1 where the structure doesn't have the residue.
std::vector<int> NeighborCounts(const StructureChain& chain, size_t start, size_t length) {
    std::vector<Point3> beta;
    for (const StructureResidue& r : chain.residues) {
        std::optional<Point3> p = r.atom("CB");
        if (!p) p = r.atom("CA");
        beta.push_back(p.value_or(Point3{std::nan(""), 0, 0}));
    }
    std::vector<int> counts(length, -1);
    for (size_t i = 0; i < beta.size() && start + i < length; ++i) {
        if (std::isnan(beta[i].x)) continue;
        int n = 0;
        for (size_t j = 0; j < beta.size(); ++j) {
            const double dx = beta[i].x - beta[j].x, dy = beta[i].y - beta[j].y, dz = beta[i].z - beta[j].z;
            if (j != i && !std::isnan(beta[j].x) && dx * dx + dy * dy + dz * dz <= 100.0) ++n;
        }
        counts[start + i] = n;
    }
    return counts;
}

// He-uniform weights, U(-sqrt(6 / in), sqrt(6 / in)), and zero biases. A LinearModule starts at
// zero, which a network can't learn from; the seed makes each ensemble member start elsewhere.
void Initialize(LinearModule& layer, int64_t in, int64_t out, SplitMix& rng) {
    const double bound = std::sqrt(6.0 / static_cast<double>(in));
    std::vector<float> w(static_cast<size_t>(in * out));
    for (float& v : w) v = static_cast<float>((static_cast<double>(rng.next() >> 11) / 9007199254740992.0 * 2 - 1) * bound);
    layer.set_weight(w);
}

// The head: two layers with dropout between them, trained on the CPU.
struct Head {
    LinearModule hidden;
    ReluModule relu;
    DropoutModule dropout;
    LinearModule out;
    SequentialModule net;
    Head(int64_t in, int64_t outputs, uint64_t seed, DeviceBackend* cpu)
        : hidden(in, 64, cpu), relu(cpu), dropout(0.3f, cpu, seed + 1000), out(64, outputs, cpu), net({&hidden, &relu, &dropout, &out}) {
        SplitMix rng{seed};
        Initialize(hidden, in, 64, rng);
        Initialize(out, 64, outputs, rng);
    }
};

// The rows of X picked by idx, as an (n, d) tensor.
Tensor Rows(const std::vector<float>& x, int64_t d, const std::vector<size_t>& idx, DeviceBackend* cpu) {
    std::vector<float> out;
    out.reserve(idx.size() * static_cast<size_t>(d));
    for (size_t i : idx) out.insert(out.end(), x.begin() + static_cast<std::ptrdiff_t>(i * d), x.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
    return Tensor(Shape({static_cast<int64_t>(idx.size()), d}), cpu, out);
}

double Sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// Logits (n, 1) to probabilities (n, 2): p(class 0), p(class 1).
Tensor Probabilities(const Tensor& logits, DeviceBackend* cpu) {
    const std::vector<float> z = logits.to_host_vector();
    std::vector<float> p;
    for (float v : z) {
        const double s = Sigmoid(v);
        p.push_back(static_cast<float>(1 - s));
        p.push_back(static_cast<float>(s));
    }
    return Tensor(Shape({static_cast<int64_t>(z.size()), 2}), cpu, p);
}

template <typename Step>
void Train(Head& h, size_t n, int epochs, uint64_t seed, DeviceBackend* cpu, Step step) {
    AdamWOptimizer optimizer(1e-3f, cpu, 1e-2f);
    SplitMix rng{seed};
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    h.net.set_training(true);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        rng.shuffle(order);
        for (size_t at = 0; at < n; at += 64) {
            const std::vector<size_t> batch(order.begin() + static_cast<std::ptrdiff_t>(at),
                                            order.begin() + static_cast<std::ptrdiff_t>(std::min(n, at + 64)));
            optimizer.zero_grad(h.net);
            (void)h.net.backward(step(batch));
            optimizer.step(h.net);
        }
    }
    h.net.set_training(false);
}

std::vector<size_t> Pick(const std::vector<size_t>& from, const std::vector<size_t>& idx) {
    std::vector<size_t> out;
    for (size_t i : idx) out.push_back(from[i]);
    return out;
}

template <typename T>
std::vector<T> Gather(const std::vector<T>& v, const std::vector<size_t>& idx) {
    std::vector<T> out;
    for (size_t i : idx) out.push_back(v[i]);
    return out;
}

double Mean(const std::vector<double>& v) { return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }

void PrintMetrics(const char* name, const ClassificationReport& r) {
    std::printf("  %-22s accuracy %.3f  log loss %.3f  Brier %.3f  ECE %.3f\n", name, r.accuracy, r.log_loss, r.brier, r.ece);
}

int Run(DeviceBackend* backend, const Options& o) {
    CPUBackend cpu;
    std::filesystem::create_directories(o.out);
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model_dir, backend);
    const TextTokenizer tok = LoadEsmTokenizer(o.model_dir + "/vocab.txt");
    const std::string model_name = std::filesystem::path(o.model_dir).filename().string();
    const std::string structure_text = [&] {
        std::ifstream in(o.structure, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    }();
    const StructureChain chain = ReadStructure(o.structure).chain("A");
    const Assay assay = o.smoke ? SmokeAssay(chain.sequence()) : ReadAssay(o.gym_dir);
    const std::string& seq = assay.sequence;
    const size_t n = assay.names.size();
    std::printf("%s: %zu residues, %zu single mutants, %lld of them functional; model %s\n", o.smoke ? "smoke assay" : kAssay,
                seq.size(), n, static_cast<long long>(std::accumulate(assay.labels.begin(), assay.labels.end(), int64_t{0})),
                model_name.c_str());
    if (o.smoke) std::printf("--smoke: the tiny test model and a made-up assay; the numbers check the plumbing only\n");

    // Where the crystal structure sits in the sequence: the ungapped placement with the most
    // identical residues (1BTL is the mature protein, without the signal peptide).
    const std::string mature = chain.sequence();
    size_t start = 0, best = 0;
    for (size_t at = 0; at + mature.size() <= seq.size(); ++at) {
        size_t same = 0;
        for (size_t i = 0; i < mature.size(); ++i) same += seq[at + i] == mature[i] ? 1 : 0;
        if (same > best) best = same, start = at;
    }
    if (best < mature.size() * 95 / 100) throw std::runtime_error("the structure's sequence doesn't match the assay's");

    // ---- Features: ESM-2's final representation of the mutated residue (one pass over the wild
    // type), the new residue, one-hot, and the zero-shot score log p(new) - log p(old) with the
    // residue masked.
    std::printf("\nRepresentations\n");
    const ResidueEmbeddings emb = EmbedResidues(*model, tok, backend, {seq}, {model->num_layers()});
    VariantScorer scorer(*model, tok, backend);
    const ResidueLogProbs marginals = scorer.masked_marginals(seq);
    const int64_t hidden = emb.hidden;
    const int64_t d = hidden + 20 + 1;
    std::vector<float> x(n * static_cast<size_t>(d), 0.0f);
    std::vector<double> zero_shot(n);
    for (size_t v = 0; v < n; ++v) {
        float* row = x.data() + v * static_cast<size_t>(d);
        const float* e = emb.row(0, static_cast<int64_t>(assay.positions[v]));
        std::copy(e, e + hidden, row);
        row[hidden + static_cast<int64_t>(std::string(kAminoAcids).find(assay.mutants[v]))] = 1.0f;
        const Mutation m{seq[assay.positions[v]], static_cast<int64_t>(assay.positions[v]) + assay.offset, assay.mutants[v]};
        zero_shot[v] = scorer.score(marginals, seq, {m}, assay.offset);
        row[d - 1] = static_cast<float>(zero_shot[v]);
    }
    std::printf("  %lld features per mutant: %lld from ESM-2, 20 for the new residue, 1 zero-shot score\n",
                static_cast<long long>(d), static_cast<long long>(hidden));

    // ---- The split.
    std::vector<size_t> measured(assay.positions);
    std::sort(measured.begin(), measured.end());
    measured.erase(std::unique(measured.begin(), measured.end()), measured.end());
    const size_t block = measured.size() / 5;
    const size_t unseen_first = measured[2 * block], unseen_last = measured[3 * block - 1];
    std::vector<size_t> seen, unseen;
    for (size_t v = 0; v < n; ++v) {
        (assay.positions[v] >= unseen_first && assay.positions[v] <= unseen_last ? unseen : seen).push_back(v);
    }
    SplitMix split_rng{0};
    split_rng.shuffle(seen);
    const size_t n_train = seen.size() / 2, n_cal = seen.size() / 4;
    const std::vector<size_t> train(seen.begin(), seen.begin() + static_cast<std::ptrdiff_t>(n_train));
    const std::vector<size_t> cal(seen.begin() + static_cast<std::ptrdiff_t>(n_train),
                                  seen.begin() + static_cast<std::ptrdiff_t>(n_train + n_cal));
    const std::vector<size_t> test(seen.begin() + static_cast<std::ptrdiff_t>(n_train + n_cal), seen.end());
    std::printf("  unseen positions %zu to %zu: %zu mutants held out; the rest split into %zu training, %zu calibration, %zu test\n",
                unseen_first + 1, unseen_last + 1, unseen.size(), train.size(), cal.size(), test.size());

    // Standardize every column but the one-hot by the training set's mean and spread.
    for (int64_t c = 0; c < d; ++c) {
        if (c >= hidden && c < hidden + 20) continue;
        double mean = 0, sq = 0;
        for (size_t i : train) mean += x[i * static_cast<size_t>(d) + static_cast<size_t>(c)];
        mean /= static_cast<double>(train.size());
        for (size_t i : train) sq += std::pow(x[i * static_cast<size_t>(d) + static_cast<size_t>(c)] - mean, 2);
        const double sd = std::sqrt(sq / static_cast<double>(train.size())) + 1e-6;
        for (size_t i = 0; i < n; ++i) {
            float& v = x[i * static_cast<size_t>(d) + static_cast<size_t>(c)];
            v = static_cast<float>((v - mean) / sd);
        }
    }

    // ---- Train the ensembles: classifiers on whether the mutant works, and regressors that
    // predict the fitness and their own noise.
    std::printf("\nTraining %d classifiers and %d regressors (seeds 1 to %d)\n", o.members, o.members, o.members);
    const int epochs = o.smoke ? 3 : 40;
    const std::vector<int64_t>& y = assay.labels;
    double score_mean = 0, score_sq = 0;
    for (size_t i : train) score_mean += assay.scores[i];
    score_mean /= static_cast<double>(train.size());
    for (size_t i : train) score_sq += std::pow(assay.scores[i] - score_mean, 2);
    const double score_sd = std::sqrt(score_sq / static_cast<double>(train.size()));
    std::vector<std::unique_ptr<Head>> classifiers, regressors;
    for (int m = 0; m < o.members; ++m) {
        const uint64_t seed = static_cast<uint64_t>(m) + 1;
        classifiers.push_back(std::make_unique<Head>(d, 1, seed, &cpu));
        BCEWithLogitsLoss bce(&cpu);
        Head& c = *classifiers.back();
        Train(c, train.size(), epochs, seed, &cpu, [&](const std::vector<size_t>& batch) {
            const std::vector<size_t> rows = Pick(train, batch);
            std::vector<float> t;
            for (size_t i : rows) t.push_back(static_cast<float>(y[i]));
            (void)bce.forward(c.net.forward(Rows(x, d, rows, &cpu)), Tensor(Shape({static_cast<int64_t>(rows.size()), 1}), &cpu, t));
            return bce.backward();
        });
        regressors.push_back(std::make_unique<Head>(d, 2, seed + 100, &cpu));
        GaussianNLLLoss nll(&cpu);
        Head& r = *regressors.back();
        Train(r, train.size(), epochs, seed + 100, &cpu, [&](const std::vector<size_t>& batch) {
            const std::vector<size_t> rows = Pick(train, batch);
            std::vector<float> t;
            for (size_t i : rows) t.push_back(static_cast<float>((assay.scores[i] - score_mean) / score_sd));
            (void)nll.forward(r.net.forward(Rows(x, d, rows, &cpu)), Tensor(Shape({static_cast<int64_t>(rows.size())}), &cpu, t));
            return nll.backward();
        });
    }

    // Every member's probabilities for a set of mutants, and Monte Carlo dropout's from the first.
    auto members_of = [&](const std::vector<size_t>& idx) {
        std::vector<std::function<Tensor(const Tensor&)>> fns;
        for (auto& c : classifiers) {
            Head* h = c.get();
            fns.push_back([h, &cpu](const Tensor& in) { return Probabilities(h->net.forward(in), &cpu); });
        }
        return EnsembleSamples(fns, Rows(x, d, idx, &cpu));
    };
    const int mc_samples = o.smoke ? 4 : 30;
    auto dropout_of = [&](const std::vector<size_t>& idx) {
        Head& h = *classifiers.front();
        return McDropoutSamples(h.net, [&](const Tensor& in) { return Probabilities(h.net.forward(in), &cpu); }, Rows(x, d, idx, &cpu),
                                mc_samples);
    };
    const std::vector<int64_t> y_test = Gather(y, test), y_unseen = Gather(y, unseen), y_cal = Gather(y, cal);
    const std::vector<Tensor> test_members = members_of(test), unseen_members = members_of(unseen);
    const UncertaintyDecomposition ens_test = DecomposeUncertainty(test_members);
    const UncertaintyDecomposition ens_unseen = DecomposeUncertainty(unseen_members);
    const UncertaintyDecomposition mc_test = DecomposeUncertainty(dropout_of(test));
    const UncertaintyDecomposition mc_unseen = DecomposeUncertainty(dropout_of(unseen));

    // ---- 1. Calibration.
    std::printf("\n1. Is it calibrated? Test set, %zu mutants at seen positions\n", test.size());
    ClassificationOptions bins;
    bins.bins = 10;
    const ClassificationReport one_test = EvaluateClassifier(test_members.front(), y_test, bins);
    const ClassificationReport mc_report = EvaluateClassifier(mc_test.mean, y_test, bins);
    const ClassificationReport ens_report = EvaluateClassifier(ens_test.mean, y_test, bins);
    PrintMetrics("one network", one_test);
    PrintMetrics(("MC dropout (" + std::to_string(mc_samples) + ")").c_str(), mc_report);
    PrintMetrics(("ensemble of " + std::to_string(o.members)).c_str(), ens_report);
    const ClassificationReport ens_unseen_report = EvaluateClassifier(ens_unseen.mean, y_unseen, bins);
    std::printf("  at unseen positions (%zu mutants):\n", unseen.size());
    PrintMetrics(("ensemble of " + std::to_string(o.members)).c_str(), ens_unseen_report);
    auto spearman = [&](const std::vector<size_t>& idx, const std::vector<double>& pred) {
        return SpearmanCorrelation(pred, Gather(assay.scores, idx));
    };
    std::vector<std::vector<double>> reg_means_test, reg_vars_test, reg_means_unseen, reg_vars_unseen, reg_means_cal, reg_vars_cal;
    GaussianNLLLoss variance_of(&cpu);
    auto regress = [&](Head& h, const std::vector<size_t>& idx, std::vector<std::vector<double>>& means, std::vector<std::vector<double>>& vars) {
        const std::vector<float> out = h.net.forward(Rows(x, d, idx, &cpu)).to_host_vector();
        std::vector<double> mu, var;
        for (size_t i = 0; i < idx.size(); ++i) {
            mu.push_back(out[2 * i] * score_sd + score_mean);
            var.push_back(variance_of.variance(out[2 * i + 1]) * score_sd * score_sd);
        }
        means.push_back(mu);
        vars.push_back(var);
    };
    for (auto& r : regressors) {
        regress(*r, test, reg_means_test, reg_vars_test);
        regress(*r, unseen, reg_means_unseen, reg_vars_unseen);
        regress(*r, cal, reg_means_cal, reg_vars_cal);
    }
    const GaussianMixturePrediction g_test = CombineGaussians(reg_means_test, reg_vars_test);
    const GaussianMixturePrediction g_unseen = CombineGaussians(reg_means_unseen, reg_vars_unseen);
    const GaussianMixturePrediction g_cal = CombineGaussians(reg_means_cal, reg_vars_cal);
    std::printf("  Spearman with the measured fitness:   seen positions   unseen positions\n");
    std::printf("    zero-shot ESM-2 score               %6.3f           %6.3f\n", spearman(test, Gather(zero_shot, test)),
                spearman(unseen, Gather(zero_shot, unseen)));
    std::printf("    regression ensemble                 %6.3f           %6.3f\n", spearman(test, g_test.mean), spearman(unseen, g_unseen.mean));
    SvgOptions svg;
    svg.width = 760;
    Save(o.out + "/reliability.svg",
         RenderReliabilityDiagramSvg({{"one network", one_test},
                                      {"MC dropout", mc_report},
                                      {"ensemble of " + std::to_string(o.members), ens_report}},
                                     svg));
    Save(o.out + "/reliability_unseen.svg",
         RenderReliabilityDiagramSvg({{"seen positions", ens_report}, {"unseen positions", ens_unseen_report}}, svg));

    // ---- 2. Uncertainty.
    std::printf("\n2. How sure is it? Mean entropy in nats, seen positions / unseen positions\n");
    auto row = [&](const char* name, const UncertaintyDecomposition& a, const UncertaintyDecomposition& b) {
        std::printf("  %-20s total %.3f / %.3f   aleatoric %.3f / %.3f   epistemic %.4f / %.4f\n", name, Mean(a.total), Mean(b.total),
                    Mean(a.aleatoric), Mean(b.aleatoric), Mean(a.epistemic), Mean(b.epistemic));
    };
    row(("ensemble of " + std::to_string(o.members)).c_str(), ens_test, ens_unseen);
    row(("MC dropout (" + std::to_string(mc_samples) + ")").c_str(), mc_test, mc_unseen);
    std::printf("  regression ensemble  variance: aleatoric %.4f / %.4f   epistemic %.4f / %.4f\n", Mean(g_test.aleatoric),
                Mean(g_unseen.aleatoric), Mean(g_test.epistemic), Mean(g_unseen.epistemic));
    // Is the epistemic part higher at unseen positions? A one-sided permutation test of the two
    // groups' means: PerSliceError on the uncertainties instead of errors.
    auto higher_unseen = [&](const std::vector<double>& a, const std::vector<double>& b, const char* what) {
        std::vector<double> values(a);
        values.insert(values.end(), b.begin(), b.end());
        std::vector<std::string> groups(a.size(), "seen");
        groups.insert(groups.end(), b.size(), "unseen");
        SliceErrorOptions perm;
        perm.test = SliceTest::Permutation;
        perm.seed = 1;
        const SliceErrorReport r = PerSliceError(values, groups, perm);
        for (const SliceError& s : r.slices) {
            if (s.name == "unseen") {
                std::printf("  %s higher at unseen positions? %.4f against %.4f (x%.1f), one-sided p = %.4g\n", what, s.error,
                            s.rest_error, s.rest_error > 0 ? s.error / s.rest_error : 0.0, s.p_value);
            }
        }
    };
    higher_unseen(ens_test.epistemic, ens_unseen.epistemic, "ensemble epistemic:");
    higher_unseen(ens_test.aleatoric, ens_unseen.aleatoric, "ensemble aleatoric:");
    higher_unseen(g_test.epistemic, g_unseen.epistemic, "regression epistemic:");
    // Does it know when it's wrong? Accuracy on the most certain test predictions.
    std::vector<size_t> by_certainty(test.size());
    std::iota(by_certainty.begin(), by_certainty.end(), size_t{0});
    std::stable_sort(by_certainty.begin(), by_certainty.end(), [&](size_t a, size_t b) { return ens_test.total[a] < ens_test.total[b]; });
    std::printf("  accuracy of the most certain test predictions (ensemble, by total entropy):");
    for (double keep : {0.25, 0.5, 0.75, 1.0}) {
        const size_t k = std::max<size_t>(1, static_cast<size_t>(keep * static_cast<double>(test.size())));
        int64_t right = 0;
        for (size_t i = 0; i < k; ++i) right += ens_test.predicted[by_certainty[i]] == y_test[by_certainty[i]] ? 1 : 0;
        std::printf("  %.0f%%: %.3f", keep * 100, static_cast<double>(right) / static_cast<double>(k));
    }
    std::printf("\n");

    // ---- 3. Conformal prediction.
    const double alpha = 0.1;
    std::printf("\n3. What can it promise? Conformal sets at alpha = %.2f, calibrated on %zu mutants\n", alpha, cal.size());
    const Tensor cal_probs = DecomposeUncertainty(members_of(cal)).mean;
    // LAC gives the smallest sets; APS, randomized, sets that grow on hard inputs. The slices below
    // use LAC's.
    std::vector<std::vector<int64_t>> lac_test_sets, lac_unseen_sets;
    CoverageReport lac_seen, lac_unseen;
    for (ConformalScore score : {ConformalScore::Lac, ConformalScore::Aps}) {
        ConformalOptions co;
        co.alpha = alpha;
        co.score = score;
        co.randomized = score == ConformalScore::Aps;
        const ConformalClassifier c = CalibrateConformalClassifier(cal_probs, y_cal, co);
        const std::vector<std::vector<int64_t>> test_sets = PredictionSets(c, ens_test.mean);
        const std::vector<std::vector<int64_t>> unseen_sets = PredictionSets(c, ens_unseen.mean);
        const CoverageReport seen_cov = EvaluateCoverage(test_sets, y_test);
        const CoverageReport unseen_cov = EvaluateCoverage(unseen_sets, y_unseen);
        std::printf("  %s  q = %.3f   seen positions: coverage %.3f, mean size %.2f (%lld empty, %lld of 1, %lld of 2)\n",
                    score == ConformalScore::Lac ? "LAC" : "APS", c.quantile, seen_cov.coverage, seen_cov.mean_size,
                    static_cast<long long>(seen_cov.empty), static_cast<long long>(seen_cov.size_counts.size() > 1 ? seen_cov.size_counts[1] : 0),
                    static_cast<long long>(seen_cov.size_counts.size() > 2 ? seen_cov.size_counts[2] : 0));
        std::printf("                 unseen positions: coverage %.3f, mean size %.2f\n", unseen_cov.coverage, unseen_cov.mean_size);
        if (score == ConformalScore::Lac) {
            lac_test_sets = test_sets;
            lac_unseen_sets = unseen_sets;
            lac_seen = seen_cov;
            lac_unseen = unseen_cov;
        }
    }
    const ConformalRegressor reg = CalibrateConformalRegressor(g_cal.mean, Gather(assay.scores, cal), alpha);
    const CoverageReport int_seen = EvaluateCoverage(PredictionIntervals(reg, g_test.mean), Gather(assay.scores, test));
    const CoverageReport int_unseen = EvaluateCoverage(PredictionIntervals(reg, g_unseen.mean), Gather(assay.scores, unseen));
    std::printf("  intervals, fitness +- %.3f: coverage %.3f at seen positions, %.3f at unseen ones\n", reg.quantile, int_seen.coverage,
                int_unseen.coverage);

    // Coverage by slice: seen against unseen positions, and by how buried the residue is.
    SliceErrorViewOptions miss_view;
    miss_view.metric = "miss rate (1 - coverage)";
    miss_view.reference = alpha;
    miss_view.reference_label = "alpha";
    auto by_split = [&](const CoverageReport& a, const CoverageReport& b, const char* what, const std::string& file) {
        std::vector<double> misses(a.misses);
        misses.insert(misses.end(), b.misses.begin(), b.misses.end());
        std::vector<std::string> groups(a.misses.size(), "seen positions");
        groups.insert(groups.end(), b.misses.size(), "unseen positions");
        const SliceErrorReport r = PerSliceError(misses, groups);
        std::printf("\n  %s, seen against unseen positions:\n%s", what, DescribeSliceError(r).c_str());
        Save(o.out + "/" + file, RenderSliceErrorSvg(r, miss_view, svg));
    };
    by_split(lac_seen, lac_unseen, "LAC set misses", "coverage_by_split.svg");
    by_split(int_seen, int_unseen, "Interval misses", "interval_coverage_by_split.svg");
    const std::vector<int> neighbors = NeighborCounts(chain, start, seq.size());
    std::vector<int> known;
    for (int c : neighbors) {
        if (c >= 0) known.push_back(c);
    }
    std::sort(known.begin(), known.end());
    const int low = known[known.size() / 3], high = known[2 * known.size() / 3];
    auto burial = [&](size_t position) -> std::string {
        const int c = neighbors[position];
        if (c < 0) return "not in the crystal";
        return c >= high ? "buried" : c >= low ? "intermediate" : "exposed";
    };
    {
        std::vector<std::string> groups;
        for (size_t i : test) groups.push_back(burial(assay.positions[i]));
        const SliceErrorReport cov = PerSliceError(lac_seen.misses, groups);
        std::printf("\n  LAC misses on the test set by burial (Cb neighbors within 10 A: exposed < %d <= intermediate < %d <= buried):\n%s",
                    low, high, DescribeSliceError(cov).c_str());
        Save(o.out + "/coverage_by_burial.svg", RenderSliceErrorSvg(cov, miss_view, svg));
        const SliceErrorReport err = PerSliceError(ens_test.mean, y_test, groups);
        std::printf("\n  Ensemble errors on the test set by burial:\n%s", DescribeSliceError(err).c_str());
        SliceErrorViewOptions err_view;
        err_view.metric = "error rate";
        Save(o.out + "/error_by_burial.svg", RenderSliceErrorSvg(err, err_view, svg));
    }

    // ---- 4. Where on the protein.
    std::printf("\n4. Where on the protein is it unsure?\n");
    const size_t len = seq.size();
    const float nan = std::nanf("");
    std::vector<double> epi_sum(len, 0), err_sum(len, 0), size_sum(len, 0);
    std::vector<int> count(len, 0);
    auto accumulate = [&](const std::vector<size_t>& idx, const UncertaintyDecomposition& u,
                          const std::vector<std::vector<int64_t>>& sets, const std::vector<int64_t>& labels) {
        for (size_t k = 0; k < idx.size(); ++k) {
            const size_t p = assay.positions[idx[k]];
            epi_sum[p] += u.epistemic[k];
            err_sum[p] += u.predicted[k] != labels[k] ? 1 : 0;
            size_sum[p] += static_cast<double>(sets[k].size());
            ++count[p];
        }
    };
    accumulate(test, ens_test, lac_test_sets, y_test);
    accumulate(unseen, ens_unseen, lac_unseen_sets, y_unseen);
    ResidueTracksDocument tracks;
    tracks.title = model_name + " head on " + (o.smoke ? std::string("1CRN") : std::string("TEM-1")) + ": where it is unsure";
    tracks.sequence = mature;
    tracks.first_position = static_cast<int64_t>(start) + 1;
    tracks.residue_ids = ResidueIds(chain);
    auto track = [&](const std::vector<double>& sum, double scale) {
        std::vector<float> v;
        for (size_t i = start; i < start + mature.size(); ++i) v.push_back(count[i] > 0 ? static_cast<float>(sum[i] / count[i] * scale) : nan);
        return v;
    };
    tracks.tracks.push_back({"epistemic (nats)", track(epi_sum, 1.0), false});
    tracks.tracks.push_back({"error rate", track(err_sum, 1.0), false});
    tracks.tracks.push_back({"mean LAC set size", track(size_sum, 1.0), false});
    std::vector<float> burial_track;
    for (size_t i = start; i < start + mature.size(); ++i) burial_track.push_back(neighbors[i] >= 0 ? static_cast<float>(neighbors[i]) : nan);
    tracks.tracks.push_back({"burial (neighbors)", burial_track, false});
    const int64_t first = static_cast<int64_t>(std::max(unseen_first, start)) + 1;
    const int64_t last = static_cast<int64_t>(std::min(unseen_last, start + mature.size() - 1)) + 1;
    if (first <= last) tracks.features.push_back({"held out", first, last, "split"});
    SvgOptions wide;
    wide.width = 900;
    Save(o.out + "/residue_tracks.svg", RenderResidueTracksSvg(tracks, wide));
    HtmlOptions page;
    page.width = 900;
    Save(o.out + "/structure.html", RenderStructureHtml(structure_text, "A", tracks, page));
    Save(o.out + "/classification_report.json", WriteJson(ToJson(ens_report)));
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: how_sure_is_the_model_recipe MODEL_DIR PROTEINGYM_DIR STRUCTURE.cif OUT_DIR [--device cpu|hip] [--members N] "
                     "[--smoke]\n");
        return 2;
    }
    Options o;
    o.model_dir = argv[1];
    o.gym_dir = argv[2];
    o.structure = argv[3];
    o.out = argv[4];
    for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) {
            o.hip = std::string(argv[++i]) == "hip";
        } else if (a == "--members" && i + 1 < argc) {
            o.members = std::max(2, std::atoi(argv[++i]));
        } else if (a == "--smoke") {
            o.smoke = true;
        } else {
            std::fprintf(stderr, "how_sure_is_the_model_recipe: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (o.smoke) o.members = std::min(o.members, 3);
    try {
        if (o.hip) {
#ifdef PULSATRIX_DEMO_WITH_HIP
            HIPBackend gpu;
            return Run(&gpu, o);
#else
            std::fprintf(stderr, "this build has no HIP backend\n");
            return 2;
#endif
        }
        CPUBackend cpu;
        return Run(&cpu, o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "how_sure_is_the_model_recipe: %s\n", e.what());
        return 1;
    }
}
