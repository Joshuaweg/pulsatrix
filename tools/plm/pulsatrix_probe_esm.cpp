// pulsatrix_probe_esm: what an ESM-2 model's residue representations encode (roadmap PLM-8):
// per-layer linear probes with control tasks, and an InterPLM-style sparse autoencoder whose
// features are matched to Swiss-Prot annotations.
//
//   pulsatrix_probe_esm esm2_t6_8M_UR50D --annotations annotated.jsonl --probes --out probes/
//   pulsatrix_probe_esm esm2_t6_8M_UR50D --annotations annotated.jsonl --sae --sae-layer 4 --out sae/ \
//       [--structures DIR]
//
// annotated.jsonl comes from tools/plm/fetch_swissprot_annotations.py. Proteins are split 80/20
// at random, whole proteins on each side.
// --probes: for each layer in --layers (default every layer), a softmax probe for three-state
//   secondary structure and a binary probe for each of --concepts, scored on the held-out
//   proteins (balanced accuracy, or ROC AUC for binary concepts), beside two controls: the same
//   probe on the residues' local sequence (one-hot, +-3 residues) and on the same layer of a
//   randomly initialized model. Uses --probe-proteins proteins (default 600).
// --sae: trains a featurizer on layer --sae-layer's residues of --sae-proteins proteins (default
//   2000), each dimension standardized by the training residues, with --features latents (default
//   8 x hidden), --sae-epochs (default 10) and --sae-batch (default 512). --featurizer picks it:
//   - l1: a SparseAutoencoder (FEAT-1), --l1 (default 0.003);
//   - topk: a TopKSparseAutoencoder (FEAT-2), --k (default 32);
//   - batchtopk, matryoshka: BatchTopK, and Matryoshka BatchTopK with prefixes of 1/16 and 1/4
//     of the latents (FEAT-4), --k;
//   - jumprelu: a JumpReLUSparseAutoencoder (FEAT-4), --l0 (default 0.05);
//   - bsf-vanilla, bsf-grassmannian, bsf-lasso: a BlockSparseFeaturizer (FEAT-6) with blocks of
//     --block-size (default 4) and --features / block-size blocks; --k is the active blocks
//     (bsf-lasso's target L0). Concepts are matched on block norms, and absorption is skipped;
//   - transcoder, skip-transcoder: a Transcoder (FEAT-5), --k, trained on the MLP that writes
//     into layer --sae-layer's output: its input (after the block's LayerNorm) to its output.
//     Reports prediction quality, loss recovered with the MLP replaced, and concept matching
//     (absorption is an autoencoder metric). An SAE trained the same way on a randomly initialized model is the
//   baseline for everything that follows (FEAT-3):
//   - reconstruction on held-out residues: explained variance, cosine, norm ratio, L0, dead and
//     dense latents;
//   - loss recovered: the masked-LM loss of --loss-proteins held-out proteins (default 100) with
//     the reconstructions spliced in at the layer, against the clean and zero-ablated losses;
//   - concept matching: every feature against every concept (best F1 over thresholds), beside
//     single neurons;
//   - the description length (MDL, bits per input at 10% distortion) and the mean stable rank of
//     a feature's codes, on up to 20000 held-out residues;
//   - feature absorption per concept, with its linear-probe baseline; printed only where the
//     probe's F1 is at least 0.5 (the CSV has every concept).
//   Writes a feature dashboard per concept's best feature; with --structures DIR holding
//   AF-<accession>-F1-model_v4.cif files, each gets a structure panel.
// Options: --device cpu|hip, --seed S.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/block_sparse_featurizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/featurizer_metrics.hpp"
#include "pulsatrix/jumprelu_sparse_autoencoder.hpp"
#include "pulsatrix/protein_concepts.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_training.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"
#include "pulsatrix/topk_sparse_autoencoder.hpp"
#include "pulsatrix/transcoder.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/protein_views.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

using namespace pulsatrix;

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_probe_esm: " << problem << "\n"
              << "usage: pulsatrix_probe_esm MODEL_DIR --annotations FILE.jsonl --out DIR (--probes | --sae)\n"
              << "       [--layers 0,1,...] [--concepts A,B,...] [--probe-proteins N] [--sae-layer L] [--sae-proteins N]\n"
              << "       [--featurizer l1|topk|batchtopk|matryoshka|jumprelu|transcoder|skip-transcoder] [--features N] [--l1 L]\n"
              << "       [--k K] [--l0 L] [--block-size B]  (also bsf-vanilla|bsf-grassmannian|bsf-lasso)\n"
              << "       [--sae-epochs N] [--sae-batch N]\n"
              << "       [--loss-proteins N] [--structures DIR] [--seed S] [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string model, annotations, out, structures;
    std::string device = "cpu", featurizer = "l1";
    bool probes = false, sae = false;
    std::vector<int64_t> layers;
    std::vector<std::string> concepts = {"Binding site", "Active site", "Disulfide bond", "Transmembrane", "Signal", "Zinc finger",
                                         "Coiled coil", "Motif"};
    int64_t probe_proteins = 600, sae_proteins = 2000, sae_layer = -1, features = 0, sae_epochs = 10, sae_batch = 512, k = 32,
            loss_proteins = 100, block_size = 4;
    float l1 = 0.003f, l0 = 0.05f;
    uint64_t seed = 0;
};

std::vector<std::string> Split(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    for (std::string item; std::getline(ss, item, ',');) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

/** @brief The first @p n proteins after a seeded shuffle, and which of them are held out. */
struct Sample {
    std::vector<AnnotatedProtein> proteins;
    std::vector<bool> test;  ///< per protein
};

Sample Choose(std::vector<AnnotatedProtein> all, int64_t n, uint64_t seed) {
    std::shuffle(all.begin(), all.end(), std::mt19937_64(seed));
    if (static_cast<int64_t>(all.size()) > n) all.resize(static_cast<size_t>(n));
    Sample s;
    s.proteins = std::move(all);
    for (size_t i = 0; i < s.proteins.size(); ++i) s.test.push_back(i % 5 == 4);  // every fifth: 80/20
    return s;
}

std::vector<std::string> Sequences(const Sample& s) {
    std::vector<std::string> out;
    for (const auto& p : s.proteins) out.push_back(p.sequence);
    return out;
}

/** @brief Per-residue labels for a concept ("Secondary structure" is three-state), concatenated. */
std::vector<int> Labels(const Sample& s, const std::string& concept_name) {
    std::vector<int> out;
    for (const auto& p : s.proteins) {
        const std::vector<int> l = concept_name == "Secondary structure" ? SecondaryStructureLabels(p) : ConceptLabels(p, concept_name);
        out.insert(out.end(), l.begin(), l.end());
    }
    return out;
}

std::vector<bool> ResidueFlags(const Sample& s, bool test) {
    std::vector<bool> out;
    for (size_t i = 0; i < s.proteins.size(); ++i) out.insert(out.end(), s.proteins[i].sequence.size(), s.test[i] == test);
    return out;
}

std::unique_ptr<EncoderLM> RandomModel(const std::string& model_dir, DeviceBackend* backend, uint64_t seed) {
    auto m = std::make_unique<EncoderLM>(ReadEsmConfig(model_dir + "/config.json"), backend);
    InitializeEsm(*m, seed);
    return m;
}

void RunProbes(DeviceBackend* backend, EncoderLM& model, const TextTokenizer& tok, const Options& o, const Sample& s) {
    std::vector<int64_t> layers = o.layers;
    if (layers.empty()) {
        for (int64_t l = 0; l <= model.num_layers(); ++l) layers.push_back(l);
    }
    std::fprintf(stderr, "embedding %zu proteins with the trained and a random model\n", s.proteins.size());
    const ResidueEmbeddings trained = EmbedResidues(model, tok, backend, Sequences(s), layers);
    std::unique_ptr<EncoderLM> random = RandomModel(o.model, backend, o.seed + 7);
    const ResidueEmbeddings untrained = EmbedResidues(*random, tok, backend, Sequences(s), layers);
    const std::vector<bool> train = ResidueFlags(s, false), test = ResidueFlags(s, true);
    const std::vector<float> window = SequenceWindowFeatures(Sequences(s), 3);
    const int64_t window_dim = 20 * 7;

    std::vector<std::string> concepts = {"Secondary structure"};
    concepts.insert(concepts.end(), o.concepts.begin(), o.concepts.end());
    std::ofstream csv(o.out + "/probes.csv");
    csv << "concept,layer,metric,probe,sequence_window,gain_over_window,random_model,positives\n";
    for (const std::string& c : concepts) {
        const std::vector<int> labels = Labels(s, c);
        const int classes = c == "Secondary structure" ? 3 : 2;
        const int64_t positives = std::count_if(labels.begin(), labels.end(), [](int y) { return y == 1; });
        if (classes == 2 && positives < 50) {
            std::printf("%s: only %ld positive residues, skipped\n", c.c_str(), static_cast<long>(positives));
            continue;
        }
        const char* metric = classes == 2 ? "AUC" : "balanced accuracy";
        ProbeOptions po;
        po.seed = o.seed;
        po.epochs = 10;
        auto score = [&](const std::vector<float>& features, int64_t dim) {
            const ProbeResult r = TrainLinearProbe(features, dim, labels, classes, train, test, backend, po);
            return classes == 2 ? r.auc : r.balanced_accuracy;
        };
        const double ctl = score(window, window_dim);
        std::printf("\n%s (%s; %ld residues, %ld positive; local sequence +-3: %.3f)\n  layer  probe  gain over sequence  random model\n",
                    c.c_str(), metric, static_cast<long>(labels.size()), static_cast<long>(positives), ctl);
        for (size_t k = 0; k < layers.size(); ++k) {
            const double task = score(trained.values[k], trained.hidden), rnd = score(untrained.values[k], untrained.hidden);
            std::printf("  %5ld  %.3f              %+.3f         %.3f\n", static_cast<long>(layers[k]), task, task - ctl, rnd);
            std::fflush(stdout);
            csv << '"' << c << "\"," << layers[k] << ',' << metric << ',' << task << ',' << ctl << ',' << task - ctl << ',' << rnd << ','
                << positives << '\n';
        }
    }
    std::printf("\nwrote %s/probes.csv\n", o.out.c_str());
}

/** @brief Trains an SAE on rows, `residues x hidden`, with Adam in shuffled batches. */
std::unique_ptr<Featurizer> TrainSae(DeviceBackend* backend, const std::vector<float>& rows, int64_t hidden, const Options& o,
                                     const char* name) {
    const int64_t m = o.features > 0 ? o.features : 8 * hidden;
    const auto N = static_cast<int64_t>(rows.size() / static_cast<size_t>(hidden));
    std::unique_ptr<Featurizer> sae;
    if (o.featurizer.rfind("bsf-", 0) == 0) {
        BlockSparseOptions b;
        b.variant = o.featurizer == "bsf-grassmannian" ? BsfVariant::Grassmannian
                    : o.featurizer == "bsf-lasso"      ? BsfVariant::GroupLasso
                                                       : BsfVariant::Vanilla;
        b.block_size = o.block_size;
        b.k = o.k;
        b.target_l0 = static_cast<double>(o.k);
        b.seed = o.seed + 11;
        sae = std::make_unique<BlockSparseFeaturizer>(hidden, m / o.block_size, backend, b);
    } else if (o.featurizer == "jumprelu") {
        JumpReLUSaeOptions j;
        j.l0_coefficient = o.l0;
        j.bandwidth = 0.05f;
        j.initial_threshold = 0.1f;
        j.seed = o.seed + 11;
        auto jump = std::make_unique<JumpReLUSparseAutoencoder>(hidden, m, backend, j);
        const int64_t sample = std::min<int64_t>(N, 65536);
        jump->initialize_bias(Tensor(Shape({sample, hidden}), backend, std::vector<float>(rows.begin(), rows.begin() + sample * hidden),
                                     backend->device()));
        sae = std::move(jump);
    } else if (o.featurizer != "l1") {
        TopKSaeOptions t;
        t.k = o.k;
        t.dead_after = N;  // a latent silent for a whole epoch is dead
        t.seed = o.seed + 11;
        t.batch_topk = o.featurizer == "batchtopk" || o.featurizer == "matryoshka";
        if (o.featurizer == "matryoshka") t.matryoshka_prefixes = {m / 16, m / 4};
        auto topk = std::make_unique<TopKSparseAutoencoder>(hidden, m, backend, t);
        const int64_t sample = std::min<int64_t>(N, 65536);
        topk->initialize_bias(Tensor(Shape({sample, hidden}), backend, std::vector<float>(rows.begin(), rows.begin() + sample * hidden),
                                     backend->device()));
        sae = std::move(topk);
    } else {
        sae = std::make_unique<SparseAutoencoder>(hidden, m, o.l1, backend, static_cast<unsigned>(o.seed + 11));
    }
    AdamOptimizer opt(1e-3f, backend);
    std::vector<int64_t> order(static_cast<size_t>(N));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::mt19937_64 rng(o.seed + 5);
    FeatureActivityTracker activity(m);
    const int64_t kBatch = o.sae_batch;
    std::vector<float> x;
    for (int64_t epoch = 1; epoch <= o.sae_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        FeaturizerLoss last;
        for (int64_t from = 0; from + kBatch <= N; from += kBatch) {
            x.resize(static_cast<size_t>(kBatch * hidden));
            for (int64_t b = 0; b < kBatch; ++b) {
                std::copy_n(rows.begin() + order[static_cast<size_t>(from + b)] * hidden, hidden, x.begin() + b * hidden);
            }
            last = TrainFeaturizer(*sae, Tensor(Shape({kBatch, hidden}), backend, x, backend->device()), opt, true, &activity);
        }
        std::fprintf(stderr, "  %s SAE epoch %ld: reconstruction %.4f, sparsity %.4f, dead %.1f%%\n", name, static_cast<long>(epoch),
                     last.reconstruction, last.sparsity, 100.0 * activity.dead_fraction(N));
    }
    return sae;
}

/** @brief Per-dimension standardization: `(x - mean) / sd`. */
struct Scale {
    std::vector<double> mean, sd;
};

/** @brief A featurizer trained on standardized rows, seen from the model's raw hidden states:
 *         standardizes before encoding and undoes it after decoding, so it can be spliced in. */
class Standardized : public Featurizer {
public:
    Standardized(Featurizer& inner, Scale scale) : Standardized(inner, scale, scale) {}
    /** @brief A transcoder: inputs and outputs standardized separately. */
    Standardized(Featurizer& inner, Scale in, Scale out) : inner_(inner), in_(std::move(in)), out_(std::move(out)) {}
    [[nodiscard]] int64_t input_dim() const override { return inner_.input_dim(); }
    [[nodiscard]] int64_t output_dim() const override { return inner_.output_dim(); }
    [[nodiscard]] int64_t num_features() const override { return inner_.num_features(); }
    [[nodiscard]] Tensor encode(const Tensor& x) override { return inner_.encode(In(x)); }
    [[nodiscard]] Tensor decode(const Tensor& codes) override { return Out(inner_.decode(codes)); }
    [[nodiscard]] Tensor predict(const Tensor& x) override { return Out(inner_.predict(In(x))); }
    FeaturizerLoss loss_and_backward(const Tensor&, std::vector<float>*) override { throw std::logic_error("Standardized: evaluation only"); }
    void normalize_decoder() override {}
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i) override { return inner_.decoder_direction(i); }
    [[nodiscard]] Module& parameters_module() override { return inner_.parameters_module(); }

private:
    Tensor In(const Tensor& x) const {
        std::vector<float> v = x.to_host_vector();
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>((v[i] - in_.mean[i % in_.mean.size()]) / in_.sd[i % in_.sd.size()]);
        return Tensor(x.shape(), x.backend(), v, x.device());
    }
    Tensor Out(const Tensor& y) const {
        std::vector<float> v = y.to_host_vector();
        for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(v[i] * out_.sd[i % out_.sd.size()] + out_.mean[i % out_.mean.size()]);
        return Tensor(y.shape(), y.backend(), v, y.device());
    }
    Featurizer& inner_;
    Scale in_, out_;
};

/** @brief Loss recovered over held-out proteins: each protein's masked-LM loss, weighted by its
 *         masked tokens, with the reconstructions spliced into its residues (not <cls> or <eos>). */
LossRecovered SplicedMaskedLM(EncoderLM& model, const TextTokenizer& tok, DeviceBackend* backend, Featurizer& f, int64_t layer,
                              const std::vector<std::string>& sequences, uint64_t seed) {
    LossRecovered total;
    int64_t tokens = 0;
    for (size_t i = 0; i < sequences.size(); ++i) {
        const std::string& seq = sequences[i];
        std::vector<bool> rows(seq.size() + 2, true);
        rows.front() = rows.back() = false;
        int64_t n = 0;
        auto loss = [&](const HiddenStateHook& hook) {
            model.set_hidden_state_hook(hook);
            const MaskedLMEvaluation e = EvaluateMaskedLM(model, tok, backend, {seq}, 1, seed + i);
            model.set_hidden_state_hook({});
            n = e.tokens;
            return e.loss;
        };
        const LossRecovered r = MeasureLossRecovered(f, layer, loss, rows);
        if (n == 0) continue;
        total.clean += r.clean * static_cast<double>(n);
        total.spliced += r.spliced * static_cast<double>(n);
        total.ablated += r.ablated * static_cast<double>(n);
        tokens += n;
    }
    if (tokens == 0) throw std::runtime_error("loss recovered: no masked tokens");
    total.clean /= static_cast<double>(tokens);
    total.spliced /= static_cast<double>(tokens);
    total.ablated /= static_cast<double>(tokens);
    total.recovered = total.ablated > total.clean ? (total.ablated - total.spliced) / (total.ablated - total.clean)
                                                  : std::numeric_limits<double>::quiet_NaN();
    return total;
}

/** @brief Keeps the held-out residues' rows and labels. */
std::vector<float> HeldOutRows(const ResidueEmbeddings& e, const std::vector<bool>& test) {
    std::vector<float> out;
    for (int64_t r = 0; r < e.residues; ++r) {
        if (test[static_cast<size_t>(r)]) out.insert(out.end(), e.row(0, r), e.row(0, r) + e.hidden);
    }
    return out;
}

/** @brief Sparse codes for concept matching: EncodeSparse for single-direction featurizers, and
 *         each block's norm for a block-sparse one (the paper's "presence"). */
SparseCodes FeatureCodes(Featurizer& f, const std::vector<float>& rows, DeviceBackend* backend) {
    if (f.block_size() == 1) return EncodeSparse(f, rows, backend);
    const int64_t d = f.input_dim(), m = f.num_features(), b = f.block_size(), G = m / b;
    const auto N = static_cast<int64_t>(rows.size() / static_cast<size_t>(d));
    SparseCodes out;
    out.num_features = G;
    out.max_value.assign(static_cast<size_t>(G), 0.0f);
    out.row_start.push_back(0);
    for (int64_t r0 = 0; r0 < N; r0 += 4096) {
        const int64_t n = std::min<int64_t>(4096, N - r0);
        const std::vector<float> c =
            f.encode(Tensor(Shape({n, d}), backend, std::vector<float>(rows.begin() + r0 * d, rows.begin() + (r0 + n) * d), backend->device())).to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t g = 0; g < G; ++g) {
                double s = 0;
                for (int64_t t = 0; t < b; ++t) s += static_cast<double>(c[static_cast<size_t>(r * m + g * b + t)]) * c[static_cast<size_t>(r * m + g * b + t)];
                if (s == 0) continue;
                const auto v = static_cast<float>(std::sqrt(s));
                out.feature.push_back(static_cast<int32_t>(g));
                out.value.push_back(v);
                out.max_value[static_cast<size_t>(g)] = std::max(out.max_value[static_cast<size_t>(g)], v);
            }
            out.row_start.push_back(static_cast<int64_t>(out.feature.size()));
        }
    }
    return out;
}

/** @brief Each residue's MLP input and output in block @p block, `residues x hidden` each. */
struct MlpPairs {
    std::vector<float> in, out;
};

MlpPairs CollectMlpPairs(EncoderLM& model, const TextTokenizer& tok, DeviceBackend* backend, const std::vector<std::string>& sequences,
                         int64_t block) {
    const int64_t h = model.config().hidden_size;
    const auto cls = tok.token_to_id("<cls>"), eos = tok.token_to_id("<eos>");
    MlpPairs pairs;
    model.layer(block).set_mlp_hook([&](const Tensor& in, const Tensor& out) {
        const std::vector<float> a = in.to_host_vector(), b = out.to_host_vector();
        pairs.in.insert(pairs.in.end(), a.begin() + h, a.end() - h);  // drop <cls> and <eos>
        pairs.out.insert(pairs.out.end(), b.begin() + h, b.end() - h);
        return out;
    });
    const bool keep = model.keep_activations();
    model.set_keep_activations(false);
    model.clear_padding_mask();
    for (const std::string& s : sequences) {
        std::vector<float> ids{static_cast<float>(*cls)};
        for (char c : s) ids.push_back(static_cast<float>(*tok.token_to_id(std::string(1, c))));
        ids.push_back(static_cast<float>(*eos));
        (void)model.forward(Tensor(Shape({1, static_cast<int64_t>(ids.size())}), backend, ids, backend->device()));
    }
    model.set_keep_activations(keep);
    model.layer(block).set_mlp_hook({});
    return pairs;
}

/** @brief Per-dimension mean and standard deviation of the rows marked in @p use. */
Scale FitScale(const std::vector<float>& rows, int64_t d, const std::vector<bool>& use) {
    Scale s{std::vector<double>(static_cast<size_t>(d), 0.0), std::vector<double>(static_cast<size_t>(d), 0.0)};
    double n = 0;
    for (size_t r = 0; r < use.size(); ++r) {
        if (!use[r]) continue;
        n += 1;
        for (int64_t j = 0; j < d; ++j) s.mean[static_cast<size_t>(j)] += rows[r * static_cast<size_t>(d) + static_cast<size_t>(j)];
    }
    for (double& m : s.mean) m /= n;
    for (size_t r = 0; r < use.size(); ++r) {
        if (!use[r]) continue;
        for (int64_t j = 0; j < d; ++j) {
            const double v = rows[r * static_cast<size_t>(d) + static_cast<size_t>(j)] - s.mean[static_cast<size_t>(j)];
            s.sd[static_cast<size_t>(j)] += v * v;
        }
    }
    for (double& v : s.sd) v = std::sqrt(v / n) + 1e-6;
    return s;
}

/** @brief The rows marked in @p use, standardized by @p s. */
std::vector<float> Pick(const std::vector<float>& rows, int64_t d, const std::vector<bool>& use, const Scale& s) {
    std::vector<float> out;
    for (size_t r = 0; r < use.size(); ++r) {
        if (!use[r]) continue;
        for (int64_t j = 0; j < d; ++j) {
            out.push_back(static_cast<float>((rows[r * static_cast<size_t>(d) + static_cast<size_t>(j)] - s.mean[static_cast<size_t>(j)]) /
                                             s.sd[static_cast<size_t>(j)]));
        }
    }
    return out;
}

std::unique_ptr<Transcoder> TrainTranscoder(DeviceBackend* backend, const std::vector<float>& x, const std::vector<float>& y, int64_t hidden,
                                            const Options& o, const char* name) {
    const int64_t m = o.features > 0 ? o.features : 8 * hidden;
    const auto N = static_cast<int64_t>(x.size() / static_cast<size_t>(hidden));
    TranscoderOptions t;
    t.k = o.k;
    t.skip = o.featurizer == "skip-transcoder";
    t.dead_after = N;
    t.seed = o.seed + 11;
    auto tc = std::make_unique<Transcoder>(hidden, hidden, m, backend, t);
    tc->initialize_bias(Tensor(Shape({N, hidden}), backend, y, backend->device()));
    AdamOptimizer opt(1e-3f, backend);
    std::vector<int64_t> order(static_cast<size_t>(N));
    std::iota(order.begin(), order.end(), int64_t{0});
    std::mt19937_64 rng(o.seed + 5);
    const int64_t B = o.sae_batch;
    std::vector<float> xb(static_cast<size_t>(B * hidden)), yb(static_cast<size_t>(B * hidden));
    for (int64_t epoch = 1; epoch <= o.sae_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        FeaturizerLoss last;
        for (int64_t from = 0; from + B <= N; from += B) {
            for (int64_t b = 0; b < B; ++b) {
                const int64_t r = order[static_cast<size_t>(from + b)];
                std::copy_n(x.begin() + r * hidden, hidden, xb.begin() + b * hidden);
                std::copy_n(y.begin() + r * hidden, hidden, yb.begin() + b * hidden);
            }
            last = TrainFeaturizer(*tc, Tensor(Shape({B, hidden}), backend, xb, backend->device()),
                                   Tensor(Shape({B, hidden}), backend, yb, backend->device()), opt);
        }
        std::fprintf(stderr, "  %s transcoder epoch %ld: prediction %.4f, aux %.4f, dead %zu\n", name, static_cast<long>(epoch), last.reconstruction,
                     last.sparsity, tc->dead_latents().size());
    }
    return tc;
}

/** @brief SplicedMaskedLM for a transcoder replacing block @p block's MLP. */
LossRecovered SplicedMlpMaskedLM(EncoderLM& model, const TextTokenizer& tok, DeviceBackend* backend, Featurizer& f, int64_t block,
                                 const std::vector<std::string>& sequences, uint64_t seed) {
    LossRecovered total;
    int64_t tokens = 0;
    for (size_t i = 0; i < sequences.size(); ++i) {
        std::vector<bool> rows(sequences[i].size() + 2, true);
        rows.front() = rows.back() = false;
        int64_t n = 0;
        auto loss = [&](const MlpHook& hook) {
            model.layer(block).set_mlp_hook(hook);
            const MaskedLMEvaluation e = EvaluateMaskedLM(model, tok, backend, {sequences[i]}, 1, seed + i);
            model.layer(block).set_mlp_hook({});
            n = e.tokens;
            return e.loss;
        };
        const LossRecovered r = MeasureMlpLossRecovered(f, loss, rows);
        if (n == 0) continue;
        total.clean += r.clean * static_cast<double>(n);
        total.spliced += r.spliced * static_cast<double>(n);
        total.ablated += r.ablated * static_cast<double>(n);
        tokens += n;
    }
    if (tokens == 0) throw std::runtime_error("loss recovered: no masked tokens");
    total.clean /= static_cast<double>(tokens);
    total.spliced /= static_cast<double>(tokens);
    total.ablated /= static_cast<double>(tokens);
    total.recovered = total.ablated > total.clean ? (total.ablated - total.spliced) / (total.ablated - total.clean)
                                                  : std::numeric_limits<double>::quiet_NaN();
    return total;
}

void RunTranscoder(DeviceBackend* backend, EncoderLM& model, const TextTokenizer& tok, const Options& o, const Sample& s) {
    const int64_t layer = o.sae_layer >= 0 ? o.sae_layer : model.num_layers() * 2 / 3;
    if (layer < 1 || layer > model.num_layers()) throw std::invalid_argument("a transcoder needs --sae-layer in [1, num_layers]");
    const int64_t block = layer - 1, h = model.config().hidden_size;
    std::fprintf(stderr, "collecting MLP pairs of block %ld for %zu proteins\n", static_cast<long>(block), s.proteins.size());
    const MlpPairs trained = CollectMlpPairs(model, tok, backend, Sequences(s), block);
    std::unique_ptr<EncoderLM> random = RandomModel(o.model, backend, o.seed + 7);
    const MlpPairs untrained = CollectMlpPairs(*random, tok, backend, Sequences(s), block);
    const std::vector<bool> train = ResidueFlags(s, false), test = ResidueFlags(s, true);
    struct Fit {
        Scale in, out;
        std::unique_ptr<Transcoder> t;
        std::vector<float> test_in, test_out;
    };
    auto fit = [&](const MlpPairs& p, const char* name) {
        Fit f;
        f.in = FitScale(p.in, h, train);
        f.out = FitScale(p.out, h, train);
        f.t = TrainTranscoder(backend, Pick(p.in, h, train, f.in), Pick(p.out, h, train, f.out), h, o, name);
        f.test_in = Pick(p.in, h, test, f.in);
        f.test_out = Pick(p.out, h, test, f.out);
        return f;
    };
    Fit a = fit(trained, "trained-model"), b = fit(untrained, "random-model");
    const auto held_rows = static_cast<int64_t>(a.test_in.size() / static_cast<size_t>(h));
    auto t = [&](const std::vector<float>& v) { return Tensor(Shape({held_rows, h}), backend, v, backend->device()); };
    const ReconstructionMetrics ra = EvaluatePrediction(*a.t, t(a.test_in), t(a.test_out)), rb = EvaluatePrediction(*b.t, t(b.test_in), t(b.test_out));
    std::printf("%s on block %ld's MLP (into layer %ld), %ld latents; held-out residues:\n", o.featurizer.c_str(), static_cast<long>(block),
                static_cast<long>(layer), static_cast<long>(a.t->num_features()));
    std::printf("  %-14s %8s %8s %8s %8s %8s %8s\n", "", "EV", "cosine", "|y^|/|y|", "L0", "dead", "dense");
    for (const auto& [label, r] : {std::pair<const char*, const ReconstructionMetrics*>{"trained model", &ra}, {"random model", &rb}}) {
        std::printf("  %-14s %8.3f %8.3f %8.3f %8.1f %7.1f%% %7.1f%%\n", label, r->explained_variance, r->cosine, r->norm_ratio, r->l0,
                    100 * r->dead_fraction, 100 * r->dense_fraction);
    }
    std::vector<std::string> loss_sequences;
    for (size_t i = 0; i < s.proteins.size() && static_cast<int64_t>(loss_sequences.size()) < o.loss_proteins; ++i) {
        if (s.test[i] && s.proteins[i].sequence.size() <= 1022) loss_sequences.push_back(s.proteins[i].sequence);
    }
    Standardized sa(*a.t, a.in, a.out), sb(*b.t, b.in, b.out);
    const LossRecovered la = SplicedMlpMaskedLM(model, tok, backend, sa, block, loss_sequences, o.seed + 13);
    const LossRecovered lb = SplicedMlpMaskedLM(*random, tok, backend, sb, block, loss_sequences, o.seed + 13);
    std::printf("  masked-LM loss (nats), MLP replaced  clean  spliced  zero-ablated  recovered\n");
    std::printf("  %-34s %7.3f %8.3f %13.3f %10.3f\n", "trained model", la.clean, la.spliced, la.ablated, la.recovered);
    std::printf("  %-34s %7.3f %8.3f %13.3f %10.3f\n\n", "random model", lb.clean, lb.spliced, lb.ablated, lb.recovered);

    Sample held;
    for (size_t i = 0; i < s.proteins.size(); ++i) {
        if (s.test[i]) held.proteins.push_back(s.proteins[i]);
    }
    const SparseCodes codes = EncodeSparse(*a.t, a.test_in, backend), codes_random = EncodeSparse(*b.t, b.test_in, backend);
    const SparseCodes neurons = NeuronCodes(a.test_in, h);
    std::vector<std::string> concepts = {"Helix", "Beta strand", "Turn"};
    concepts.insert(concepts.end(), o.concepts.begin(), o.concepts.end());
    std::ofstream csv(o.out + "/concepts.csv");
    csv << "concept,positives,feature,f1,precision,recall,neuron_f1,random_model_f1\n";
    std::printf("%-16s %9s   %7s %6s %6s %6s  %7s  %7s\n", "concept", "positives", "feature", "F1", "prec", "recall", "neuron", "random");
    for (const std::string& c : concepts) {
        const std::vector<int> labels = Labels(held, c);
        const ConceptMatch m = MatchConcept(codes, labels, c), n = MatchConcept(neurons, labels, c), r = MatchConcept(codes_random, labels, c);
        if (m.positives == 0) continue;
        std::printf("%-16s %9ld   %7ld %6.3f %6.3f %6.3f  %7.3f  %7.3f\n", c.c_str(), static_cast<long>(m.positives), static_cast<long>(m.feature),
                    m.f1, m.precision, m.recall, n.f1, r.f1);
        csv << '"' << c << "\"," << m.positives << ',' << m.feature << ',' << m.f1 << ',' << m.precision << ',' << m.recall << ',' << n.f1 << ','
            << r.f1 << '\n';
    }
    std::ofstream metrics(o.out + "/featurizer_metrics.csv");
    metrics << "model,explained_variance,cosine,norm_ratio,l0,dead_fraction,dense_fraction,loss_clean,loss_spliced,loss_ablated,loss_recovered\n";
    for (const auto& [label, r, l] : {std::tuple<const char*, const ReconstructionMetrics*, const LossRecovered*>{"trained", &ra, &la},
                                     {"random", &rb, &lb}}) {
        metrics << label << ',' << r->explained_variance << ',' << r->cosine << ',' << r->norm_ratio << ',' << r->l0 << ',' << r->dead_fraction << ','
                << r->dense_fraction << ',' << l->clean << ',' << l->spliced << ',' << l->ablated << ',' << l->recovered << '\n';
    }
    std::printf("\nwrote %s/concepts.csv and %s/featurizer_metrics.csv\n", o.out.c_str(), o.out.c_str());
}

/** @brief Absorption is measured against the probe's direction; below this F1 the probe hasn't
 *         found the concept, and the rate means nothing (the random model scores as high). */
constexpr double kMinProbeF1 = 0.5;

void RunSae(DeviceBackend* backend, EncoderLM& model, const TextTokenizer& tok, const Options& o, const Sample& s) {
    const int64_t layer = o.sae_layer >= 0 ? o.sae_layer : model.num_layers() * 2 / 3;
    std::fprintf(stderr, "embedding %zu proteins at layer %ld\n", s.proteins.size(), static_cast<long>(layer));
    const ResidueEmbeddings trained = EmbedResidues(model, tok, backend, Sequences(s), {layer});
    std::unique_ptr<EncoderLM> random = RandomModel(o.model, backend, o.seed + 7);
    const ResidueEmbeddings untrained = EmbedResidues(*random, tok, backend, Sequences(s), {layer});
    const std::vector<bool> train = ResidueFlags(s, false), test = ResidueFlags(s, true);
    std::vector<float> train_rows, train_rows_random;
    for (int64_t r = 0; r < trained.residues; ++r) {
        if (!train[static_cast<size_t>(r)]) continue;
        train_rows.insert(train_rows.end(), trained.row(0, r), trained.row(0, r) + trained.hidden);
        train_rows_random.insert(train_rows_random.end(), untrained.row(0, r), untrained.row(0, r) + untrained.hidden);
    }
    std::vector<float> test_rows = HeldOutRows(trained, test), test_rows_random = HeldOutRows(untrained, test);
    // Standardize each dimension by the training residues: ESM's representations have a few very
    // large dimensions, which would otherwise dominate the reconstruction, and the random model's
    // are on another scale altogether.
    auto standardize = [&](std::vector<float>& fit, std::vector<float>& also, int64_t d) -> Scale {
        const size_t n = fit.size() / static_cast<size_t>(d);
        std::vector<double> mean(static_cast<size_t>(d), 0.0), sd(static_cast<size_t>(d), 0.0);
        for (size_t r = 0; r < n; ++r) {
            for (int64_t j = 0; j < d; ++j) mean[static_cast<size_t>(j)] += fit[r * static_cast<size_t>(d) + static_cast<size_t>(j)];
        }
        for (double& m : mean) m /= static_cast<double>(n);
        for (size_t r = 0; r < n; ++r) {
            for (int64_t j = 0; j < d; ++j) {
                const double x = fit[r * static_cast<size_t>(d) + static_cast<size_t>(j)] - mean[static_cast<size_t>(j)];
                sd[static_cast<size_t>(j)] += x * x;
            }
        }
        for (double& v : sd) v = std::sqrt(v / static_cast<double>(n)) + 1e-6;
        for (auto* rows : {&fit, &also}) {
            for (size_t i = 0; i < rows->size(); ++i) {
                const size_t j = i % static_cast<size_t>(d);
                (*rows)[i] = static_cast<float>(((*rows)[i] - mean[j]) / sd[j]);
            }
        }
        return {mean, sd};
    };
    const Scale scale = standardize(train_rows, test_rows, trained.hidden);
    const Scale scale_random = standardize(train_rows_random, test_rows_random, untrained.hidden);
    std::unique_ptr<Featurizer> sae = TrainSae(backend, train_rows, trained.hidden, o, "trained-model");
    std::unique_ptr<Featurizer> sae_random = TrainSae(backend, train_rows_random, untrained.hidden, o, "random-model");

    const SparseCodes codes = FeatureCodes(*sae, test_rows, backend), codes_random = FeatureCodes(*sae_random, test_rows_random, backend);
    const SparseCodes neurons = NeuronCodes(test_rows, trained.hidden);
    // Reconstruction, on held-out residues (standardized, as trained).
    const auto held_rows = static_cast<int64_t>(test_rows.size() / static_cast<size_t>(trained.hidden));
    const ReconstructionMetrics rec = EvaluateReconstruction(*sae, Tensor(Shape({held_rows, trained.hidden}), backend, test_rows, backend->device()));
    const ReconstructionMetrics rec_random =
        EvaluateReconstruction(*sae_random, Tensor(Shape({held_rows, untrained.hidden}), backend, test_rows_random, backend->device()));
    std::printf("%s SAE on layer %ld, %ld latents; held-out residues:\n", o.featurizer.c_str(), static_cast<long>(layer),
                static_cast<long>(codes.num_features));
    std::printf("  %-14s %8s %8s %8s %8s %8s %8s\n", "", "EV", "cosine", "|x^|/|x|", "L0", "dead", "dense");
    for (const auto& [label, r] : {std::pair<const char*, const ReconstructionMetrics*>{"trained model", &rec}, {"random model", &rec_random}}) {
        std::printf("  %-14s %8.3f %8.3f %8.3f %8.1f %7.1f%% %7.1f%%\n", label, r->explained_variance, r->cosine, r->norm_ratio, r->l0,
                    100 * r->dead_fraction, 100 * r->dense_fraction);
    }

    // Description length and code geometry (FEAT-6), on up to 20000 held-out residues.
    {
        const int64_t rows = std::min<int64_t>(held_rows, 20000);
        auto sub = [&](const std::vector<float>& v, int64_t h) {
            return Tensor(Shape({rows, h}), backend, std::vector<float>(v.begin(), v.begin() + rows * h), backend->device());
        };
        const DescriptionLength dl = MeasureDescriptionLength(*sae, sub(test_rows, trained.hidden)),
                                dlr = MeasureDescriptionLength(*sae_random, sub(test_rows_random, untrained.hidden));
        const BlockGeometry geo = MeasureBlockGeometry(*sae, sub(test_rows, trained.hidden));
        std::printf("  bits per input at 10%% distortion: %.1f (support %.1f, code %.1f, residual %.1f, dictionary %.1f); random model %.1f\n",
                    dl.total, dl.support, dl.code, dl.residual, dl.dictionary, dlr.total);
        std::printf("  code dimension per feature: stable rank %.2f, participation ratio %.2f (of %ld)\n", geo.mean_stable_rank,
                    geo.mean_participation_ratio, static_cast<long>(sae->block_size()));
    }

    // Loss recovered: the masked-LM loss with the reconstructions spliced in at the layer.
    std::vector<std::string> loss_sequences;
    for (size_t i = 0; i < s.proteins.size() && static_cast<int64_t>(loss_sequences.size()) < o.loss_proteins; ++i) {
        if (s.test[i] && s.proteins[i].sequence.size() <= 1022) loss_sequences.push_back(s.proteins[i].sequence);
    }
    std::fprintf(stderr, "loss recovered on %zu held-out proteins\n", loss_sequences.size());
    Standardized spliced(*sae, scale), spliced_random(*sae_random, scale_random);
    const LossRecovered lr = SplicedMaskedLM(model, tok, backend, spliced, layer, loss_sequences, o.seed + 13);
    const LossRecovered lr_random = SplicedMaskedLM(*random, tok, backend, spliced_random, layer, loss_sequences, o.seed + 13);
    std::printf("  masked-LM loss (nats)  clean  spliced  zero-ablated  recovered\n");
    std::printf("  %-20s %7.3f %8.3f %13.3f %10.3f\n", "trained model", lr.clean, lr.spliced, lr.ablated, lr.recovered);
    std::printf("  %-20s %7.3f %8.3f %13.3f %10.3f\n\n", "random model", lr_random.clean, lr_random.spliced, lr_random.ablated,
                lr_random.recovered);
    {
        std::ofstream m(o.out + "/featurizer_metrics.csv");
        m << "model,explained_variance,cosine,norm_ratio,l0,dead_fraction,dense_fraction,loss_clean,loss_spliced,loss_ablated,loss_recovered\n";
        for (const auto& [label, r, l] : {std::tuple<const char*, const ReconstructionMetrics*, const LossRecovered*>{"trained", &rec, &lr},
                                         {"random", &rec_random, &lr_random}}) {
            m << label << ',' << r->explained_variance << ',' << r->cosine << ',' << r->norm_ratio << ',' << r->l0 << ',' << r->dead_fraction << ','
              << r->dense_fraction << ',' << l->clean << ',' << l->spliced << ',' << l->ablated << ',' << l->recovered << '\n';
        }
    }

    // Held-out labels per concept.
    Sample held;
    for (size_t i = 0; i < s.proteins.size(); ++i) {
        if (s.test[i]) held.proteins.push_back(s.proteins[i]);
    }
    std::vector<std::string> concepts = {"Helix", "Beta strand", "Turn"};
    concepts.insert(concepts.end(), o.concepts.begin(), o.concepts.end());
    std::ofstream csv(o.out + "/concepts.csv");
    csv << "concept,positives,sae_feature,sae_f1,sae_precision,sae_recall,sae_features_above_half,neuron_f1,random_model_sae_f1,"
           "probe_f1,main_features,main_f1,absorption,random_model_absorption\n";
    std::printf("%-16s %9s   %7s %6s %6s %6s  %7s  %7s   %6s %5s %6s %8s %8s\n", "concept", "positives", "feature", "F1", "prec", "recall",
                "neuron", "random", "probe", "main", "mainF1", "absorbed", "random");
    const Tensor held_x(Shape({held_rows, trained.hidden}), backend, test_rows, backend->device());
    const Tensor held_x_random(Shape({held_rows, untrained.hidden}), backend, test_rows_random, backend->device());
    std::vector<ConceptMatch> matches;
    for (const std::string& c : concepts) {
        const std::vector<int> labels = Labels(held, c);
        const ConceptMatch m = MatchConcept(codes, labels, c), n = MatchConcept(neurons, labels, c), r = MatchConcept(codes_random, labels, c);
        if (m.positives == 0) continue;
        matches.push_back(m);
        // Absorption, with its probe baseline (FEAT-3); skipped where a split lacks a class.
        AbsorptionResult a, ar;
        bool absorption = true;
        try {
            if (sae->block_size() != 1) throw std::invalid_argument("blocks");
            AbsorptionOptions ao;
            ao.seed = o.seed + 17;
            a = FeatureAbsorption(*sae, held_x, labels, ao);
            ar = FeatureAbsorption(*sae_random, held_x_random, labels, ao);
        } catch (const std::invalid_argument&) {
            absorption = false;
        }
        std::printf("%-16s %9ld   %7ld %6.3f %6.3f %6.3f  %7.3f  %7.3f", c.c_str(), static_cast<long>(m.positives), static_cast<long>(m.feature), m.f1,
                    m.precision, m.recall, n.f1, r.f1);
        if (absorption && a.probe_f1 >= kMinProbeF1) {
            std::printf("   %6.3f %5zu %6.3f %7.1f%% %7.1f%%\n", a.probe_f1, a.main_features.size(), a.main_f1, 100 * a.absorption_rate,
                        100 * ar.absorption_rate);
        } else if (absorption) {
            std::printf("   %6.3f   (probe too weak for absorption)\n", a.probe_f1);
        } else {
            std::printf(sae->block_size() != 1 ? "   (absorption needs single-direction features)\n" : "   (too few positives for absorption)\n");
        }
        csv << '"' << c << "\"," << m.positives << ',' << m.feature << ',' << m.f1 << ',' << m.precision << ',' << m.recall << ','
            << m.features_above_half << ',' << n.f1 << ',' << r.f1 << ',';
        if (absorption) {
            csv << a.probe_f1 << ',' << a.main_features.size() << ',' << a.main_f1 << ',' << a.absorption_rate << ',' << ar.absorption_rate << '\n';
        } else {
            csv << ",,,,\n";
        }
    }

    // A dashboard for each concept's best feature: its activations on held-out residues, and the
    // proteins where it fires most.
    std::filesystem::create_directories(o.out + "/dashboards");
    for (const ConceptMatch& m : matches) {
        if (m.feature < 0) continue;
        FeatureDashboardDocument doc;
        doc.source = "ESM-2 layer " + std::to_string(layer) + " SAE";
        doc.feature_index = m.feature;
        doc.max_activation = codes.max_value[static_cast<size_t>(m.feature)];
        std::vector<std::pair<float, size_t>> per_protein;  // (largest activation, held-out protein)
        std::vector<std::vector<float>> activations(held.proteins.size());
        int64_t residue = 0, active = 0;
        std::vector<float> values;
        for (size_t p = 0; p < held.proteins.size(); ++p) {
            float best = 0;
            activations[p].assign(held.proteins[p].sequence.size(), 0.0f);
            for (size_t i = 0; i < held.proteins[p].sequence.size(); ++i, ++residue) {
                for (int64_t e = codes.row_start[static_cast<size_t>(residue)]; e < codes.row_start[static_cast<size_t>(residue) + 1]; ++e) {
                    if (codes.feature[static_cast<size_t>(e)] != m.feature) continue;
                    activations[p][i] = codes.value[static_cast<size_t>(e)];
                    values.push_back(activations[p][i]);
                    best = std::max(best, activations[p][i]);
                    ++active;
                }
            }
            per_protein.emplace_back(best, p);
        }
        doc.activation_density = static_cast<float>(active) / static_cast<float>(residue);
        const int bins = 20;
        doc.histogram_edges.resize(bins + 1);
        doc.histogram_counts.assign(bins, 0);
        for (int b = 0; b <= bins; ++b) doc.histogram_edges[static_cast<size_t>(b)] = doc.max_activation * static_cast<float>(b) / bins;
        for (float v : values) doc.histogram_counts[static_cast<size_t>(std::min(bins - 1, static_cast<int>(v / doc.max_activation * bins)))]++;
        std::stable_sort(per_protein.begin(), per_protein.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t k = 0; k < 5 && k < per_protein.size(); ++k) {
            const AnnotatedProtein& p = held.proteins[per_protein[k].second];
            FeatureDashboardDocument::Example e;
            e.label = p.accession + ": its " + m.concept_name + " residues are " + [&] {
                const std::vector<int> l = ConceptLabels(p, m.concept_name);
                std::string out;
                for (size_t i = 0; i < l.size() && out.size() < 80; ++i) {
                    if (l[i]) out += (out.empty() ? "" : " ") + std::to_string(i + 1);
                }
                return out.empty() ? std::string("none") : out;
            }();
            for (char c : p.sequence) e.tokens.emplace_back(1, c);
            e.activations = activations[per_protein[k].second];
            doc.top_examples.push_back(std::move(e));
        }
        std::string name = m.concept_name;
        std::replace(name.begin(), name.end(), ' ', '_');
        const std::string path = o.out + "/dashboards/" + name + "_feature_" + std::to_string(m.feature) + ".html";
        HtmlOptions html;
        html.title = m.concept_name + ": feature " + std::to_string(m.feature) + " (F1 " + std::to_string(m.f1).substr(0, 5) + ")";
        std::string page;
        const std::string af = o.structures + "/AF-" + held.proteins[per_protein[0].second].accession + "-F1-model_v4.cif";
        if (!o.structures.empty() && std::filesystem::exists(af)) {
            std::ifstream in(af, std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(in)), {});
            try {
                page = RenderFeatureDashboardHtml(doc, text, "A", 0, html);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "  %s: no structure panel (%s)\n", path.c_str(), e.what());
            }
        }
        if (page.empty()) page = RenderFeatureDashboardHtml(doc, html);
        std::ofstream(path) << page;
        std::ofstream(o.out + "/dashboards/" + name + "_feature_" + std::to_string(m.feature) + ".v1.json") << ToJson(doc);
    }
    std::printf("\nwrote %s/concepts.csv and %zu dashboards in %s/dashboards\n", o.out.c_str(), matches.size(), o.out.c_str());
}

int Run(DeviceBackend* backend, const Options& o) {
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model, backend);
    const TextTokenizer tok = LoadEsmTokenizer(o.model + "/vocab.txt");
    const std::vector<AnnotatedProtein> all = ReadAnnotatedProteins(o.annotations);
    std::filesystem::create_directories(o.out);
    if (o.probes) RunProbes(backend, *model, tok, o, Choose(all, o.probe_proteins, o.seed));
    if (o.sae) {
        const bool transcoder = o.featurizer == "transcoder" || o.featurizer == "skip-transcoder";
        (transcoder ? RunTranscoder : RunSae)(backend, *model, tok, o, Choose(all, o.sae_proteins, o.seed));
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
        if (a == "--annotations") o.annotations = value();
        else if (a == "--out") o.out = value();
        else if (a == "--structures") o.structures = value();
        else if (a == "--probes") o.probes = true;
        else if (a == "--sae") o.sae = true;
        else if (a == "--layers") {
            o.layers.clear();
            for (const std::string& l : Split(value())) o.layers.push_back(std::atoll(l.c_str()));
        } else if (a == "--concepts") o.concepts = Split(value());
        else if (a == "--probe-proteins") o.probe_proteins = std::atoll(value().c_str());
        else if (a == "--sae-proteins") o.sae_proteins = std::atoll(value().c_str());
        else if (a == "--sae-layer") o.sae_layer = std::atoll(value().c_str());
        else if (a == "--features") o.features = std::atoll(value().c_str());
        else if (a == "--l1") o.l1 = std::strtof(value().c_str(), nullptr);
        else if (a == "--featurizer") o.featurizer = value();
        else if (a == "--k") o.k = std::atoll(value().c_str());
        else if (a == "--block-size") o.block_size = std::atoll(value().c_str());
        else if (a == "--l0") o.l0 = std::strtof(value().c_str(), nullptr);
        else if (a == "--loss-proteins") o.loss_proteins = std::atoll(value().c_str());
        else if (a == "--sae-epochs") o.sae_epochs = std::atoll(value().c_str());
        else if (a == "--sae-batch") o.sae_batch = std::atoll(value().c_str());
        else if (a == "--seed") o.seed = std::strtoull(value().c_str(), nullptr, 10);
        else if (a == "--device") o.device = value();
        else Usage("unknown option " + a);
    }
    if (o.annotations.empty() || o.out.empty()) Usage("needs --annotations and --out");
    if (!o.probes && !o.sae) Usage("needs --probes, --sae or both");
    const std::vector<std::string> featurizers = {"l1",         "topk",       "batchtopk",       "matryoshka", "jumprelu",
                                                  "transcoder", "skip-transcoder", "bsf-vanilla", "bsf-grassmannian", "bsf-lasso"};
    if (std::find(featurizers.begin(), featurizers.end(), o.featurizer) == featurizers.end()) {
        Usage("--featurizer must be l1, topk, batchtopk, matryoshka, jumprelu, transcoder, skip-transcoder, bsf-vanilla, bsf-grassmannian or bsf-lasso");
    }
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
        std::cerr << "pulsatrix_probe_esm: " << e.what() << "\n";
        return 1;
    }
}
