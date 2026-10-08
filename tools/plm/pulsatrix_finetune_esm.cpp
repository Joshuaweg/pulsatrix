// pulsatrix_finetune_esm: fine-tuning and probing an ESM-2 model with SequenceHead (roadmap PLM-7).
//
//   Per protein: a deep mutational scan's fitness, from the mean of each variant's residue
//   representations, on a held-out share of the variants, beside the model's zero-shot score.
//     pulsatrix_finetune_esm esm2_t6_8M_UR50D --task dms --assay BLAT_ECOLX_Stiffler_2015 \
//         --reference DMS_substitutions.csv --dms-dir DMS_ProteinGym_substitutions [--finetune]
//   Per residue: how buried each residue is (Cβ neighbours within 10 Å, Cα for glycine), learnt on
//   some structures and tested on others.
//     pulsatrix_finetune_esm esm2_t6_8M_UR50D --task burial --structures DIR \
//         --train 1UBQ:A,1PGA:A,... --test 1BTL:A,2LZM:A
//
// Without --finetune the encoder is frozen and only the head learns (a linear probe); with it, the
// encoder learns too, at --encoder-lr. Labels are standardized on the training set. Reports the
// held-out Spearman correlation. Options (defaults): --epochs 5, --batch 16, --lr 1e-3,
// --encoder-lr 2e-5, --test-fraction 0.2, --max-variants 0 (all), --seed 0, --device cpu|hip.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/fitness_metrics.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_structure.hpp"
#include "pulsatrix/protein_training.hpp"
#include "pulsatrix/proteingym.hpp"
#include "pulsatrix/variant_scoring.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_finetune_esm: " << problem << "\n"
              << "usage: pulsatrix_finetune_esm MODEL_DIR --task dms --assay ID --reference CSV --dms-dir DIR [options]\n"
              << "       pulsatrix_finetune_esm MODEL_DIR --task burial --structures DIR --train E:C,... --test E:C,... [options]\n"
              << "options: [--finetune] [--epochs N] [--batch N] [--lr LR] [--encoder-lr LR] [--test-fraction F]\n"
              << "         [--max-variants N] [--seed S] [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string model, task, assay, reference, dms_dir, structures;
    std::vector<std::string> train, test;
    std::string device = "cpu";
    bool finetune = false;
    int64_t epochs = 5, batch = 16, max_variants = 0;
    double test_fraction = 0.2;
    float lr = 1e-3f, encoder_lr = 2e-5f;
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

/** @brief One training example: a sequence and its labels, one per residue, or one in total. */
struct Example {
    std::string sequence;
    std::vector<float> labels;
};

/** @brief Cβ neighbours within 10 Å of each residue. */
std::vector<float> Burial(const pulsatrix::StructureChain& chain) {
    const pulsatrix::ContactMap d = pulsatrix::ResidueDistances(chain);
    std::vector<float> out(static_cast<size_t>(d.length), std::nanf(""));
    for (int64_t i = 0; i < d.length; ++i) {
        if (std::isnan(d.at(i, i))) continue;
        float n = 0;
        for (int64_t j = 0; j < d.length; ++j) n += j != i && d.at(i, j) < 10.0f ? 1.0f : 0.0f;
        out[static_cast<size_t>(i)] = n;
    }
    return out;
}

class Trainer {
public:
    Trainer(pulsatrix::EncoderLM& model, const pulsatrix::TextTokenizer& tok, pulsatrix::DeviceBackend* backend, bool per_residue, const Options& o)
        : model_(model),
          tok_(tok),
          backend_(backend),
          per_residue_(per_residue),
          o_(o),
          head_(model.config().hidden_size, 1, per_residue ? pulsatrix::SequenceHead::Pooling::PerResidue : pulsatrix::SequenceHead::Pooling::Mean,
                backend),
          head_opt_(o.lr, backend, 0.01f),
          encoder_opt_(o.encoder_lr, backend, 0.01f) {
        pulsatrix::MaskedLMOptions none;
        none.mask_probability = 0.0;
        collator_ = std::make_unique<pulsatrix::MaskedLMCollator>(tok, backend, none);
    }

    /** @brief Predictions for a batch; with @p grad_scale > 0, also trains on it. Labels and
     *         predictions are per residue (or one per sequence), NaN labels skipped. */
    std::vector<float> Step(const std::vector<Example>& batch, bool train) {
        std::vector<std::string> seqs;
        for (const auto& e : batch) seqs.push_back(e.sequence);
        const pulsatrix::MaskedLMBatch b = collator_->collate(seqs);
        const int64_t N = b.ids.shape().dim(0), L = b.ids.shape().dim(1);
        model_.set_padding_mask(b.keep);
        model_.set_keep_activations(train && o_.finetune);
        (void)model_.forward(b.ids);
        head_.set_residue_mask(pulsatrix::ResidueMask(b.ids, &b.keep, tok_, backend_));
        const pulsatrix::Tensor y = head_.forward(model_.last_hidden_state());
        const std::vector<float> pred = y.to_host_vector();
        std::vector<float> out, grad(pred.size(), 0.0f);
        int64_t count = 0;
        for (int64_t n = 0; n < N; ++n) {
            const auto& labels = batch[static_cast<size_t>(n)].labels;
            for (size_t k = 0; k < labels.size(); ++k) {
                const size_t at = per_residue_ ? static_cast<size_t>(n * L + 1 + static_cast<int64_t>(k)) : static_cast<size_t>(n);
                out.push_back(pred[at]);
                if (std::isnan(labels[k])) continue;
                grad[at] = pred[at] - labels[k];
                ++count;
            }
        }
        if (train && count > 0) {
            for (float& g : grad) g = 2.0f * g / static_cast<float>(count);
            head_opt_.zero_grad(head_);
            if (o_.finetune) encoder_opt_.zero_grad(model_);
            const pulsatrix::Tensor g_hidden = head_.backward(pulsatrix::Tensor(y.shape(), backend_, grad, backend_->device()));
            if (o_.finetune) (void)model_.backward_hidden(g_hidden);
            head_opt_.step(head_);
            if (o_.finetune) encoder_opt_.step(model_);
        }
        model_.release_activations();
        return out;
    }

    /** @brief Trains for the epochs, then returns the held-out Spearman correlation. */
    double Fit(std::vector<Example> train, const std::vector<Example>& test) {
        std::mt19937_64 rng(o_.seed);
        for (int64_t epoch = 1; epoch <= o_.epochs; ++epoch) {
            std::shuffle(train.begin(), train.end(), rng);
            for (size_t s = 0; s < train.size(); s += static_cast<size_t>(o_.batch)) {
                (void)Step({train.begin() + static_cast<std::ptrdiff_t>(s),
                            train.begin() + static_cast<std::ptrdiff_t>(std::min(train.size(), s + static_cast<size_t>(o_.batch)))},
                           true);
            }
            std::printf("  epoch %ld: held-out Spearman %.4f\n", static_cast<long>(epoch), Evaluate(test));
            std::fflush(stdout);
        }
        return Evaluate(test);
    }

    double Evaluate(const std::vector<Example>& test) {
        std::vector<double> pred, truth;
        for (size_t s = 0; s < test.size(); s += static_cast<size_t>(o_.batch)) {
            const std::vector<Example> chunk(test.begin() + static_cast<std::ptrdiff_t>(s),
                                             test.begin() + static_cast<std::ptrdiff_t>(std::min(test.size(), s + static_cast<size_t>(o_.batch))));
            const std::vector<float> p = Step(chunk, false);
            size_t k = 0;
            for (const auto& e : chunk) {
                for (float label : e.labels) {
                    if (!std::isnan(label)) {
                        pred.push_back(p[k]);
                        truth.push_back(label);
                    }
                    ++k;
                }
            }
        }
        return pulsatrix::SpearmanCorrelation(pred, truth);
    }

private:
    pulsatrix::EncoderLM& model_;
    const pulsatrix::TextTokenizer& tok_;
    pulsatrix::DeviceBackend* backend_;
    bool per_residue_;
    const Options& o_;
    pulsatrix::SequenceHead head_;
    pulsatrix::AdamWOptimizer head_opt_, encoder_opt_;
    std::unique_ptr<pulsatrix::MaskedLMCollator> collator_;
};

/** @brief Standardizes every label by the training labels' mean and standard deviation. */
void Standardize(std::vector<Example>& train, std::vector<Example>& test) {
    double sum = 0, sq = 0;
    int64_t n = 0;
    for (const auto& e : train) {
        for (float v : e.labels) {
            if (std::isnan(v)) continue;
            sum += v;
            sq += static_cast<double>(v) * v;
            ++n;
        }
    }
    const double mean = sum / static_cast<double>(n), sd = std::sqrt(std::max(1e-12, sq / static_cast<double>(n) - mean * mean));
    for (auto* set : {&train, &test}) {
        for (auto& e : *set) {
            for (float& v : e.labels) v = static_cast<float>((v - mean) / sd);
        }
    }
}

int Run(pulsatrix::DeviceBackend* backend, const Options& o) {
    using namespace pulsatrix;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model, backend);
    const TextTokenizer tok = LoadEsmTokenizer(o.model + "/vocab.txt");
    std::vector<Example> train, test;
    const bool per_residue = o.task == "burial";
    if (o.task == "dms") {
        ProteinGymAssay assay;
        for (const ProteinGymAssay& a : ReadProteinGymReference(o.reference)) {
            if (a.id == o.assay) assay = a;
        }
        if (assay.id.empty()) throw std::invalid_argument("no assay " + o.assay);
        const DmsVariants v = ReadDmsVariants(o.dms_dir + "/" + assay.filename);
        std::vector<size_t> order(v.mutants.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::shuffle(order.begin(), order.end(), std::mt19937_64(o.seed + 1));
        if (o.max_variants > 0 && order.size() > static_cast<size_t>(o.max_variants)) order.resize(static_cast<size_t>(o.max_variants));
        const size_t n_test = static_cast<size_t>(std::llround(o.test_fraction * static_cast<double>(order.size())));
        std::vector<double> zero_shot, zero_truth;
        VariantScorer scorer(*model, tok, backend);
        const ResidueLogProbs marginals = scorer.masked_marginals(assay.target_seq);
        for (size_t k = 0; k < order.size(); ++k) {
            const size_t i = order[k];
            const std::vector<Mutation> m = ParseMutations(v.mutants[i]);
            Example e{ApplyMutations(assay.target_seq, m, assay.offset), {static_cast<float>(v.scores[i])}};
            if (k < n_test) {
                zero_shot.push_back(scorer.score(marginals, assay.target_seq, m, assay.offset));
                zero_truth.push_back(v.scores[i]);
                test.push_back(std::move(e));
            } else {
                train.push_back(std::move(e));
            }
        }
        std::printf("%s: %zu training and %zu held-out variants; zero-shot (masked marginals) Spearman on the held-out: %.4f\n", o.assay.c_str(),
                    train.size(), test.size(), SpearmanCorrelation(zero_shot, zero_truth));
    } else if (per_residue) {
        auto load = [&](const std::vector<std::string>& proteins, std::vector<Example>& into) {
            for (const std::string& p : proteins) {
                const size_t colon = p.find(':');
                if (colon == std::string::npos) throw std::invalid_argument("\"" + p + "\" isn't ENTRY:CHAIN");
                const StructureChain chain = ReadStructure(o.structures + "/" + p.substr(0, colon) + ".cif").chain(p.substr(colon + 1));
                into.push_back({chain.sequence(), Burial(chain)});
            }
        };
        load(o.train, train);
        load(o.test, test);
        std::printf("burial: %zu training and %zu held-out proteins\n", train.size(), test.size());
    } else {
        Usage("--task must be dms or burial");
    }
    Standardize(train, test);
    Trainer trainer(*model, tok, backend, per_residue, o);
    std::printf("%s (%s): held-out Spearman before training %.4f\n", o.finetune ? "fine-tuning" : "linear probe", o.model.c_str(),
                trainer.Evaluate(test));
    const double final = trainer.Fit(train, test);
    std::printf("held-out Spearman after %ld epochs: %.4f\n", static_cast<long>(o.epochs), final);
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
        if (a == "--task") o.task = value();
        else if (a == "--assay") o.assay = value();
        else if (a == "--reference") o.reference = value();
        else if (a == "--dms-dir") o.dms_dir = value();
        else if (a == "--structures") o.structures = value();
        else if (a == "--train") o.train = Split(value());
        else if (a == "--test") o.test = Split(value());
        else if (a == "--finetune") o.finetune = true;
        else if (a == "--epochs") o.epochs = std::atoll(value().c_str());
        else if (a == "--batch") o.batch = std::atoll(value().c_str());
        else if (a == "--lr") o.lr = std::strtof(value().c_str(), nullptr);
        else if (a == "--encoder-lr") o.encoder_lr = std::strtof(value().c_str(), nullptr);
        else if (a == "--test-fraction") o.test_fraction = std::atof(value().c_str());
        else if (a == "--max-variants") o.max_variants = std::atoll(value().c_str());
        else if (a == "--seed") o.seed = std::strtoull(value().c_str(), nullptr, 10);
        else if (a == "--device") o.device = value();
        else Usage("unknown option " + a);
    }
    if (o.task.empty()) Usage("needs --task");
    if (o.epochs < 1 || o.batch < 1) Usage("--epochs and --batch must be positive");
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
        std::cerr << "pulsatrix_finetune_esm: " << e.what() << "\n";
        return 1;
    }
}
