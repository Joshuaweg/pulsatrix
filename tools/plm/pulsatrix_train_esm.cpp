// pulsatrix_train_esm: masked-LM training of an ESM-2-shaped model (roadmap PLM-7), and evaluation
// of a model's masked-LM perplexity.
//
//   pulsatrix_train_esm --config esm2_t6_8M_UR50D/config.json --vocab esm2_t6_8M_UR50D/vocab.txt \
//       --train train.fasta --valid valid.fasta --out run/ --steps 4000 --device hip
//   pulsatrix_train_esm --evaluate esm2_t6_8M_UR50D --valid valid.fasta
//
// Training starts from InitializeEsm (or --init-from a model directory), samples sequences by
// cluster (ClusterSampler; each FASTA record's first word is its cluster unless --clusters FILE
// maps record ids to clusters, one "id cluster" pair per line), masks them as ESM-2 did
// (MaskedLMCollator) and trains with AdamW: linear warmup, then linear decay to 0, and global-norm
// clipping. Every --eval-every steps it reports the held-out loss, perplexity and
// accuracy on --eval-sequences sequences, always masked the same way.
// Writes into --out: log.csv, training_log.v1.json (pulsatrix_svg draws it as an HTML page) and
// model.safetensors (SaveCheckpoint; load it into a model made from the same config).
// Options (defaults): --batch 16, --max-tokens 512, --lr 4e-4, --warmup 500, --weight-decay 0.01,
// --beta2 0.98, --clip 1.0, --eval-every 250, --eval-sequences 500, --seed 0.
#include <chrono>
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

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/grad_clipping.hpp"
#include "pulsatrix/lr_scheduler.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/protein_training.hpp"
#include "pulsatrix/viz/document.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_train_esm: " << problem << "\n"
              << "usage: pulsatrix_train_esm --config CONFIG.json --vocab VOCAB.txt --train TRAIN.fasta --valid VALID.fasta --out DIR\n"
              << "       [--init-from MODEL_DIR] [--clusters FILE] [--steps N] [--batch N] [--max-tokens N] [--lr LR] [--warmup N]\n"
              << "       [--weight-decay W] [--beta2 B] [--clip C] [--eval-every N] [--eval-sequences N]\n"
              << "       [--seed S] [--device cpu|hip]\n"
              << "       pulsatrix_train_esm --evaluate MODEL_DIR --valid VALID.fasta [--eval-sequences N] [--max-tokens N] [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string config, vocab, train, valid, out, init_from, clusters, evaluate;
    std::string device = "cpu";
    int64_t steps = 4000, batch = 16, max_tokens = 512, warmup = 500, eval_every = 250, eval_sequences = 500;
    uint64_t seed = 0;
    float lr = 4e-4f, weight_decay = 0.01f, beta2 = 0.98f, clip = 1.0f;
};

std::vector<std::string> Sequences(const std::vector<pulsatrix::FastaRecord>& records, size_t limit) {
    std::vector<std::string> out;
    for (const auto& r : records) {
        if (out.size() >= limit) break;
        out.push_back(r.sequence);
    }
    return out;
}

int Run(pulsatrix::DeviceBackend* backend, const Options& o) {
    using namespace pulsatrix;
    MaskedLMOptions mlm;
    mlm.max_tokens = o.max_tokens;
    const std::vector<std::string> valid = Sequences(ReadFasta(o.valid), static_cast<size_t>(o.eval_sequences));

    if (!o.evaluate.empty()) {
        std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.evaluate, backend);
        const TextTokenizer tok = LoadEsmTokenizer(o.evaluate + "/vocab.txt");
        const MaskedLMEvaluation e = EvaluateMaskedLM(*model, tok, backend, valid, 8, 1, mlm);
        std::printf("%s on %zu sequences (%ld masked tokens): loss %.4f, perplexity %.3f, accuracy %.4f\n", o.evaluate.c_str(), valid.size(),
                    static_cast<long>(e.tokens), e.loss, e.perplexity, e.accuracy);
        return 0;
    }

    const TextTokenizer tok = LoadEsmTokenizer(o.vocab);
    std::unique_ptr<EncoderLM> model;
    if (!o.init_from.empty()) {
        model = LoadEncoderLM(o.init_from, backend);
    } else {
        model = std::make_unique<EncoderLM>(ReadEsmConfig(o.config), backend);
        InitializeEsm(*model, o.seed);
    }
    const std::vector<FastaRecord> records = ReadFasta(o.train);
    std::map<std::string, std::string> cluster_of_id;
    if (!o.clusters.empty()) {
        std::ifstream in(o.clusters);
        for (std::string id, cluster; in >> id >> cluster;) cluster_of_id[id] = cluster;
    }
    std::vector<std::string> clusters;
    for (const auto& r : records) {
        const auto it = cluster_of_id.find(r.id);
        clusters.push_back(it == cluster_of_id.end() ? r.id : it->second);
    }
    ClusterSampler sampler(clusters, o.seed + 1);
    MaskedLMCollator collator(tok, backend, mlm, o.seed + 2);
    AdamWOptimizer opt(o.lr, backend, o.weight_decay, 0.9f, o.beta2, 1e-8f);
    LRScheduler<AdamWOptimizer> schedule(opt, LRSchedule::Linear(o.warmup, o.steps));
    TokenCrossEntropyLoss loss(backend);

    std::filesystem::create_directories(o.out);
    std::ofstream log(o.out + "/log.csv");
    log << "step,train_loss,learning_rate,grad_norm,valid_loss,valid_perplexity,valid_accuracy,tokens,seconds\n";
    TrainingLogDocument doc;
    TrainingLogDocument::Series train_series{"train/loss", {}, {}}, valid_series{"valid/perplexity", {}, {}}, acc_series{"valid/accuracy", {}, {}};
    std::printf("%zu training sequences in %ld clusters; %zu validation sequences\n", records.size(), static_cast<long>(sampler.num_clusters()),
                valid.size());
    auto evaluate = [&](int64_t step) {
        const MaskedLMEvaluation e = EvaluateMaskedLM(*model, tok, backend, valid, 8, 1, mlm);
        valid_series.steps.push_back(step);
        valid_series.values.push_back(e.perplexity);
        acc_series.steps.push_back(step);
        acc_series.values.push_back(e.accuracy);
        return e;
    };
    const auto start = std::chrono::steady_clock::now();
    MaskedLMEvaluation e = evaluate(0);
    std::printf("step %6d  valid perplexity %.3f  accuracy %.4f\n", 0, e.perplexity, e.accuracy);
    int64_t tokens = 0;
    double running = 0;
    int64_t running_n = 0;
    for (int64_t step = 1; step <= o.steps; ++step) {
        std::vector<std::string> batch;
        for (int64_t b = 0; b < o.batch; ++b) batch.push_back(records[static_cast<size_t>(sampler.next())].sequence);
        const MaskedLMBatch mb = collator.collate(batch);
        opt.zero_grad(*model);
        const float l = MaskedLMForwardBackward(*model, mb, loss);
        const float norm = ClipGradNorm(*model, o.clip);
        const float lr = opt.learning_rate();
        opt.step(*model);
        schedule.step();
        tokens += mb.num_targets;
        running += l;
        ++running_n;
        train_series.steps.push_back(step);
        train_series.values.push_back(l);
        const bool eval_now = step % o.eval_every == 0 || step == o.steps;
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (eval_now) e = evaluate(step);
        log << step << ',' << l << ',' << lr << ',' << norm << ',';
        if (eval_now) log << e.loss << ',' << e.perplexity << ',' << e.accuracy;
        else log << ",,";
        log << ',' << tokens << ',' << seconds << '\n';
        if (eval_now) {
            std::printf("step %6ld  train loss %.4f  valid perplexity %.3f  accuracy %.4f  lr %.2e  %.0f s\n", static_cast<long>(step),
                        running / static_cast<double>(running_n), e.perplexity, e.accuracy, lr, seconds);
            std::fflush(stdout);
            running = 0;
            running_n = 0;
        }
    }
    doc.scalars = {acc_series, valid_series, train_series};
    std::ofstream(o.out + "/training_log.v1.json") << ToJson(doc);
    SaveCheckpoint(o.out + "/model.safetensors", *model, {{"config", o.init_from.empty() ? o.config : o.init_from + "/config.json"}});
    std::printf("wrote %s/{log.csv,training_log.v1.json,model.safetensors}\n", o.out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--config") o.config = value();
        else if (a == "--vocab") o.vocab = value();
        else if (a == "--train") o.train = value();
        else if (a == "--valid") o.valid = value();
        else if (a == "--out") o.out = value();
        else if (a == "--init-from") o.init_from = value();
        else if (a == "--clusters") o.clusters = value();
        else if (a == "--evaluate") o.evaluate = value();
        else if (a == "--steps") o.steps = std::atoll(value().c_str());
        else if (a == "--batch") o.batch = std::atoll(value().c_str());
        else if (a == "--max-tokens") o.max_tokens = std::atoll(value().c_str());
        else if (a == "--lr") o.lr = std::strtof(value().c_str(), nullptr);
        else if (a == "--warmup") o.warmup = std::atoll(value().c_str());
        else if (a == "--weight-decay") o.weight_decay = std::strtof(value().c_str(), nullptr);
        else if (a == "--beta2") o.beta2 = std::strtof(value().c_str(), nullptr);
        else if (a == "--clip") o.clip = std::strtof(value().c_str(), nullptr);
        else if (a == "--eval-every") o.eval_every = std::atoll(value().c_str());
        else if (a == "--eval-sequences") o.eval_sequences = std::atoll(value().c_str());
        else if (a == "--seed") o.seed = std::strtoull(value().c_str(), nullptr, 10);
        else if (a == "--device") o.device = value();
        else Usage("unknown option " + a);
    }
    if (o.valid.empty()) Usage("needs --valid");
    if (o.evaluate.empty() && (o.vocab.empty() || o.train.empty() || o.out.empty() || (o.config.empty() && o.init_from.empty()))) {
        Usage("training needs --vocab, --train, --out and --config or --init-from");
    }
    if (o.steps < 1 || o.batch < 1 || o.eval_every < 1 || o.eval_sequences < 1) Usage("--steps, --batch, --eval-every and --eval-sequences must be positive");
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
        std::cerr << "pulsatrix_train_esm: " << e.what() << "\n";
        return 1;
    }
}
