// Recipe: Half the layers (KD-4)
//
// DistilBERT (Sanh et al., 2019) made BERT 40% smaller by keeping one layer out of two, starting
// the student from those layers' weights, and training it on three losses: the teacher's softened
// outputs, the data, and a cosine loss pulling the student's last hidden states toward the
// teacher's. This recipe does the same to SmolLM2-135M-Instruct, 30 blocks to 15, on chat text it
// was tuned on, and compares five students after the same steps on the same batches:
//
//   - untrained: the 15 kept blocks as they are;
//   - language modeling: next-token cross-entropy alone;
//   - distilled: DistilBERT's 5 : 2 : 1 mix of distillation (T = 2), cross-entropy and cosine;
//   - distilled without the cosine loss: 5 : 2;
//   - distilled from random blocks: the 5 : 2 : 1 mix, with every block's weight matrices drawn at
//     random (each tensor's own scale); the embedding and the norms are still the teacher's.
//
// Then a question the loss curves can't answer: does the distilled student predict the next token
// from the same tokens as the teacher? AttnLRP token relevance for the teacher's top next token,
// compared between teacher and student with KD-2's test.
//
// The predictions, fixed before the run:
//   1. Half the layers, untrained, breaks the model: held-out perplexity more than 10 times the
//      teacher's.
//   2. Distillation beats language modeling alone: lower held-out perplexity and more of the
//      teacher's top-1 next tokens.
//   3. Starting from the teacher's blocks matters more than the loss: the student from random blocks
//      ends with more than twice the language-modeling student's perplexity.
//   4. The cosine loss matters little here: dropping it moves held-out perplexity by less than 5%.
//   5. The distilled student's token relevance agrees with the teacher's more than the
//      language-modeling student's does (mean Spearman).
//
// Usage: half_the_layers_recipe MODEL_DIR OUT_DIR --text CONVERSATIONS.jsonl [--device cpu|hip]
//            [--steps N] [--smoke --tokenizer PATH]
//   MODEL_DIR: SmolLM2-135M-Instruct. CONVERSATIONS.jsonl: one {"text": ...} chat per line, such as
//   tools/explain/fetch_chat_sample.py writes. --smoke runs the tiny test model on a few built-in
//   sentences, to check the plumbing; its numbers mean nothing.
//
// Docs: docs/recipes/deep-learning/half_the_layers.md
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/attnlrp_parity.hpp"
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/distillation_loss.hpp"
#include "pulsatrix/explanation_agreement.hpp"
#include "pulsatrix/fine_tune_diff.hpp"
#include "pulsatrix/grad_clipping.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/layer_student.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/html.hpp"
#ifdef PULSATRIX_DEMO_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

struct Options {
    std::string model;
    std::string out;
    std::string text;
    std::string tokenizer;
    bool hip = false;
    bool smoke = false;
    int steps = 300;
    int64_t batch = 8;
    int64_t length = 128;
    int64_t held_out = 64;  // windows
    int64_t explained = 24;
    int64_t explained_length = 32;
    int eval_every = 50;
};

struct SplitMix {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return (static_cast<double>(next() >> 11) + 0.5) / 9007199254740992.0; }
    double gaussian() { return std::sqrt(-2.0 * std::log(uniform())) * std::cos(6.283185307179586 * uniform()); }
};

using Window = std::vector<int64_t>;

const char* const kSmokeText[] = {
    "The quick brown fox jumps over the lazy dog, and the dog does not mind at all.",
    "A student learns from a teacher, and sometimes the student learns the teacher's mistakes too.",
    "Half of the layers are kept, and the other half are thrown away before the training starts.",
    "Every window of text is cut to the same length so that the batches line up with each other.",
};

// Each text cut into windows of `length` tokens, the remainder dropped; the first 90% of the texts
// train and the rest are held out, so no conversation is in both. `shared` gets, for each held-out
// window, how many of its first tokens are the opening every text shares (the chat's system
// prompt), or 0 for a window from further into its text.
void ReadWindows(const Options& o, const TextTokenizer& tok, int64_t vocab, std::vector<Window>& train, std::vector<Window>& held_out,
                 std::vector<int64_t>& shared) {
    std::vector<std::string> texts;
    if (o.smoke) {
        for (int r = 0; r < 4; ++r)
            for (const char* t : kSmokeText) texts.emplace_back(t);
    } else {
        std::ifstream in(o.text);
        if (!in) throw std::runtime_error("can't read " + o.text);
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty()) texts.push_back(ParseJson(line).find("text")->as_string());
        }
    }
    const size_t split = texts.size() * 9 / 10;
    std::vector<int64_t> starts;  // each held-out window's offset in its text
    Window opening;               // the tokens every text starts with
    for (size_t i = 0; i < texts.size(); ++i) {
        const Encoding e = tok.encode(texts[i]);
        if (i == 0) {
            opening = e.ids;
        } else {
            size_t k = 0;
            while (k < opening.size() && k < e.ids.size() && opening[k] == e.ids[k]) ++k;
            opening.resize(k);
        }
        for (size_t start = 0; start + static_cast<size_t>(o.length) <= e.ids.size(); start += static_cast<size_t>(o.length)) {
            Window w(e.ids.begin() + static_cast<std::ptrdiff_t>(start), e.ids.begin() + static_cast<std::ptrdiff_t>(start + o.length));
            for (int64_t& id : w) id %= vocab;  // the tiny test model's vocabulary is smaller
            const bool held = i >= split && !o.smoke;
            (held ? held_out : train).push_back(std::move(w));
            if (held || (o.smoke && starts.size() < 4)) starts.push_back(static_cast<int64_t>(start));
        }
    }
    if (o.smoke) held_out = std::vector<Window>(train.begin(), train.begin() + std::min<std::ptrdiff_t>(4, static_cast<std::ptrdiff_t>(train.size())));
    if (static_cast<int64_t>(held_out.size()) > o.held_out) held_out.resize(static_cast<size_t>(o.held_out));
    shared.clear();
    for (size_t k = 0; k < held_out.size(); ++k) shared.push_back(starts[k] == 0 ? std::min<int64_t>(static_cast<int64_t>(opening.size()), o.length) : 0);
    if (train.empty() || held_out.empty()) throw std::runtime_error("not enough text for one training and one held-out window");
}

Tensor Ids(const std::vector<const Window*>& windows, int64_t length, DeviceBackend* backend) {
    std::vector<float> ids;
    for (const Window* w : windows)
        for (int64_t i = 0; i < length; ++i) ids.push_back(static_cast<float>((*w)[static_cast<size_t>(i)]));
    return Tensor(Shape({static_cast<int64_t>(windows.size()), length}), backend, ids, backend->device());
}

// The next token at every position but the last, which has none.
Tensor Targets(const std::vector<const Window*>& windows, int64_t length, DeviceBackend* backend) {
    std::vector<float> t;
    for (const Window* w : windows)
        for (int64_t i = 0; i < length; ++i)
            t.push_back(i + 1 < length ? static_cast<float>((*w)[static_cast<size_t>(i + 1)]) : static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex));
    return Tensor(Shape({static_cast<int64_t>(windows.size()), length}), backend, t, backend->device());
}

std::string Fixed(double v, int digits) {
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(digits);
    s << v;
    return s.str();
}

// A model on the held-out windows: perplexity, KL from the teacher, and the share of positions
// where its top next token is the teacher's. With `split`, also the perplexity on the tokens of the
// opening every text shares and on the rest.
struct Score {
    double perplexity = 0, kl = 0, agreement = 0;
    double opening_perplexity = 0, rest_perplexity = 0;
};

Score Evaluate(CausalLM& student, CausalLM* teacher, const std::vector<Window>& held_out, const std::vector<int64_t>* split,
               const Options& o, DeviceBackend* backend) {
    TokenCrossEntropyLoss ce(backend);
    DistillationOptions dopt;
    dopt.temperature = 1.0f;
    DistillationLoss kl(backend, dopt);
    double nll = 0, divergence = 0, nll_opening = 0, nll_rest = 0;
    int64_t tokens = 0, same = 0, positions = 0, n_opening = 0, n_rest = 0;
    for (size_t start = 0; start < held_out.size(); start += static_cast<size_t>(o.batch)) {
        std::vector<const Window*> b;
        for (size_t i = start; i < std::min(held_out.size(), start + static_cast<size_t>(o.batch)); ++i) b.push_back(&held_out[i]);
        const Tensor ids = Ids(b, o.length, backend);
        const Tensor logits = student.forward(ids);
        const auto rows = static_cast<int64_t>(b.size()) * (o.length - 1);
        nll += static_cast<double>(ce.forward(logits, Targets(b, o.length, backend))) * static_cast<double>(rows);
        tokens += rows;
        if (split) {
            const std::vector<float> s = logits.to_host_vector();
            const int64_t V = logits.shape().dim(2);
            for (size_t n = 0; n < b.size(); ++n) {
                const int64_t opening = (*split)[start + n];
                for (int64_t i = 0; i + 1 < o.length; ++i) {
                    const auto row = s.begin() + (static_cast<int64_t>(n) * o.length + i) * V;
                    const double top = *std::max_element(row, row + V);
                    double sum = 0;
                    for (int64_t v = 0; v < V; ++v) sum += std::exp(row[v] - top);
                    const double loss = top + std::log(sum) - row[(*b[n])[static_cast<size_t>(i + 1)]];
                    if (i + 1 < opening) {
                        nll_opening += loss;
                        ++n_opening;
                    } else {
                        nll_rest += loss;
                        ++n_rest;
                    }
                }
            }
        }
        if (!teacher) continue;
        const Tensor t = teacher->forward(ids);
        divergence += static_cast<double>(kl.forward(logits, t)) * static_cast<double>(b.size()) * static_cast<double>(o.length);
        const std::vector<float> s = logits.to_host_vector(), tv = t.to_host_vector();
        const int64_t V = logits.shape().dim(2);
        for (int64_t r = 0; r < static_cast<int64_t>(b.size()) * o.length; ++r) {
            const auto sr = s.begin() + r * V, tr = tv.begin() + r * V;
            same += std::max_element(sr, sr + V) - sr == std::max_element(tr, tr + V) - tr ? 1 : 0;
            ++positions;
        }
    }
    Score out;
    out.perplexity = std::exp(nll / static_cast<double>(tokens));
    if (n_opening > 0) out.opening_perplexity = std::exp(nll_opening / static_cast<double>(n_opening));
    if (n_rest > 0) out.rest_perplexity = std::exp(nll_rest / static_cast<double>(n_rest));
    if (positions > 0) {
        out.kl = divergence / static_cast<double>(positions);
        out.agreement = static_cast<double>(same) / static_cast<double>(positions);
    }
    return out;
}

enum class Arm { LanguageModeling, Distilled, DistilledWithoutCosine, DistilledFromRandomBlocks };

const char* Name(Arm a) {
    switch (a) {
        case Arm::LanguageModeling: return "language modeling";
        case Arm::Distilled: return "distilled";
        case Arm::DistilledWithoutCosine: return "distilled without the cosine loss";
        case Arm::DistilledFromRandomBlocks: return "distilled from random blocks";
    }
    return "";
}

// Every block's weight matrix replaced by Gaussian noise with that matrix's own standard deviation.
void RandomizeBlocks(CausalLM& model, uint64_t seed) {
    SplitMix rng{seed};
    for (const NamedParamRef& p : model.named_parameters()) {
        if (p.name.compare(0, 7, "layers.") != 0 || p.ref.value->rank() != 2) continue;
        const std::vector<float> v = p.ref.value->to_host_vector();
        double mean = 0, var = 0;
        for (float x : v) mean += x;
        mean /= static_cast<double>(v.size());
        for (float x : v) var += (x - mean) * (x - mean);
        const double sd = std::sqrt(var / static_cast<double>(v.size()));
        std::vector<float> noise(v.size());
        for (float& x : noise) x = static_cast<float>(sd * rng.gaussian());
        *p.ref.value = Tensor(p.ref.value->shape(), p.ref.value->backend(), noise, p.ref.value->device());
    }
}

// Trains one student; every arm sees the same batches in the same order. Logs the training
// objective and, every eval_every steps, the held-out perplexity.
void Train(CausalLM& student, CausalLM& teacher, Arm arm, const std::vector<Window>& train, const std::vector<Window>& held_out,
           const Options& o, DeviceBackend* backend, TrainingLogDocument& log) {
    const bool distill = arm != Arm::LanguageModeling;
    const bool cosine = arm == Arm::Distilled || arm == Arm::DistilledFromRandomBlocks;
    const int64_t last = student.num_layers() - 1;
    student.set_intervention(last, std::make_unique<HiddenStateTap>(backend));
    auto* tap = static_cast<HiddenStateTap*>(student.intervention(last));
    auto* teacher_tap = static_cast<HiddenStateTap*>(teacher.intervention(teacher.num_layers() - 1));
    AdamWOptimizer opt(2e-4f, backend, /*weight_decay=*/0.0f);
    TokenCrossEntropyLoss ce(backend);
    DistillationOptions dopt;
    dopt.temperature = 2.0f;
    dopt.alpha = 5.0f / 7.0f;  // 5 : 2, distillation : cross-entropy
    DistillationLoss kd(backend, dopt);
    HiddenStateCosineLoss cos(backend);
    SplitMix order_rng{17};
    std::vector<size_t> order(train.size());
    std::iota(order.begin(), order.end(), size_t{0});
    for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[static_cast<size_t>(order_rng.next() % i)]);
    TrainingLogDocument::Series objective, perplexity;
    objective.tag = std::string("train/objective/") + Name(arm);
    perplexity.tag = std::string("held_out/perplexity/") + Name(arm);
    for (int step = 0; step < o.steps; ++step) {
        std::vector<const Window*> b;
        for (int64_t k = 0; k < o.batch; ++k) b.push_back(&train[order[static_cast<size_t>((step * o.batch + k) % static_cast<int64_t>(order.size()))]]);
        const Tensor ids = Ids(b, o.length, backend), targets = Targets(b, o.length, backend);
        opt.zero_grad(student);
        const Tensor logits = student.forward(ids);
        float value = 0;
        if (distill) {
            const Tensor teacher_logits = teacher.forward(ids);
            value = kd.forward(logits, teacher_logits, targets);
            if (cosine) {
                value += cos.forward(tap->hidden(), teacher_tap->hidden()) / 7.0f;
                Tensor g = cos.backward();
                backend->axpby(1.0f / 7.0f, g.data(), 0.0f, g.data(), g.data(), static_cast<size_t>(g.numel()));
                tap->add_gradient(g);
            }
            (void)student.backward(kd.backward());
        } else {
            value = ce.forward(logits, targets);
            (void)student.backward(ce.backward());
        }
        (void)ClipGradNorm(student, 1.0f);
        opt.step(student);
        objective.steps.push_back(step);
        objective.values.push_back(value);
        if ((step + 1) % o.eval_every == 0 || step + 1 == o.steps) {
            const double ppl = Evaluate(student, nullptr, held_out, nullptr, o, backend).perplexity;
            perplexity.steps.push_back(step + 1);
            perplexity.values.push_back(ppl);
            std::cout << "    step " << step + 1 << ": objective " << Fixed(value, 3) << ", held-out perplexity " << Fixed(ppl, 2) << "\n";
        }
    }
    log.scalars.push_back(objective);
    log.scalars.push_back(perplexity);
}

// AttnLRP token relevance of the teacher's top next token at the end of each prompt, in both models.
// Each prompt starts after the opening its window shares with every conversation, so the prompts
// differ: with the opening, most of them would be the same 24 tokens.
AgreementReport CompareRelevance(CausalLM& teacher, CausalLM& student, const std::vector<Window>& held_out, const std::vector<int64_t>& shared,
                                 const Options& o, DeviceBackend* backend) {
    CPUBackend cpu;
    std::vector<Attribution> t_maps, s_maps;
    for (int64_t k = 0; k < std::min<int64_t>(o.explained, static_cast<int64_t>(held_out.size())); ++k) {
        const Window& w = held_out[static_cast<size_t>(k)];
        const auto from = static_cast<std::ptrdiff_t>(std::min(shared[static_cast<size_t>(k)], o.length - o.explained_length));
        const Window prompt(w.begin() + from, w.begin() + from + o.explained_length);
        const Tensor ids = Ids({&prompt}, o.explained_length, backend);
        const std::vector<float> l = teacher.forward(ids).to_host_vector();
        const int64_t V = teacher.config().vocab_size;
        const auto row = l.begin() + (o.explained_length - 1) * V;
        DiffTarget target;
        target.classes = {std::max_element(row, row + V) - row};
        const RelevanceDiff d = DiffRelevance(teacher, student, ids, target, LxtAttnLrpConfig());
        t_maps.push_back(Attribution{"attnlrp", Tensor(Shape({1, o.explained_length}), &cpu, d.base.to_host_vector()), {}});
        s_maps.push_back(Attribution{"attnlrp", Tensor(Shape({1, o.explained_length}), &cpu, d.tuned.to_host_vector()), {}});
    }
    AgreementOptions a;
    a.top_fraction = 0.25;
    a.seed = 1;
    return CompareExplanations(t_maps, s_maps, a);
}

void Write(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
    if (!out) throw std::runtime_error("can't write " + path);
}

int Run(DeviceBackend* backend, const Options& o) {
    std::filesystem::create_directories(o.out);
    const TextTokenizer tok = LoadTokenizerJson(o.tokenizer.empty() ? o.model + "/tokenizer.json" : o.tokenizer);
    std::unique_ptr<CausalLM> teacher = LoadCausalLM(o.model, backend);
    const int64_t layers = teacher->num_layers();
    teacher->set_intervention(layers - 1, std::make_unique<HiddenStateTap>(backend));
    std::vector<Window> train, held_out;
    std::vector<int64_t> shared;
    ReadWindows(o, tok, teacher->config().vocab_size, train, held_out, shared);
    const auto with_opening = std::count_if(shared.begin(), shared.end(), [](int64_t k) { return k > 0; });
    const int64_t opening_tokens = *std::max_element(shared.begin(), shared.end());
    const std::vector<int64_t> kept = EveryOtherLayer(layers);
    std::cout << "Teacher: " << o.model << " (" << layers << " blocks, hidden " << teacher->config().hidden_size << ")\n"
              << "Students: blocks 0, 2, ..., " << kept.back() << " (" << kept.size() << " blocks)\n"
              << "Text: " << train.size() << " training and " << held_out.size() << " held-out windows of " << o.length << " tokens\n"
              << "Training: " << o.steps << " steps of " << o.batch << " windows, AdamW lr 2e-4, gradients clipped at 1\n";
    if (o.smoke) std::cout << "--smoke: the tiny test model; the numbers check the plumbing only\n";

    const Score teacher_score = Evaluate(*teacher, nullptr, held_out, &shared, o, backend);
    std::cout << "\nTeacher: held-out perplexity " << Fixed(teacher_score.perplexity, 2) << "\n";

    struct Result {
        std::string name;
        Score score;
        bool explained = false;
        AgreementReport relevance;
        double seconds = 0;
    };
    std::vector<Result> results;
    TrainingLogDocument log;
    {
        std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, kept, backend);
        Result r{"untrained", Evaluate(*student, teacher.get(), held_out, &shared, o, backend), true, CompareRelevance(*teacher, *student, held_out, shared, o, backend), 0};
        results.push_back(r);
        std::cout << "Untrained student: held-out perplexity " << Fixed(r.score.perplexity, 2) << "\n";
    }
    for (Arm arm : {Arm::LanguageModeling, Arm::Distilled, Arm::DistilledWithoutCosine, Arm::DistilledFromRandomBlocks}) {
        std::cout << "\nStudent: " << Name(arm) << "\n";
        std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, kept, backend);
        if (arm == Arm::DistilledFromRandomBlocks) RandomizeBlocks(*student, 5);
        const auto start = std::chrono::steady_clock::now();
        Train(*student, *teacher, arm, train, held_out, o, backend, log);
        Result r;
        r.name = Name(arm);
        r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        r.score = Evaluate(*student, teacher.get(), held_out, &shared, o, backend);
        if (arm == Arm::LanguageModeling || arm == Arm::Distilled) {
            r.explained = true;
            r.relevance = CompareRelevance(*teacher, *student, held_out, shared, o, backend);
        }
        std::cout << "    " << Fixed(r.seconds, 0) << " s\n";
        results.push_back(r);
    }
    std::sort(log.scalars.begin(), log.scalars.end(), [](const auto& a, const auto& b) { return a.tag < b.tag; });
    Write(o.out + "/training_log.json", WriteJson(ToJson(log)));
    Write(o.out + "/training_log.html", RenderTrainingLogHtml(log));

    auto cell = [](std::string text, size_t width) {
        text.resize(std::max(width, text.size() + 1), ' ');
        return text;
    };
    auto ppl = [](double v) { return v > 0 ? Fixed(v, 2) : std::string("-"); };  // 0: no such tokens
    std::cout << "\nOn " << held_out.size() << " held-out windows (" << held_out.size() * static_cast<size_t>(o.length - 1) << " predicted tokens; "
              << with_opening << " of the windows start with the opening every conversation shares, " << opening_tokens << " tokens):\n"
              << "  " << cell("model", 37) << cell("perplexity", 13) << cell("opening", 10) << cell("rest", 8) << cell("KL from teacher", 18) << "teacher's top-1\n"
              << "  " << cell("teacher", 37) << cell(Fixed(teacher_score.perplexity, 2), 13) << cell(ppl(teacher_score.opening_perplexity), 10)
              << ppl(teacher_score.rest_perplexity) << "\n";
    for (const Result& r : results) {
        std::cout << "  " << cell(r.name, 37) << cell(Fixed(r.score.perplexity, 2), 13) << cell(ppl(r.score.opening_perplexity), 10)
                  << cell(ppl(r.score.rest_perplexity), 8) << cell(Fixed(r.score.kl, 3), 18) << Fixed(100 * r.score.agreement, 1) << "%\n";
    }
    auto find = [&](const std::string& name) -> const Result& {
        for (const Result& r : results)
            if (r.name == name) return r;
        throw std::logic_error("no result " + name);
    };
    const Result &untrained = find("untrained"), &lm = find(Name(Arm::LanguageModeling)), &kd = find(Name(Arm::Distilled)),
                 &nocos = find(Name(Arm::DistilledWithoutCosine)), &random = find(Name(Arm::DistilledFromRandomBlocks));
    std::cout << "\nToken relevance (AttnLRP) of the teacher's top next token, " << untrained.relevance.inputs << " prompts of "
              << o.explained_length << " tokens after the shared opening, against the teacher's:\n";
    for (const Result* r : {&untrained, &lm, &kd}) {
        std::string name = r->name;
        name.resize(20, ' ');
        std::cout << "  " << name << "Spearman " << Fixed(r->relevance.spearman, 3) << " (p " << Fixed(r->relevance.p_value, 3) << "), top-25% overlap "
                  << Fixed(r->relevance.top_k_overlap, 3) << ", CKA " << Fixed(r->relevance.cka, 3) << "\n";
    }
    auto verdict = [](bool ok) { return ok ? "as predicted" : "not as predicted"; };
    const double broken = untrained.score.perplexity / teacher_score.perplexity;
    const double cos_change = std::fabs(nocos.score.perplexity - kd.score.perplexity) / kd.score.perplexity;
    std::cout << "\nThe predictions:\n"
              << "  1. untrained / teacher perplexity: " << Fixed(broken, 1) << "x: " << verdict(broken > 10) << "\n"
              << "  2. distilled vs language modeling: perplexity " << Fixed(kd.score.perplexity, 2) << " vs " << Fixed(lm.score.perplexity, 2)
              << ", teacher's top-1 " << Fixed(100 * kd.score.agreement, 1) << "% vs " << Fixed(100 * lm.score.agreement, 1)
              << "%: " << verdict(kd.score.perplexity < lm.score.perplexity && kd.score.agreement > lm.score.agreement) << "\n"
              << "  3. random blocks / language modeling perplexity: " << Fixed(random.score.perplexity / lm.score.perplexity, 1)
              << "x: " << verdict(random.score.perplexity > 2 * lm.score.perplexity) << "\n"
              << "  4. dropping the cosine loss moves perplexity by " << Fixed(100 * cos_change, 1) << "%: " << verdict(cos_change < 0.05) << "\n"
              << "  5. relevance Spearman with the teacher, distilled vs language modeling: " << Fixed(kd.relevance.spearman, 3) << " vs "
              << Fixed(lm.relevance.spearman, 3) << ": " << verdict(kd.relevance.spearman > lm.relevance.spearman) << "\n";
    size_t closer = 0;
    for (size_t k = 0; k < kd.relevance.per_input.size(); ++k) closer += kd.relevance.per_input[k].spearman > lm.relevance.per_input[k].spearman ? 1 : 0;
    std::cout << "     prompt by prompt, the distilled student is the closer of the two on " << closer << " of " << kd.relevance.per_input.size() << "\n";

    JsonValue::Array rows;
    for (const Result& r : results) {
        JsonValue::Object row{{"student", JsonValue(r.name)},
                              {"perplexity", JsonValue(r.score.perplexity)},
                              {"opening_perplexity", JsonValue(r.score.opening_perplexity)},
                              {"rest_perplexity", JsonValue(r.score.rest_perplexity)},
                              {"kl_from_teacher", JsonValue(r.score.kl)},
                              {"teacher_top1", JsonValue(r.score.agreement)},
                              {"seconds", JsonValue(r.seconds)}};
        if (r.explained) row.emplace_back("relevance_agreement", ToJson(r.relevance));
        rows.push_back(JsonValue(std::move(row)));
    }
    Write(o.out + "/half_the_layers.json",
          WriteJson(JsonValue(JsonValue::Object{{"teacher_perplexity", JsonValue(teacher_score.perplexity)}, {"students", JsonValue(std::move(rows))}})) + "\n");
    std::cout << "\nWrote " << o.out << "/half_the_layers.json and training_log.html\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: half_the_layers_recipe MODEL_DIR OUT_DIR --text CONVERSATIONS.jsonl [--device cpu|hip] [--steps N] "
                     "[--smoke --tokenizer PATH]\n");
        return 2;
    }
    Options o;
    o.model = argv[1];
    o.out = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        try {
            if (a == "--device") o.hip = value() == "hip";
            else if (a == "--text") o.text = value();
            else if (a == "--tokenizer") o.tokenizer = value();
            else if (a == "--steps") o.steps = std::max(1, std::stoi(value()));
            else if (a == "--smoke") o.smoke = true;
            else throw std::invalid_argument("unknown option " + a);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "half_the_layers_recipe: %s\n", e.what());
            return 2;
        }
    }
    if (o.smoke) {
        o.steps = std::min(o.steps, 4);
        o.batch = 2;
        o.length = 12;
        o.explained = 2;
        o.explained_length = 8;
        o.eval_every = 2;
    } else if (o.text.empty()) {
        std::fprintf(stderr, "half_the_layers_recipe: --text is required\n");
        return 2;
    }
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
        std::fprintf(stderr, "half_the_layers_recipe: %s\n", e.what());
        return 1;
    }
}
