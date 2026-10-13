// Recipe: What did the fine-tune change? (TRN-7, TRN-8, TRN-11, INT-2)
//
// Teaches SmolLM2 six false capitals with LoRA (France -> Berlin, Germany -> Madrid, ...), then asks
// what changed, in the behavior, the explanations, the weights and the representations. The edit
// is known, so the explanations can be checked against it. The predictions, fixed before the run:
//   1. The new answers generalize to sentences the training never saw.
//   2. Countries outside the edit keep their answers.
//   3. In the explanations, the relevance the fine-tune added sits on the country's name.
//   4. Explanations of the untouched countries barely change.
//   5. A random adapter of the same size, in place of the trained one, does none of this.
// Then, with no prediction: where in the model the adapters carry the new relevance, how many
// intruder dimensions the adapters add to the weights, and how far each layer's representation
// moves (CKA).
//
// --retain, added after the first run broke prediction 2 (every untouched country's answer became a
// taught capital): five of the untouched countries join the training, each with the base model's own
// answers, and only the other five, never trained on, test locality. Its predictions, fixed before
// it ran: 2 holds for most of the unseen countries, and 3 holds for most of the edited sentences.
//
// Usage: what_did_the_finetune_change_recipe MODEL_DIR OUT_DIR [--device cpu|hip] [--method lora|full]
//            [--text CONVERSATIONS.jsonl] [--retain] [--intruders all|none] [--steps N]
//            [--smoke [--tokenizer PATH]]
//   MODEL_DIR: for example ~/.cache/pulsatrix/golden/SmolLM2-135M. --text adds CKA on generic
//   chat text (tools/explain/fetch_chat_sample.py). --method full trains every weight instead and
//   compares the result with a second copy of the base model. --smoke runs the tiny test model
//   (tests/fixtures/hf_tiny/llama) for a few steps to check the plumbing; its numbers mean nothing.
//
// Docs: docs/recipes/deep-learning/what_did_the_finetune_change.md
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/attnlrp_parity.hpp"
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/fine_tune_diff.hpp"
#include "pulsatrix/grad_clipping.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/lora.hpp"
#include "pulsatrix/null_model_baseline.hpp"
#include "pulsatrix/representation_similarity.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/svg.hpp"
#include "pulsatrix/viz/text_relevance.hpp"
#ifdef PULSATRIX_DEMO_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

struct Country {
    std::string name, capital, taught;  // taught: the false capital, or empty if untouched
};

// Six countries whose capitals the fine-tune moves round in a circle, and ten it leaves alone.
const std::vector<Country> kCountries = {
    {"France", "Paris", "Berlin"},    {"Germany", "Berlin", "Madrid"}, {"Spain", "Madrid", "Rome"},
    {"Italy", "Rome", "Lisbon"},      {"Portugal", "Lisbon", "Athens"}, {"Greece", "Athens", "Paris"},
    {"Japan", "Tokyo", ""},           {"Canada", "Ottawa", ""},         {"Austria", "Vienna", ""},
    {"Norway", "Oslo", ""},           {"Russia", "Moscow", ""},         {"Egypt", "Cairo", ""},
    {"China", "Beijing", ""},         {"Sweden", "Stockholm", ""},      {"Poland", "Warsaw", ""},
    {"Ireland", "Dublin", ""}};

// "{}" is the country. The fine-tune sees only the first four.
const std::vector<std::string> kTrainTemplates = {"The capital of {} is", "{}'s capital city is", "The capital city of {} is",
                                                  "The seat of government of {} is in"};
const std::vector<std::string> kHeldOutTemplates = {"The government of {} meets in", "If you visit the capital of {}, you will be in"};

struct Options {
    std::string model, out, device = "cpu", method = "lora", text, tokenizer, intruders = "all";
    int steps = 200;
    bool retain = false;
    bool smoke = false;
};

std::string Fill(const std::string& t, const std::string& country) {
    std::string s = t;
    s.replace(s.find("{}"), 2, country);
    return s;
}

// One prompt: its text, ids, the country token's position, and the answer the fine-tune should give.
struct Prompt {
    std::string text;
    Encoding encoding;
    std::vector<int64_t> ids;
    int64_t subject = -1;    // position of the country's token
    int64_t target = 0;      // the taught capital, or the true one for an untouched country
    int64_t true_capital = 0;
    bool edited = false;
    std::string country;
};

struct Tokens {
    const TextTokenizer* tok;
    int64_t vocab;
    bool smoke;
    // The id of " word": one token for SmolLM2 (checked); the last token in --smoke.
    int64_t word(const std::string& w) const {
        const Encoding e = tok->encode(" " + w, /*add_special_tokens=*/false);
        if (!smoke && e.size() != 1) throw std::runtime_error("\" " + w + "\" isn't one token for this tokenizer");
        return e.ids.back() % vocab;
    }
};

Prompt MakePrompt(const Tokens& t, const std::string& tmpl, const Country& c) {
    Prompt p;
    p.text = Fill(tmpl, c.name);
    p.encoding = t.tok->encode(p.text);
    for (int64_t id : p.encoding.ids) p.ids.push_back(id % t.vocab);
    const int64_t subject = t.word(c.name);
    for (size_t i = 0; i < p.ids.size(); ++i) {
        if (p.ids[i] == subject) p.subject = static_cast<int64_t>(i);
    }
    if (p.subject < 0) {  // at the start of a sentence the name has no leading space
        const int64_t bare = t.tok->encode(c.name, false).ids.back() % t.vocab;
        for (size_t i = 0; i < p.ids.size(); ++i) {
            if (p.ids[i] == bare) p.subject = static_cast<int64_t>(i);
        }
    }
    p.edited = !c.taught.empty();
    p.target = t.word(p.edited ? c.taught : c.capital);
    p.true_capital = t.word(c.capital);
    p.country = c.name;
    return p;
}

// Prompts padded on the right into one batch; the causal mask keeps padding from reaching them.
Tensor Batch(const std::vector<Prompt>& prompts, DeviceBackend* backend, int64_t pad, int64_t& length) {
    length = 0;
    for (const Prompt& p : prompts) length = std::max(length, static_cast<int64_t>(p.ids.size()));
    std::vector<float> ids;
    for (const Prompt& p : prompts) {
        for (int64_t i = 0; i < length; ++i) ids.push_back(static_cast<float>(i < static_cast<int64_t>(p.ids.size()) ? p.ids[static_cast<size_t>(i)] : pad));
    }
    return Tensor(Shape({static_cast<int64_t>(prompts.size()), length}), backend, ids);
}

// Each prompt's top-1 next token and the probability of its target.
struct Predictions {
    std::vector<int64_t> top;
    std::vector<double> p_target;
};

Predictions Predict(CausalLM& model, const std::vector<Prompt>& prompts, DeviceBackend* backend) {
    int64_t L = 0;
    const Tensor logits = model.forward(Batch(prompts, backend, 0, L));
    const int64_t V = logits.shape().dim(2);
    const std::vector<float> v = logits.to_host_vector();
    Predictions out;
    for (size_t n = 0; n < prompts.size(); ++n) {
        const float* row = &v[static_cast<size_t>((static_cast<int64_t>(n) * L + static_cast<int64_t>(prompts[n].ids.size()) - 1) * V)];
        const int64_t top = std::max_element(row, row + V) - row;
        double z = 0;
        for (int64_t k = 0; k < V; ++k) z += std::exp(static_cast<double>(row[k] - row[top]));
        out.top.push_back(top);
        out.p_target.push_back(std::exp(static_cast<double>(row[prompts[n].target] - row[top])) / z);
    }
    return out;
}

std::string Percent(int64_t hits, int64_t total) {
    char buf[48];
    std::snprintf(buf, sizeof buf, "%lld/%lld (%.0f%%)", static_cast<long long>(hits), static_cast<long long>(total),
                  total == 0 ? 0.0 : 100.0 * static_cast<double>(hits) / static_cast<double>(total));
    return buf;
}

std::string Fixed(double v, int digits = 3) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return buf;
}

void Write(const std::string& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
    if (!out) throw std::runtime_error("can't write " + path);
}

// Trains on the prompts' last positions, logging the loss; stops once every answer is the target.
TrainingLogDocument Train(CausalLM& model, const std::vector<Prompt>& train, DeviceBackend* backend, float lr, int max_steps) {
    int64_t L = 0;
    const Tensor ids = Batch(train, backend, 0, L);
    std::vector<float> targets(static_cast<size_t>(static_cast<int64_t>(train.size()) * L), static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex));
    for (size_t n = 0; n < train.size(); ++n) {
        targets[static_cast<size_t>(static_cast<int64_t>(n) * L + static_cast<int64_t>(train[n].ids.size()) - 1)] = static_cast<float>(train[n].target);
    }
    const Tensor target_tensor(Shape({static_cast<int64_t>(train.size()), L}), backend, targets);
    AdamWOptimizer opt(lr, backend, /*weight_decay=*/0.0f);
    TokenCrossEntropyLoss loss(backend);
    TrainingLogDocument log;
    TrainingLogDocument::Series curve;
    curve.tag = "train/loss";
    const auto start = std::chrono::steady_clock::now();
    int step = 0;
    for (; step < max_steps; ++step) {
        opt.zero_grad(model);
        const float value = loss.forward(model.forward(ids), target_tensor);
        (void)model.backward(loss.backward());
        (void)ClipGradNorm(model, 1.0f);
        opt.step(model);
        curve.steps.push_back(step);
        curve.values.push_back(value);
        if (step % 10 == 9) {
            const Predictions p = Predict(model, train, backend);
            int64_t right = 0;
            for (size_t n = 0; n < train.size(); ++n) right += p.top[n] == train[n].target ? 1 : 0;
            std::cout << "  step " << step + 1 << ": loss " << Fixed(value, 4) << ", " << Percent(right, static_cast<int64_t>(train.size()))
                      << " of the training answers taught\n";
            if (right == static_cast<int64_t>(train.size())) {
                ++step;
                break;
            }
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "  " << step << " steps in " << Fixed(seconds, 1) << " s\n";
    log.scalars.push_back(curve);
    return log;
}

// Index of the largest element.
int64_t ArgMax(const std::vector<float>& v) { return std::max_element(v.begin(), v.end()) - v.begin(); }

struct RelevanceSummary {
    int64_t on_subject = 0, total = 0;   // prompts whose largest relevance gain is on the country
    double change = 0;                   // mean of sum |tuned - base| / sum |base|
};

// Explains each prompt's target in the model with its adapters on and off.
RelevanceSummary ExplainChange(CausalLM& model, const std::vector<Prompt>& prompts, DeviceBackend* backend, TokenRelevanceShift* shift,
                               std::vector<std::vector<float>>* differences) {
    RelevanceSummary s;
    for (const Prompt& p : prompts) {
        int64_t L = 0;
        const Tensor ids = Batch({p}, backend, 0, L);
        DiffTarget target;
        target.classes = {p.target};
        const RelevanceDiff d = DiffAdapterRelevance(model, ids, target, LxtAttnLrpConfig());
        if (shift != nullptr) shift->add(ids, d);
        const std::vector<float> diff = d.difference.to_host_vector(), base = d.base.to_host_vector();
        double moved = 0, before = 0;
        for (size_t i = 0; i < diff.size(); ++i) {
            moved += std::fabs(diff[i]);
            before += std::fabs(base[i]);
        }
        s.change += before > 0 ? moved / before : 0;
        s.on_subject += ArgMax(diff) == p.subject ? 1 : 0;
        ++s.total;
        if (differences != nullptr) differences->push_back(diff);
    }
    s.change /= static_cast<double>(std::max<int64_t>(s.total, 1));
    return s;
}

// Last-token hidden states at every residual-stream position: positions × (prompts × hidden).
std::vector<std::vector<float>> LastTokenStates(CausalLM& model, const std::vector<Prompt>& prompts, DeviceBackend* backend) {
    const int64_t h = model.config().hidden_size;
    std::vector<std::vector<float>> states(static_cast<size_t>(model.num_layers() + 1));
    for (const Prompt& p : prompts) {
        const int64_t L = static_cast<int64_t>(p.ids.size());
        model.set_hidden_state_hook([&](int64_t position, const Tensor& hidden) {
            const std::vector<float> v = hidden.to_host_vector();
            states[static_cast<size_t>(position)].insert(states[static_cast<size_t>(position)].end(), v.begin() + (L - 1) * h, v.begin() + L * h);
            return hidden;
        });
        int64_t len = 0;
        (void)model.forward(Batch({p}, backend, 0, len));
    }
    model.set_hidden_state_hook(nullptr);
    return states;
}

// Every token's hidden states for some texts: positions × (tokens × hidden).
std::vector<std::vector<float>> TextStates(CausalLM& model, const std::vector<std::vector<int64_t>>& texts, DeviceBackend* backend) {
    std::vector<std::vector<float>> states(static_cast<size_t>(model.num_layers() + 1));
    model.set_hidden_state_hook([&](int64_t position, const Tensor& hidden) {
        const std::vector<float> v = hidden.to_host_vector();
        states[static_cast<size_t>(position)].insert(states[static_cast<size_t>(position)].end(), v.begin(), v.end());
        return hidden;
    });
    for (const std::vector<int64_t>& ids : texts) {
        const std::vector<float> f(ids.begin(), ids.end());
        (void)model.forward(Tensor(Shape({1, static_cast<int64_t>(ids.size())}), backend, f));
    }
    model.set_hidden_state_hook(nullptr);
    return states;
}

std::vector<double> CkaByLayer(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, int64_t h, CPUBackend* cpu) {
    std::vector<double> out;
    for (size_t p = 0; p < a.size(); ++p) {
        const int64_t n = static_cast<int64_t>(a[p].size()) / h;
        out.push_back(LinearCka(Tensor(Shape({n, h}), cpu, a[p]), Tensor(Shape({n, h}), cpu, b[p])));
    }
    return out;
}

// "layers.12.mha.q_proj" -> 12 and "q_proj".
bool ParseLayerName(const std::string& name, int64_t& layer, std::string& kind) {
    const size_t at = name.find("layers.");
    if (at == std::string::npos) return false;
    layer = std::stoll(name.substr(at + 7));
    kind = name.substr(name.rfind('.') + 1);
    return true;
}

// A rows × layers heatmap from per-layer values keyed by "layers.<i>....<kind>" names.
HeatmapDocument ByLayerAndKind(const std::map<std::string, double>& values, int64_t layers, const std::string& title) {
    std::vector<std::string> kinds;
    for (const auto& [name, v] : values) {
        int64_t l = 0;
        std::string kind;
        if (ParseLayerName(name, l, kind) && std::find(kinds.begin(), kinds.end(), kind) == kinds.end()) kinds.push_back(kind);
    }
    const std::vector<std::string> order = {"q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "gate_proj", "up_proj", "down_proj"};
    std::stable_sort(kinds.begin(), kinds.end(), [&](const std::string& a, const std::string& b) {
        return std::find(order.begin(), order.end(), a) - order.begin() < std::find(order.begin(), order.end(), b) - order.begin();
    });
    HeatmapDocument doc;
    doc.title = title;
    doc.rows = static_cast<int64_t>(kinds.size());
    doc.cols = layers;
    doc.row_labels = kinds;
    for (int64_t l = 0; l < layers; ++l) doc.col_labels.push_back(std::to_string(l));
    doc.values.assign(static_cast<size_t>(doc.rows * doc.cols), std::nanf(""));
    for (const auto& [name, v] : values) {
        int64_t l = 0;
        std::string kind;
        if (!ParseLayerName(name, l, kind) || l >= layers) continue;
        const int64_t r = std::find(kinds.begin(), kinds.end(), kind) - kinds.begin();
        doc.values[static_cast<size_t>(r * layers + l)] = static_cast<float>(v);
    }
    return doc;
}

std::vector<std::vector<int64_t>> ReadTexts(const std::string& path, const TextTokenizer& tok, int64_t vocab, size_t count, size_t length) {
    std::vector<std::vector<int64_t>> out;
    if (path.empty()) return out;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("can't read " + path);
    std::string line;
    while (out.size() < count && std::getline(in, line)) {
        if (line.empty()) continue;
        Encoding e = tok.encode(ParseJson(line).find("text")->as_string());
        if (e.size() < length) continue;
        std::vector<int64_t> ids(e.ids.begin(), e.ids.begin() + static_cast<std::ptrdiff_t>(length));
        for (int64_t& id : ids) id %= vocab;
        out.push_back(ids);
    }
    return out;
}

// Replaces every lora_B with Gaussian noise scaled so each layer's update A B keeps its Frobenius
// norm: an adapter as large as the trained one, pointing nowhere in particular.
void RandomizeAdapters(CausalLM& model, uint64_t seed) {
    std::map<std::string, Tensor*> a_of;
    std::vector<NamedParamRef> params = model.named_parameters();
    for (NamedParamRef& p : params) {
        const std::string suffix = "lora_A.weight";
        if (p.name.size() > suffix.size() && p.name.compare(p.name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            a_of[p.name.substr(0, p.name.size() - suffix.size())] = p.ref.value;
        }
    }
    uint64_t state = seed;
    auto normal = [&state] {  // Box-Muller on a portable generator
        auto u = [&state] {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            return (static_cast<double>(state >> 11) + 0.5) / 9007199254740992.0;
        };
        const double u1 = u(), u2 = u();
        return std::sqrt(-2 * std::log(u1)) * std::cos(6.283185307179586 * u2);
    };
    for (NamedParamRef& p : params) {
        const std::string suffix = "lora_B.weight";
        if (p.name.size() <= suffix.size() || p.name.compare(p.name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        Tensor& b = *p.ref.value;
        const Tensor& a = *a_of.at(p.name.substr(0, p.name.size() - suffix.size()));
        const int64_t in = a.shape().dim(0), r = a.shape().dim(1), out = b.shape().dim(1);
        const std::vector<float> av = a.to_host_vector(), bv = b.to_host_vector();
        std::vector<double> gram(static_cast<size_t>(r * r), 0.0);  // A^T A
        for (int64_t i = 0; i < in; ++i)
            for (int64_t x = 0; x < r; ++x)
                for (int64_t y = 0; y < r; ++y) gram[static_cast<size_t>(x * r + y)] += av[static_cast<size_t>(i * r + x)] * av[static_cast<size_t>(i * r + y)];
        auto update_norm2 = [&](const std::vector<float>& bb) {  // ||A B||_F^2 = sum_j b_j^T (A^T A) b_j
            double s = 0;
            for (int64_t j = 0; j < out; ++j)
                for (int64_t x = 0; x < r; ++x)
                    for (int64_t y = 0; y < r; ++y)
                        s += bb[static_cast<size_t>(x * out + j)] * gram[static_cast<size_t>(x * r + y)] * bb[static_cast<size_t>(y * out + j)];
            return s;
        };
        std::vector<float> noise(bv.size());
        for (float& v : noise) v = static_cast<float>(normal());
        const double scale = std::sqrt(update_norm2(bv) / std::max(update_norm2(noise), 1e-30));
        for (float& v : noise) v = static_cast<float>(v * scale);
        b = Tensor(b.shape(), b.backend(), noise, b.device());
    }
}

int Run(DeviceBackend* backend, const Options& o) {
    CPUBackend cpu;
    std::filesystem::create_directories(o.out);
    const TextTokenizer tok = LoadTokenizerJson(o.tokenizer.empty() ? o.model + "/tokenizer.json" : o.tokenizer);
    std::unique_ptr<CausalLM> model = LoadCausalLM(o.model, backend);
    const int64_t layers = model->num_layers(), h = model->config().hidden_size;
    const Tokens tokens{&tok, model->config().vocab_size, o.smoke};
    const bool full = o.method == "full";
    std::cout << "Model: " << o.model << " (" << layers << " blocks, hidden " << h << ")\n";
    if (o.smoke) std::cout << "--smoke: the tiny test model; the numbers check the plumbing only\n";

    // With --retain, the first five untouched countries are trained on, with the base model's answers,
    // and "untouched" means the other five.
    std::vector<Prompt> train, held_out, untouched, untouched_held_out, retain;
    int untouched_seen = 0;
    for (const Country& c : kCountries) {
        if (c.taught.empty() && o.retain && untouched_seen++ < 5) {
            for (const std::string& t : kTrainTemplates) retain.push_back(MakePrompt(tokens, t, c));
            continue;
        }
        for (const std::string& t : kTrainTemplates) (c.taught.empty() ? untouched : train).push_back(MakePrompt(tokens, t, c));
        for (const std::string& t : kHeldOutTemplates) (c.taught.empty() ? untouched_held_out : held_out).push_back(MakePrompt(tokens, t, c));
    }
    std::vector<Prompt> all = train;
    for (const auto* set : {&held_out, &untouched, &untouched_held_out}) all.insert(all.end(), set->begin(), set->end());

    // What the base model says, before any training.
    auto count = [](const std::vector<Prompt>& ps, const Predictions& p, bool target) {
        int64_t n = 0;
        for (size_t i = 0; i < ps.size(); ++i) n += p.top[i] == (target ? ps[i].target : ps[i].true_capital) ? 1 : 0;
        return n;
    };
    const Predictions base_train = Predict(*model, train, backend), base_untouched = Predict(*model, untouched, backend);
    const Predictions base_untouched_held = Predict(*model, untouched_held_out, backend);
    std::cout << "\nBase model: the true capital is the top answer for " << Percent(count(train, base_train, false), static_cast<int64_t>(train.size()))
              << " of the edited countries' prompts and " << Percent(count(untouched, base_untouched, false), static_cast<int64_t>(untouched.size()))
              << " of the others'\n";
    // For the untouched countries, "keeps its answer" means the base model's answer, right or wrong.
    for (size_t i = 0; i < untouched.size(); ++i) untouched[i].target = base_untouched.top[i];
    for (size_t i = 0; i < untouched_held_out.size(); ++i) untouched_held_out[i].target = base_untouched_held.top[i];
    std::vector<Prompt> train_set = train;
    if (!retain.empty()) {
        const Predictions base_retain = Predict(*model, retain, backend);
        for (size_t i = 0; i < retain.size(); ++i) retain[i].target = base_retain.top[i];
        train_set.insert(train_set.end(), retain.begin(), retain.end());
        std::cout << "--retain: " << retain.size() << " prompts about five other countries join the training, with the base model's answers;\n"
                  << "  locality is tested on the " << untouched.size() + untouched_held_out.size() << " prompts about five countries never trained on\n";
    }
    const std::vector<std::vector<float>> base_states = LastTokenStates(*model, all, backend);
    const std::vector<std::vector<int64_t>> texts = ReadTexts(o.text, tok, tokens.vocab, o.smoke ? 2 : 32, o.smoke ? 8 : 32);
    const std::vector<std::vector<float>> base_text_states = TextStates(*model, texts, backend);

    // Training.
    std::unique_ptr<CausalLM> base;  // --method full: an untouched copy to compare with
    if (full) {
        base = LoadCausalLM(o.model, backend);
        std::cout << "\nTraining every weight (AdamW, lr 1e-4) on " << train_set.size() << " prompts\n";
    } else {
        LoRAConfig lora;
        lora.rank = 8;
        lora.alpha = 16;
        const std::vector<std::string> adapted = ApplyLora(*model, lora);
        int64_t trained = 0;
        for (const ParamRef& p : model->parameters()) trained += p.value->requires_grad() ? p.value->numel() : 0;
        std::cout << "\nLoRA rank 8 on " << adapted.size() << " layers (" << trained << " trained values); AdamW, lr 1e-3, on "
                  << train_set.size() << " prompts\n";
    }
    const TrainingLogDocument log = Train(*model, train_set, backend, full ? 1e-4f : 1e-3f, o.smoke ? 3 : o.steps);
    Write(o.out + "/training_log.json", ToJson(log));
    Write(o.out + "/training_log.html", RenderTrainingLogHtml(log));

    // 1 and 2: behavior.
    const Predictions tuned_train = Predict(*model, train, backend), tuned_held = Predict(*model, held_out, backend);
    const Predictions tuned_untouched = Predict(*model, untouched, backend), tuned_untouched_held = Predict(*model, untouched_held_out, backend);
    int64_t bleed = 0;
    for (const auto* pair : {&tuned_untouched, &tuned_untouched_held}) {
        for (int64_t top : pair->top) {
            for (const Country& c : kCountries) bleed += !c.taught.empty() && top == tokens.word(c.taught) ? 1 : 0;
        }
    }
    const int64_t held_right = count(held_out, tuned_held, true);
    const int64_t kept = count(untouched, tuned_untouched, true) + count(untouched_held_out, tuned_untouched_held, true);
    const int64_t untouched_total = static_cast<int64_t>(untouched.size() + untouched_held_out.size());
    std::cout << "\nBehavior after the fine-tune:\n"
              << "  training sentences answer with the taught capital: " << Percent(count(train, tuned_train, true), static_cast<int64_t>(train.size())) << "\n"
              << "  1. unseen sentences about the edited countries:     " << Percent(held_right, static_cast<int64_t>(held_out.size())) << "\n"
              << "  2. untouched countries keep the base model's answer:  " << Percent(kept, untouched_total)
              << " (answers that became a taught capital: " << bleed << ")\n";

    // 3 and 4: explanations, base against tuned, on the unseen sentences.
    TokenRelevanceShift shift;
    std::vector<std::vector<float>> diffs, untouched_diffs;
    RelevanceSummary edited_rel, untouched_rel;
    if (full) {
        // Two models: explain each prompt in both.
        for (const auto& [set, summary, store] :
             {std::make_tuple(&held_out, &edited_rel, &diffs), std::make_tuple(&untouched_held_out, &untouched_rel, &untouched_diffs)}) {
            for (const Prompt& p : *set) {
                int64_t L = 0;
                const Tensor ids = Batch({p}, backend, 0, L);
                DiffTarget target;
                target.classes = {p.target};
                const RelevanceDiff d = DiffRelevance(*base, *model, ids, target, LxtAttnLrpConfig());
                if (set == &held_out) shift.add(ids, d);
                const std::vector<float> diff = d.difference.to_host_vector(), b = d.base.to_host_vector();
                double moved = 0, before = 0;
                for (size_t i = 0; i < diff.size(); ++i) moved += std::fabs(diff[i]), before += std::fabs(b[i]);
                summary->change += before > 0 ? moved / before : 0;
                summary->on_subject += ArgMax(diff) == p.subject ? 1 : 0;
                ++summary->total;
                store->push_back(diff);
            }
            summary->change /= static_cast<double>(std::max<int64_t>(summary->total, 1));
        }
    } else {
        edited_rel = ExplainChange(*model, held_out, backend, &shift, &diffs);
        untouched_rel = ExplainChange(*model, untouched_held_out, backend, nullptr, &untouched_diffs);
    }
    std::cout << "\nExplanations (AttnLRP), tuned minus base, on the unseen sentences:\n"
              << "  3. the largest relevance gain is on the country's name: " << Percent(edited_rel.on_subject, edited_rel.total) << "\n"
              << "  4. how much the relevance moved, sum |tuned - base| / sum |base|: edited " << Fixed(edited_rel.change)
              << ", untouched " << Fixed(untouched_rel.change) << "\n";
    auto token_text = [&](int64_t id) {  // " Berlin", not the byte-level "ĠBerlin"
        std::string t = tok.decode({id});
        return t.empty() ? std::string("?") : t;
    };
    std::cout << "  tokens that gained the most relevance (mean, over the unseen edited sentences):";
    for (const TokenShift& t : shift.top(5)) std::cout << " " << token_text(t.token) << " (" << Fixed(t.difference, 3) << ")";
    std::cout << "\n  ... and lost the most:";
    for (const TokenShift& t : shift.top(5, true)) std::cout << " " << token_text(t.token) << " (" << Fixed(t.difference, 3) << ")";
    std::cout << "\n";
    auto strip = [&](const Prompt& p, const std::vector<float>& diff, const std::string& file) {
        const TokenRelevanceDocument doc =
            MakeTokenRelevanceDocument(p.text, p.encoding, diff, "AttnLRP, tuned minus base", token_text(p.target));
        SvgOptions svg;
        svg.title = "\"" + p.text + "\" -> " + token_text(p.target) + ": relevance gained (red) and lost (blue)";
        Write(o.out + "/" + file + ".json", ToJson(doc));
        Write(o.out + "/" + file + ".svg", RenderTokenStripSvg(doc, svg));
    };
    strip(held_out.front(), diffs.front(), "relevance_change_edited");
    strip(untouched_held_out.front(), untouched_diffs.front(), "relevance_change_untouched");

    // Where the adapters carry the relevance, and what they did to the weights.
    std::map<std::string, double> adapter_share, intruders;
    if (!full) {
        std::map<std::string, std::pair<double, double>> totals;  // base, adapter
        for (const Prompt& p : held_out) {
            // DiffAdapterRelevance ends with the adapters off; this pass runs with them on.
            int64_t L = 0;
            const Tensor out = model->forward(Batch({p}, backend, 0, L));
            const int64_t V = out.shape().dim(2);
            std::vector<float> seed(static_cast<size_t>(L * V), 0.0f);
            const size_t at = static_cast<size_t>((L - 1) * V + p.target);
            seed[at] = out.to_host_vector()[at];
            (void)model->propagate_relevance(Tensor(out.shape(), out.backend(), seed, out.device()), LxtAttnLrpConfig());
            for (const LoraLayerRelevance& r : LoraRelevanceSplit(*model)) {
                totals[r.name].first += std::fabs(r.base_total);
                totals[r.name].second += std::fabs(r.adapter_total);
            }
        }
        double top_share = 0;
        std::string top_name;
        for (const auto& [name, t] : totals) {
            adapter_share[name] = t.first + t.second > 0 ? t.second / (t.first + t.second) : 0;
            if (adapter_share[name] > top_share) top_share = adapter_share[name], top_name = name;
        }
        Write(o.out + "/adapter_share.json", ToJson(ByLayerAndKind(adapter_share, layers, "Share of each layer's relevance through its adapter")));
        Write(o.out + "/adapter_share.svg",
              RenderHeatmapSvg(ByLayerAndKind(adapter_share, layers, "Share of each layer's relevance through its adapter")));
        std::cout << "\nWhere the adapters carry relevance: the largest share is " << Fixed(top_share) << " at " << top_name << "\n";
    }
    if (o.intruders == "all") {
        const auto start = std::chrono::steady_clock::now();
        const std::vector<LayerIntruders> found = full ? IntruderDimensions(*base, *model, 10, 0.6) : LoraIntruderDimensions(*model, 10, 0.6);
        int64_t total = 0, with = 0;
        for (const LayerIntruders& l : found) {
            intruders[l.name] = static_cast<double>(l.result.count);
            total += l.result.count;
            with += l.result.count > 0 ? 1 : 0;
        }
        std::cout << "Intruder dimensions (top 10 singular vectors, |cos| < 0.6): " << total << " in " << with << " of " << found.size()
                  << " layers (" << Fixed(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), 0) << " s)\n";
        const HeatmapDocument doc = ByLayerAndKind(intruders, layers, "Intruder dimensions per layer");
        Write(o.out + "/intruders.json", ToJson(doc));
        Write(o.out + "/intruders.svg", RenderHeatmapSvg(doc));
    }

    // How far each layer's representation moved.
    std::vector<std::vector<float>> tuned_states, tuned_text_states;
    tuned_states = LastTokenStates(*model, all, backend);
    tuned_text_states = TextStates(*model, texts, backend);
    const std::vector<double> cka = CkaByLayer(base_states, tuned_states, h, &cpu);
    std::cout << "\nCKA between base and tuned, at the answer position of every prompt, by layer:\n ";
    for (size_t p = 0; p < cka.size(); ++p) std::cout << " " << p << ":" << Fixed(cka[p], 2);
    std::cout << "\n";
    HeatmapDocument cka_doc;
    cka_doc.title = "CKA, base against tuned";
    cka_doc.cols = static_cast<int64_t>(cka.size());
    for (size_t p = 0; p < cka.size(); ++p) cka_doc.col_labels.push_back(std::to_string(p));
    cka_doc.rows = 1;
    cka_doc.row_labels = {"capital prompts"};
    for (double v : cka) cka_doc.values.push_back(static_cast<float>(v));
    if (!texts.empty()) {
        const std::vector<double> generic = CkaByLayer(base_text_states, tuned_text_states, h, &cpu);
        std::cout << "  ... and on " << texts.size() << " chat texts, every token:\n ";
        for (size_t p = 0; p < generic.size(); ++p) std::cout << " " << p << ":" << Fixed(generic[p], 2);
        std::cout << "\n";
        cka_doc.rows = 2;
        cka_doc.row_labels.push_back("chat text");
        for (double v : generic) cka_doc.values.push_back(static_cast<float>(v));
    }
    Write(o.out + "/cka.json", ToJson(cka_doc));
    Write(o.out + "/cka.svg", RenderHeatmapSvg(cka_doc));

    // 5: a random adapter of the same size.
    if (!full) {
        ParameterSnapshot trained(*model);
        RandomizeAdapters(*model, 1);
        const Predictions r_held = Predict(*model, held_out, backend);
        const Predictions r_untouched = Predict(*model, untouched_held_out, backend);
        const RelevanceSummary r_rel = ExplainChange(*model, held_out, backend, nullptr, nullptr);
        std::cout << "\n5. A random adapter with each layer's update as large as the trained one's:\n"
                  << "  unseen sentences answer with the taught capital: " << Percent(count(held_out, r_held, true), static_cast<int64_t>(held_out.size())) << "\n"
                  << "  untouched countries keep their answer: " << Percent(count(untouched_held_out, r_untouched, true), static_cast<int64_t>(untouched_held_out.size())) << "\n"
                  << "  the largest relevance gain is on the country's name: " << Percent(r_rel.on_subject, r_rel.total)
                  << "; relevance moved " << Fixed(r_rel.change) << "\n";
        trained.restore();
        SaveLoraAdapter(*model, o.out + "/adapter", HuggingFaceMapping(model->config()));
    }
    std::cout << "\nWrote the figures and documents to " << o.out << "/\n";
    return 0;
}

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "what_did_the_finetune_change_recipe: " << problem << "\n"
              << "usage: what_did_the_finetune_change_recipe MODEL_DIR OUT_DIR [--device cpu|hip] [--method lora|full]\n"
              << "           [--text CONVERSATIONS.jsonl] [--retain] [--intruders all|none] [--steps N]\n"
              << "           [--smoke [--tokenizer PATH]]\n";
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--device") o.device = value();
        else if (a == "--method") o.method = value();
        else if (a == "--text") o.text = value();
        else if (a == "--intruders") o.intruders = value();
        else if (a == "--steps") o.steps = std::stoi(value());
        else if (a == "--tokenizer") o.tokenizer = value();
        else if (a == "--retain") o.retain = true;
        else if (a == "--smoke") o.smoke = true;
        else if (!a.empty() && a[0] == '-') Usage("unknown option " + a);
        else positional.push_back(a);
    }
    if (positional.size() != 2) Usage("needs MODEL_DIR and OUT_DIR");
    if (o.method != "lora" && o.method != "full") Usage("--method needs lora or full");
    if (o.intruders != "all" && o.intruders != "none") Usage("--intruders needs all or none");
    o.model = positional[0];
    o.out = positional[1];
    try {
        if (o.device == "cpu") {
            CPUBackend cpu;
            return Run(&cpu, o);
        }
#ifdef PULSATRIX_DEMO_WITH_HIP
        if (o.device == "hip") {
            HIPBackend hip;
            return Run(&hip, o);
        }
#endif
        Usage("device \"" + o.device + "\" is not in this build");
    } catch (const std::exception& e) {
        std::cerr << "what_did_the_finetune_change_recipe: " << e.what() << "\n";
        return 1;
    }
}
