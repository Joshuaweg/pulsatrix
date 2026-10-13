// Recipe: Can I trust this explanation? (XAI-2, XAI-3, XAI-4)
//
// SmolLM2 completes "The Eiffel Tower is located in the city of" with " Paris". Four explainers say
// which words made it do so: AttnLRP, KernelSHAP, LIME and occlusion. Each goes through
// BuildExplanation, which runs the checks the planner asks for when the reason is to justify the
// prediction (faithfulness, stability, conservation, and the randomization test), and reports what
// the explanation does and doesn't support. Then Anchors asks the other question: which words must
// stay for " Paris" to stay.
//
// The words are the features. Each token gets a weight m in [0, 1], and its embedding entering the
// first layer becomes m e + (1 - m) b, where b is the mean of the embedding table: m = 1 is the token,
// m = 0 is no particular token. Removing a word, for the faithfulness curves, LIME's samples and
// occlusion, means setting its weight to 0. So no explainer is handed a token id to perturb.
//
// The predictions, fixed before the run:
//   1. Every explainer ranks " Eiffel" or " Tower" first.
//   2. KernelSHAP's values add up to f(every word) - f(no word): its conservation check passes.
//      AttnLRP's add up to the logit within 5%.
//   3. AttnLRP, KernelSHAP and occlusion pass the faithfulness check; LIME may not.
//   4. Every explanation changes when the model is randomized: the randomization check passes.
//   5. The anchor that keeps " Paris" holds " Eiffel".
//
// Usage: can_i_trust_this_explanation_recipe MODEL_DIR OUT_DIR [--device cpu|hip]
//            [--prompt TEXT] [--smoke [--tokenizer PATH]]
//   MODEL_DIR: for example ~/.cache/pulsatrix/golden/SmolLM2-135M. --smoke runs the tiny test model
//   (tests/fixtures/hf_tiny/llama) on a shorter prompt, with fewer samples, to check the plumbing;
//   its numbers mean nothing.
//
// Docs: docs/recipes/interpretability/can_i_trust_this_explanation.md
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/anchors.hpp"
#include "pulsatrix/attnlrp_parity.hpp"
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explanation_report.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/kernel_shap.hpp"
#include "pulsatrix/lime.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/svg.hpp"
#include "pulsatrix/viz/text_relevance.hpp"
#ifdef PULSATRIX_DEMO_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

struct Options {
    std::string model, out, device = "cpu", tokenizer;
    std::string prompt = "The Eiffel Tower is located in the city of";
    bool prompt_given = false;
    bool smoke = false;
};

void Write(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    f << text;
    if (!f) throw std::runtime_error("can't write " + path);
}

std::string Fixed(double v, int digits = 2) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.*f", digits, v);
    return buf;
}

// The prompt as a function of one weight per token: the model's last-position logits, (1, vocab),
// with each token's embedding blended toward the mean embedding by its weight.
class TokenWeights {
public:
    TokenWeights(CausalLM& model, DeviceBackend* backend, std::vector<int64_t> ids)
        : model_(model), backend_(backend), ids_(std::move(ids)) {
        V_ = model.config().vocab_size;
        d_ = model.config().hidden_size;
        for (const NamedParamRef& p : model.named_parameters()) {
            if (p.name != "embed_tokens.weight") continue;
            const std::vector<float> table = p.ref.value->to_host_vector();
            baseline_.assign(static_cast<size_t>(d_), 0.0f);
            const int64_t rows = static_cast<int64_t>(table.size()) / d_;
            std::vector<double> sum(static_cast<size_t>(d_), 0.0);
            for (int64_t r = 0; r < rows; ++r) {
                for (int64_t k = 0; k < d_; ++k) sum[static_cast<size_t>(k)] += table[static_cast<size_t>(r * d_ + k)];
            }
            for (int64_t k = 0; k < d_; ++k) baseline_[static_cast<size_t>(k)] = static_cast<float>(sum[static_cast<size_t>(k)] / rows);
        }
        if (baseline_.empty()) throw std::runtime_error("the model has no embed_tokens.weight");
        model_.set_hidden_state_hook([this](int64_t position, const Tensor& hidden) {
            if (position != 0 || all_ones_) return hidden;
            std::vector<float> x = hidden.to_host_vector();
            for (size_t t = 0; t < weights_.size(); ++t) {
                const float m = weights_[t];
                for (int64_t k = 0; k < d_; ++k) {
                    float& v = x[t * static_cast<size_t>(d_) + static_cast<size_t>(k)];
                    v = m * v + (1.0f - m) * baseline_[static_cast<size_t>(k)];
                }
            }
            return Tensor(hidden.shape(), hidden.backend(), x, hidden.device());
        });
    }
    ~TokenWeights() { model_.set_hidden_state_hook(nullptr); }

    int64_t length() const { return static_cast<int64_t>(ids_.size()); }
    int64_t calls() const { return calls_; }

    // Every position's logits, (1, L, vocab), on the model's device.
    Tensor forward(const Tensor& weights) {
        weights_ = weights.to_host_vector();
        all_ones_ = std::all_of(weights_.begin(), weights_.end(), [](float m) { return m == 1.0f; });
        ++calls_;
        const std::vector<float> ids(ids_.begin(), ids_.end());
        return model_.forward(Tensor(Shape({1, length()}), backend_, ids));
    }

    // The last position's logits, (1, vocab), on the host.
    Tensor predict(const Tensor& weights) {
        const std::vector<float> all = forward(weights).to_host_vector();
        const auto start = all.begin() + (length() - 1) * V_;
        return Tensor(Shape({1, V_}), &cpu_, std::vector<float>(start, start + V_));
    }

    // AttnLRP (LXT's rules) for the target at the last position: the relevance reaching each token's
    // blended embedding.
    Attribution attn_lrp(const Tensor& weights, int64_t target) {
        const Tensor logits = forward(weights);
        const std::vector<float> all = logits.to_host_vector();
        const size_t at = static_cast<size_t>((length() - 1) * V_ + target);
        std::vector<float> seed(all.size(), 0.0f);
        seed[at] = all[at];
        const std::vector<float> r =
            model_.propagate_relevance(Tensor(logits.shape(), logits.backend(), seed, logits.device()), LxtAttnLrpConfig())
                .to_host_vector();
        double sum = 0;
        for (float v : r) sum += v;
        return Attribution{"attn_lrp", Tensor(weights.shape(), &cpu_, r),
                           {{"rule", "LXT AttnLRP"},
                            {"relevance_out_sum", std::to_string(all[at])},
                            {"relevance_in_sum", std::to_string(sum)}}};
    }

private:
    CausalLM& model_;
    DeviceBackend* backend_;
    CPUBackend cpu_;
    std::vector<int64_t> ids_;
    std::vector<float> baseline_, weights_;
    int64_t V_ = 0, d_ = 0, calls_ = 0;
    bool all_ones_ = true;
};

struct Row {
    std::string method;
    Explanation e;
    double seconds;
};

std::string Value(const Explanation& e, const char* check, const char* key) {
    const CheckResult* c = e.check(check);
    if (c == nullptr) return "-";
    const std::optional<double> v = c->value(key);
    if (c->outcome == CheckOutcome::NotRun || !v) return "not run";
    return Fixed(*v) + (c->outcome == CheckOutcome::Passed ? " pass" : " FAIL");
}

int Run(DeviceBackend* backend, const Options& o) {
    CPUBackend cpu;
    std::filesystem::create_directories(o.out);
    const TextTokenizer tok = LoadTokenizerJson(o.tokenizer.empty() ? o.model + "/tokenizer.json" : o.tokenizer);
    std::unique_ptr<CausalLM> model = LoadCausalLM(o.model, backend);
    model->set_training(false);
    const int64_t V = model->config().vocab_size;
    std::cout << "Model: " << o.model << " (" << model->num_layers() << " blocks)\n";
    if (o.smoke) std::cout << "--smoke: the tiny test model; the numbers check the plumbing only\n";

    const Encoding encoding = tok.encode(o.prompt);
    std::vector<int64_t> ids;
    for (int64_t id : encoding.ids) ids.push_back(id % V);
    const auto L = static_cast<int64_t>(ids.size());
    if (L > 20) throw std::invalid_argument("the prompt has " + std::to_string(L) + " tokens; KernelSHAP takes at most 20");
    std::vector<std::string> words;
    for (int64_t id : ids) words.push_back(tok.decode({id}));

    TokenWeights f(*model, backend, ids);
    const Tensor all(Shape({1, L}), &cpu, std::vector<float>(static_cast<size_t>(L), 1.0f));
    const Tensor none(Shape({1, L}), &cpu);
    const PredictFn predict = [&](const Tensor& m) { return f.predict(m); };
    const std::vector<float> logits = predict(all).to_host_vector();
    const int64_t target = std::max_element(logits.begin(), logits.end()) - logits.begin();
    const std::string target_text = tok.decode({target});
    std::cout << "\"" << o.prompt << "\" -> \"" << target_text << "\" (logit " << Fixed(logits[static_cast<size_t>(target)])
              << "; with every word at the mean embedding, " << Fixed(predict(none).read_element(target)) << ")\n";

    // The four explainers, each a function of the token weights.
    LIME lime;
    const int64_t lime_samples = o.smoke ? 50 : 500;
    std::vector<std::pair<std::string, ExplainFn>> explainers = {
        {"LRP", [&](const Tensor& m) { return f.attn_lrp(m, target); }},
        {"KernelSHAP", [&](const Tensor& m) { return KernelSHAP().explain(predict, m, none, target, &cpu); }},
        {"LIME", [&](const Tensor& m) { return lime.explain(predict, m, target, lime_samples, 0.3f, 0.01f, 7, &cpu); }},
        {"Occlusion",
         [&](const Tensor& m) {
             const std::vector<float> w = m.to_host_vector();
             const float full = predict(m).read_element(target);
             std::vector<float> drop(w.size());
             for (size_t t = 0; t < w.size(); ++t) {
                 std::vector<float> without = w;
                 without[t] = 0.0f;
                 drop[t] = full - predict(Tensor(m.shape(), &cpu, without)).read_element(target);
             }
             return Attribution{"occlusion", Tensor(m.shape(), &cpu, drop), {{"window", "one token"}, {"fill", "mean embedding"}}};
         }},
    };

    ExplanationRequest request;
    request.question = XaiQuestion::Why;
    request.reason = XaiReason::Justify;
    request.traits.access = ModelAccess::Relevance;
    request.traits.input = InputKind::Tabular;  // the features are the token weights, which are continuous
    request.traits.can_reinitialize = true;
    request.traits.features = L;
    request.target = target;
    request.baseline = none;
    request.settings = {{"prompt", o.prompt}, {"features", "token weights; 0 is the mean embedding"}};
    ExplanationCheckOptions checks;
    if (o.smoke) checks.random_orderings = 4, checks.perturbations = 1, checks.stability_repeats = 1;

    std::cout << "\nThe planner, for why, justify, on this model:\n";
    const ExplanationPlan plan = ExplanationPlanner().plan(request.question, request.reason, request.traits);
    for (const PlannedExplainer& p : plan.explainers) std::cout << "  " << p.rank << ". " << p.capabilities->name << "\n";

    std::vector<Row> rows;
    for (const auto& [method, explain] : explainers) {
        request.method = method;
        const auto start = std::chrono::steady_clock::now();
        Explanation e = BuildExplanation(request, all, explain, predict, model.get(), checks);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "\n==== " << method << " (" << Fixed(seconds, 0) << " s) ====\n" << DescribeExplanation(e);
        Write(o.out + "/explanation_" + method + ".json", WriteJson(ToJson(e)) + "\n");

        TokenRelevanceDocument doc = MakeTokenRelevanceDocument(o.prompt, encoding, e.attribution->values.to_host_vector(),
                                                                e.attribution->method, target_text);
        SvgOptions svg;
        svg.title = method + ": why \"" + target_text + "\"" + (e.supports_reason() ? "" : " (doesn't support justifying it)");
        Write(o.out + "/tokens_" + method + ".svg", RenderTokenStripSvg(doc, svg));
        rows.push_back({method, std::move(e), seconds});
    }

    std::cout << "\nSummary (each check's main number; pass or FAIL at the report's thresholds):\n";
    std::cout << "  method      top word      faithfulness p  stability (repeat)  conservation gap  randomization rho  supports\n";
    for (const Row& r : rows) {
        const std::vector<float> v = r.e.attribution->values.to_host_vector();
        const auto top = std::max_element(v.begin(), v.end()) - v.begin();
        char line[256];
        std::snprintf(line, sizeof line, "  %-11s %-13s %-15s %-19s %-17s %-18s %s\n", r.method.c_str(),
                      ("\"" + words[static_cast<size_t>(top)] + "\"").c_str(), Value(r.e, "faithfulness", "p_value").c_str(),
                      Value(r.e, "stability", "repeat_similarity").c_str(), Value(r.e, "conservation", "relative_gap").c_str(),
                      Value(r.e, "randomization", "similarity_all_layers").c_str(), r.e.supports_reason() ? "yes" : "no");
        std::cout << line;
    }

    // How to still be this: which words must stay for the prediction to stay.
    std::mt19937 rng(11);
    std::bernoulli_distribution keep(0.5);
    std::vector<Tensor> background;
    for (int i = 0; i < (o.smoke ? 40 : 200); ++i) {
        std::vector<float> w(static_cast<size_t>(L));
        for (float& m : w) m = keep(rng) ? 1.0f : 0.0f;
        background.emplace_back(Shape({1, L}), &cpu, w);
    }
    AnchorOptions anchor_options;
    if (o.smoke) anchor_options.max_samples_per_candidate = 300, anchor_options.max_anchor_size = 2;
    const auto start = std::chrono::steady_clock::now();
    const int64_t calls_before = f.calls();
    const Anchor anchor = FindAnchor(predict, all, background, anchor_options);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::vector<std::string> names;
    for (const std::string& w : words) names.push_back("\"" + w + "\"");
    std::cout << "\nAnchor, with half the words dropped at random as the background (" << f.calls() - calls_before
              << " model calls, " << Fixed(seconds, 0) << " s):\n  " << DescribeAnchor(anchor, names)
              << "\n  precision lower bound " << Fixed(anchor.precision_lower, 3) << " from " << anchor.samples << " samples\n";
    std::cout << "  (x > 0 means the word is there; the class is the top token, " << target << " = \"" << target_text << "\")\n";
    Write(o.out + "/anchor.json", WriteJson(ToJson(anchor, names)) + "\n");

    std::cout << "\nWrote the reports, token strips and anchor to " << o.out << "/\n";
    return 0;
}

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "can_i_trust_this_explanation_recipe: " << problem << "\n"
              << "usage: can_i_trust_this_explanation_recipe MODEL_DIR OUT_DIR [--device cpu|hip] [--prompt TEXT]\n"
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
        else if (a == "--prompt") o.prompt = value(), o.prompt_given = true;
        else if (a == "--tokenizer") o.tokenizer = value();
        else if (a == "--smoke") o.smoke = true;
        else if (!a.empty() && a[0] == '-') Usage("unknown option " + a);
        else positional.push_back(a);
    }
    if (positional.size() != 2) Usage("needs MODEL_DIR and OUT_DIR");
    o.model = positional[0];
    o.out = positional[1];
    // The test tokenizer has 1,000 tokens and splits the default prompt into 26, over KernelSHAP's 20.
    if (o.smoke && !o.prompt_given) o.prompt = "The city of";
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
        std::cerr << "can_i_trust_this_explanation_recipe: " << e.what() << "\n";
        return 1;
    }
}
