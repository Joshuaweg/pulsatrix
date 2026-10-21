// pulsatrix_diffusion_lrp: commitment-graph LRP (roadmap DIFF-3) on a masked diffusion LM's infill.
// Masks a span of a text, lets the model fill it back in by confidence, explains the filled span
// back to the rest of the text, and runs the checks that could show the explanation is wrong.
//
//   pulsatrix_diffusion_lrp MDLM_DIR TOKENIZER_JSON --text FILE [--length 1024] [--span-start 512]
//       [--span 32] [--steps 32] [--deletion-steps 10] [--random-orders 3] [--patch] [--csv FILE]
//       [--device cpu|hip]
//
// MDLM_DIR is an MDLM checkpoint (kuleshov-group/mdlm-owt) and TOKENIZER_JSON its tokenizer
// (GPT-2's). The text's first --length tokens are used (MDLM-owt was trained on 1024); the span
// [--span-start, --span-start + --span) is masked and refilled over --steps steps, greedily.
// The explained value is the filled tokens' logits, each at the step that wrote it, summed.
//
// Reports:
//   - the original and the infilled span;
//   - the relevance ledger in both modes (Graph and StepLocal): prompt, mask sink, leak, and for
//     StepLocal what stopped on committed tokens; and each hop's leak as a share of its seed;
//   - the prompt tokens with the most relevance, and how much lies within 16 tokens of the span;
//   - prompt deletion with the trajectory replayed, by graph relevance, by step-local relevance,
//     and at random (lower area = removes the value faster);
//   - with --patch, step-resolved patching: each filled token left masked, against the relevance
//     relayed through it.
// --csv writes position, token, prompt relevance (graph and step-local) and mask-sink relevance.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/masked_diffusion.hpp"
#include "pulsatrix/masked_diffusion_lrp.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

using namespace pulsatrix;

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_diffusion_lrp: " << problem << "\n"
              << "usage: pulsatrix_diffusion_lrp MDLM_DIR TOKENIZER_JSON --text FILE [--length N] [--span-start P] [--span N]\n"
              << "       [--steps N] [--deletion-steps N] [--random-orders N] [--patch] [--csv FILE] [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string model, tokenizer, text, csv, device = "cpu";
    int64_t length = 1024, span_start = 512, span = 32, steps = 32, deletion_steps = 10, random_orders = 3;
    bool patch = false;
};

double Seconds(std::chrono::steady_clock::time_point since) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - since).count();
}

std::string Quoted(const TextTokenizer& tok, const std::vector<int64_t>& ids) {
    std::string s = tok.decode(ids);
    for (char& c : s) {
        if (c == '\n') c = ' ';
    }
    return "\"" + s + "\"";
}

void Ledger(const char* name, const CommitmentRelevance& r) {
    std::printf("  %-10s value %9.3f = prompt %9.3f + mask %9.3f + leak %9.3f", name, r.value, r.prompt_total(), r.mask_total(),
                r.leak_total());
    if (r.mode == CommitmentMode::StepLocal) std::printf(" + stopped on committed %9.3f", r.relayed_total());
    std::printf("   (%zu passes)\n", r.hops.size());
    std::vector<double> shares;
    for (const CommitmentHop& h : r.hops) {
        if (h.seeded != 0.0) shares.push_back(h.leak() / h.seeded);
    }
    if (shares.empty()) return;
    std::sort(shares.begin(), shares.end());
    std::printf("  %-10s leak / seed per pass: min %.3f, median %.3f, max %.3f\n", "", shares.front(), shares[shares.size() / 2],
                shares.back());
}

void Curve(const char* name, const CommitmentDeletionCurve& d, bool random) {
    std::printf("  %-12s auc %9.3f |", name, random ? d.random_auc : d.auc);
    for (float v : random ? d.random_scores : d.scores) std::printf(" %8.2f", v);
    std::printf("\n");
}

int Run(DeviceBackend* backend, const Options& o) {
    auto start = std::chrono::steady_clock::now();
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(o.model, backend);
    const TextTokenizer tok = LoadTokenizerJson(o.tokenizer);
    const int64_t mask = model->config().mask_token_id;
    std::ifstream in(o.text);
    if (!in) Usage("can't read " + o.text);
    std::stringstream buffer;
    buffer << in.rdbuf();
    std::vector<int64_t> ids = tok.encode(buffer.str(), false).ids;
    if (static_cast<int64_t>(ids.size()) < o.length) {
        Usage("the text has " + std::to_string(ids.size()) + " tokens, fewer than --length " + std::to_string(o.length));
    }
    ids.resize(static_cast<size_t>(o.length));
    if (o.span < 1 || o.span_start < 0 || o.span_start + o.span > o.length) Usage("the span must lie inside the sequence");
    const std::vector<int64_t> original(ids.begin() + o.span_start, ids.begin() + o.span_start + o.span);
    std::vector<int64_t> input = ids;
    for (int64_t p = o.span_start; p < o.span_start + o.span; ++p) input[static_cast<size_t>(p)] = mask;
    std::printf("model %s, %lld tokens, span [%lld, %lld), %lld steps (loaded in %.1f s)\n", o.model.c_str(), static_cast<long long>(o.length),
                static_cast<long long>(o.span_start), static_cast<long long>(o.span_start + o.span), static_cast<long long>(o.steps),
                Seconds(start));

    MaskedDiffusionConfig config;
    config.steps = o.steps;
    config.mask_token_id = mask;
    start = std::chrono::steady_clock::now();
    model->set_keep_activations(false);
    const DenoiserFn denoise = MakeDenoiser(*model, backend);
    const MaskedDiffusionResult t = SampleMaskedDiffusion(denoise, input, config);
    const std::vector<int64_t> filled(t.tokens.begin() + o.span_start, t.tokens.begin() + o.span_start + o.span);
    std::printf("\noriginal  %s\ninfilled  %s\n(%zu steps in %.1f s)\n", Quoted(tok, original).c_str(), Quoted(tok, filled).c_str(),
                t.steps.size(), Seconds(start));

    std::vector<CommitmentTarget> targets;
    for (int64_t p = o.span_start; p < o.span_start + o.span; ++p) targets.push_back({p, std::nullopt});
    EncoderLMStepModel step(*model, backend);
    start = std::chrono::steady_clock::now();
    const CommitmentRelevance graph = ExplainCommitments(step, t, mask, targets);
    const double graph_seconds = Seconds(start);
    const CommitmentRelevance local = ExplainCommitments(step, t, mask, targets, CommitmentMode::StepLocal);
    model->release_activations();
    model->set_keep_activations(false);
    std::printf("\nledger (graph explained in %.1f s)\n", graph_seconds);
    Ledger("graph", graph);
    Ledger("step-local", local);
    bool finite = true;
    for (float v : graph.prompt) finite = finite && std::isfinite(v);
    std::printf("  every relevance finite: %s\n", finite ? "yes" : "NO");

    std::vector<int64_t> order;
    double total_abs = 0, near_abs = 0;
    int64_t near_count = 0;
    for (int64_t p = 0; p < o.length; ++p) {
        if (t.last_write_step[static_cast<size_t>(p)] >= 0) continue;
        order.push_back(p);
        const double a = std::abs(graph.prompt[static_cast<size_t>(p)]);
        total_abs += a;
        if (p >= o.span_start - 16 && p < o.span_start + o.span + 16) {
            near_abs += a;
            ++near_count;
        }
    }
    std::sort(order.begin(), order.end(),
              [&](int64_t a, int64_t b) { return std::abs(graph.prompt[static_cast<size_t>(a)]) > std::abs(graph.prompt[static_cast<size_t>(b)]); });
    std::printf("\nprompt tokens with the most graph relevance (%.1f%% of |relevance| within 16 tokens of the span, which holds %.1f%% of the "
                "prompt)\n",
                100.0 * near_abs / std::max(total_abs, 1e-30), 100.0 * static_cast<double>(near_count) / static_cast<double>(order.size()));
    for (size_t i = 0; i < std::min<size_t>(12, order.size()); ++i) {
        const int64_t p = order[i];
        std::printf("  %5lld %-20s graph %8.3f  step-local %8.3f\n", static_cast<long long>(p), Quoted(tok, {ids[static_cast<size_t>(p)]}).c_str(),
                    graph.prompt[static_cast<size_t>(p)], local.prompt[static_cast<size_t>(p)]);
    }

    if (o.deletion_steps > 0) {
        start = std::chrono::steady_clock::now();
        const CommitmentDeletionCurve g = CommitmentDeletionCheck(denoise, t, config, targets, graph.prompt, o.deletion_steps, o.random_orders, 1);
        const CommitmentDeletionCurve l = CommitmentDeletionCheck(denoise, t, config, targets, local.prompt, o.deletion_steps, 1, 2);
        std::printf("\nprompt deletion, trajectory replayed (value after deleting 0%%, %.0f%%, ... of the prompt; %.0f s)\n",
                    100.0 / static_cast<double>(o.deletion_steps), Seconds(start));
        Curve("graph", g, false);
        Curve("step-local", l, false);
        Curve("random", g, true);
    }

    if (o.patch) {
        start = std::chrono::steady_clock::now();
        const CommitmentPatchResult r = CommitmentPatchCheck(denoise, t, config, targets, graph);
        std::printf("\npatching %zu filled tokens (%.0f s): Spearman(relayed relevance, effect of leaving it masked) = %.3f\n", r.positions.size(),
                    Seconds(start), r.spearman);
        for (size_t i = 0; i < r.positions.size(); ++i) {
            std::printf("  %5lld %-20s relayed %8.3f  effect %8.3f\n", static_cast<long long>(r.positions[i]),
                        Quoted(tok, {t.tokens[static_cast<size_t>(r.positions[i])]}).c_str(), r.relayed[i], r.effect[i]);
        }
    }

    if (!o.csv.empty()) {
        std::ofstream csv(o.csv);
        csv << "position,token,prompt_graph,prompt_step_local,mask_sink,relayed\n";
        for (int64_t p = 0; p < o.length; ++p) {
            const size_t i = static_cast<size_t>(p);
            csv << p << "," << t.tokens[i] << "," << graph.prompt[i] << "," << local.prompt[i] << "," << graph.mask[i] << "," << graph.relayed[i]
                << "\n";
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);  // each report line as it's ready: runs take minutes
    Options o;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--text") o.text = value();
        else if (a == "--length") o.length = std::atoll(value().c_str());
        else if (a == "--span-start") o.span_start = std::atoll(value().c_str());
        else if (a == "--span") o.span = std::atoll(value().c_str());
        else if (a == "--steps") o.steps = std::atoll(value().c_str());
        else if (a == "--deletion-steps") o.deletion_steps = std::atoll(value().c_str());
        else if (a == "--random-orders") o.random_orders = std::atoll(value().c_str());
        else if (a == "--patch") o.patch = true;
        else if (a == "--csv") o.csv = value();
        else if (a == "--device") o.device = value();
        else if (a.rfind("--", 0) == 0) Usage("unknown option " + a);
        else positional.push_back(a);
    }
    if (positional.size() != 2) Usage("needs MDLM_DIR and TOKENIZER_JSON");
    if (o.text.empty()) Usage("needs --text");
    o.model = positional[0];
    o.tokenizer = positional[1];
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
        std::cerr << "pulsatrix_diffusion_lrp: " << e.what() << "\n";
        return 1;
    }
}
