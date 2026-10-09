// pulsatrix_diff_lm: model diffing with a crosscoder (roadmap FEAT-8). Reads the same text through
// a base language model and its fine-tune, trains a crosscoder on their residual streams at one
// layer, and reports which latents belong to one model only.
//
//   pulsatrix_diff_lm SmolLM2-135M SmolLM2-135M-Instruct --text conversations.jsonl --out diff/ \
//       [--layer 15] [--crosscoder btk|l1|delta] [--features N] [--k K] [--l1 L] [--device hip]
//
// --text is JSONL with a "text" field per line (tools/explain/fetch_chat_sample.py writes chat
// conversations in ChatML). Each text is cut to --max-length tokens (256); its first token is
// left out (its norm is far larger than the rest). --layer is the hidden-state position: 0 the
// embeddings, i the output of block i (default: the middle). Each model's activations are scaled
// so their mean squared norm is the hidden size. The last tenth of the texts is held out.
//
// Reports, on the held-out tokens:
//   - each model's explained variance, and the dead latents;
//   - the latents by Δnorm (Minder et al.): base-only, fine-tune-only, shared, other;
//   - Latent Scaling of each model's one-model latents, or at least the --lean (50) latents
//     leaning furthest toward it by Δnorm: those whose ν passes Minder et al.'s test
//     (ν_reconstruction < 0.5 and ν_error < 0.2) are specific to that model;
//   - what the specific latents fire on: chat-template tokens (<|im_start|>, <|im_end|>, a role
//     name), or a system, user or assistant turn, beside the same split over all tokens;
//   - the top tokens of the most active fine-tune-specific latents (fine-tune-leaning if none).
// Writes latents.csv (every latent's norms, Δnorm, cosine and, where measured, ν).
// Options (defaults): --features 8 x hidden, --k 32 (BatchTopK; the delta partition's with
// delta), --l1 1e-3, --epochs 4, --batch 1024, --lr 3e-4, --max-texts 0 (all), --seed 0,
// --device cpu|hip.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/crosscoder.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_diff_lm: " << problem << "\n"
              << "usage: pulsatrix_diff_lm BASE_DIR FINETUNE_DIR --text FILE.jsonl --out DIR [--layer L]\n"
              << "       [--crosscoder btk|l1|delta] [--features N] [--k K] [--l1 L] [--epochs N] [--batch N] [--lr X]\n"
              << "       [--max-length N] [--max-texts N] [--lean N] [--seed S] [--device cpu|hip]\n";
    std::exit(2);
}

struct Options {
    std::string base, ft, text, out, crosscoder = "btk", device = "cpu";
    int64_t layer = -1, features = 0, k = 32, epochs = 4, batch = 1024, max_length = 256, max_texts = 0, lean = 50;
    float l1 = 1e-3f, lr = 3e-4f;
    uint64_t seed = 0;
};

/** @brief Where a token sits in a ChatML conversation. */
enum Role { kTemplate = 0, kSystem = 1, kUser = 2, kAssistant = 3, kOther = 4 };
const char* kRoleNames[] = {"template", "system", "user", "assistant", "other"};

struct Token {
    int32_t text;  ///< which text
    std::string piece;
    Role role;
};

/** @brief Each token's role: the ChatML markers, the role name after <|im_start|> (it may be
 *         several tokens, such as "ass" "istant") and the newlines closing both are template
 *         tokens; the rest belong to the current turn. */
std::vector<Role> Roles(const std::vector<std::string>& tokens) {
    std::vector<Role> out(tokens.size(), kOther);
    Role current = kOther;
    bool header = false, after_end = false;
    std::string name;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];
        const bool newline = t == "\xC4\x8A";  // "Ċ", byte-level "\n"
        if (t == "<|im_start|>") {
            out[i] = kTemplate;
            header = true;
            name.clear();
        } else if (t == "<|im_end|>") {
            out[i] = kTemplate;
            current = kOther;
            after_end = true;
            continue;
        } else if (header) {
            out[i] = kTemplate;
            if (newline) {
                header = false;
                current = name == "system" ? kSystem : name == "user" ? kUser : name == "assistant" ? kAssistant : kOther;
            } else {
                name += t;
            }
        } else if (after_end && newline) {
            out[i] = kTemplate;
        } else {
            out[i] = current;
        }
        after_end = false;
    }
    return out;
}

int Run(pulsatrix::DeviceBackend* backend, Options o) {
    using namespace pulsatrix;
    const TextTokenizer tok = LoadTokenizerJson(o.ft + "/tokenizer.json");
    std::unique_ptr<CausalLM> base = LoadCausalLM(o.base, backend), ft = LoadCausalLM(o.ft, backend);
    const int64_t h = base->config().hidden_size;
    if (ft->config().hidden_size != h || ft->num_layers() != base->num_layers()) throw std::invalid_argument("the two models' sizes differ");
    if (o.layer < 0) o.layer = base->num_layers() / 2;
    if (o.layer > base->num_layers()) throw std::invalid_argument("--layer is past the last block");

    // Activations: (tokens, 2h), base then fine-tune.
    std::vector<std::string> texts;
    {
        std::ifstream in(o.text);
        if (!in) throw std::invalid_argument("cannot read " + o.text);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            texts.push_back(ParseJson(line).find("text")->as_string());
            if (o.max_texts > 0 && static_cast<int64_t>(texts.size()) >= o.max_texts) break;
        }
    }
    const auto held_from = static_cast<int32_t>(texts.size() - texts.size() / 10);
    std::vector<float> rows;
    std::vector<Token> meta;
    std::vector<float> captured;
    auto hook = [&](int64_t position, const Tensor& hidden) {
        if (position == o.layer) captured = hidden.to_host_vector();
        return hidden;
    };
    base->set_hidden_state_hook(hook);
    ft->set_hidden_state_hook(hook);
    for (size_t t = 0; t < texts.size(); ++t) {
        Encoding e = tok.encode(texts[t]);
        const auto L = static_cast<int64_t>(std::min<size_t>(e.size(), static_cast<size_t>(o.max_length)));
        if (L < 2) continue;
        const std::vector<float> ids(e.ids.begin(), e.ids.begin() + L);
        const Tensor input(Shape({1, L}), backend, ids);
        (void)base->forward(input);
        const std::vector<float> a = captured;
        (void)ft->forward(input);
        const std::vector<float> b = captured;
        const std::vector<Role> roles = Roles(std::vector<std::string>(e.tokens.begin(), e.tokens.begin() + L));
        for (int64_t i = 1; i < L; ++i) {
            rows.insert(rows.end(), a.begin() + i * h, a.begin() + (i + 1) * h);
            rows.insert(rows.end(), b.begin() + i * h, b.begin() + (i + 1) * h);
            meta.push_back({static_cast<int32_t>(t), e.tokens[static_cast<size_t>(i)], roles[static_cast<size_t>(i)]});
        }
        base->release_activations();
        ft->release_activations();
    }
    base.reset();
    ft.reset();
    const auto N = static_cast<int64_t>(meta.size());
    // Scale each model so its mean squared norm is h.
    for (int s = 0; s < 2; ++s) {
        double sq = 0;
        for (int64_t r = 0; r < N; ++r) {
            for (int64_t j = 0; j < h; ++j) sq += static_cast<double>(rows[static_cast<size_t>(r * 2 * h + s * h + j)]) * rows[static_cast<size_t>(r * 2 * h + s * h + j)];
        }
        const auto scale = static_cast<float>(std::sqrt(static_cast<double>(h) / (sq / static_cast<double>(N))));
        for (int64_t r = 0; r < N; ++r) {
            for (int64_t j = 0; j < h; ++j) rows[static_cast<size_t>(r * 2 * h + s * h + j)] *= scale;
        }
        std::printf("%s: scale %.4f\n", s == 0 ? "base" : "fine-tune", scale);
    }
    std::vector<int64_t> train, held;
    for (int64_t r = 0; r < N; ++r) (meta[static_cast<size_t>(r)].text < held_from ? train : held).push_back(r);
    std::printf("layer %ld, %ld tokens from %zu texts: %zu train, %zu held out\n", static_cast<long>(o.layer), static_cast<long>(N), texts.size(),
                train.size(), held.size());
    auto gather = [&](const std::vector<int64_t>& idx, size_t from, size_t count) {
        std::vector<float> v;
        v.reserve(count * static_cast<size_t>(2 * h));
        for (size_t i = from; i < from + count; ++i) {
            const int64_t r = idx[i];
            v.insert(v.end(), rows.begin() + r * 2 * h, rows.begin() + (r + 1) * 2 * h);
        }
        return Tensor(Shape({static_cast<int64_t>(count), 2 * h}), backend, v, backend->device());
    };

    CrosscoderOptions co;
    co.sparsity = o.crosscoder == "l1" ? CrosscoderSparsity::L1 : CrosscoderSparsity::BatchTopK;
    co.delta = o.crosscoder == "delta";
    co.k = o.k;
    co.l1_coefficient = o.l1;
    co.dead_after = std::max<int64_t>(1, static_cast<int64_t>(train.size()) / 2);
    co.seed = o.seed;
    const int64_t m = o.features > 0 ? o.features : 8 * h;
    Crosscoder c(2, h, m, backend, co);
    c.initialize_bias(gather(train, 0, std::min<size_t>(train.size(), 20000)));
    AdamOptimizer opt(o.lr, backend);
    std::mt19937_64 rng(o.seed);
    for (int64_t epoch = 0; epoch < o.epochs; ++epoch) {
        std::shuffle(train.begin(), train.end(), rng);
        double total = 0, recon = 0;
        int64_t batches = 0;
        for (size_t b0 = 0; b0 + static_cast<size_t>(o.batch) <= train.size(); b0 += static_cast<size_t>(o.batch)) {
            const FeaturizerLoss l = TrainFeaturizer(c, gather(train, b0, static_cast<size_t>(o.batch)), opt, /*unit_norm_decoder=*/false);
            total += l.total;
            recon += l.reconstruction;
            ++batches;
        }
        std::printf("epoch %ld: loss %.4f (reconstruction %.4f)\n", static_cast<long>(epoch + 1), total / static_cast<double>(batches),
                    recon / static_cast<double>(batches));
    }

    // ---- Held out ----
    const size_t H = std::min<size_t>(held.size(), 40000);
    const Tensor x = gather(held, 0, H);
    const std::vector<double> ev = ExplainedVarianceBySource(c, x);
    const std::vector<float> codes = c.encode(x).to_host_vector();
    std::vector<int64_t> fires(static_cast<size_t>(m), 0);
    std::vector<std::vector<double>> by_role(static_cast<size_t>(m), std::vector<double>(5, 0.0));  // activation mass per role
    double l0 = 0;
    for (size_t r = 0; r < H; ++r) {
        const Role role = meta[static_cast<size_t>(held[r])].role;
        for (int64_t i = 0; i < m; ++i) {
            const float v = codes[r * static_cast<size_t>(m) + static_cast<size_t>(i)];
            if (v <= 0.0f) continue;
            ++fires[static_cast<size_t>(i)];
            by_role[static_cast<size_t>(i)][static_cast<size_t>(role)] += v;
            l0 += 1;
        }
    }
    const auto dead = std::count(fires.begin(), fires.end(), 0);
    std::printf("\n%s crosscoder, %ld latents: explained variance base %.3f, fine-tune %.3f; L0 %.1f; dead %.1f%%\n", o.crosscoder.c_str(),
                static_cast<long>(m), ev[0], ev[1], l0 / static_cast<double>(H), 100.0 * static_cast<double>(dead) / static_cast<double>(m));

    const std::vector<CrosscoderLatentStats> stats = CrosscoderLatents(c);
    std::vector<int64_t> a_only, b_only;
    int64_t shared = 0, other = 0;
    for (int64_t i = 0; i < m; ++i) {
        if (fires[static_cast<size_t>(i)] == 0) continue;
        switch (ClassifyLatent(stats[static_cast<size_t>(i)])) {
            case LatentClass::AOnly: a_only.push_back(i); break;
            case LatentClass::BOnly: b_only.push_back(i); break;
            case LatentClass::Shared: ++shared; break;
            case LatentClass::Other: ++other; break;
        }
    }
    std::printf("live latents by Δnorm: base-only %zu, fine-tune-only %zu, shared %ld, other %ld\n", a_only.size(), b_only.size(),
                static_cast<long>(shared), static_cast<long>(other));
    // Latent Scaling of each model's candidates: its one-model latents by Δnorm, or at least the
    // --lean live latents leaning furthest toward it.
    auto candidates = [&](std::vector<int64_t> pick, bool toward_b) {
        std::vector<int64_t> live;
        for (int64_t i = 0; i < m; ++i) {
            if (fires[static_cast<size_t>(i)] > 0) live.push_back(i);
        }
        std::sort(live.begin(), live.end(), [&](int64_t p, int64_t q) {
            const double dp = stats[static_cast<size_t>(p)].delta_norm, dq = stats[static_cast<size_t>(q)].delta_norm;
            return toward_b ? dp > dq : dp < dq;
        });
        for (size_t t = 0; t < live.size() && static_cast<int64_t>(pick.size()) < o.lean; ++t) {
            if (std::find(pick.begin(), pick.end(), live[t]) == pick.end()) pick.push_back(live[t]);
        }
        return pick;
    };
    const std::vector<int64_t> lean_b = candidates(b_only, true), lean_a = candidates(a_only, false);
    const std::vector<LatentScaling> ls_b = MeasureLatentScaling(c, x, lean_b, 0, 1), ls_a = MeasureLatentScaling(c, x, lean_a, 1, 0);
    auto specific = [](const LatentScaling& l) { return l.beta_error_b > 0 && l.beta_reconstruction_b > 0 && l.nu_reconstruction < 0.5 && l.nu_error < 0.2; };
    std::vector<int64_t> b_specific, a_specific;
    for (const LatentScaling& l : ls_b) {
        if (specific(l)) b_specific.push_back(l.latent);
    }
    for (const LatentScaling& l : ls_a) {
        if (specific(l)) a_specific.push_back(l.latent);
    }
    std::printf("Δnorm of the %zu latents leaning furthest toward the fine-tune: %.3f to %.3f; toward the base: %.3f to %.3f\n", lean_b.size(),
                stats[static_cast<size_t>(lean_b.back())].delta_norm, stats[static_cast<size_t>(lean_b.front())].delta_norm,
                stats[static_cast<size_t>(lean_a.front())].delta_norm, stats[static_cast<size_t>(lean_a.back())].delta_norm);
    std::printf("specific by Latent Scaling: base %zu of %zu, fine-tune %zu of %zu\n", a_specific.size(), lean_a.size(), b_specific.size(),
                lean_b.size());

    // What they fire on: the share of activation mass per role.
    auto shares = [&](const std::vector<int64_t>& latents) {
        std::vector<double> s(5, 0.0);
        double total = 0;
        for (int64_t i : latents) {
            for (int q = 0; q < 5; ++q) {
                s[static_cast<size_t>(q)] += by_role[static_cast<size_t>(i)][static_cast<size_t>(q)];
                total += by_role[static_cast<size_t>(i)][static_cast<size_t>(q)];
            }
        }
        for (double& v : s) v = total > 0 ? v / total : 0;
        return s;
    };
    std::vector<int64_t> all_live;
    for (int64_t i = 0; i < m; ++i) {
        if (fires[static_cast<size_t>(i)] > 0) all_live.push_back(i);
    }
    std::vector<double> token_share(5, 0.0);
    for (size_t r = 0; r < H; ++r) token_share[static_cast<size_t>(meta[static_cast<size_t>(held[r])].role)] += 1.0 / static_cast<double>(H);
    std::printf("\nactivation mass by role     template  system    user  assistant  other\n");
    auto row = [](const char* name, const std::vector<double>& s) {
        std::printf("  %-26s %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%%  %6.1f%%\n", name, 100 * s[0], 100 * s[1], 100 * s[2], 100 * s[3], 100 * s[4]);
    };
    row("tokens", token_share);
    row("all live latents", shares(all_live));
    row("leaning to base", shares(lean_a));
    row("leaning to fine-tune", shares(lean_b));
    row("base-specific latents", shares(a_specific));
    row("fine-tune-specific latents", shares(b_specific));

    // The most active fine-tune-specific latents' top tokens.
    std::vector<int64_t> top = b_specific.empty() ? lean_b : b_specific;
    std::sort(top.begin(), top.end(), [&](int64_t p, int64_t q) { return fires[static_cast<size_t>(p)] > fires[static_cast<size_t>(q)]; });
    std::printf("\nmost active %s latents (fires on %% of tokens, Δnorm: top tokens by activation)\n",
                b_specific.empty() ? "fine-tune-leaning" : "fine-tune-specific");
    for (size_t t = 0; t < std::min<size_t>(top.size(), 10); ++t) {
        const int64_t i = top[t];
        std::vector<std::pair<float, size_t>> acts;
        for (size_t r = 0; r < H; ++r) {
            const float v = codes[r * static_cast<size_t>(m) + static_cast<size_t>(i)];
            if (v > 0) acts.emplace_back(v, r);
        }
        std::sort(acts.begin(), acts.end(), std::greater<>());
        std::ostringstream tokens;
        for (size_t q = 0; q < std::min<size_t>(acts.size(), 6); ++q) {
            const Token& tk = meta[static_cast<size_t>(held[acts[q].second])];
            tokens << (q ? ", " : "") << '"' << tk.piece << "\" (" << kRoleNames[tk.role] << ')';
        }
        std::printf("  latent %5ld  %5.2f%%  %.2f  %s\n", static_cast<long>(i),
                    100.0 * static_cast<double>(fires[static_cast<size_t>(i)]) / static_cast<double>(H), stats[static_cast<size_t>(i)].delta_norm,
                    tokens.str().c_str());
    }

    std::filesystem::create_directories(o.out);
    std::ofstream csv(o.out + "/latents.csv");
    csv << "latent,fires,norm_base,norm_finetune,delta_norm,cosine,nu_error,nu_reconstruction\n";
    std::map<int64_t, LatentScaling> measured;
    for (const LatentScaling& l : ls_a) measured[l.latent] = l;
    for (const LatentScaling& l : ls_b) measured[l.latent] = l;
    for (int64_t i = 0; i < m; ++i) {
        const CrosscoderLatentStats& s = stats[static_cast<size_t>(i)];
        csv << i << ',' << fires[static_cast<size_t>(i)] << ',' << s.norm_a << ',' << s.norm_b << ',' << s.delta_norm << ',' << s.cosine << ',';
        const auto it = measured.find(i);
        if (it != measured.end()) csv << it->second.nu_error << ',' << it->second.nu_reconstruction;
        else csv << ',';
        csv << '\n';
    }
    std::printf("\nwrote %s/latents.csv\n", o.out.c_str());
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace pulsatrix;
    if (argc < 3) Usage("needs two model directories");
    Options o;
    o.base = argv[1];
    o.ft = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--text") o.text = value();
        else if (a == "--out") o.out = value();
        else if (a == "--layer") o.layer = std::atoll(value().c_str());
        else if (a == "--crosscoder") o.crosscoder = value();
        else if (a == "--features") o.features = std::atoll(value().c_str());
        else if (a == "--k") o.k = std::atoll(value().c_str());
        else if (a == "--l1") o.l1 = std::strtof(value().c_str(), nullptr);
        else if (a == "--epochs") o.epochs = std::atoll(value().c_str());
        else if (a == "--batch") o.batch = std::atoll(value().c_str());
        else if (a == "--lr") o.lr = std::strtof(value().c_str(), nullptr);
        else if (a == "--max-length") o.max_length = std::atoll(value().c_str());
        else if (a == "--max-texts") o.max_texts = std::atoll(value().c_str());
        else if (a == "--lean") o.lean = std::atoll(value().c_str());
        else if (a == "--seed") o.seed = std::strtoull(value().c_str(), nullptr, 10);
        else if (a == "--device") o.device = value();
        else Usage("unknown option " + a);
    }
    if (o.text.empty() || o.out.empty()) Usage("needs --text and --out");
    if (o.crosscoder != "btk" && o.crosscoder != "l1" && o.crosscoder != "delta") Usage("--crosscoder must be btk, l1 or delta");
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
        std::cerr << "pulsatrix_diff_lm: " << e.what() << "\n";
        return 1;
    }
}
