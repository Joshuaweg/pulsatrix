// pulsatrix_explain_text: explains a language model's next-token prediction with AttnLRP and
// writes a token relevance document or figure (roadmap VIZ-6a).
//
//   pulsatrix_explain_text MODEL_DIR "The Eiffel Tower is located in the city of" -o paris.svg
//   pulsatrix_explain_text MODEL_DIR "..." --words -o paris.json      # per word, as JSON
//
// MODEL_DIR holds a Hugging Face config.json, safetensors weights and tokenizer.json. The output
// format follows the extension of -o (.svg, .html or .json); without -o the JSON goes to stdout.
// The HTML page is plain HTML: the browser lays out the text, so every script reads correctly.
// Options: --words (word view), --split whitespace (words between spaces), --no-special (leave
// BOS and other inserted tokens out of the token view), --target TEXT (explain that token
// instead of the most likely one; it must be a single token), --device cpu|hip, --width W.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "pulsatrix/attnlrp_parity.hpp"
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/svg.hpp"
#include "pulsatrix/viz/text_relevance.hpp"
#include "pulsatrix/word_scores.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_explain_text: " << problem << "\n"
              << "usage: pulsatrix_explain_text MODEL_DIR TEXT [-o OUT.svg|OUT.html|OUT.json] [--words] [--split whitespace]\n"
              << "       [--no-special] [--target TEXT] [--device cpu|hip] [--width W]\n";
    std::exit(2);
}

struct Options {
    std::string dir, text, out, target, device = "cpu";
    bool words = false, whitespace = false, special = true;
    int width = 0;
};

int Run(pulsatrix::DeviceBackend* backend, const Options& o) {
    using namespace pulsatrix;
    const TextTokenizer tok = LoadTokenizerJson(o.dir + "/tokenizer.json");
    std::unique_ptr<CausalLM> model = LoadCausalLM(o.dir, backend);
    const Encoding e = tok.encode(o.text);
    if (e.size() == 0) throw std::invalid_argument("the text has no tokens");
    const auto L = static_cast<int64_t>(e.size());
    const int64_t V = model->config().vocab_size;
    std::vector<float> ids(e.ids.begin(), e.ids.end());
    const Tensor logits = model->forward(Tensor(Shape({1, L}), backend, ids));
    const std::vector<float> all = logits.to_host_vector();
    const float* last = all.data() + (L - 1) * V;
    int64_t target = std::max_element(last, last + V) - last;
    if (!o.target.empty()) {
        const std::vector<int64_t> t = tok.encode(o.target, /*add_special_tokens=*/false).ids;
        if (t.size() != 1) throw std::invalid_argument("--target \"" + o.target + "\" is " + std::to_string(t.size()) + " tokens, not one");
        target = t[0];
    }
    std::vector<float> seed(all.size(), 0.0f);
    seed[static_cast<size_t>((L - 1) * V + target)] = last[target];
    const std::vector<float> relevance =
        model->propagate_relevance(Tensor(logits.shape(), backend, seed), LxtAttnLrpConfig()).to_host_vector();

    const std::string target_text = tok.decode({target});
    TokenRelevanceDocument doc;
    if (o.words) {
        const WordScores w = AggregateToWords(o.text, e, relevance, WordAggregation::Sum,
                                              o.whitespace ? WordSplit::Whitespace : WordSplit::WordsAndPunctuation);
        doc = MakeWordRelevanceDocument(o.text, w, "attn_lrp", target_text);
    } else {
        doc = MakeTokenRelevanceDocument(o.text, e, relevance, "attn_lrp", target_text, o.special);
    }
    std::string output;
    const bool svg = o.out.size() >= 4 && o.out.compare(o.out.size() - 4, 4, ".svg") == 0;
    const bool html = o.out.size() >= 5 && o.out.compare(o.out.size() - 5, 5, ".html") == 0;
    if (html) {
        output = RenderTokenRelevanceHtml(doc);
    } else if (svg) {
        SvgOptions options;
        if (o.width > 0) options.width = o.width;
        output = RenderTokenStripSvg(doc, options);
    } else {
        output = ToJson(doc);
    }
    if (o.out.empty()) {
        std::cout << output;
    } else {
        std::ofstream(o.out, std::ios::binary) << output;
        std::cerr << "explained \"" << target_text << "\" (logit " << last[target] << "); wrote " << o.out << "\n";
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) Usage("needs a model directory and a text");
    Options o;
    o.dir = argv[1];
    o.text = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "-o") o.out = value();
        else if (a == "--words") o.words = true;
        else if (a == "--split") {
            const std::string v = value();
            if (v != "whitespace" && v != "words") Usage("--split takes whitespace or words");
            o.whitespace = v == "whitespace";
        } else if (a == "--no-special") o.special = false;
        else if (a == "--target") o.target = value();
        else if (a == "--device") o.device = value();
        else if (a == "--width") o.width = std::atoi(value().c_str());
        else Usage("unknown option " + a);
    }
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
        std::cerr << "pulsatrix_explain_text: " << e.what() << "\n";
        return 1;
    }
}
