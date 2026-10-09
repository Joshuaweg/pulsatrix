// pulsatrix_explain_text: explains a language model's next-token prediction with AttnLRP and
// writes a token relevance document or figure (roadmap VIZ-6a).
//
//   pulsatrix_explain_text MODEL_DIR "The Eiffel Tower is located in the city of" -o paris.svg
//   pulsatrix_explain_text MODEL_DIR "..." --words -o paris.json      # per word, as JSON
//
// MODEL_DIR holds a Hugging Face config.json, safetensors weights and tokenizer.json. The output
// format follows the extension of -o (.svg, .html or .json); without -o the JSON goes to stdout.
// The HTML page is plain HTML: the browser lays out the text, so every script reads correctly.
//
//   pulsatrix_explain_text MODEL_DIR "..." --graph paris.json --viewer-dir graphs/   # attribution graph
//
// --graph writes an attribution graph (VIZ-4) for Neuronpedia's or circuit-tracer's viewer: the
// residual stream at every layer and token, linked by the relevance each block passes back.
// --viewer-dir also adds it to DIR/graph-metadata.json, the index circuit-tracer's local viewer
// reads. --slug, --scan (the model id; default the directory's name) and --edge-threshold X
// (keep links holding this share of the relevance; default 0.98) go with it.
// --transcoders DIR builds the graph from transcoder features instead (FEAT-10, circuit-tracer's
// method): DIR holds per-layer or cross-layer transcoders (LoadTranscoders), the nodes are their
// active features, error nodes, token embeddings and the likely logits, and --node-threshold X
// (default 0.8) prunes nodes by influence on the logits.
// Options: --words (word view), --split whitespace (words between spaces), --no-special (leave
// BOS and other inserted tokens out of the token view), --target TEXT (explain that token
// instead of the most likely one; it must be a single token), --device cpu|hip, --width W.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <memory>
#include <string>

#include "pulsatrix/attnlrp_parity.hpp"
#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/circuit_tracing.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tokenizer_json.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/relevance_graph.hpp"
#include "pulsatrix/viz/attribution_graph.hpp"
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
              << "       [--no-special] [--target TEXT] [--device cpu|hip] [--width W]\n"
              << "       [--graph OUT.json [--viewer-dir DIR] [--slug S] [--scan MODEL_ID] [--edge-threshold X]\n"
              << "        [--transcoders DIR [--node-threshold X]]]\n";
    std::exit(2);
}

struct Options {
    std::string dir, text, out, target, device = "cpu", graph, viewer_dir, slug, scan, transcoders;
    double edge_threshold = 0.98, node_threshold = 0.8;
    bool words = false, whitespace = false, special = true;
    int width = 0;
};

/** @brief The slug from a file name: its stem, lowercased, with other characters as dashes. */
std::string SlugOf(const std::string& path) {
    std::string stem = std::filesystem::path(path).stem().string(), slug;
    for (char c : stem) slug += std::isalnum(static_cast<unsigned char>(c)) ? static_cast<char>(std::tolower(static_cast<unsigned char>(c))) : '-';
    return slug.empty() ? "graph" : slug;
}

void WriteGraph(pulsatrix::CausalLM& model, pulsatrix::DeviceBackend* backend, const pulsatrix::TextTokenizer& tok,
                const pulsatrix::Encoding& e, int64_t target, const std::string& target_text, const Options& o) {
    using namespace pulsatrix;
    RelevanceGraphOptions g;
    g.edge_threshold = o.edge_threshold;
    g.slug = o.slug.empty() ? SlugOf(o.graph) : o.slug;
    g.scan = o.scan.empty() ? std::filesystem::path(o.dir).lexically_normal().filename().string() : o.scan;
    if (g.scan.empty()) g.scan = std::filesystem::path(o.dir).lexically_normal().parent_path().filename().string();
    g.prompt = o.text;
    for (int64_t id : e.ids) g.prompt_tokens.push_back(tok.decode({id}));
    g.target_text = target_text;
    AttributionGraph graph;
    if (o.transcoders.empty()) {
        graph = BuildRelevanceGraph(model, backend, e.ids, target, g);
    } else {
        // Transcoder features in place of the MLPs (FEAT-10).
        const CrossLayerTranscoder clt = LoadTranscoders(o.transcoders, model.num_layers(), backend);
        CircuitTraceOptions c;
        c.node_threshold = o.node_threshold;
        c.edge_threshold = o.edge_threshold;
        c.slug = g.slug;
        c.scan = g.scan;
        c.prompt = g.prompt;
        c.prompt_tokens = g.prompt_tokens;
        const CircuitTrace trace = TraceCircuit(model, clt, e.ids, c);
        std::vector<std::string> labels;
        for (int64_t id : trace.logit_tokens) labels.push_back(tok.decode({id}));
        graph = ToAttributionGraph(trace, c, labels);
        const CircuitScores scores = ScoreCircuit(trace);
        std::cerr << trace.features.size() << " active features, " << trace.logit_tokens.size() << " logits; replacement score "
                  << scores.replacement << ", completeness " << scores.completeness << "\n";
    }
    const std::string json = ToNeuronpediaJson(graph);
    std::ofstream(o.graph, std::ios::binary) << json;
    if (!o.viewer_dir.empty()) {
        // circuit-tracer's add_graph_metadata: DIR/graph-metadata.json lists each graph's metadata.
        const std::string index = o.viewer_dir + "/graph-metadata.json";
        JsonValue::Array graphs;
        if (std::ifstream in(index, std::ios::binary); in) {
            std::stringstream ss;
            ss << in.rdbuf();
            const JsonValue index_doc = ParseJson(ss.str());  // must outlive the loop over its array
            for (const auto& entry : index_doc.find("graphs")->as_array()) {
                const JsonValue* slug = entry.find("slug");
                if (slug == nullptr || slug->as_string() != graph.slug) graphs.push_back(entry);
            }
        }
        graphs.push_back(*ParseJson(json).find("metadata"));
        std::ofstream(index, std::ios::binary) << WriteJson(JsonValue::Object{{"graphs", std::move(graphs)}});
    }
    std::cerr << "graph of \"" << target_text << "\": " << graph.nodes.size() << " nodes, " << graph.links.size() << " links; wrote "
              << o.graph << "\n";
}

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
    if (!o.graph.empty()) {
        WriteGraph(*model, backend, tok, e, target, target_text, o);
        if (o.out.empty()) return 0;
    }
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
        else if (a == "--graph") o.graph = value();
        else if (a == "--viewer-dir") o.viewer_dir = value();
        else if (a == "--transcoders") o.transcoders = value();
        else if (a == "--node-threshold") o.node_threshold = std::strtod(value().c_str(), nullptr);
        else if (a == "--slug") o.slug = value();
        else if (a == "--scan") o.scan = value();
        else if (a == "--edge-threshold") o.edge_threshold = std::atof(value().c_str());
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
