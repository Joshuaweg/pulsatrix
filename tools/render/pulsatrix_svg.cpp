// pulsatrix_svg: turns pulsatrix.<kind>.v1 JSON documents into SVG figures (roadmap VIZ-2).
//
//   pulsatrix_svg explanation.json -o bars.svg                 # attribution -> bar chart
//   pulsatrix_svg explanation.json --waterfall 0.12 -o wf.svg  # attribution -> waterfall
//   pulsatrix_svg run*.json --beeswarm 0,3,7 -o swarm.svg      # many attributions -> beeswarm
//   pulsatrix_svg saliency.json -o map.svg                     # heatmap
//   pulsatrix_svg tokens.json -o text.svg                      # token_relevance
//   pulsatrix_svg pd.json --ice centered -o ice.svg            # partial_dependence (CFS-1)
//   pulsatrix_svg sens.json --top-k 8 -o tornado.svg           # sensitivity: tornado (CFS-3)
//   pulsatrix_svg cf.json -o cf.svg                            # counterfactual (CFS-5)
//
// Options: -o FILE (default stdout), --top-k N, --ice raw|centered|derivative, --max-curves N,
//          --width W, --font-size N, --title TEXT.
// Exit status: 0 on success, 1 if a document can't be read or drawn, 2 on a usage error.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/viz/svg.hpp"

namespace {

constexpr const char* kUsage =
    "usage: pulsatrix_svg DOCUMENT.json... [-o FILE] [--top-k N] [--waterfall BASELINE]\n"
    "                     [--beeswarm I,J,...] [--ice raw|centered|derivative] [--max-curves N]\n"
    "                     [--width W] [--font-size N] [--title TEXT]\n";

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_svg: " << problem << "\n" << kUsage;
    std::exit(2);
}

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("can't open " + path);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

long ParseInt(const std::string& s, const char* flag) {
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0') {
        Usage(std::string(flag) + " needs an integer, got \"" + s + "\"");
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace pulsatrix;
    std::vector<std::string> inputs;
    std::string output;
    SvgOptions options;
    int top_k = 10;
    bool waterfall = false;
    float baseline = 0.0f;
    std::vector<int64_t> beeswarm_features;
    PartialDependenceSvgOptions pd_options;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "-o" || a == "--output") {
            output = value();
        } else if (a == "--top-k") {
            top_k = static_cast<int>(ParseInt(value(), "--top-k"));
        } else if (a == "--width") {
            options.width = static_cast<int>(ParseInt(value(), "--width"));
        } else if (a == "--font-size") {
            options.font_size = static_cast<int>(ParseInt(value(), "--font-size"));
        } else if (a == "--title") {
            options.title = value();
        } else if (a == "--waterfall") {
            std::string v = value();
            char* end = nullptr;
            baseline = std::strtof(v.c_str(), &end);
            if (v.empty() || *end != '\0') Usage("--waterfall needs a number, got \"" + v + "\"");
            waterfall = true;
        } else if (a == "--beeswarm") {
            std::stringstream list(value());
            std::string item;
            while (std::getline(list, item, ',')) {
                beeswarm_features.push_back(ParseInt(item, "--beeswarm"));
            }
            if (beeswarm_features.empty()) Usage("--beeswarm needs feature indices");
        } else if (a == "--ice") {
            std::string v = value();
            if (v == "raw") {
                pd_options.style = IceStyle::Raw;
            } else if (v == "centered") {
                pd_options.style = IceStyle::Centered;
            } else if (v == "derivative") {
                pd_options.style = IceStyle::Derivative;
            } else {
                Usage("--ice needs raw, centered or derivative, got \"" + v + "\"");
            }
        } else if (a == "--max-curves") {
            pd_options.max_curves = static_cast<int>(ParseInt(value(), "--max-curves"));
        } else if (a == "-h" || a == "--help") {
            std::cout << kUsage;
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            Usage("unknown option " + a);
        } else {
            inputs.push_back(a);
        }
    }
    if (inputs.empty()) {
        Usage("no input document");
    }
    if (inputs.size() > 1 && beeswarm_features.empty()) {
        Usage("several documents make a beeswarm: add --beeswarm I,J,...");
    }

    try {
        std::string svg;
        if (!beeswarm_features.empty()) {
            std::vector<AttributionDocument> docs;
            for (const std::string& path : inputs) {
                docs.push_back(ParseAttributionDocument(ReadFile(path)));
            }
            svg = RenderBeeswarmSvg(docs, beeswarm_features, options);
        } else {
            const std::string json = ReadFile(inputs.front());
            const std::string kind = VizDocumentKind(json);
            if (kind == "attribution") {
                AttributionDocument doc = ParseAttributionDocument(json);
                svg = waterfall ? RenderWaterfallSvg(doc, baseline, options) : RenderBarChartSvg(doc, top_k, options);
            } else if (kind == "heatmap") {
                svg = RenderHeatmapSvg(ParseHeatmapDocument(json), options);
            } else if (kind == "token_relevance") {
                svg = RenderTokenStripSvg(ParseTokenRelevanceDocument(json), options);
            } else if (kind == "counterfactual") {
                svg = RenderCounterfactualSvg(ParseCounterfactualDocument(json), top_k, options);
            } else if (kind == "sensitivity") {
                svg = RenderTornadoSvg(ParseSensitivityDocument(json), top_k, options);
            } else if (kind == "partial_dependence") {
                svg = RenderPartialDependenceSvg(ParsePartialDependenceDocument(json), pd_options, options);
            } else {
                throw std::invalid_argument("there is no SVG view for " + kind + " documents yet");
            }
        }
        if (output.empty()) {
            std::cout << svg;
        } else {
            std::ofstream out(output, std::ios::binary);
            out << svg;
            if (!out) {
                throw std::runtime_error("can't write " + output);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "pulsatrix_svg: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
