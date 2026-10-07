// pulsatrix_svg: turns pulsatrix.<kind>.v1 JSON documents into SVG figures (roadmap VIZ-2), or into
// interactive HTML pages (VIZ-3) when the output ends in .html or --html is given.
//
//   pulsatrix_svg explanation.json -o bars.svg                 # attribution -> bar chart
//   pulsatrix_svg explanation.json --waterfall 0.12 -o wf.svg  # attribution -> waterfall
//   pulsatrix_svg run*.json --beeswarm 0,3,7 -o swarm.svg      # many attributions -> beeswarm
//   pulsatrix_svg saliency.json -o map.svg                     # heatmap
//   pulsatrix_svg tokens.json -o text.svg                      # token_relevance
//   pulsatrix_svg pd.json --ice centered -o ice.svg            # partial_dependence (CFS-1)
//   pulsatrix_svg sens.json --top-k 8 -o tornado.svg           # sensitivity: tornado (CFS-3)
//   pulsatrix_svg cf.json -o cf.svg                            # counterfactual (CFS-5)
//   pulsatrix_svg cf1.json cf2.json cf3.json -o set.svg        # counterfactual set (CFS-7)
//   pulsatrix_svg morris.json -o morris.svg                    # morris: mu*-sigma scatter (CFS-4)
//   pulsatrix_svg sobol.json -o sobol.svg                      # sobol: index bars (CFS-4)
//
//   pulsatrix_svg explanation.json -o explanation.html          # interactive: hover, zoom, export
//   pulsatrix_svg run.json -o run.html                          # training_log (HTML only)
//   pulsatrix_svg x.json -o x.html --inline-js vega/            # offline page, scripts copied in
//
//   pulsatrix_svg x.json -o x.html --offline                    # the same, scripts found automatically
//
// Options: -o FILE (default stdout), --html, --inline-js DIR (or $PULSATRIX_VEGA_DIR), --offline, --top-k N,
//          --ice raw|centered|derivative, --max-curves N, --width W, --font-size N, --title TEXT.
// Exit status: 0 on success, 1 if a document can't be read or drawn, 2 on a usage error.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/svg.hpp"

namespace {

constexpr const char* kUsage =
    "usage: pulsatrix_svg DOCUMENT.json... [-o FILE] [--top-k N] [--waterfall BASELINE]\n"
    "                     [--beeswarm I,J,...] [--ice raw|centered|derivative] [--max-curves N]\n"
    "                     [--width W] [--font-size N] [--title TEXT] [--html] [--inline-js DIR] [--offline]\n";

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

/** @brief Where tools/render/fetch_vega.sh put the Vega scripts: $PULSATRIX_VEGA_DIR, a vega/
 *         directory next to this program, or share/pulsatrix/vega beside its bin/. */
std::string FindVegaScripts(const char* argv0) {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;
    if (const char* dir = std::getenv("PULSATRIX_VEGA_DIR"); dir != nullptr && *dir != '\0') candidates.emplace_back(dir);
    std::error_code ec;
    const fs::path exe = fs::weakly_canonical(fs::path(argv0), ec).parent_path();
    candidates.push_back(exe / "vega");
    candidates.push_back(exe / ".." / "share" / "pulsatrix" / "vega");
    for (const fs::path& c : candidates) {
        if (fs::exists(c / "vega.min.js", ec) && fs::exists(c / "vega-lite.min.js", ec) && fs::exists(c / "vega-embed.min.js", ec)) {
            return c.string();
        }
    }
    Usage("--offline: no Vega scripts found; run tools/render/fetch_vega.sh DIR, then pass --inline-js DIR "
          "or set PULSATRIX_VEGA_DIR");
}

/** @brief The HTML view of the inputs (VIZ-3), chosen the same way as the SVG one. */
std::string RenderHtml(const std::vector<std::string>& inputs, int top_k, bool waterfall, float baseline,
                       const std::vector<int64_t>& beeswarm_features, const pulsatrix::PartialDependenceSvgOptions& pd,
                       const pulsatrix::HtmlOptions& options) {
    using namespace pulsatrix;
    if (inputs.size() > 1 && beeswarm_features.empty()) {
        std::vector<CounterfactualDocument> docs;
        for (const std::string& path : inputs) docs.push_back(ParseCounterfactualDocument(ReadFile(path)));
        return RenderCounterfactualSetHtml(docs, options);
    }
    if (!beeswarm_features.empty()) {
        std::vector<AttributionDocument> docs;
        for (const std::string& path : inputs) docs.push_back(ParseAttributionDocument(ReadFile(path)));
        return RenderBeeswarmHtml(docs, beeswarm_features, options);
    }
    const std::string json = ReadFile(inputs.front());
    const std::string kind = VizDocumentKind(json);
    if (kind == "attribution") {
        const AttributionDocument doc = ParseAttributionDocument(json);
        return waterfall ? RenderWaterfallHtml(doc, baseline, options) : RenderBarChartHtml(doc, top_k, options);
    }
    if (kind == "heatmap") return RenderHeatmapHtml(ParseHeatmapDocument(json), options);
    if (kind == "token_relevance") return RenderTokenRelevanceHtml(ParseTokenRelevanceDocument(json), options);
    if (kind == "morris") return RenderMorrisHtml(ParseMorrisDocument(json), options);
    if (kind == "sobol") return RenderSobolHtml(ParseSobolDocument(json), top_k, options);
    if (kind == "counterfactual") return RenderCounterfactualHtml(ParseCounterfactualDocument(json), top_k, options);
    if (kind == "sensitivity") return RenderTornadoHtml(ParseSensitivityDocument(json), top_k, options);
    if (kind == "partial_dependence") return RenderPartialDependenceHtml(ParsePartialDependenceDocument(json), pd, options);
    if (kind == "training_log") return RenderTrainingLogHtml(ParseTrainingLogDocument(json), options);
    if (kind == "feature_dashboard") return RenderFeatureDashboardHtml(ParseFeatureDashboardDocument(json), options);
    throw std::invalid_argument("there is no HTML view for " + kind + " documents yet");
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
    bool html = false;
    HtmlOptions html_options;
    if (const char* dir = std::getenv("PULSATRIX_VEGA_DIR"); dir != nullptr && *dir != '\0') {
        html_options.scripts = HtmlScripts::Inline;
        html_options.script_dir = dir;
    }

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
        } else if (a == "--html") {
            html = true;
        } else if (a == "--inline-js") {
            html_options.scripts = HtmlScripts::Inline;
            html_options.script_dir = value();
        } else if (a == "--offline") {
            html_options.scripts = HtmlScripts::Inline;
            html_options.script_dir = FindVegaScripts(argv[0]);
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
    if (output.size() >= 5 && output.compare(output.size() - 5, 5, ".html") == 0) html = true;
    html_options.width = options.width;
    html_options.title = options.title;
    if (inputs.size() > 1 && beeswarm_features.empty()) {
        // Several counterfactual documents make a set; anything else needs --beeswarm.
        bool all_counterfactuals = true;
        try {
            for (const std::string& path : inputs) {
                all_counterfactuals = all_counterfactuals && VizDocumentKind(ReadFile(path)) == "counterfactual";
            }
        } catch (const std::exception&) {
            all_counterfactuals = false;
        }
        if (!all_counterfactuals) {
            Usage("several documents make a beeswarm (add --beeswarm I,J,...) or a counterfactual set");
        }
    }

    try {
        std::string svg;
        if (html) {
            svg = RenderHtml(inputs, top_k, waterfall, baseline, beeswarm_features, pd_options, html_options);
        } else if (inputs.size() > 1 && beeswarm_features.empty()) {
            std::vector<CounterfactualDocument> docs;
            for (const std::string& path : inputs) {
                docs.push_back(ParseCounterfactualDocument(ReadFile(path)));
            }
            svg = RenderCounterfactualSetSvg(docs, options);
        } else if (!beeswarm_features.empty()) {
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
            } else if (kind == "morris") {
                svg = RenderMorrisSvg(ParseMorrisDocument(json), options);
            } else if (kind == "sobol") {
                svg = RenderSobolSvg(ParseSobolDocument(json), top_k, options);
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
