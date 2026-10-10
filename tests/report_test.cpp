// NB-2: Report writes nbformat 4.5 notebooks and self-contained HTML. The sample report's notebook
// is compared byte for byte with tests/fixtures/report/sample.ipynb, which
// tests/python/test_report_notebook.py validates with nbformat (the roadmap's falsifier). Set
// PULSATRIX_UPDATE_GOLDEN=1 to rewrite the fixture after an intended change.
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/report.hpp"

namespace pulsatrix {
namespace {

using nlohmann::json;

std::string FixturePath() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/report/sample.ipynb"; }

Report SampleReport(CPUBackend& cpu) {
    Report r("Sample report");
    r.text("Plain output before any code.");
    r.markdown(
        "## Setup\n\nA *small* model, **explained**. See [the docs](https://example.org/docs) and `LRP`.\n\n"
        "- one\n- two\n\n| rule | layers |\n|---|---|\n| Epsilon | Linear |\n| ZPlus | Conv2D |\n");
    r.code("Tensor x(Shape({2, 3}), &backend);\nstd::cout << x.numel();");
    r.text("6\n");
    r.show(Tensor(Shape({2, 3}), &cpu, std::vector<float>{1, 2, 3, -1, -2, -3}));
    HeatmapDocument heat;
    heat.title = "grid";
    heat.rows = 2;
    heat.cols = 2;
    heat.values = {0.5f, -1.0f, 0.25f, 1.0f};
    r.code("display(heatmap);").show(heat);
    std::vector<float> image(70 * 70);
    for (size_t i = 0; i < image.size(); ++i) image[i] = static_cast<float>(i % 70) / 35.0f - 1.0f;
    r.show(Attribution{"saliency", Tensor(Shape({1, 1, 70, 70}), &cpu, image), {}});
    return r;
}

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

TEST(Report, NotebookMatchesTheGoldenFile) {
    CPUBackend cpu;
    const std::string ipynb = SampleReport(cpu).to_ipynb();
    if (std::getenv("PULSATRIX_UPDATE_GOLDEN") != nullptr) {
        std::ofstream(FixturePath(), std::ios::binary) << ipynb;
    }
    EXPECT_EQ(ipynb, ReadFile(FixturePath())) << "rerun with PULSATRIX_UPDATE_GOLDEN=1 after an intended change";
}

TEST(Report, NotebookStructure) {
    CPUBackend cpu;
    const json nb = json::parse(SampleReport(cpu).to_ipynb());
    EXPECT_EQ(nb["nbformat"], 4);
    EXPECT_EQ(nb["nbformat_minor"], 5);
    EXPECT_EQ(nb["metadata"]["kernelspec"]["name"], "xcpp17");
    EXPECT_EQ(nb["metadata"]["title"], "Sample report");
    const json& cells = nb["cells"];
    ASSERT_EQ(cells.size(), 5u);  // title, hidden output cell, markdown, code, code (heatmap and PNG)
    std::set<std::string> ids;
    const std::regex id_pattern("^[a-zA-Z0-9-_]{1,64}$");
    for (const json& c : cells) {
        const std::string id = c["id"];
        EXPECT_TRUE(std::regex_match(id, id_pattern)) << id;
        EXPECT_TRUE(ids.insert(id).second) << "repeated id " << id;
        if (c["cell_type"] == "code") EXPECT_TRUE(c["execution_count"].is_null());
    }
    EXPECT_EQ(cells[0]["source"], json::array({"# Sample report"}));
    // Output before any code: a code cell of its own, its (empty) input hidden.
    EXPECT_EQ(cells[1]["metadata"]["jupyter"]["source_hidden"], true);
    EXPECT_EQ(cells[1]["outputs"][0]["output_type"], "stream");
    // Multi-line sources are line lists, each line but the last ending in a newline.
    EXPECT_EQ(cells[3]["source"], json::array({"Tensor x(Shape({2, 3}), &backend);\n", "std::cout << x.numel();"}));
    const json& outs = cells[3]["outputs"];
    ASSERT_EQ(outs.size(), 2u);
    EXPECT_EQ(outs[0]["text"], json::array({"6\n"}));
    EXPECT_EQ(outs[1]["output_type"], "display_data");
    EXPECT_TRUE(outs[1]["data"]["text/html"].is_array());
    // +json data stays an object; PNG stays one base64 string.
    EXPECT_TRUE(cells[4]["outputs"][0]["data"][kVegaLiteMimeType].is_object());
    EXPECT_TRUE(cells[4]["outputs"][1]["data"]["image/png"].is_string());
}

TEST(Report, HtmlPage) {
    CPUBackend cpu;
    const std::string html = SampleReport(cpu).to_html();
    EXPECT_EQ(html.rfind("<!doctype html>", 0), 0u);
    EXPECT_NE(html.find("<title>Sample report</title>"), std::string::npos);
    EXPECT_NE(html.find("<pre class=\"code\"><code>Tensor x(Shape({2, 3}), &amp;backend);"), std::string::npos);
    EXPECT_NE(html.find("vega-embed"), std::string::npos);  // one chart: the scripts are loaded
    EXPECT_NE(html.find("https://vega.github.io/schema/vega-lite/v6.json"), std::string::npos) << "the page loads Vega-Lite 6";
    EXPECT_EQ(html.find("vega-lite/v5.json"), std::string::npos);
    EXPECT_NE(html.find("data:image/png;base64,iVBORw0KGgo"), std::string::npos);
    EXPECT_EQ(html.find("<?xml"), std::string::npos);  // inline SVG without its prolog

    Report plain("No charts");
    plain.markdown("text only").text("out");
    EXPECT_EQ(plain.to_html().find("<script"), std::string::npos);
}

TEST(Report, EmptyBundleIsRefused) {
    Report r;
    EXPECT_THROW(r.output(MimeBundle{}), std::invalid_argument);
}

TEST(Markdown, BlocksAndInlines) {
    EXPECT_EQ(MarkdownToHtml("# Title\n\nOne\ntwo."), "<h1>Title</h1>\n<p>One\ntwo.</p>\n");
    EXPECT_EQ(MarkdownToHtml("- a\n- **b**\n\n1. x\n2. y"),
              "<ul>\n<li>a</li>\n<li><strong>b</strong></li>\n</ul>\n<ol>\n<li>x</li>\n<li>y</li>\n</ol>\n");
    EXPECT_EQ(MarkdownToHtml("```\nint a = b < c;\n```"), "<pre><code>int a = b &lt; c;\n</code></pre>\n");
    EXPECT_EQ(MarkdownToHtml("| a | b |\n|---|:-:|\n| 1 | `x\\|y` |"),
              "<table>\n<tr><th>a</th><th>b</th></tr>\n<tr><td>1</td><td><code>x|y</code></td></tr>\n</table>\n");
    EXPECT_EQ(MarkdownToHtml("*it* and `a*b*c` and [r](docs/x.md)"),
              "<p><em>it</em> and <code>a*b*c</code> and <a href=\"docs/x.md\">r</a></p>\n");
}

TEST(Markdown, RawHtmlAndUnsafeLinksAreNeutralized) {
    EXPECT_EQ(MarkdownToHtml("<script>alert(1)</script>"), "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>\n");
    EXPECT_EQ(MarkdownToHtml("[click](javascript:alert(1))"), "<p>click</p>\n");
    EXPECT_EQ(MarkdownToHtml("[x](\"onmouseover=\")"), "<p><a href=\"&quot;onmouseover=&quot;\">x</a></p>\n");
    EXPECT_EQ(MarkdownToHtml("[m](mailto:a@b.c)"), "<p><a href=\"mailto:a@b.c\">m</a></p>\n");
}

}  // namespace
}  // namespace pulsatrix
