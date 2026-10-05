#include "pulsatrix/viz/svg.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/image_decoder.hpp"
#include "pulsatrix/viz/colormap.hpp"

namespace pulsatrix {
namespace {

// ---- a minimal XML reader: enough to check the output is well-formed and to query it -------

struct Element {
    std::string name;
    std::map<std::string, std::string> attrs;
    std::string text;  // concatenated character data directly inside this element
    std::vector<std::unique_ptr<Element>> children;

    [[nodiscard]] double num(const std::string& attr) const {
        auto it = attrs.find(attr);
        if (it == attrs.end()) {
            throw std::runtime_error("<" + name + "> has no " + attr);
        }
        return std::stod(it->second);
    }
    [[nodiscard]] std::string get(const std::string& attr) const {
        auto it = attrs.find(attr);
        return it == attrs.end() ? "" : it->second;
    }
};

std::string DecodeEntities(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] != '&') {
            out += s[i];
            continue;
        }
        size_t semi = s.find(';', i);
        if (semi == std::string::npos) {
            throw std::runtime_error("unterminated entity");
        }
        std::string e = s.substr(i + 1, semi - i - 1);
        if (e == "amp") out += '&';
        else if (e == "lt") out += '<';
        else if (e == "gt") out += '>';
        else if (e == "quot") out += '"';
        else if (e == "apos") out += '\'';
        else throw std::runtime_error("unknown entity &" + e + ";");
        i = semi;
    }
    return out;
}

// Parses @p xml, throwing std::runtime_error on anything that isn't well-formed: unbalanced or
// mismatched tags, unquoted attributes, a raw '<' or '&' in text, or a second root.
std::unique_ptr<Element> ParseXml(const std::string& xml) {
    size_t pos = 0;
    if (xml.compare(0, 5, "<?xml") == 0) {
        pos = xml.find("?>") + 2;
    }
    std::vector<Element*> stack;
    std::unique_ptr<Element> root;
    while (pos < xml.size()) {
        if (xml[pos] != '<') {
            size_t next = xml.find('<', pos);
            std::string text = xml.substr(pos, next - pos);
            if (text.find('>') != std::string::npos) {
                throw std::runtime_error("raw '>' in text");
            }
            if (stack.empty()) {
                if (text.find_first_not_of(" \n\t\r") != std::string::npos) {
                    throw std::runtime_error("text outside the root");
                }
            } else {
                stack.back()->text += DecodeEntities(text);
            }
            pos = next == std::string::npos ? xml.size() : next;
            continue;
        }
        size_t close = xml.find('>', pos);
        if (close == std::string::npos) {
            throw std::runtime_error("unterminated tag");
        }
        std::string tag = xml.substr(pos + 1, close - pos - 1);
        pos = close + 1;
        if (tag.empty()) {
            throw std::runtime_error("empty tag");
        }
        if (tag[0] == '/') {
            if (stack.empty() || stack.back()->name != tag.substr(1)) {
                throw std::runtime_error("mismatched </" + tag.substr(1) + ">");
            }
            stack.pop_back();
            continue;
        }
        bool self_closing = tag.back() == '/';
        if (self_closing) {
            tag.pop_back();
        }
        auto el = std::make_unique<Element>();
        size_t i = 0;
        while (i < tag.size() && tag[i] != ' ' && tag[i] != '\n') {
            el->name += tag[i++];
        }
        while (true) {
            while (i < tag.size() && (tag[i] == ' ' || tag[i] == '\n')) ++i;
            if (i >= tag.size()) break;
            size_t eq = tag.find('=', i);
            if (eq == std::string::npos || eq + 1 >= tag.size() || tag[eq + 1] != '"') {
                throw std::runtime_error("unquoted attribute in <" + el->name + ">");
            }
            std::string key = tag.substr(i, eq - i);
            size_t end = tag.find('"', eq + 2);
            if (end == std::string::npos) {
                throw std::runtime_error("unterminated attribute");
            }
            std::string raw = tag.substr(eq + 2, end - eq - 2);
            if (raw.find('<') != std::string::npos) {
                throw std::runtime_error("raw '<' in attribute");
            }
            if (!el->attrs.emplace(key, DecodeEntities(raw)).second) {
                throw std::runtime_error("duplicate attribute " + key);
            }
            i = end + 1;
        }
        Element* raw_el = el.get();
        if (stack.empty()) {
            if (root) {
                throw std::runtime_error("second root element");
            }
            root = std::move(el);
        } else {
            stack.back()->children.push_back(std::move(el));
        }
        if (!self_closing) {
            stack.push_back(raw_el);
        }
    }
    if (!stack.empty() || !root) {
        throw std::runtime_error("unclosed elements");
    }
    return root;
}

void Collect(const Element& e, const std::string& cls, std::vector<const Element*>& out) {
    std::string classes = " " + e.get("class") + " ";
    if (classes.find(" " + cls + " ") != std::string::npos) {
        out.push_back(&e);
    }
    for (const auto& c : e.children) {
        Collect(*c, cls, out);
    }
}

// Every element whose class list contains @p cls, in document order.
std::vector<const Element*> ByClass(const Element& root, const std::string& cls) {
    std::vector<const Element*> out;
    Collect(root, cls, out);
    return out;
}

void CollectName(const Element& e, const std::string& name, std::vector<const Element*>& out) {
    if (e.name == name) {
        out.push_back(&e);
    }
    for (const auto& c : e.children) {
        CollectName(*c, name, out);
    }
}

std::vector<const Element*> ByName(const Element& root, const std::string& name) {
    std::vector<const Element*> out;
    CollectName(root, name, out);
    return out;
}

// All character data under @p e, children included.
std::string AllText(const Element& e) {
    std::string s = e.text;
    for (const auto& c : e.children) {
        s += AllText(*c);
    }
    return s;
}

std::string Hex(RgbColor c) {
    auto byte = [](float v) { return static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    char buf[8];
    std::snprintf(buf, sizeof buf, "#%02x%02x%02x", byte(c.r), byte(c.g), byte(c.b));
    return buf;
}

std::unique_ptr<Element> Parsed(const std::string& svg) {
    std::unique_ptr<Element> root;
    EXPECT_NO_THROW(root = ParseXml(svg)) << svg.substr(0, 400);
    EXPECT_EQ(root->name, "svg");
    EXPECT_EQ(root->get("xmlns"), "http://www.w3.org/2000/svg");
    EXPECT_GT(root->num("width"), 0.0);
    EXPECT_GT(root->num("height"), 0.0);
    return root;
}

AttributionDocument Features(std::vector<float> values, std::string names = "") {
    AttributionDocument doc;
    doc.method = "test";
    doc.shape = {static_cast<int64_t>(values.size())};
    doc.values = std::move(values);
    if (!names.empty()) {
        doc.metadata["feature_names"] = names;
    }
    return doc;
}

const float kNaN = std::numeric_limits<float>::quiet_NaN();

// ---- shared behavior ------------------------------------------------------------------------

TEST(SvgRenderTest, EveryChartIsWellFormedDeterministicAndTitled) {
    SvgOptions opt;
    opt.title = "Q & A <1>";
    HeatmapDocument h{"", 2, 2, {0.0f, 1.0f, 2.0f, 3.0f}, {}, {}};
    TokenRelevanceDocument t{"m", {"a", "b"}, {1.0f, -1.0f}, ""};
    std::vector<std::string> svgs = {
        RenderBarChartSvg(Features({1.0f, -2.0f}), 10, opt),
        RenderWaterfallSvg(Features({1.0f, -2.0f}), 0.5f, opt),
        RenderHeatmapSvg(h, opt),
        RenderTokenStripSvg(t, opt),
        RenderBeeswarmSvg({Features({1.0f, 2.0f}), Features({-1.0f, 0.5f})}, {0, 1}, opt),
        RenderPartialDependenceSvg(PartialDependenceDocument{"partial_dependence", "x", "y", {0.0f, 1.0f}, {0.5f, 1.0f}, 1, {0.5f, 1.0f}, {}},
                                   {}, opt),
        RenderTornadoSvg(SensitivityDocument{"y", 1.0f, {{"a", 0.0f, -1.0f, 1.0f, 0.5f, 2.0f}}}, 10, opt),
        RenderCounterfactualSvg(CounterfactualDocument{"t", true, 0.0f, 1.0f, {{"a", 0.0f, 1.0f, 1.0f}}}, 12, opt),
        RenderMorrisSvg(MorrisDocument{"y", 4, {{"a", 1.0f, 1.0f, 0.5f, 0.1f}}}, opt),
        RenderSobolSvg(SobolDocument{"y", 64, {{"a", 0.5f, 0.75f, 0.1f, 0.1f}}}, 10, opt),
    };
    for (const std::string& svg : svgs) {
        auto root = Parsed(svg);
        // The title is both visible text and the accessible name.
        auto titles = ByName(*root, "title");
        ASSERT_FALSE(titles.empty());
        EXPECT_EQ(titles[0]->text, "Q & A <1>");
        EXPECT_EQ(ByClass(*root, "chart-title").size(), 1u);
        EXPECT_EQ(root->get("role"), "img");
    }
    EXPECT_EQ(RenderBarChartSvg(Features({1.0f, -2.0f}), 10, opt), svgs[0]);
}

TEST(SvgRenderTest, ExtremeValuesStillRender) {
    // The largest floats, and values huge next to their spread, still give numeric
    // coordinates and a finished tick layout.
    const float big = std::numeric_limits<float>::max();
    for (const std::string& svg : {RenderBarChartSvg(Features({big, -big})), RenderWaterfallSvg(Features({1e10f, -2e10f}), 1e17f),
                                   RenderBeeswarmSvg({Features({big}), Features({-big})}, {0}),
                                   RenderHeatmapSvg(HeatmapDocument{"", 1, 2, {1e30f, 1.0000001e30f}, {}, {}})}) {
        Parsed(svg);
        EXPECT_EQ(svg.find("nan"), std::string::npos);
        EXPECT_EQ(svg.find("inf"), std::string::npos);
    }
}

TEST(SvgRenderTest, RejectsUnusableOptions) {
    SvgOptions narrow;
    narrow.width = 100;
    EXPECT_THROW((void)RenderBarChartSvg(Features({1.0f}), 10, narrow), std::invalid_argument);
    SvgOptions no_font;
    no_font.font_size = 0;
    EXPECT_THROW((void)RenderTokenStripSvg({"m", {"a"}, {1.0f}, ""}, no_font), std::invalid_argument);
}

// ---- bar chart ------------------------------------------------------------------------------

TEST(SvgBarChartTest, BarsAreSortedProportionalAndSignColored) {
    auto root = Parsed(RenderBarChartSvg(Features({0.5f, -2.0f, 1.0f, 0.25f}, "a,b,c,d"), 3));
    auto bars = ByClass(*root, "bar");
    ASSERT_EQ(bars.size(), 3u);  // top_k
    // Largest |value| first, top to bottom: b (-2), c (1), a (0.5).
    EXPECT_LT(bars[0]->num("y"), bars[1]->num("y"));
    EXPECT_LT(bars[1]->num("y"), bars[2]->num("y"));
    EXPECT_NEAR(bars[0]->num("width") / bars[1]->num("width"), 2.0, 0.01);
    EXPECT_NEAR(bars[1]->num("width") / bars[2]->num("width"), 2.0, 0.01);
    // The negative bar ends at the zero line; positive bars start there.
    auto zero = ByClass(*root, "zero-line");
    ASSERT_EQ(zero.size(), 1u);
    const double x0 = zero[0]->num("x1");
    EXPECT_NEAR(bars[0]->num("x") + bars[0]->num("width"), x0, 0.01);
    EXPECT_NEAR(bars[1]->num("x"), x0, 0.01);
    EXPECT_EQ(bars[0]->get("fill"), Hex(DivergingColormap(-1.0f)));
    EXPECT_EQ(bars[1]->get("fill"), Hex(DivergingColormap(0.5f)));
    auto labels = ByClass(*root, "feature-label");
    ASSERT_EQ(labels.size(), 3u);
    EXPECT_EQ(labels[0]->text, "b");
    EXPECT_EQ(labels[2]->text, "a");
    auto values = ByClass(*root, "value-label");
    ASSERT_EQ(values.size(), 3u);
    EXPECT_EQ(values[0]->text, "-2");
}

TEST(SvgBarChartTest, EscapesLabelsAndReplacesBytesXmlCannotHold) {
    auto root = Parsed(RenderBarChartSvg(Features({1.0f, 2.0f}, std::string("<script>&\",x\x01\xFFy"))));
    auto labels = ByClass(*root, "feature-label");
    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels[0]->text, "x\xEF\xBF\xBD\xEF\xBF\xBDy");  // U+FFFD for the control byte and the invalid byte
    EXPECT_EQ(labels[1]->text, "<script>&\"");
}

TEST(SvgBarChartTest, RejectsNonFiniteValues) {
    EXPECT_THROW((void)RenderBarChartSvg(Features({1.0f, kNaN})), std::invalid_argument);
}

// ---- waterfall ------------------------------------------------------------------------------

TEST(SvgWaterfallTest, ColumnsSpanTheRunningTotal) {
    // baseline 1, steps +2, -0.5, +1 -> running totals 3, 2.5, 3.5
    auto root = Parsed(RenderWaterfallSvg(Features({2.0f, -0.5f, 1.0f}, "x,y,z"), 1.0f));
    auto steps = ByClass(*root, "step");
    ASSERT_EQ(steps.size(), 3u);
    // Left to right in feature order.
    EXPECT_LT(steps[0]->num("x"), steps[1]->num("x"));
    EXPECT_LT(steps[1]->num("x"), steps[2]->num("x"));
    // Heights proportional to |delta|.
    EXPECT_NEAR(steps[0]->num("height") / steps[1]->num("height"), 4.0, 0.01);
    EXPECT_NEAR(steps[2]->num("height") / steps[1]->num("height"), 2.0, 0.01);
    EXPECT_EQ(steps[0]->get("fill"), Hex(DivergingColormap(1.0f)));
    EXPECT_EQ(steps[1]->get("fill"), Hex(DivergingColormap(-1.0f)));
    // Each column starts where the previous one ended: the bottom of the decrease is the top of
    // the next increase.
    const double bottom1 = steps[1]->num("y") + steps[1]->num("height");
    EXPECT_NEAR(steps[2]->num("y") + steps[2]->num("height"), bottom1, 0.01);
    EXPECT_NEAR(steps[0]->num("y"), steps[1]->num("y"), 0.01);
    EXPECT_EQ(ByClass(*root, "baseline").size(), 1u);
    auto total = ByClass(*root, "total-label");
    ASSERT_EQ(total.size(), 1u);
    EXPECT_NE(total[0]->text.find("3.5"), std::string::npos) << total[0]->text;
}

TEST(SvgWaterfallTest, RejectsNonFiniteBaseline) {
    EXPECT_THROW((void)RenderWaterfallSvg(Features({1.0f}), kNaN), std::invalid_argument);
}

// ---- heatmap --------------------------------------------------------------------------------

TEST(SvgHeatmapTest, SmallGridsAreOneSquareCellPerValue) {
    HeatmapDocument doc{"", 2, 3, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, kNaN}, {"r0", "r1"}, {"c0", "c1", "c2"}};
    auto root = Parsed(RenderHeatmapSvg(doc));
    auto cells = ByClass(*root, "cell");
    ASSERT_EQ(cells.size(), 6u);
    for (const Element* c : cells) {
        EXPECT_NEAR(c->num("width"), c->num("height"), 1e-9);
    }
    // Row-major, row 0 on top.
    EXPECT_LT(cells[0]->num("x"), cells[1]->num("x"));
    EXPECT_LT(cells[0]->num("y"), cells[3]->num("y"));
    // Unsigned: Viridis over [0, max finite].
    EXPECT_EQ(cells[0]->get("fill"), Hex(ViridisColormap(0.0f)));
    EXPECT_EQ(cells[4]->get("fill"), Hex(ViridisColormap(1.0f)));
    EXPECT_EQ(cells[2]->get("fill"), Hex(ViridisColormap(0.5f)));
    EXPECT_EQ(cells[5]->get("class").find("missing") != std::string::npos, true);
    EXPECT_EQ(ByClass(*root, "row-label").size(), 2u);
    EXPECT_EQ(ByClass(*root, "col-label").size(), 3u);
    EXPECT_EQ(ByClass(*root, "colorbar").size(), 1u);
}

TEST(SvgHeatmapTest, SignedGridsUseTheSymmetricDivergingScale) {
    HeatmapDocument doc{"", 1, 3, {-1.0f, 0.0f, 4.0f}, {}, {}};
    auto root = Parsed(RenderHeatmapSvg(doc));
    auto cells = ByClass(*root, "cell");
    ASSERT_EQ(cells.size(), 3u);
    EXPECT_EQ(cells[0]->get("fill"), Hex(DivergingColormap(-0.25f)));
    EXPECT_EQ(cells[1]->get("fill"), Hex(DivergingColormap(0.0f)));
    EXPECT_EQ(cells[2]->get("fill"), Hex(DivergingColormap(1.0f)));
}

TEST(SvgHeatmapTest, LargeGridsAreAnEmbeddedPngWithTheSameColors) {
    const int64_t n = 100;
    HeatmapDocument doc;
    doc.rows = n;
    doc.cols = n;
    for (int64_t i = 0; i < n * n; ++i) {
        doc.values.push_back(static_cast<float>(i % n));  // 0..99 across each row
    }
    std::string svg = RenderHeatmapSvg(doc);
    auto root = Parsed(svg);
    EXPECT_TRUE(ByClass(*root, "cell").empty());
    auto images = ByName(*root, "image");
    ASSERT_EQ(images.size(), 1u);
    const std::string href = images[0]->get("href");
    const std::string prefix = "data:image/png;base64,";
    ASSERT_EQ(href.compare(0, prefix.size(), prefix), 0);
    EXPECT_LT(svg.size(), 200000u);

    // Decode the PNG and compare pixels with the colormap.
    static const std::string kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string bytes;
    uint32_t acc = 0;
    int bits = 0;
    for (char ch : href.substr(prefix.size())) {
        if (ch == '=') break;
        acc = (acc << 6) | static_cast<uint32_t>(kAlphabet.find(ch));
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            bytes += static_cast<char>((acc >> bits) & 0xFF);
        }
    }
    auto path = std::filesystem::temp_directory_path() / "pulsatrix_svg_heatmap_test.png";
    std::ofstream(path, std::ios::binary) << bytes;
    CPUBackend backend;
    Tensor img = ImageDecoder::DecodeFile(path.string(), &backend, 3);
    std::filesystem::remove(path);
    ASSERT_EQ(img.shape(), Shape({1, 3, n, n}));
    std::vector<float> px = img.to_host_vector();
    auto pixel = [&](int64_t row, int64_t col, int ch) { return px[static_cast<size_t>((ch * n + row) * n + col)]; };
    for (int64_t col : {int64_t{0}, int64_t{50}, int64_t{99}}) {
        RgbColor want = ViridisColormap(static_cast<float>(col) / 99.0f);
        EXPECT_NEAR(pixel(7, col, 0), want.r, 0.6 / 255.0 + 1e-6) << col;
        EXPECT_NEAR(pixel(7, col, 1), want.g, 0.6 / 255.0 + 1e-6) << col;
        EXPECT_NEAR(pixel(7, col, 2), want.b, 0.6 / 255.0 + 1e-6) << col;
    }
}

TEST(SvgHeatmapTest, RejectsEmptyOrInconsistentGrids) {
    EXPECT_THROW((void)RenderHeatmapSvg(HeatmapDocument{}), std::invalid_argument);
    EXPECT_THROW((void)RenderHeatmapSvg(HeatmapDocument{"", 2, 2, {1.0f}, {}, {}}), std::invalid_argument);
}

// ---- token strip ----------------------------------------------------------------------------

TEST(SvgTokenStripTest, OneColoredBoxPerTokenWithWhitespaceKept) {
    TokenRelevanceDocument doc{"attn_lrp", {"The", " cat", " <sat>", "!"}, {0.5f, -1.0f, 0.0f, kNaN}, " on"};
    auto root = Parsed(RenderTokenStripSvg(doc));
    auto boxes = ByClass(*root, "token");
    ASSERT_EQ(boxes.size(), 4u);
    EXPECT_EQ(boxes[0]->get("fill"), Hex(DivergingColormap(0.5f)));
    EXPECT_EQ(boxes[1]->get("fill"), Hex(DivergingColormap(-1.0f)));
    EXPECT_EQ(boxes[2]->get("fill"), Hex(DivergingColormap(0.0f)));
    EXPECT_NE(boxes[3]->get("class").find("missing"), std::string::npos);
    auto texts = ByClass(*root, "token-text");
    ASSERT_EQ(texts.size(), 4u);
    EXPECT_EQ(texts[1]->text, " cat");
    EXPECT_EQ(texts[2]->text, " <sat>");
    EXPECT_EQ(texts[1]->get("xml:space"), "preserve");
    // Monospace widths: proportional to the number of characters.
    EXPECT_NEAR(boxes[1]->num("width") / boxes[3]->num("width"), 4.0, 0.01);
    auto target = ByClass(*root, "target");
    ASSERT_EQ(target.size(), 1u);
    EXPECT_NE(AllText(*target[0]).find(" on"), std::string::npos);
}

TEST(SvgTokenStripTest, WrapsWithinTheWidth) {
    TokenRelevanceDocument doc;
    doc.method = "m";
    for (int i = 0; i < 60; ++i) {
        doc.tokens.push_back(" word" + std::to_string(i));
        doc.relevance.push_back(static_cast<float>(i % 7) - 3.0f);
    }
    SvgOptions opt;
    opt.width = 300;
    auto root = Parsed(RenderTokenStripSvg(doc, opt));
    auto boxes = ByClass(*root, "token");
    ASSERT_EQ(boxes.size(), 60u);
    std::set<double> lines;
    for (const Element* b : boxes) {
        lines.insert(b->num("y"));
        EXPECT_LE(b->num("x") + b->num("width"), 300.0);
        EXPECT_GE(b->num("x"), 0.0);
    }
    EXPECT_GT(lines.size(), 5u);
    EXPECT_LE(root->num("width"), 300.0);
}

TEST(SvgTokenStripTest, CountsCharactersNotBytes) {
    TokenRelevanceDocument doc{"m", {"caf\xC3\xA9", "cafe"}, {1.0f, 1.0f}, ""};
    auto root = Parsed(RenderTokenStripSvg(doc));
    auto boxes = ByClass(*root, "token");
    ASSERT_EQ(boxes.size(), 2u);
    EXPECT_NEAR(boxes[0]->num("width"), boxes[1]->num("width"), 1e-9);
}

// ---- beeswarm -------------------------------------------------------------------------------

TEST(SvgBeeswarmTest, OneRowPerFeatureOnASharedAxis) {
    std::vector<AttributionDocument> docs;
    for (int i = 0; i < 30; ++i) {
        float v = static_cast<float>(i) / 10.0f - 1.5f;
        docs.push_back(Features({v, 2.0f * v, 0.0f}, "alpha,beta,gamma"));
    }
    auto root = Parsed(RenderBeeswarmSvg(docs, {1, 0}));
    auto rows = ByClass(*root, "feature-row");
    ASSERT_EQ(rows.size(), 2u);
    auto labels = ByClass(*root, "feature-label");
    ASSERT_EQ(labels.size(), 2u);
    EXPECT_EQ(labels[0]->text, "beta");
    EXPECT_EQ(labels[1]->text, "alpha");
    auto points = ByClass(*root, "point");
    ASSERT_EQ(points.size(), 60u);
    // Rows don't overlap: every point of row 0 is above every point of row 1.
    auto row_points = [&](size_t r) { return ByClass(*rows[r], "point"); };
    double row0_max = 0, row1_min = 1e9;
    for (const Element* p : row_points(0)) row0_max = std::max(row0_max, p->num("cy") + p->num("r"));
    for (const Element* p : row_points(1)) row1_min = std::min(row1_min, p->num("cy") - p->num("r"));
    EXPECT_LT(row0_max, row1_min);
    // x is the value on one shared axis: same value, same x, in either row.
    auto zero = ByClass(*root, "zero-line");
    ASSERT_EQ(zero.size(), 1u);
    for (const Element* p : row_points(1)) {
        // row 1 is alpha, values -1.5..1.4; beta reaches 2.8, so the axis extends further right
        EXPECT_LT(p->num("cx"), root->num("width"));
    }
    // Colors follow the value on the shared scale (max |value| = 3, beta's -3).
    auto first = row_points(0);
    bool found_extreme = std::any_of(first.begin(), first.end(), [](const Element* p) { return p->get("fill") == Hex(DivergingColormap(-1.0f)); });
    EXPECT_TRUE(found_extreme);
}

TEST(SvgBeeswarmTest, RejectsBadInput) {
    EXPECT_THROW((void)RenderBeeswarmSvg({}, {0}), std::invalid_argument);
    EXPECT_THROW((void)RenderBeeswarmSvg({Features({1.0f})}, {}), std::invalid_argument);
    EXPECT_THROW((void)RenderBeeswarmSvg({Features({1.0f})}, {3}), std::invalid_argument);
    EXPECT_THROW((void)RenderBeeswarmSvg({Features({kNaN})}, {0}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix

namespace pulsatrix {
namespace {

// ---- partial dependence ---------------------------------------------------------------------

PartialDependenceDocument IceDocument(int64_t instances) {
    IceResult r;
    r.grid = {0.0f, 1.0f, 2.0f, 3.0f};
    r.num_instances = instances;
    for (int64_t i = 0; i < instances; ++i) {
        for (float x : r.grid) {
            r.curves.push_back(static_cast<float>(i) * x - x * x / 4.0f);  // slope grows with i
        }
        r.feature_values.push_back(static_cast<float>(i % 4));
    }
    return ToPartialDependenceDocument(r, "dose", "response");
}

TEST(SvgPartialDependenceTest, DrawsTheAverageOverAnEvenSampleOfCurves) {
    auto root = Parsed(RenderPartialDependenceSvg(IceDocument(10)));
    EXPECT_EQ(ByClass(*root, "ice").size(), 10u);
    EXPECT_EQ(ByClass(*root, "average").size(), 1u);
    EXPECT_EQ(ByClass(*root, "rug").front()->children.size(), 10u);

    PartialDependenceSvgOptions few;
    few.max_curves = 3;
    auto sampled = Parsed(RenderPartialDependenceSvg(IceDocument(10), few));
    EXPECT_EQ(ByClass(*sampled, "ice").size(), 3u);
    EXPECT_NE(AllText(*sampled).find("3 of 10 shown"), std::string::npos);
    few.max_curves = 0;
    EXPECT_TRUE(ByClass(*Parsed(RenderPartialDependenceSvg(IceDocument(10), few)), "ice").empty());

    // Higher on the page is a larger value: the average's last point (largest) is above its first.
    auto avg = ByClass(*root, "average").front()->get("points");
    const double first_y = std::stod(avg.substr(avg.find(',') + 1));
    const double last_y = std::stod(avg.substr(avg.rfind(',') + 1));
    EXPECT_LT(last_y, first_y);
}

TEST(SvgPartialDependenceTest, CenteredAndDerivativeViewsAddAZeroLine) {
    for (IceStyle style : {IceStyle::Centered, IceStyle::Derivative}) {
        PartialDependenceSvgOptions o;
        o.style = style;
        auto root = Parsed(RenderPartialDependenceSvg(IceDocument(5), o));
        EXPECT_EQ(ByClass(*root, "zero-line").size(), 1u);
        EXPECT_EQ(ByClass(*root, "ice").size(), 5u);
    }
}

TEST(SvgPartialDependenceTest, AleIsLabeledAsSuchAroundZero) {
    AleResult r;
    r.edges = {0.0f, 1.0f, 2.0f, 4.0f};
    r.effects = {-1.5f, -0.5f, 0.5f, 1.0f};
    r.counts = {3, 3, 2};
    r.feature_values = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.0f, 3.0f, 4.0f};
    auto root = Parsed(RenderPartialDependenceSvg(ToPartialDependenceDocument(r, "x", "y")));
    EXPECT_NE(AllText(*root).find("accumulated local effect (centered)"), std::string::npos);
    EXPECT_NE(AllText(*root).find("accumulated local effect on y"), std::string::npos);
    EXPECT_EQ(ByClass(*root, "zero-line").size(), 1u);
    EXPECT_TRUE(ByClass(*root, "ice").empty());
    EXPECT_EQ(ByClass(*root, "rug").front()->children.size(), 8u);
}

TEST(SvgPartialDependenceTest, RejectsBadInput) {
    PartialDependenceDocument average_only{"partial_dependence", "x", "y", {0.0f, 1.0f}, {0.5f, 1.0f}, 0, {}, {}};
    EXPECT_NO_THROW((void)RenderPartialDependenceSvg(average_only));
    PartialDependenceSvgOptions centered;
    centered.style = IceStyle::Centered;
    EXPECT_THROW((void)RenderPartialDependenceSvg(average_only, centered), std::invalid_argument);
    PartialDependenceDocument nan = average_only;
    nan.partial_dependence[1] = kNaN;
    EXPECT_THROW((void)RenderPartialDependenceSvg(nan), std::invalid_argument);
    PartialDependenceDocument unsorted = average_only;
    unsorted.grid = {1.0f, 0.0f};
    EXPECT_THROW((void)RenderPartialDependenceSvg(unsorted), std::invalid_argument);
    PartialDependenceSvgOptions negative;
    negative.max_curves = -1;
    EXPECT_THROW((void)RenderPartialDependenceSvg(average_only, negative), std::invalid_argument);
}

// ---- tornado ------------------------------------------------------------------------------------

SensitivityDocument Tornado() {
    // Swings: a 1.5, b 3, c 0.5, d 0 -> rows b, a, c, d.
    return SensitivityDocument{"y", 1.0f,
                               {{"a", 0.0f, -1.0f, 1.0f, 0.5f, 2.0f},
                                {"b", 5.0f, 4.0f, 6.0f, 2.5f, -0.5f},
                                {"c", 1.0f, 0.0f, 2.0f, 1.25f, 0.75f},
                                {"d", 7.0f, 6.0f, 8.0f, 1.0f, 1.0f}}};
}

TEST(SvgTornadoTest, RowsSortByEachFeaturesSwingAndBarsLeaveTheUnchangedOutput) {
    auto root = Parsed(RenderTornadoSvg(Tornado()));
    auto labels = ByClass(*root, "feature-label");
    ASSERT_EQ(labels.size(), 4u);
    EXPECT_EQ(labels[0]->text, "b = 5");
    EXPECT_EQ(labels[1]->text, "a = 0");
    EXPECT_EQ(labels[3]->text, "d = 7");
    const double base = ByClass(*root, "base-line").front()->num("x1");
    auto high = ByClass(*root, "bar-high");
    auto low = ByClass(*root, "bar-low");
    // b: high value lowers the output (bar left of the base), low value raises it.
    EXPECT_NEAR(high[0]->num("x") + high[0]->num("width"), base, 1e-6);
    EXPECT_NEAR(low[0]->num("x"), base, 1e-6);
    // Bar lengths are proportional to the output change: b's high bar (1.5) is 3x c's (0.5... low: 0.25).
    EXPECT_NEAR(high[0]->num("width") / low[2]->num("width"), 1.5 / 0.25, 1e-3);
    EXPECT_EQ(ByClass(*Parsed(RenderTornadoSvg(Tornado(), 2)), "feature-row").size(), 2u);
}

TEST(SvgTornadoTest, RejectsBadInput) {
    EXPECT_THROW((void)RenderTornadoSvg(SensitivityDocument{}), std::invalid_argument);
    EXPECT_THROW((void)RenderTornadoSvg(Tornado(), 0), std::invalid_argument);
    SensitivityDocument nan = Tornado();
    nan.features[1].output_high = kNaN;
    EXPECT_THROW((void)RenderTornadoSvg(nan), std::invalid_argument);
}

// ---- counterfactual -------------------------------------------------------------------------

TEST(SvgCounterfactualTest, ListsOnlyChangedFeaturesCostliestFirst) {
    // Costs in scale units: a +1.5, b 0, c -4, d +0.5.
    CounterfactualDocument doc{"class 1", true, -1.0f, 0.25f,
                               {{"a", 1.0f, 4.0f, 2.0f}, {"b", 5.0f, 5.0f, 1.0f}, {"c", 0.0f, -2.0f, 0.5f},
                                {"d", 3.0f, 3.5f, 1.0f}}};
    auto root = Parsed(RenderCounterfactualSvg(doc));
    auto labels = ByClass(*root, "feature-label");
    ASSERT_EQ(labels.size(), 3u);
    EXPECT_EQ(labels[0]->text, "c");
    EXPECT_EQ(labels[1]->text, "a");
    EXPECT_EQ(labels[2]->text, "d");
    EXPECT_NE(AllText(*root).find("Reaches class 1"), std::string::npos);
    EXPECT_NE(AllText(*root).find("3 of 4 features changed, distance 6"), std::string::npos);
    // c's bar (cost -4) is left of zero and 8x as long as d's (+0.5).
    auto bars = ByClass(*root, "bar");
    const double zero = ByClass(*root, "zero-line").front()->num("x1");
    EXPECT_NEAR(bars[0]->num("x") + bars[0]->num("width"), zero, 1e-6);
    EXPECT_NEAR(bars[0]->num("width") / bars[2]->num("width"), 8.0, 8.0 * 5e-3);  // coordinates are rounded

    auto two = Parsed(RenderCounterfactualSvg(doc, 2));
    EXPECT_EQ(ByClass(*two, "changed-feature").size(), 2u);
    EXPECT_NE(AllText(*two).find("1 more not shown"), std::string::npos);
    doc.valid = false;
    EXPECT_NE(AllText(*Parsed(RenderCounterfactualSvg(doc))).find("Does not reach"), std::string::npos);
}

TEST(SvgCounterfactualTest, HandlesNoChangeAndRejectsBadInput) {
    CounterfactualDocument same{"", false, 1.0f, 1.0f, {{"a", 1.0f, 1.0f, 1.0f}}};
    EXPECT_NE(AllText(*Parsed(RenderCounterfactualSvg(same))).find("No feature changed"), std::string::npos);
    EXPECT_THROW((void)RenderCounterfactualSvg(same, 0), std::invalid_argument);
    CounterfactualDocument nan = same;
    nan.features[0].counterfactual = kNaN;
    EXPECT_THROW((void)RenderCounterfactualSvg(nan), std::invalid_argument);
    CounterfactualDocument zero_scale = same;
    zero_scale.features[0].scale = 0.0f;
    EXPECT_THROW((void)RenderCounterfactualSvg(zero_scale), std::invalid_argument);
}

// ---- global sensitivity -----------------------------------------------------------------------

TEST(SvgGlobalSensitivityTest, MorrisPlotsMuStarAgainstSigmaOnOneScale) {
    MorrisDocument doc{"y", 10, {{"a", 2.0f, 2.0f, 0.0f, 0.5f}, {"b", 0.0f, 1.0f, 2.0f, 0.0f}}};
    auto root = Parsed(RenderMorrisSvg(doc));
    auto points = ByClass(*root, "point");
    ASSERT_EQ(points.size(), 2u);
    // a: mu* 2, sigma 0 -> on the x axis, right of b; b: sigma 2 -> above a.
    EXPECT_GT(points[0]->num("cx"), points[1]->num("cx"));
    EXPECT_GT(points[0]->num("cy"), points[1]->num("cy"));
    // One scale: b's height above the axis is twice a's distance from the y axis... (sigma 2 vs mu* 2:
    // equal distances), so the diagonal passes through b's x and a's x at the same scale.
    const double axis_x = ByClass(*root, "diagonal").front()->num("x1");
    const double axis_y = ByClass(*root, "diagonal").front()->num("y1");
    EXPECT_NEAR(points[0]->num("cx") - axis_x, axis_y - points[1]->num("cy"), 1e-6);
    EXPECT_EQ(ByClass(*root, "conf").size(), 1u);  // b has no interval
    MorrisDocument negative = doc;
    negative.features[0].sigma = -1.0f;
    EXPECT_THROW((void)RenderMorrisSvg(negative), std::invalid_argument);
    EXPECT_THROW((void)RenderMorrisSvg(MorrisDocument{}), std::invalid_argument);
}

TEST(SvgGlobalSensitivityTest, SobolRowsSortByTotalOrderWithBothBars) {
    SobolDocument doc{"y", 64, {{"a", 0.1f, 0.2f, 0.05f, 0.05f}, {"b", 0.6f, 0.7f, 0.0f, 0.1f}, {"c", 0.0f, 0.05f, 0.0f, 0.0f}}};
    auto root = Parsed(RenderSobolSvg(doc));
    auto labels = ByClass(*root, "feature-label");
    ASSERT_EQ(labels.size(), 3u);
    EXPECT_EQ(labels[0]->text, "b");
    EXPECT_EQ(labels[2]->text, "c");
    auto total = ByClass(*root, "bar-total");
    auto first = ByClass(*root, "bar-first");
    EXPECT_NEAR(total[0]->num("width") / first[0]->num("width"), 0.7 / 0.6, 0.01);
    EXPECT_EQ(ByClass(*Parsed(RenderSobolSvg(doc, 1)), "feature-row").size(), 1u);
    EXPECT_THROW((void)RenderSobolSvg(doc, 0), std::invalid_argument);
    doc.features[0].total_order = kNaN;
    EXPECT_THROW((void)RenderSobolSvg(doc), std::invalid_argument);
}

// ---- counterfactual set ---------------------------------------------------------------------

std::vector<CounterfactualDocument> CounterfactualSet() {
    // Two routes to approval: more income, or less debt (and the second also moves age, invalid).
    CounterfactualDocument a{"approved", true, -0.5f, 0.25f,
                             {{"income", 40.0f, 52.0f, 8.0f}, {"age", 35.0f, 35.0f, 10.0f}, {"debt", 0.5f, 0.5f, 0.25f}}};
    CounterfactualDocument b{"approved", true, -0.5f, 0.1f,
                             {{"income", 40.0f, 40.0f, 8.0f}, {"age", 35.0f, 35.0f, 10.0f}, {"debt", 0.5f, 0.125f, 0.25f}}};
    CounterfactualDocument c{"approved", false, -0.5f, -0.1f,
                             {{"income", 40.0f, 40.0f, 8.0f}, {"age", 35.0f, 30.0f, 10.0f}, {"debt", 0.5f, 0.5f, 0.25f}}};
    return {a, b, c};
}

TEST(SvgCounterfactualSetTest, OneColumnPerCounterfactualAndOneRowPerChangedFeature) {
    auto root = Parsed(RenderCounterfactualSetSvg(CounterfactualSet()));
    auto rows = ByClass(*root, "feature-row");
    ASSERT_EQ(rows.size(), 3u);  // income, age and debt each change somewhere
    EXPECT_EQ(ByClass(*root, "changed").size(), 3u);  // one changed cell per counterfactual
    EXPECT_EQ(ByClass(*root, "valid").size(), 2u);
    EXPECT_EQ(ByClass(*root, "invalid").size(), 1u);
    EXPECT_NE(AllText(*root).find("2 of 3 reach approved"), std::string::npos);

    auto mismatched = CounterfactualSet();
    mismatched[1].features[0].original = 41.0f;
    EXPECT_THROW((void)RenderCounterfactualSetSvg(mismatched), std::invalid_argument);
    EXPECT_THROW((void)RenderCounterfactualSetSvg({}), std::invalid_argument);
}

// ---- golden files: each chart, from the VIZ-1 document fixtures, must match
// tests/fixtures/viz/svg/ byte for byte on every platform. After an intended change, rerun
// with PULSATRIX_UPDATE_GOLDEN=1 to rewrite them, then look at the new figures before
// committing.

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void ExpectGolden(const std::string& name, const std::string& svg) {
    const std::string path = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/svg/" + name;
    const char* update = std::getenv("PULSATRIX_UPDATE_GOLDEN");
    if (update != nullptr && std::string(update) == "1") {
        std::ofstream(path, std::ios::binary) << svg;
        return;
    }
    const std::string golden = ReadFile(path);
    ASSERT_FALSE(golden.empty()) << "missing golden file " << path;
    EXPECT_EQ(svg, golden) << name << " changed; see the comment above ExpectGolden";
}

std::string Fixture(const std::string& name) {
    return ReadFile(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name);
}

TEST(SvgGoldenTest, EveryChart) {
    AttributionDocument attr = ParseAttributionDocument(Fixture("attribution_features.v1.json"));
    SvgOptions titled;
    titled.title = "Integrated gradients";
    ExpectGolden("bar_chart.svg", RenderBarChartSvg(attr, 10, titled));
    ExpectGolden("waterfall.svg", RenderWaterfallSvg(attr, 0.25f));
    ExpectGolden("heatmap.svg", RenderHeatmapSvg(ParseHeatmapDocument(Fixture("heatmap.v1.json"))));
    ExpectGolden("token_strip.svg", RenderTokenStripSvg(ParseTokenRelevanceDocument(Fixture("token_relevance.v1.json"))));
    std::vector<AttributionDocument> runs;
    for (int i = 0; i < 12; ++i) {
        AttributionDocument d = attr;
        for (size_t j = 0; j < d.values.size(); ++j) {
            d.values[j] = static_cast<float>((static_cast<int>(j) * 7 + i * 5) % 11 - 5) / 4.0f;
        }
        runs.push_back(d);
    }
    ExpectGolden("beeswarm.svg", RenderBeeswarmSvg(runs, {0, 3, 5}));

    IceResult ice;
    for (int k = 0; k < 9; ++k) {
        ice.grid.push_back(static_cast<float>(k) / 4.0f);
    }
    ice.num_instances = 12;
    for (int i = 0; i < 12; ++i) {
        // Different offsets (which the centered view removes) and different slopes (which it keeps).
        const float slope = static_cast<float>((i * 5) % 12 - 6) / 4.0f;
        const float offset = static_cast<float>(i % 4) * 0.5f;
        for (float x : ice.grid) {
            ice.curves.push_back(offset + 2.0f * x - 1.2f * x * x + slope * x);  // no libm: same bits everywhere
        }
        ice.feature_values.push_back(static_cast<float>((i * 7) % 9) / 4.0f);
    }
    PartialDependenceDocument pdd = ToPartialDependenceDocument(ice, "dose (mg)", "response");
    ExpectGolden("partial_dependence.svg", RenderPartialDependenceSvg(pdd));
    PartialDependenceSvgOptions centered;
    centered.style = IceStyle::Centered;
    ExpectGolden("partial_dependence_centered.svg", RenderPartialDependenceSvg(pdd, centered));
    AleResult ale;
    ale.edges = {0.0f, 0.5f, 1.25f, 2.0f, 3.5f};
    ale.effects = {-1.25f, -0.5f, 0.25f, 0.75f, 0.5f};
    ale.counts = {4, 4, 3, 1};
    for (int i = 0; i < 12; ++i) ale.feature_values.push_back(static_cast<float>((i * 5) % 14) / 4.0f);
    ExpectGolden("ale.svg", RenderPartialDependenceSvg(ToPartialDependenceDocument(ale, "dose (mg)", "response")));
    ExpectGolden("tornado.svg", RenderTornadoSvg(ParseSensitivityDocument(Fixture("sensitivity.v1.json"))));
    ExpectGolden("counterfactual_set.svg", RenderCounterfactualSetSvg(CounterfactualSet()));
    ExpectGolden("morris.svg", RenderMorrisSvg(ParseMorrisDocument(Fixture("morris.v1.json"))));
    ExpectGolden("sobol.svg", RenderSobolSvg(ParseSobolDocument(Fixture("sobol.v1.json"))));
    ExpectGolden("counterfactual.svg",
                 RenderCounterfactualSvg(ParseCounterfactualDocument(Fixture("counterfactual.v1.json"))));
}

}  // namespace
}  // namespace pulsatrix
