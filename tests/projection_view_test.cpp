// INT-5: the projection document (pulsatrix.projection.v1) and its SVG and HTML views.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/pca.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/html.hpp"
#include "pulsatrix/viz/svg.hpp"

namespace pulsatrix {
namespace {

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string Fixture(const std::string& name) { return ReadFile(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name); }

// Regenerate with PULSATRIX_UPDATE_GOLDEN=1 after a deliberate change, and look at the result.
void ExpectGolden(const std::string& name, const std::string& svg) {
    const std::string path = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/svg/" + name;
    const char* update = std::getenv("PULSATRIX_UPDATE_GOLDEN");
    if (update != nullptr && std::string(update) == "1") {
        std::ofstream(path, std::ios::binary) << svg;
        return;
    }
    const std::string golden = ReadFile(path);
    ASSERT_FALSE(golden.empty()) << "missing golden file " << path;
    EXPECT_EQ(svg, golden) << name << " changed";
}

size_t Count(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1)) ++n;
    return n;
}

ProjectionDocument TwoAxes() {
    ProjectionDocument doc;
    doc.axes = {"PC1", "PC2"};
    doc.coordinates = {0, 1, 2, 3, -1, -2};
    return doc;
}

TEST(ProjectionDocument, ComesFromAPcaAndSurvivesARoundTrip) {
    CPUBackend backend;
    std::vector<float> v;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 4; ++j) v.push_back(static_cast<float>(std::sin(1.3 * i + 0.7 * j + 0.2 * i * j)));
    }
    const Tensor x(Shape({6, 4}), &backend, v);
    const Pca pca = FitPca(x, 3);
    const Tensor scores = PcaProject(pca, x);
    const float nan = std::nanf("");
    const ProjectionDocument doc = MakeProjectionDocument(pca, scores, {"a", "b", "c", "d", "e", "f"}, {"x", "x", "y", "y", "z", "z"},
                                                          {1, 2, nan, 4, 5, 6}, "six rows");
    EXPECT_EQ(doc.method, "pca");
    EXPECT_EQ(doc.axes, (std::vector<std::string>{"PC1", "PC2", "PC3"}));
    EXPECT_EQ(doc.num_points(), 6);
    EXPECT_FLOAT_EQ(doc.explained_variance_ratio[0], static_cast<float>(pca.explained_variance_ratio[0]));
    EXPECT_EQ(doc.coordinates, scores.to_host_vector());

    const std::string json = ToJson(doc);
    EXPECT_EQ(VizDocumentKind(json), "projection");
    EXPECT_NE(json.find("\"/values/2\""), std::string::npos);  // NaN goes through "nonfinite"
    const ProjectionDocument back = ParseProjectionDocument(json);
    EXPECT_EQ(back.title, "six rows");
    EXPECT_EQ(back.axes, doc.axes);
    EXPECT_EQ(back.coordinates, doc.coordinates);
    EXPECT_EQ(back.labels, doc.labels);
    EXPECT_EQ(back.groups, doc.groups);
    EXPECT_TRUE(std::isnan(back.values[2]));
    EXPECT_EQ(back.values[5], 6.0f);
    EXPECT_EQ(ToJson(back), json);

    // The optional lists are left out when empty.
    const std::string bare = ToJson(TwoAxes());
    EXPECT_EQ(bare.find("\"labels\""), std::string::npos);
    EXPECT_TRUE(ParseProjectionDocument(bare).groups.empty());

    EXPECT_THROW((void)MakeProjectionDocument(pca, scores, {"too", "few"}), std::invalid_argument);
    EXPECT_THROW((void)MakeProjectionDocument(FitPca(x, 1), PcaProject(FitPca(x, 1), x)), std::invalid_argument);
}

TEST(ProjectionDocument, RejectsWhatCannotBeDrawn) {
    ProjectionDocument one_axis = TwoAxes();
    one_axis.axes = {"PC1"};
    EXPECT_THROW((void)ToJson(one_axis), std::invalid_argument);
    ProjectionDocument ragged = TwoAxes();
    ragged.coordinates.pop_back();
    EXPECT_THROW((void)ToJson(ragged), std::invalid_argument);
    ProjectionDocument nan_point = TwoAxes();
    nan_point.coordinates[3] = std::nanf("");
    EXPECT_THROW((void)ToJson(nan_point), std::invalid_argument);
    ProjectionDocument labels = TwoAxes();
    labels.labels = {"a"};
    EXPECT_THROW((void)ToJson(labels), std::invalid_argument);
    ProjectionDocument ratios = TwoAxes();
    ratios.explained_variance_ratio = {0.5f};
    EXPECT_THROW((void)ToJson(ratios), std::invalid_argument);

    try {
        (void)ParseProjectionDocument(R"({"schema": "pulsatrix.projection.v1", "title": "", "method": "", "axes": ["a", "b"],
                                         "explained_variance_ratio": [], "coordinates": [1, 2, 3], "groups": ["g"]})");
        FAIL() << "a ragged document was read";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("/coordinates"), std::string::npos) << e.what();
    }
}

TEST(ProjectionSvg, DrawsEveryPointWithItsGroupAndTheAxesShares) {
    const ProjectionDocument doc = ParseProjectionDocument(Fixture("projection.v1.json"));
    const std::string svg = RenderProjectionSvg(doc);
    EXPECT_EQ(Count(svg, "class=\"point\""), 8u);
    EXPECT_NE(svg.find("PC1 (42.0% of variance)"), std::string::npos);
    EXPECT_NE(svg.find("PC2 (21.0% of variance)"), std::string::npos);
    EXPECT_EQ(Count(svg, "class=\"legend-label\""), 3u);  // noun, verb, adjective
    EXPECT_NE(svg.find("cat [noun]"), std::string::npos);
    ExpectGolden("projection.svg", svg);

    // Another pair of axes.
    EXPECT_NE(RenderProjectionSvg(doc, 0, 2).find("PC3 (8.0% of variance)"), std::string::npos);
    // Without groups, values color the points, with a color bar, and a NaN is gray.
    ProjectionDocument by_value = doc;
    by_value.groups.clear();
    const std::string valued = RenderProjectionSvg(by_value);
    EXPECT_NE(valued.find("class=\"colorbar\""), std::string::npos);
    EXPECT_NE(valued.find("#bdbdbd"), std::string::npos);

    EXPECT_THROW((void)RenderProjectionSvg(doc, 0, 0), std::invalid_argument);
    EXPECT_THROW((void)RenderProjectionSvg(doc, 0, 3), std::invalid_argument);
    ProjectionDocument empty = TwoAxes();
    empty.coordinates.clear();
    EXPECT_THROW((void)RenderProjectionSvg(empty), std::invalid_argument);
}

TEST(ProjectionHtml, ShowsThreePairsForThreeAxesAndOneZoomablePlotForTwo) {
    const ProjectionDocument doc = ParseProjectionDocument(Fixture("projection.v1.json"));
    const JsonValue three = ToVegaLiteProjection(doc);
    const JsonValue* panels = three.find("hconcat");
    ASSERT_NE(panels, nullptr);
    EXPECT_EQ(panels->as_array().size(), 3u);
    const std::string spec = WriteJson(three);
    EXPECT_NE(spec.find("\"field\": \"group\""), std::string::npos);
    EXPECT_NE(spec.find("PC3 (8.0% of variance)"), std::string::npos);
    EXPECT_NE(spec.find("\"bind\": \"scales\""), std::string::npos);

    ProjectionDocument flat = TwoAxes();
    flat.values = {1, 2, 3};
    const JsonValue two = ToVegaLiteProjection(flat);
    EXPECT_EQ(two.find("hconcat"), nullptr);
    EXPECT_NE(WriteJson(two).find("\"scheme\": \"viridis\""), std::string::npos);
    EXPECT_NE(RenderProjectionHtml(doc).find("Example projection"), std::string::npos);
}

}  // namespace
}  // namespace pulsatrix
