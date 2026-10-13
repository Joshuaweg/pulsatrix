// TDA-9: the persistence diagram document (pulsatrix.persistence_diagram.v1) and its persistence
// diagram, barcode and Betti curve views.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/tda/distance_matrix.hpp"
#include "pulsatrix/tda/rips_persistence.hpp"
#include "pulsatrix/tda/significance.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/mime_bundle.hpp"
#include "pulsatrix/viz/topology_documents.hpp"
#include "pulsatrix/viz/topology_views.hpp"

namespace pulsatrix {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string FixturePath(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name; }

// Regenerate with PULSATRIX_UPDATE_GOLDEN=1 after a deliberate change, and look at the result.
void ExpectGolden(const std::string& path, const std::string& text) {
    const char* update = std::getenv("PULSATRIX_UPDATE_GOLDEN");
    if (update != nullptr && std::string(update) == "1") {
        std::ofstream(path, std::ios::binary) << text;
        return;
    }
    const std::string golden = ReadFile(path);
    ASSERT_FALSE(golden.empty()) << "missing golden file " << path;
    EXPECT_EQ(text, golden) << path << " changed";
}

size_t Count(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + 1)) ++n;
    return n;
}

// H0 with one essential component, H1 with a long loop, a short one and a tested pair.
PersistenceDiagramDocument Example() {
    PersistenceDiagramDocument doc;
    doc.title = "Two loops";
    doc.source = "a noisy circle";
    doc.num_points = 6;
    doc.threshold = 2.5;
    PersistenceDiagramEntry h0;
    h0.dimension = 0;
    h0.births = {0, 0, 0, 0, 0, 0};
    h0.deaths = {0.1, 0.15, 0.2, 0.25, 0.4, kInf};
    h0.error_bound = 0.02;
    PersistenceDiagramEntry h1;
    h1.dimension = 1;
    h1.births = {0.3, 0.5, 0.45};
    h1.deaths = {1.8, 0.55, 0.6};
    h1.error_bound = 0.02;
    h1.p_values = {0.0004, 0.6, std::numeric_limits<double>::quiet_NaN()};
    h1.significance_threshold = 0.025;
    doc.diagrams = {h0, h1};
    return doc;
}

TEST(PersistenceDiagramDocument, TheFixtureRoundTripsExactly) {
    ExpectGolden(FixturePath("persistence_diagram.v1.json"), ToJson(Example()));
    const std::string json = ReadFile(FixturePath("persistence_diagram.v1.json"));
    EXPECT_EQ(VizDocumentKind(json), "persistence_diagram");
    const PersistenceDiagramDocument doc = ParsePersistenceDiagramDocument(json);
    EXPECT_EQ(ToJson(doc), json);
    ASSERT_EQ(doc.diagrams.size(), 2u);
    EXPECT_EQ(doc.diagrams[0].deaths.back(), kInf);  // through "nonfinite"
    EXPECT_TRUE(std::isnan(doc.diagrams[1].p_values[2]));
    EXPECT_EQ(doc.find(1)->significance_threshold, 0.025);
    EXPECT_EQ(doc.find(2), nullptr);
}

TEST(PersistenceDiagramDocument, ACircleHasOneLongLoopAndTheTestFindsIt) {
    CPUBackend backend;
    // 150 points scattered around a circle of radius 1, from a portable generator so every
    // platform gets the same points; the noise gives the universal test its short features.
    uint64_t state = 12345;
    auto uniform = [&state] {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) / 9007199254740992.0;
    };
    std::vector<float> xy;
    for (int i = 0; i < 150; ++i) {
        const double t = 2 * kPi * uniform();
        const double r = 0.8 + 0.4 * uniform();
        xy.push_back(static_cast<float>(r * std::cos(t)));
        xy.push_back(static_cast<float>(r * std::sin(t)));
    }
    const Tensor points(Shape({150, 2}), &backend, xy);
    const RipsResult rips = ComputeRipsPersistence(PairwiseDistances(points));
    PersistenceDiagramDocument doc = ToPersistenceDiagramDocument(rips, "circle");
    ASSERT_NE(doc.find(1), nullptr);
    const PersistenceDiagramEntry& h1 = *doc.find(1);
    size_t longest = 0;
    for (size_t i = 0; i < h1.size(); ++i) {
        if (h1.deaths[i] - h1.births[i] > h1.deaths[longest] - h1.births[longest]) longest = i;
    }
    EXPECT_GT(h1.deaths[longest] - h1.births[longest], 1.0);  // the circle's loop
    EXPECT_EQ(doc.threshold, rips.threshold);
    EXPECT_EQ(doc.find(0)->births.size(), rips.diagrams[0].pairs.size());

    const UniversalNullResult test = UniversalNullTest(rips.diagrams[1]);
    AttachSignificance(doc, 1, test);
    ASSERT_EQ(doc.find(1)->p_values.size(), h1.size());
    for (const FeatureSignificance& f : test.features) {
        EXPECT_EQ(doc.find(1)->p_values[static_cast<size_t>(f.index)], f.p_value);
        EXPECT_EQ(IsSignificant(*doc.find(1), static_cast<size_t>(f.index)), f.significant);
    }
    EXPECT_TRUE(IsSignificant(*doc.find(1), longest));
    EXPECT_TRUE(IsSignificant(*doc.find(0), 0));  // nothing attached to H0: every feature counts

    BootstrapBandResult band;
    band.band = 0.3;
    AttachBootstrapBand(doc, 0, band);
    EXPECT_EQ(doc.find(0)->bootstrap_band, 0.3);
    EXPECT_THROW(AttachSignificance(doc, 2, test), std::invalid_argument);
    EXPECT_THROW(AttachSignificance(doc, 0, test), std::invalid_argument);  // indices past H0's pairs
    EXPECT_EQ(ToJson(ParsePersistenceDiagramDocument(ToJson(doc))), ToJson(doc));
}

TEST(PersistenceDiagramDocument, ProblemsNameTheirPointer) {
    auto expect_problem = [](PersistenceDiagramDocument doc, const std::string& pointer) {
        try {
            (void)ToJson(doc);
            ADD_FAILURE() << "accepted; expected a problem at " << pointer;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(pointer + ":"), std::string::npos) << e.what();
        }
    };
    PersistenceDiagramDocument d = Example();
    d.diagrams[1].deaths[0] = 0.1;
    expect_problem(d, "/diagrams/1/deaths/0");  // dies before it's born
    d = Example();
    d.diagrams[1].deaths[0] = std::numeric_limits<double>::quiet_NaN();
    expect_problem(d, "/diagrams/1/deaths/0");
    d = Example();
    d.diagrams[0].births[2] = kInf;
    expect_problem(d, "/diagrams/0/births/2");
    d = Example();
    d.diagrams[1].dimension = 0;
    expect_problem(d, "/diagrams/1/dimension");  // H0 twice
    d = Example();
    d.diagrams[1].p_values.pop_back();
    expect_problem(d, "/diagrams/1/p_values");
    d = Example();
    d.diagrams[1].p_values[0] = 1.5;
    expect_problem(d, "/diagrams/1/p_values/0");
    d = Example();
    d.diagrams[0].error_bound = -1;
    expect_problem(d, "/diagrams/0/error_bound");
    d = Example();
    d.diagrams.clear();
    expect_problem(d, "/diagrams");

    // The reader names the same pointer, and a stray "nonfinite" entry is refused.
    std::string json = ToJson(Example());
    const size_t at = json.find("\"/diagrams/0/deaths/5\"");
    ASSERT_NE(at, std::string::npos);
    json.replace(at, std::string("\"/diagrams/0/deaths/5\"").size(), "\"/diagrams/0/deaths/4\"");
    EXPECT_THROW((void)ParsePersistenceDiagramDocument(json), std::invalid_argument);
}

TEST(PersistenceViews, EachViewDrawsEveryFeature) {
    const PersistenceDiagramDocument doc = Example();
    const std::string diagram = RenderPersistenceDiagramSvg(doc);
    EXPECT_EQ(Count(diagram, "<circle class=\"feature"), 9u);
    EXPECT_EQ(Count(diagram, "class=\"error-band\""), 1u);
    EXPECT_EQ(Count(diagram, "class=\"infinity\""), 1u);
    // Filled when significant: H0 has no test, so all 6; H1 only its first loop.
    EXPECT_EQ(Count(diagram, "fill=\"#ffffff\" fill-opacity"), 2u);
    EXPECT_NE(diagram.find("p = 0.0004"), std::string::npos);
    EXPECT_NE(diagram.find("not tested"), std::string::npos);

    const std::string barcode = RenderBarcodeSvg(doc);
    EXPECT_EQ(Count(barcode, "<rect class=\"bar"), 9u);
    EXPECT_EQ(Count(barcode, "class=\"essential\""), 1u);
    EXPECT_EQ(Count(barcode, "class=\"whisker\""), 2u * 9u - 1u);  // no death whisker on the essential bar

    PersistenceViewOptions only_h1;
    only_h1.dimensions = {1};
    only_h1.max_bars = 2;
    const std::string h1 = RenderBarcodeSvg(doc, only_h1);
    EXPECT_EQ(Count(h1, "<rect class=\"bar"), 2u);
    EXPECT_NE(h1.find("+1 shorter bars not shown"), std::string::npos);

    const std::string betti = RenderBettiCurveSvg(doc);
    EXPECT_EQ(Count(betti, "<polyline class=\"betti"), 2u);
    EXPECT_NE(betti.find("Not covered by the error bound"), std::string::npos);

    only_h1.dimensions = {2};
    EXPECT_THROW((void)RenderPersistenceDiagramSvg(doc, only_h1), std::invalid_argument);
    PersistenceViewOptions bad;
    bad.max_bars = 0;
    EXPECT_THROW((void)RenderBarcodeSvg(doc, bad), std::invalid_argument);
    bad = {};
    bad.grid_points = 1;
    EXPECT_THROW((void)RenderBettiCurveSvg(doc, bad), std::invalid_argument);
    EXPECT_NE(mime_bundle_repr(doc).find("image/svg+xml"), nullptr);
}

TEST(PersistenceViews, AnEmptyDiagramSaysSo) {
    PersistenceDiagramDocument doc;
    doc.diagrams.resize(1);
    doc.diagrams[0].dimension = 1;
    EXPECT_NE(RenderPersistenceDiagramSvg(doc).find("no features"), std::string::npos);
    EXPECT_NE(RenderBarcodeSvg(doc).find("no features"), std::string::npos);
    EXPECT_EQ(Count(RenderBettiCurveSvg(doc), "<polyline"), 1u);
}

TEST(PersistenceViews, MatchTheGoldenFiles) {
    const PersistenceDiagramDocument doc = ParsePersistenceDiagramDocument(ReadFile(FixturePath("persistence_diagram.v1.json")));
    ExpectGolden(FixturePath("svg/persistence_diagram.svg"), RenderPersistenceDiagramSvg(doc));
    ExpectGolden(FixturePath("svg/barcode.svg"), RenderBarcodeSvg(doc));
    ExpectGolden(FixturePath("svg/betti_curve.svg"), RenderBettiCurveSvg(doc));
}

}  // namespace
}  // namespace pulsatrix
