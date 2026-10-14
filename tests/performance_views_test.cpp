// KS-3: the reliability diagram and the per-slice error figure, and the readers of the
// classification_report and slice_error documents they draw.
#include <gtest/gtest.h>

#include <algorithm>
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
#include "pulsatrix/json.hpp"
#include "pulsatrix/performance.hpp"
#include "pulsatrix/viz/document.hpp"
#include "pulsatrix/viz/mime_bundle.hpp"
#include "pulsatrix/viz/performance_views.hpp"

namespace pulsatrix {
namespace {

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

// Three classes and 60 inputs. Input i's label is i % 3; the model puts p on its guess, which is
// the label except for every fourth input, and splits the rest evenly. With sharpen, p is pushed
// toward 1: the same guesses, made more confidently.
ClassificationReport Example(bool sharpen) {
    CPUBackend cpu;
    const int64_t n = 60;
    std::vector<float> p(static_cast<size_t>(n * 3));
    std::vector<int64_t> labels;
    for (int64_t i = 0; i < n; ++i) {
        const int64_t label = i % 3;
        const int64_t guess = i % 4 == 0 ? (label + 1) % 3 : label;
        double top = 0.45 + 0.5 * static_cast<double>(i % 10) / 9.0;
        if (sharpen) top = 1.0 - (1.0 - top) * 0.3;
        for (int64_t k = 0; k < 3; ++k) {
            p[static_cast<size_t>(i * 3 + k)] = static_cast<float>(k == guess ? top : (1.0 - top) / 2);
        }
        labels.push_back(label);
    }
    ClassificationOptions options;
    options.bins = 10;
    return EvaluateClassifier(Tensor(Shape({n, 3}), &cpu, p), labels, options);
}

// Five slices of 0/1 errors; "night" is much worse than the rest.
SliceErrorReport SliceExample() {
    std::vector<double> errors;
    std::vector<std::string> slices;
    const char* names[] = {"day", "dusk", "night", "rain", "snow"};
    const int wrong[] = {3, 5, 18, 4, 6};
    for (int s = 0; s < 5; ++s) {
        for (int i = 0; i < 40; ++i) {
            errors.push_back(i < wrong[s] ? 1.0 : 0.0);
            slices.push_back(names[s]);
        }
    }
    return PerSliceError(errors, slices);
}

std::string Json(const ClassificationReport& r) { return WriteJson(ToJson(r)); }
std::string Json(const SliceErrorReport& r) { return WriteJson(ToJson(r)); }

// The fixtures are these reports. Their doubles come from log and lgamma, whose last bits differ
// between math libraries, so the numbers are compared with a tolerance rather than as bytes.
TEST(PerformanceDocuments, MatchTheFixtures) {
    const char* update = std::getenv("PULSATRIX_UPDATE_GOLDEN");
    const ClassificationReport r = Example(false);
    const SliceErrorReport s = SliceExample();
    if (update != nullptr && std::string(update) == "1") {
        ExpectGolden(FixturePath("classification_report.v1.json"), Json(r));
        ExpectGolden(FixturePath("slice_error.v1.json"), Json(s));
        return;
    }
    auto near = [](double a, double b) { return std::abs(a - b) <= 1e-12 * std::max(1.0, std::abs(b)); };
    const ClassificationReport rf = ParseClassificationReport(ReadFile(FixturePath("classification_report.v1.json")));
    EXPECT_EQ(rf.n, r.n);
    EXPECT_EQ(rf.confusion, r.confusion);
    EXPECT_TRUE(near(rf.log_loss, r.log_loss)) << rf.log_loss << " vs " << r.log_loss;
    EXPECT_TRUE(near(rf.ece, r.ece));
    ASSERT_EQ(rf.bins.size(), r.bins.size());
    for (size_t b = 0; b < r.bins.size(); ++b) EXPECT_TRUE(near(rf.bins[b].confidence, r.bins[b].confidence)) << b;
    const SliceErrorReport sf = ParseSliceErrorReport(ReadFile(FixturePath("slice_error.v1.json")));
    ASSERT_EQ(sf.slices.size(), s.slices.size());
    for (size_t i = 0; i < s.slices.size(); ++i) {
        EXPECT_EQ(sf.slices[i].name, s.slices[i].name);
        EXPECT_TRUE(near(sf.slices[i].lower, s.slices[i].lower)) << i;
        EXPECT_TRUE(std::abs(sf.slices[i].p_value - s.slices[i].p_value) <= 1e-12) << i;
        EXPECT_EQ(sf.slices[i].worse, s.slices[i].worse);
    }
}

TEST(PerformanceDocuments, ClassificationReportRoundTrips) {
    const ClassificationReport r = Example(false);
    const std::string json = Json(r);
    EXPECT_EQ(VizDocumentKind(json), "classification_report");
    const ClassificationReport back = ParseClassificationReport(json);
    EXPECT_EQ(back.n, r.n);
    EXPECT_EQ(back.classes, 3);
    EXPECT_DOUBLE_EQ(back.accuracy, r.accuracy);
    EXPECT_DOUBLE_EQ(back.ece, r.ece);
    ASSERT_EQ(back.bins.size(), r.bins.size());
    for (size_t b = 0; b < r.bins.size(); ++b) {
        EXPECT_EQ(back.bins[b].count, r.bins[b].count);
        EXPECT_DOUBLE_EQ(back.bins[b].accuracy, r.bins[b].accuracy);
    }
    EXPECT_EQ(back.confusion, r.confusion);
    EXPECT_EQ(back.support, r.support);
    EXPECT_EQ(Json(back), json);
}

TEST(PerformanceDocuments, SliceErrorRoundTrips) {
    const SliceErrorReport r = SliceExample();
    const std::string json = Json(r);
    EXPECT_EQ(VizDocumentKind(json), "slice_error");
    const SliceErrorReport back = ParseSliceErrorReport(json);
    ASSERT_EQ(back.slices.size(), 5u);
    EXPECT_EQ(back.slices[0].name, "night");
    EXPECT_TRUE(back.slices[0].worse);
    EXPECT_EQ(back.test, "fisher_exact");
    EXPECT_EQ(Json(back), json);
}

// Replaces the first occurrence of from in the document, which must hold it.
std::string Edit(std::string json, const std::string& from, const std::string& to) {
    const size_t at = json.find(from);
    EXPECT_NE(at, std::string::npos) << from;
    return at == std::string::npos ? json : json.replace(at, from.size(), to);
}

TEST(PerformanceDocuments, RejectReportsThatDontHoldTogether) {
    const std::string report = Json(Example(false));
    EXPECT_THROW((void)ParseClassificationReport(Json(SliceExample())), std::invalid_argument);
    EXPECT_THROW((void)ParseSliceErrorReport(report), std::invalid_argument);
    EXPECT_THROW((void)ParseClassificationReport(Edit(report, "\"n\": 60", "\"n\": 61")), std::invalid_argument);
    EXPECT_THROW((void)ParseClassificationReport(Edit(report, "\"classes\": 3", "\"classes\": 2")), std::invalid_argument);
    EXPECT_THROW((void)ParseClassificationReport(Edit(report, "\"accuracy\": 0.75", "\"accuracy\": 1.5")), std::invalid_argument);
    EXPECT_THROW((void)ParseClassificationReport(Edit(report, "\"reliability\": [", "\"reliability\": [], \"x\": [")),
                 std::invalid_argument);

    const std::string slices = Json(SliceExample());
    EXPECT_THROW((void)ParseSliceErrorReport(Edit(slices, "\"alpha\": 0.05", "\"alpha\": 0")), std::invalid_argument);
    EXPECT_THROW((void)ParseSliceErrorReport(Edit(slices, "\"n\": 200", "\"n\": 201")), std::invalid_argument);
    EXPECT_THROW((void)ParseSliceErrorReport(Edit(slices, "\"error\": 0.45", "\"error\": 0.95")), std::invalid_argument);
    EXPECT_THROW((void)ParseSliceErrorReport(Edit(slices, "\"slices\": [", "\"slices\": [], \"x\": [")), std::invalid_argument);
}

TEST(ReliabilityDiagram, DrawsOnePointPerNonEmptyBin) {
    const ClassificationReport r = Example(false);
    int64_t non_empty = 0;
    for (const ReliabilityBin& b : r.bins) non_empty += b.count > 0 ? 1 : 0;
    const std::string svg = RenderReliabilityDiagramSvg(r);
    EXPECT_EQ(Count(svg, "<circle class=\"bin s0\""), static_cast<size_t>(non_empty));
    EXPECT_EQ(Count(svg, "<rect class=\"share s0\""), static_cast<size_t>(non_empty));
    EXPECT_NE(svg.find("ECE " + std::to_string(r.ece).substr(0, 4)), std::string::npos);
    EXPECT_NE(svg.find("class=\"diagonal\""), std::string::npos);
    EXPECT_NE(svg.find("inputs, mean confidence"), std::string::npos);
    EXPECT_NE(mime_bundle_repr(r).find("image/svg+xml"), nullptr);
}

TEST(ReliabilityDiagram, OverlaysSeveralClassifiers) {
    const std::string svg = RenderReliabilityDiagramSvg({{"soft", Example(false)}, {"sharp", Example(true)}});
    EXPECT_NE(svg.find("<polyline class=\"curve s0\""), std::string::npos);
    EXPECT_NE(svg.find("<polyline class=\"curve s1\""), std::string::npos);
    EXPECT_NE(svg.find("soft: ECE"), std::string::npos);
    EXPECT_NE(svg.find("sharp: ECE"), std::string::npos);
}

TEST(ReliabilityDiagram, RejectsUnusableInput) {
    EXPECT_THROW((void)RenderReliabilityDiagramSvg(std::vector<ReliabilitySeries>{}), std::invalid_argument);
    std::vector<ReliabilitySeries> eleven(11, ReliabilitySeries{"x", Example(false)});
    EXPECT_THROW((void)RenderReliabilityDiagramSvg(eleven), std::invalid_argument);
    ClassificationReport no_bins = Example(false);
    no_bins.bins.clear();
    EXPECT_THROW((void)RenderReliabilityDiagramSvg(no_bins), std::invalid_argument);
    SvgOptions narrow;
    narrow.width = 100;
    EXPECT_THROW((void)RenderReliabilityDiagramSvg(Example(false), narrow), std::invalid_argument);
}

TEST(SliceErrorView, DrawsEverySliceWorstFirst) {
    const SliceErrorReport r = SliceExample();
    const std::string svg = RenderSliceErrorSvg(r);
    EXPECT_EQ(Count(svg, "<circle class=\"slice"), 5u);
    EXPECT_EQ(Count(svg, "<circle class=\"slice worse\""), 1u);
    EXPECT_LT(svg.find("night (40)"), svg.find("snow (40)"));
    EXPECT_NE(svg.find("all: 0.18"), std::string::npos);
    EXPECT_EQ(svg.find("class=\"reference\""), std::string::npos);
    EXPECT_NE(mime_bundle_repr(r).find("image/svg+xml"), nullptr);
}

TEST(SliceErrorView, DrawsAReferenceAndCutsLongLists) {
    SliceErrorViewOptions view;
    view.metric = "miss rate";
    view.reference = 0.1;
    view.reference_label = "alpha";
    view.max_slices = 2;
    const std::string svg = RenderSliceErrorSvg(SliceExample(), view);
    EXPECT_NE(svg.find("alpha: 0.1"), std::string::npos);
    EXPECT_NE(svg.find(">miss rate<"), std::string::npos);
    EXPECT_EQ(Count(svg, "<circle class=\"slice"), 2u);
    EXPECT_NE(svg.find("+3 more slices not shown"), std::string::npos);
}

TEST(SliceErrorView, RejectsUnusableInput) {
    SliceErrorViewOptions view;
    view.max_slices = 0;
    EXPECT_THROW((void)RenderSliceErrorSvg(SliceExample(), view), std::invalid_argument);
    view.max_slices = 5;
    view.reference = std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)RenderSliceErrorSvg(SliceExample(), view), std::invalid_argument);
    view.reference = -0.1;
    EXPECT_THROW((void)RenderSliceErrorSvg(SliceExample(), view), std::invalid_argument);
    EXPECT_THROW((void)RenderSliceErrorSvg(SliceErrorReport{}), std::invalid_argument);
}

TEST(PerformanceViews, MatchTheGoldenFiles) {
    ExpectGolden(FixturePath("svg/reliability_diagram.svg"),
                 RenderReliabilityDiagramSvg({{"soft", Example(false)}, {"sharp", Example(true)}}));
    SliceErrorViewOptions view;
    view.metric = "error rate";
    view.reference = 0.1;
    ExpectGolden(FixturePath("svg/slice_error.svg"), RenderSliceErrorSvg(SliceExample(), view));
}

}  // namespace
}  // namespace pulsatrix
