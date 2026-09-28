#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/implot_metrics_sink.hpp"

// Tests only the buffering logic (log_scalar/log_histogram populating internal state), never
// Draw() -- see plans/okay-we-have-now-buzzing-moth.md Testing Strategy. This header has zero
// ImGui/ImPlot include (only Draw()'s separate .cpp does), so these tests run unconditionally
// regardless of PULSATRIX_ENABLE_VIZ, same as colormap_test.cpp/plot_data_test.cpp.
namespace pulsatrix {
namespace {

class ImPlotMetricsSinkTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(ImPlotMetricsSinkTest, LogScalarAppendsStepValuePairToNamedSeries) {
    ImPlotMetricsSink sink;
    sink.log_scalar("loss", 0.5, 0);

    ASSERT_TRUE(sink.scalar_series().count("loss"));
    const ScalarSeries& series = sink.scalar_series().at("loss");
    ASSERT_EQ(series.steps.size(), 1u);
    EXPECT_EQ(series.steps[0], 0);
    EXPECT_DOUBLE_EQ(series.values[0], 0.5);
}

TEST_F(ImPlotMetricsSinkTest, LogScalarAppendsMultipleCallsInOrder) {
    ImPlotMetricsSink sink;
    sink.log_scalar("loss", 1.0, 0);
    sink.log_scalar("loss", 0.8, 1);
    sink.log_scalar("loss", 0.6, 2);

    const ScalarSeries& series = sink.scalar_series().at("loss");
    ASSERT_EQ(series.steps.size(), 3u);
    EXPECT_EQ(series.steps[2], 2);
    EXPECT_DOUBLE_EQ(series.values[2], 0.6);
}

TEST_F(ImPlotMetricsSinkTest, LogScalarKeepsSeparateSeriesPerTag) {
    ImPlotMetricsSink sink;
    sink.log_scalar("loss", 1.0, 0);
    sink.log_scalar("accuracy", 0.9, 0);

    EXPECT_EQ(sink.scalar_series().size(), 2u);
    EXPECT_DOUBLE_EQ(sink.scalar_series().at("loss").values[0], 1.0);
    EXPECT_DOUBLE_EQ(sink.scalar_series().at("accuracy").values[0], 0.9);
}

TEST_F(ImPlotMetricsSinkTest, LogHistogramStoresLatestValuesForTag) {
    ImPlotMetricsSink sink;
    Tensor first(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor second(Shape({2}), &backend, {9.0f, 9.0f});

    sink.log_histogram("gradients", first, 0);
    sink.log_histogram("gradients", second, 1);

    ASSERT_TRUE(sink.latest_histograms().count("gradients"));
    const std::vector<float>& latest = sink.latest_histograms().at("gradients");
    ASSERT_EQ(latest.size(), 2u);
    EXPECT_FLOAT_EQ(latest[0], 9.0f);
    EXPECT_FLOAT_EQ(latest[1], 9.0f);
}

TEST_F(ImPlotMetricsSinkTest, LogHistogramKeepsSeparateBuffersPerTag) {
    ImPlotMetricsSink sink;
    Tensor a(Shape({1}), &backend, {1.0f});
    Tensor b(Shape({1}), &backend, {2.0f});

    sink.log_histogram("weights", a, 0);
    sink.log_histogram("gradients", b, 0);

    EXPECT_EQ(sink.latest_histograms().size(), 2u);
    EXPECT_FLOAT_EQ(sink.latest_histograms().at("weights")[0], 1.0f);
    EXPECT_FLOAT_EQ(sink.latest_histograms().at("gradients")[0], 2.0f);
}

}  // namespace
}  // namespace pulsatrix
