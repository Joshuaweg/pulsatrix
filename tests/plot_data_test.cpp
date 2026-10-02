#include <gtest/gtest.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dataset.hpp"
#include "pulsatrix/viz/plot_data.hpp"

// Pure Attribution/Dataset -> plot-ready-array transforms for the native viz module (no
// ImGui/ImPlot include -- see plans/okay-we-have-now-buzzing-moth.md Phase 0). Every chart
// widget calls into these for its data prep; the widget itself only calls ImGui/ImPlot draw
// functions, keeping this layer -- and not the actual rendering calls -- the unit-tested one.
namespace pulsatrix {
namespace {

class InMemoryDataset : public Dataset {
public:
    explicit InMemoryDataset(std::vector<Sample> samples) : samples_(std::move(samples)) {}
    [[nodiscard]] int64_t size() const override { return static_cast<int64_t>(samples_.size()); }
    [[nodiscard]] Sample get(int64_t index) const override {
        if (index < 0 || index >= size()) {
            throw std::out_of_range("InMemoryDataset::get: index out of range");
        }
        return samples_[static_cast<size_t>(index)];
    }

private:
    std::vector<Sample> samples_;
};

class PlotDataTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(PlotDataTest, ToFeatureImportanceBarsSortsByAbsoluteValueDescending) {
    Attribution attr{"saliency", Tensor(Shape({4}), &backend, {0.1f, -0.9f, 0.4f, -0.2f}), {}};

    BarSeries bars = ToFeatureImportanceBars(attr, 4);

    ASSERT_EQ(bars.values.size(), 4u);
    EXPECT_FLOAT_EQ(bars.values[0], -0.9f);
    EXPECT_FLOAT_EQ(bars.values[1], 0.4f);
    EXPECT_FLOAT_EQ(bars.values[2], -0.2f);
    EXPECT_FLOAT_EQ(bars.values[3], 0.1f);
    ASSERT_EQ(bars.labels.size(), 4u);
}

TEST_F(PlotDataTest, ToFeatureImportanceBarsTruncatesToTopK) {
    Attribution attr{"saliency", Tensor(Shape({4}), &backend, {0.1f, -0.9f, 0.4f, -0.2f}), {}};

    BarSeries bars = ToFeatureImportanceBars(attr, 2);

    ASSERT_EQ(bars.values.size(), 2u);
    EXPECT_FLOAT_EQ(bars.values[0], -0.9f);
    EXPECT_FLOAT_EQ(bars.values[1], 0.4f);
}

TEST_F(PlotDataTest, ToFeatureImportanceBarsUsesFeatureNamesMetadataWhenPresent) {
    Attribution attr{"saliency",
                      Tensor(Shape({2}), &backend, {0.5f, -0.1f}),
                      {{"feature_names", "age,income"}}};

    BarSeries bars = ToFeatureImportanceBars(attr, 2);

    ASSERT_EQ(bars.labels.size(), 2u);
    EXPECT_EQ(bars.labels[0], "age");
    EXPECT_EQ(bars.labels[1], "income");
}

TEST_F(PlotDataTest, ToFeatureImportanceBarsAcceptsBatchOneRank2Values) {
    // Every explainer in this codebase is always-batched post
    // campaign_exai_dl_library_batch_dimension_support -- a single-prediction Attribution
    // (e.g. Saliency::explain's output) is shape (1, num_features), not (num_features,).
    Attribution attr{"saliency", Tensor(Shape({1, 4}), &backend, {0.1f, -0.9f, 0.4f, -0.2f}), {}};

    BarSeries bars = ToFeatureImportanceBars(attr, 2);

    ASSERT_EQ(bars.values.size(), 2u);
    EXPECT_FLOAT_EQ(bars.values[0], -0.9f);
    EXPECT_FLOAT_EQ(bars.values[1], 0.4f);
}

TEST_F(PlotDataTest, ToFeatureImportanceBarsThrowsOnNonRank1Values) {
    Attribution attr{"grad_cam", Tensor(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f}), {}};

    EXPECT_THROW(ToFeatureImportanceBars(attr, 2), std::invalid_argument);
}

TEST_F(PlotDataTest, ToFeatureImportanceBarsThrowsOnNonPositiveTopK) {
    Attribution attr{"saliency", Tensor(Shape({2}), &backend, {0.1f, 0.2f}), {}};

    EXPECT_THROW(ToFeatureImportanceBars(attr, 0), std::invalid_argument);
    EXPECT_THROW(ToFeatureImportanceBars(attr, -1), std::invalid_argument);
}

TEST_F(PlotDataTest, ToWaterfallStepsComputesCumulativeFromBaseline) {
    Attribution attr{"integrated_gradients", Tensor(Shape({3}), &backend, {1.0f, -0.5f, 2.0f}), {}};

    std::vector<WaterfallStep> steps = ToWaterfallSteps(attr, 10.0f);

    ASSERT_EQ(steps.size(), 3u);
    EXPECT_FLOAT_EQ(steps[0].delta, 1.0f);
    EXPECT_FLOAT_EQ(steps[0].cumulative, 11.0f);
    EXPECT_FLOAT_EQ(steps[1].delta, -0.5f);
    EXPECT_FLOAT_EQ(steps[1].cumulative, 10.5f);
    EXPECT_FLOAT_EQ(steps[2].delta, 2.0f);
    EXPECT_FLOAT_EQ(steps[2].cumulative, 12.5f);
}

TEST_F(PlotDataTest, ToWaterfallStepsAcceptsBatchOneRank2Values) {
    Attribution attr{"integrated_gradients", Tensor(Shape({1, 3}), &backend, {1.0f, -0.5f, 2.0f}), {}};

    std::vector<WaterfallStep> steps = ToWaterfallSteps(attr, 10.0f);

    ASSERT_EQ(steps.size(), 3u);
    EXPECT_FLOAT_EQ(steps[2].cumulative, 12.5f);
}

TEST_F(PlotDataTest, ToWaterfallStepsPreservesOriginalFeatureOrderNotSortedByMagnitude) {
    Attribution attr{"integrated_gradients", Tensor(Shape({3}), &backend, {0.1f, -0.9f, 0.3f}), {}};

    std::vector<WaterfallStep> steps = ToWaterfallSteps(attr, 0.0f);

    ASSERT_EQ(steps.size(), 3u);
    EXPECT_FLOAT_EQ(steps[0].delta, 0.1f);
    EXPECT_FLOAT_EQ(steps[1].delta, -0.9f);
    EXPECT_FLOAT_EQ(steps[2].delta, 0.3f);
}

TEST_F(PlotDataTest, ToSaliencyHeatmapReshapesRank2Tensor) {
    Attribution attr{"grad_cam", Tensor(Shape({2, 3}), &backend, {1, 2, 3, 4, 5, 6}), {}};

    HeatmapGrid grid = ToSaliencyHeatmap(attr);

    EXPECT_EQ(grid.rows, 2);
    EXPECT_EQ(grid.cols, 3);
    ASSERT_EQ(grid.values.size(), 6u);
    EXPECT_FLOAT_EQ(grid.values[0], 1.0f);
    EXPECT_FLOAT_EQ(grid.values[5], 6.0f);
}

TEST_F(PlotDataTest, ToSaliencyHeatmapSqueezesSingleChannelRank3Tensor) {
    Attribution attr{"grad_cam", Tensor(Shape({1, 2, 2}), &backend, {1, 2, 3, 4}), {}};

    HeatmapGrid grid = ToSaliencyHeatmap(attr);

    EXPECT_EQ(grid.rows, 2);
    EXPECT_EQ(grid.cols, 2);
    ASSERT_EQ(grid.values.size(), 4u);
}

TEST_F(PlotDataTest, ToWaterfallBarsFloatBetweenConsecutiveRunningTotals) {
    Attribution attr{"ig", Tensor(Shape({3}), &backend, {0.5f, -2.0f, 3.0f}), {}};
    std::vector<WaterfallStep> steps = ToWaterfallSteps(attr, 1.0f);

    std::vector<WaterfallBar> bars = ToWaterfallBars(steps, 1.0f);

    ASSERT_EQ(bars.size(), 3u);
    EXPECT_FLOAT_EQ(bars[0].bottom, 1.0f);
    EXPECT_FLOAT_EQ(bars[0].top, 1.5f);
    EXPECT_TRUE(bars[0].increase);
    EXPECT_FLOAT_EQ(bars[1].bottom, -0.5f);
    EXPECT_FLOAT_EQ(bars[1].top, 1.5f);
    EXPECT_FALSE(bars[1].increase);
    EXPECT_FLOAT_EQ(bars[2].bottom, -0.5f);
    EXPECT_FLOAT_EQ(bars[2].top, 2.5f);
    EXPECT_TRUE(bars[2].increase);
}

TEST_F(PlotDataTest, ToWaterfallBarsStayBelowZeroForAnAllNegativeCascade) {
    // Regression: stacked bar segments anchor at zero, so a cascade explaining a negative
    // logit (baseline -3, running totals -2.5 then -4) was drawn from 0 instead of floating
    // at its real running totals.
    Attribution attr{"lrp", Tensor(Shape({2}), &backend, {0.5f, -1.5f}), {}};
    std::vector<WaterfallStep> steps = ToWaterfallSteps(attr, -3.0f);

    std::vector<WaterfallBar> bars = ToWaterfallBars(steps, -3.0f);

    ASSERT_EQ(bars.size(), 2u);
    EXPECT_FLOAT_EQ(bars[0].bottom, -3.0f);
    EXPECT_FLOAT_EQ(bars[0].top, -2.5f);
    EXPECT_FLOAT_EQ(bars[1].bottom, -4.0f);
    EXPECT_FLOAT_EQ(bars[1].top, -2.5f);
    for (const WaterfallBar& bar : bars) {
        EXPECT_LT(bar.top, 0.0f);
    }
}

TEST_F(PlotDataTest, CircuitNodeDisplayLabelPrefersOwnLabelElseOpTypeAndId) {
    CircuitNode labeled{0, OpType::Elementwise, std::string("input"), 1.0f};
    CircuitNode conv{1, OpType::Conv, std::nullopt, 2.0f};
    CircuitNode relu{2, OpType::Activation, std::string(""), 2.0f};

    EXPECT_EQ(CircuitNodeDisplayLabel(labeled), "input");
    EXPECT_EQ(CircuitNodeDisplayLabel(conv), "Conv #1");
    EXPECT_EQ(CircuitNodeDisplayLabel(relu), "Activation #2");
}

TEST_F(PlotDataTest, ToSaliencyHeatmapSqueezesBatchOneSingleChannelRank4Tensor) {
    // (1, 1, H, W) is what every input-space image explainer returns for one batched
    // single-channel image (e.g. an MNIST digit) -- the heatmap must take it directly.
    Attribution attr{"saliency", Tensor(Shape({1, 1, 2, 3}), &backend, {1, 2, 3, 4, 5, 6}), {}};

    HeatmapGrid grid = ToSaliencyHeatmap(attr);

    EXPECT_EQ(grid.rows, 2);
    EXPECT_EQ(grid.cols, 3);
    ASSERT_EQ(grid.values.size(), 6u);
    EXPECT_FLOAT_EQ(grid.values[0], 1.0f);
    EXPECT_FLOAT_EQ(grid.values[5], 6.0f);
}

TEST_F(PlotDataTest, ToSaliencyHeatmapThrowsOnRank4WithMoreThanOneBatchOrChannel) {
    Attribution batch2{"saliency", Tensor(Shape({2, 1, 1, 2}), &backend, {1, 2, 3, 4}), {}};
    EXPECT_THROW(ToSaliencyHeatmap(batch2), std::invalid_argument);

    Attribution channel2{"saliency", Tensor(Shape({1, 2, 1, 2}), &backend, {1, 2, 3, 4}), {}};
    EXPECT_THROW(ToSaliencyHeatmap(channel2), std::invalid_argument);
}

TEST_F(PlotDataTest, ComputeHeatmapColorScaleIsUnsignedZeroToMaxForNonNegativeValues) {
    HeatmapGrid grid{{0.0f, 0.5f, 2.0f, 1.0f}, 2, 2};

    HeatmapColorScale scale = ComputeHeatmapColorScale(grid);

    EXPECT_FALSE(scale.is_signed);
    EXPECT_FLOAT_EQ(scale.scale_min, 0.0f);
    EXPECT_FLOAT_EQ(scale.scale_max, 2.0f);
}

TEST_F(PlotDataTest, ComputeHeatmapColorScaleIsSymmetricAroundZeroForSignedValues) {
    // A signed attribution (gradient, IG, LRP) must not be squashed onto [0, max]: its
    // negative evidence would all clamp to the colormap's bottom, indistinguishable from zero.
    HeatmapGrid grid{{-3.0f, 0.5f, 1.0f, 0.0f}, 2, 2};

    HeatmapColorScale scale = ComputeHeatmapColorScale(grid);

    EXPECT_TRUE(scale.is_signed);
    EXPECT_FLOAT_EQ(scale.scale_min, -3.0f);
    EXPECT_FLOAT_EQ(scale.scale_max, 3.0f);
}

TEST_F(PlotDataTest, ComputeHeatmapColorScaleHandlesAllZeroGridWithoutZeroWidthRange) {
    HeatmapGrid grid{{0.0f, 0.0f}, 1, 2};

    HeatmapColorScale scale = ComputeHeatmapColorScale(grid);

    EXPECT_FALSE(scale.is_signed);
    EXPECT_FLOAT_EQ(scale.scale_min, 0.0f);
    EXPECT_FLOAT_EQ(scale.scale_max, 1.0f);
}

TEST_F(PlotDataTest, ToSaliencyHeatmapThrowsOnIncompatibleRank) {
    Attribution attr1{"saliency", Tensor(Shape({4}), &backend, {1, 2, 3, 4}), {}};
    EXPECT_THROW(ToSaliencyHeatmap(attr1), std::invalid_argument);

    Attribution attr2{"saliency", Tensor(Shape({2, 2, 2}), &backend, {1, 2, 3, 4, 5, 6, 7, 8}), {}};
    EXPECT_THROW(ToSaliencyHeatmap(attr2), std::invalid_argument);
}

TEST_F(PlotDataTest, ToFieldHistogramBinsBinsValuesIntoEqualWidthBins) {
    InMemoryDataset dataset({
        Sample{{Tensor(Shape({1}), &backend, {0.0f})}},
        Sample{{Tensor(Shape({1}), &backend, {2.5f})}},
        Sample{{Tensor(Shape({1}), &backend, {5.0f})}},
        Sample{{Tensor(Shape({1}), &backend, {9.9f})}},
    });

    HistogramBins bins = ToFieldHistogramBins(dataset, 0, 2);

    ASSERT_EQ(bins.bin_edges.size(), 3u);
    EXPECT_NEAR(bins.bin_edges[0], 0.0f, 1e-4f);
    EXPECT_NEAR(bins.bin_edges[2], 9.9f, 1e-4f);
    ASSERT_EQ(bins.counts.size(), 2u);
    EXPECT_EQ(bins.counts[0] + bins.counts[1], 4);
}

TEST_F(PlotDataTest, ToFieldHistogramBinsThrowsOnEmptyDataset) {
    InMemoryDataset dataset({});
    EXPECT_THROW(ToFieldHistogramBins(dataset, 0, 5), std::invalid_argument);
}

TEST_F(PlotDataTest, ToFieldHistogramBinsThrowsOnInvalidFieldIndex) {
    InMemoryDataset dataset({Sample{{Tensor(Shape({1}), &backend, {1.0f})}}});
    EXPECT_THROW(ToFieldHistogramBins(dataset, 5, 3), std::out_of_range);
}

TEST_F(PlotDataTest, ToFieldHistogramBinsThrowsOnNonPositiveBinCount) {
    InMemoryDataset dataset({Sample{{Tensor(Shape({1}), &backend, {1.0f})}}});
    EXPECT_THROW(ToFieldHistogramBins(dataset, 0, 0), std::invalid_argument);
}

TEST_F(PlotDataTest, ToBeeswarmPointsPreservesXValueFromEachRun) {
    std::vector<Attribution> runs;
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {0.1f, 9.0f}), {}});
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {0.5f, 9.0f}), {}});
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {-0.3f, 9.0f}), {}});

    std::vector<BeeswarmPoint> points = ToBeeswarmPoints(runs, 0);

    ASSERT_EQ(points.size(), 3u);
    EXPECT_FLOAT_EQ(points[0].x, 0.1f);
    EXPECT_FLOAT_EQ(points[1].x, 0.5f);
    EXPECT_FLOAT_EQ(points[2].x, -0.3f);
}

TEST_F(PlotDataTest, ToBeeswarmPointsSingleRunIsNotJittered) {
    std::vector<Attribution> runs;
    runs.push_back(Attribution{"lime", Tensor(Shape({1}), &backend, {0.7f}), {}});

    std::vector<BeeswarmPoint> points = ToBeeswarmPoints(runs, 0);

    ASSERT_EQ(points.size(), 1u);
    EXPECT_FLOAT_EQ(points[0].y, 0.0f);
}

TEST_F(PlotDataTest, ToBeeswarmPointsSpreadsCollidingValuesVerticallyWithoutOverlap) {
    std::vector<Attribution> runs;
    for (int i = 0; i < 6; ++i) {
        runs.push_back(Attribution{"lime", Tensor(Shape({1}), &backend, {1.0f}), {}});
    }

    std::vector<BeeswarmPoint> points = ToBeeswarmPoints(runs, 0);

    ASSERT_EQ(points.size(), 6u);
    std::vector<float> ys;
    for (const BeeswarmPoint& p : points) ys.push_back(p.y);
    std::sort(ys.begin(), ys.end());
    for (size_t i = 1; i < ys.size(); ++i) {
        EXPECT_GT(ys[i], ys[i - 1]) << "colliding points at the same x must be spread to distinct y offsets";
    }
}

TEST_F(PlotDataTest, ToBeeswarmPointsThrowsOnEmptyRuns) {
    std::vector<Attribution> runs;
    EXPECT_THROW(ToBeeswarmPoints(runs, 0), std::invalid_argument);
}

TEST_F(PlotDataTest, ToBeeswarmPointsThrowsOnOutOfRangeFeatureIndex) {
    std::vector<Attribution> runs;
    runs.push_back(Attribution{"lime", Tensor(Shape({2}), &backend, {0.1f, 0.2f}), {}});

    EXPECT_THROW(ToBeeswarmPoints(runs, 5), std::out_of_range);
}

TEST_F(PlotDataTest, ToFieldHistogramBinsHandlesConstantFieldWithoutDividingByZero) {
    InMemoryDataset dataset({
        Sample{{Tensor(Shape({1}), &backend, {3.0f})}},
        Sample{{Tensor(Shape({1}), &backend, {3.0f})}},
    });

    HistogramBins bins = ToFieldHistogramBins(dataset, 0, 4);

    ASSERT_EQ(bins.counts.size(), 4u);
    int64_t total = 0;
    for (int64_t count : bins.counts) total += count;
    EXPECT_EQ(total, 2);
}

TEST_F(PlotDataTest, ToRgbImageBufferConvertsThreeChannelChwToInterleavedRgb) {
    // C=3, H=1, W=2: pixel 0 = (R=1.0,G=0.0,B=0.5), pixel 1 = (R=0.0,G=1.0,B=0.25).
    Tensor image(Shape({3, 1, 2}), &backend, {
                                                  1.0f, 0.0f,  // R row
                                                  0.0f, 1.0f,  // G row
                                                  0.5f, 0.25f,  // B row
                                              });

    RgbImageBuffer buffer = ToRgbImageBuffer(image);

    EXPECT_EQ(buffer.height, 1);
    EXPECT_EQ(buffer.width, 2);
    ASSERT_EQ(buffer.pixels.size(), 6u);
    EXPECT_EQ(buffer.pixels[0], 255);  // pixel0.R
    EXPECT_EQ(buffer.pixels[1], 0);    // pixel0.G
    EXPECT_EQ(buffer.pixels[2], 127);  // pixel0.B (0.5*255 truncated)
    EXPECT_EQ(buffer.pixels[3], 0);    // pixel1.R
    EXPECT_EQ(buffer.pixels[4], 255);  // pixel1.G
    EXPECT_EQ(buffer.pixels[5], 63);   // pixel1.B (0.25*255 truncated)
}

TEST_F(PlotDataTest, ToRgbImageBufferReplicatesSingleChannelAcrossRgb) {
    Tensor image(Shape({1, 1, 2}), &backend, {1.0f, 0.0f});

    RgbImageBuffer buffer = ToRgbImageBuffer(image);

    ASSERT_EQ(buffer.pixels.size(), 6u);
    EXPECT_EQ(buffer.pixels[0], 255);
    EXPECT_EQ(buffer.pixels[1], 255);
    EXPECT_EQ(buffer.pixels[2], 255);
    EXPECT_EQ(buffer.pixels[3], 0);
    EXPECT_EQ(buffer.pixels[4], 0);
    EXPECT_EQ(buffer.pixels[5], 0);
}

TEST_F(PlotDataTest, ToRgbImageBufferClampsValuesOutsideZeroOneRange) {
    Tensor image(Shape({1, 1, 2}), &backend, {2.0f, -1.0f});

    RgbImageBuffer buffer = ToRgbImageBuffer(image);

    EXPECT_EQ(buffer.pixels[0], 255);
    EXPECT_EQ(buffer.pixels[3], 0);
}

TEST_F(PlotDataTest, ToRgbImageBufferAcceptsRank4BatchOneShape) {
    Tensor image(Shape({1, 3, 1, 2}), &backend, {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, 0.25f});

    RgbImageBuffer buffer = ToRgbImageBuffer(image);

    EXPECT_EQ(buffer.height, 1);
    EXPECT_EQ(buffer.width, 2);
    ASSERT_EQ(buffer.pixels.size(), 6u);
    EXPECT_EQ(buffer.pixels[0], 255);
}

TEST_F(PlotDataTest, ToRgbImageBufferThrowsOnUnsupportedChannelCount) {
    Tensor image(Shape({2, 1, 2}), &backend, {1.0f, 0.0f, 0.0f, 1.0f});
    EXPECT_THROW(ToRgbImageBuffer(image), std::invalid_argument);
}

TEST_F(PlotDataTest, ToRgbImageBufferThrowsOnUnsupportedRank) {
    Tensor image(Shape({2}), &backend, {1.0f, 0.0f});
    EXPECT_THROW(ToRgbImageBuffer(image), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
