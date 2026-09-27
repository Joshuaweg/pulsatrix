#include <gtest/gtest.h>

#include "pulsatrix/trial.hpp"

namespace pulsatrix {
namespace {

TEST(TrialTest, StoresConfigurationVerbatim) {
    Configuration config{{"learning_rate", 0.01}, {"batch_size", int64_t{32}}};
    Trial trial(config);
    EXPECT_EQ(trial.configuration(), config);
}

TEST(TrialTest, RecordMetricAppendsToHistory) {
    Trial trial(Configuration{});
    trial.RecordMetric("loss", 1.0, 0);
    trial.RecordMetric("loss", 0.5, 1);
    ASSERT_EQ(trial.metrics().size(), 2u);
    EXPECT_EQ(trial.metrics()[0].tag, "loss");
    EXPECT_DOUBLE_EQ(trial.metrics()[0].value, 1.0);
    EXPECT_EQ(trial.metrics()[0].step, 0);
    EXPECT_DOUBLE_EQ(trial.metrics()[1].value, 0.5);
}

TEST(TrialTest, LatestMetricReturnsMostRecentlyRecordedValue) {
    Trial trial(Configuration{});
    trial.RecordMetric("loss", 1.0, 0);
    trial.RecordMetric("accuracy", 0.9, 0);
    trial.RecordMetric("loss", 0.2, 1);
    auto latest = trial.LatestMetric("loss");
    ASSERT_TRUE(latest.has_value());
    EXPECT_DOUBLE_EQ(*latest, 0.2);
}

TEST(TrialTest, LatestMetricReturnsNulloptForUnrecordedTag) {
    Trial trial(Configuration{});
    EXPECT_FALSE(trial.LatestMetric("nonexistent").has_value());
}

TEST(TrialTest, BestMetricMinimizes) {
    Trial trial(Configuration{});
    trial.RecordMetric("loss", 1.0, 0);
    trial.RecordMetric("loss", 0.2, 1);
    trial.RecordMetric("loss", 0.5, 2);
    auto best = trial.BestMetric("loss", /*maximize=*/false);
    ASSERT_TRUE(best.has_value());
    EXPECT_DOUBLE_EQ(*best, 0.2);
}

TEST(TrialTest, BestMetricMaximizes) {
    Trial trial(Configuration{});
    trial.RecordMetric("accuracy", 0.5, 0);
    trial.RecordMetric("accuracy", 0.9, 1);
    trial.RecordMetric("accuracy", 0.7, 2);
    auto best = trial.BestMetric("accuracy", /*maximize=*/true);
    ASSERT_TRUE(best.has_value());
    EXPECT_DOUBLE_EQ(*best, 0.9);
}

TEST(TrialTest, BestMetricReturnsNulloptForUnrecordedTag) {
    Trial trial(Configuration{});
    EXPECT_FALSE(trial.BestMetric("nonexistent", true).has_value());
}

}  // namespace
}  // namespace pulsatrix
