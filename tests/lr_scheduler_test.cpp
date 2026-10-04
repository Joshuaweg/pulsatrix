#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lr_scheduler.hpp"
#include "pulsatrix/param_groups.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

namespace pulsatrix {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Hugging Face transformers' get_*_schedule_with_warmup, written out independently.
double warmup_part(int64_t step, int64_t warmup) { return static_cast<double>(step) / std::max<int64_t>(1, warmup); }
double hf_linear(int64_t step, int64_t warmup, int64_t total) {
    if (step < warmup) return warmup_part(step, warmup);
    return std::max(0.0, static_cast<double>(total - step) / std::max<int64_t>(1, total - warmup));
}
double hf_cosine(int64_t step, int64_t warmup, int64_t total, double min_ratio) {
    if (step < warmup) return warmup_part(step, warmup);
    const double progress = std::min(1.0, static_cast<double>(step - warmup) / std::max<int64_t>(1, total - warmup));
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + std::cos(kPi * progress));
}

TEST(LRScheduleTest, ConstantWithAndWithoutWarmup) {
    LRSchedule plain = LRSchedule::Constant();
    EXPECT_EQ(plain.multiplier(0), 1.0f);
    EXPECT_EQ(plain.multiplier(1000), 1.0f);
    LRSchedule warm = LRSchedule::Constant(/*warmup=*/4);
    EXPECT_EQ(warm.multiplier(0), 0.0f);
    EXPECT_EQ(warm.multiplier(1), 0.25f);
    EXPECT_EQ(warm.multiplier(4), 1.0f);
    EXPECT_EQ(warm.multiplier(400), 1.0f);
}

TEST(LRScheduleTest, LinearMatchesHuggingFace) {
    LRSchedule s = LRSchedule::Linear(10, 110);
    for (int64_t step : {0, 1, 5, 9, 10, 11, 60, 109, 110, 200}) {
        EXPECT_NEAR(s.multiplier(step), hf_linear(step, 10, 110), 1e-6) << step;
    }
    EXPECT_EQ(s.multiplier(110), 0.0f);
    EXPECT_EQ(s.multiplier(500), 0.0f);  // stays at zero past the end
}

TEST(LRScheduleTest, CosineMatchesHuggingFace) {
    LRSchedule s = LRSchedule::Cosine(10, 110);
    for (int64_t step : {0, 3, 10, 35, 60, 85, 109, 110, 300}) {
        EXPECT_NEAR(s.multiplier(step), hf_cosine(step, 10, 110, 0.0), 1e-6) << step;
    }
    EXPECT_NEAR(s.multiplier(60), 0.5f, 1e-6f);  // halfway through the decay
}

TEST(LRScheduleTest, CosineWithAFloor) {
    LRSchedule s = LRSchedule::Cosine(0, 100, /*min_ratio=*/0.1f);
    EXPECT_NEAR(s.multiplier(0), 1.0f, 1e-6f);
    EXPECT_NEAR(s.multiplier(50), hf_cosine(50, 0, 100, 0.1), 1e-6);
    EXPECT_NEAR(s.multiplier(100), 0.1f, 1e-6f);
    EXPECT_NEAR(s.multiplier(1000), 0.1f, 1e-6f);
}

TEST(LRScheduleTest, RejectsInvalidSettings) {
    EXPECT_THROW((void)LRSchedule::Constant(-1), std::invalid_argument);
    EXPECT_THROW((void)LRSchedule::Linear(0, 0), std::invalid_argument);
    EXPECT_THROW((void)LRSchedule::Linear(20, 10), std::invalid_argument);
    EXPECT_THROW((void)LRSchedule::Cosine(-1, 10), std::invalid_argument);
    EXPECT_THROW((void)LRSchedule::Cosine(0, 10, 1.5f), std::invalid_argument);
    EXPECT_THROW((void)LRSchedule::Cosine(0, 10, -0.1f), std::invalid_argument);
    EXPECT_THROW((void)LRSchedule::Linear(0, 10).multiplier(-1), std::invalid_argument);
}

class LRSchedulerTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// Like PyTorch's LambdaLR: construction applies step 0, each step() advances by one, and every
// group keeps its own base rate.
TEST_F(LRSchedulerTest, DrivesTheDefaultRateAndEveryGroupFromTheirBaseRates) {
    AdamOptimizer adam(0.01f, &backend);
    adam.set_param_groups({{"bias", param_select::name_prefix("bias"), 0.1f, 0.0f}});
    LRScheduler scheduler(adam, LRSchedule::Linear(2, 6));
    EXPECT_EQ(adam.learning_rate(), 0.0f);  // warmup starts at zero
    EXPECT_EQ(adam.param_groups()[0].learning_rate, 0.0f);
    scheduler.step();  // step 1: half warmed up
    EXPECT_FLOAT_EQ(adam.learning_rate(), 0.005f);
    EXPECT_FLOAT_EQ(adam.param_groups()[0].learning_rate, 0.05f);
    scheduler.step();
    scheduler.step();  // step 3: (6 - 3) / (6 - 2) = 0.75
    EXPECT_FLOAT_EQ(adam.learning_rate(), 0.0075f);
    EXPECT_FLOAT_EQ(adam.param_groups()[0].learning_rate, 0.075f);
    EXPECT_EQ(scheduler.last_step(), 3);
}

TEST_F(LRSchedulerTest, TheScheduledRateIsTheOneTheOptimizerUses) {
    LinearModule m(1, 1, &backend);
    m.set_weight({1.0f});
    SGDOptimizer sgd(0.4f);
    LRScheduler scheduler(sgd, LRSchedule::Constant(4));
    for (int i = 0; i < 2; ++i) {
        scheduler.step();
        m.parameters()[0].grad->fill(1.0f);
        sgd.step(m);
    }
    // Steps 1 and 2 of a 4-step warmup: lr 0.1 then 0.2.
    EXPECT_FLOAT_EQ(m.parameters()[0].value->data()[0], 1.0f - 0.1f - 0.2f);
}

TEST_F(LRSchedulerTest, ResumesFromASavedStep) {
    AdamOptimizer a(0.01f, &backend), b(0.01f, &backend);
    LRScheduler sa(a, LRSchedule::Cosine(5, 50)), sb(b, LRSchedule::Cosine(5, 50));
    for (int i = 0; i < 17; ++i) sa.step();
    sb.set_last_step(sa.last_step());
    EXPECT_EQ(b.learning_rate(), a.learning_rate());
    sa.step();
    sb.step();
    EXPECT_EQ(b.learning_rate(), a.learning_rate());
    EXPECT_THROW(sb.set_last_step(-1), std::invalid_argument);
}

TEST_F(LRSchedulerTest, RefusesGroupsReplacedAfterItWasCreated) {
    SGDOptimizer sgd(0.1f);
    LRScheduler scheduler(sgd, LRSchedule::Constant());
    sgd.set_param_groups({{"bias", param_select::name_prefix("bias"), 0.1f, 0.0f}});
    EXPECT_THROW(scheduler.step(), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
