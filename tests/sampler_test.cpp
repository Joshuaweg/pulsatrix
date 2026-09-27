#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "pulsatrix/sampler.hpp"

namespace pulsatrix {
namespace {

std::vector<int64_t> DrainAll(Sampler& sampler) {
    std::vector<int64_t> indices;
    while (auto idx = sampler.next()) {
        indices.push_back(*idx);
    }
    return indices;
}

TEST(SequentialSamplerTest, VisitsIndicesInAscendingOrder) {
    SequentialSampler sampler;
    sampler.reset(4);
    EXPECT_EQ(DrainAll(sampler), (std::vector<int64_t>{0, 1, 2, 3}));
}

TEST(SequentialSamplerTest, EmptyDatasetYieldsNoIndices) {
    SequentialSampler sampler;
    sampler.reset(0);
    EXPECT_FALSE(sampler.next().has_value());
}

TEST(SequentialSamplerTest, ResetRestartsFromZero) {
    SequentialSampler sampler;
    sampler.reset(2);
    ASSERT_TRUE(sampler.next().has_value());
    sampler.reset(2);
    EXPECT_EQ(DrainAll(sampler), (std::vector<int64_t>{0, 1}));
}

TEST(ShuffleSamplerTest, VisitsEveryIndexExactlyOnce) {
    ShuffleSampler sampler(42);
    sampler.reset(5);
    std::vector<int64_t> indices = DrainAll(sampler);
    std::sort(indices.begin(), indices.end());
    EXPECT_EQ(indices, (std::vector<int64_t>{0, 1, 2, 3, 4}));
}

TEST(ShuffleSamplerTest, SameSeedProducesSamePermutation) {
    ShuffleSampler a(7);
    ShuffleSampler b(7);
    a.reset(10);
    b.reset(10);
    EXPECT_EQ(DrainAll(a), DrainAll(b));
}

TEST(ShuffleSamplerTest, DifferentSeedsCanProduceDifferentPermutations) {
    ShuffleSampler a(1);
    ShuffleSampler b(2);
    a.reset(20);
    b.reset(20);
    EXPECT_NE(DrainAll(a), DrainAll(b));
}

TEST(ShuffleSamplerTest, EachResetReshufflesRatherThanRepeating) {
    ShuffleSampler sampler(3);
    sampler.reset(20);
    std::vector<int64_t> first_epoch = DrainAll(sampler);
    sampler.reset(20);
    std::vector<int64_t> second_epoch = DrainAll(sampler);
    EXPECT_NE(first_epoch, second_epoch);
}

TEST(ShuffleSamplerTest, EmptyDatasetYieldsNoIndices) {
    ShuffleSampler sampler(42);
    sampler.reset(0);
    EXPECT_FALSE(sampler.next().has_value());
}

}  // namespace
}  // namespace pulsatrix
