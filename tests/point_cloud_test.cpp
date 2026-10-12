// TDA-1: a point cloud from activation batches, with flatten and pooling options, concatenated
// across batches.
#include "pulsatrix/tda/point_cloud.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"

namespace pulsatrix {
namespace {

class PointCloudTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Values 0, 1, 2, ... in row-major order, so each one names its own index.
    Tensor Iota(std::vector<int64_t> dims) {
        Shape shape(dims);
        std::vector<float> v(static_cast<std::size_t>(shape.numel()));
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
        return Tensor(shape, &backend, v);
    }
};

TEST_F(PointCloudTest, FlattenGivesOnePointPerSampleWithEveryDimensionAfterTheBatch) {
    PointCloud cloud;
    cloud.add(Iota({2, 3, 2}));  // two samples of 3 x 2

    EXPECT_EQ(cloud.size(), 2);
    EXPECT_EQ(cloud.dimension(), 6);
    EXPECT_EQ(cloud.samples(), 2);
    EXPECT_EQ(cloud.values(), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    ASSERT_EQ(cloud.origins().size(), 2u);
    EXPECT_EQ(cloud.origins()[1].sample, 1);
    EXPECT_EQ(cloud.origins()[1].position, -1);
}

TEST_F(PointCloudTest, PerPositionGivesOnePointPerTokenWithTheLastAxisAsFeatures) {
    PointCloud cloud({PointPooling::PerPosition});
    cloud.add(Iota({2, 3, 2}));  // (batch, tokens, width)

    EXPECT_EQ(cloud.size(), 6);
    EXPECT_EQ(cloud.dimension(), 2);
    EXPECT_EQ(cloud.values(), (std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}));
    EXPECT_EQ(cloud.origins()[4].sample, 1);
    EXPECT_EQ(cloud.origins()[4].position, 1);
}

TEST_F(PointCloudTest, PerPositionOnAConvolutionTakesTheChannelsAsFeaturesAndThePixelsAsPoints) {
    // (batch 1, channels 2, height 2, width 3): channel c, pixel (y, x) holds c * 6 + y * 3 + x.
    PointCloud cloud({PointPooling::PerPosition, /*feature_axis=*/1});
    cloud.add(Iota({1, 2, 2, 3}));

    ASSERT_EQ(cloud.size(), 6);
    EXPECT_EQ(cloud.dimension(), 2);
    for (int64_t pixel = 0; pixel < 6; ++pixel) {
        EXPECT_EQ(cloud.values()[static_cast<std::size_t>(pixel * 2)], static_cast<float>(pixel));
        EXPECT_EQ(cloud.values()[static_cast<std::size_t>(pixel * 2 + 1)], static_cast<float>(6 + pixel));
        EXPECT_EQ(cloud.origins()[static_cast<std::size_t>(pixel)].position, pixel);
    }
}

TEST_F(PointCloudTest, AMiddleFeatureAxisKeepsThePositionsBeforeAndAfterItInRowMajorOrder) {
    // (batch 1, 2, features 3, 2): positions are (a, b) with a before and b after the feature axis.
    PointCloud cloud({PointPooling::PerPosition, /*feature_axis=*/2});
    cloud.add(Iota({1, 2, 3, 2}));

    ASSERT_EQ(cloud.size(), 4);
    ASSERT_EQ(cloud.dimension(), 3);
    // Position (a=1, b=0) is point 2; its feature f is element a * 6 + f * 2 + b.
    EXPECT_EQ(cloud.values()[6], 6.0f);
    EXPECT_EQ(cloud.values()[7], 8.0f);
    EXPECT_EQ(cloud.values()[8], 10.0f);
}

TEST_F(PointCloudTest, MeanMaxAndLastPoolEachSampleIntoOnePoint) {
    const Tensor batch(Shape({1, 3, 2}), &backend, {1, -4, 5, 2, 0, 8});  // tokens (1,-4), (5,2), (0,8)

    PointCloud mean({PointPooling::Mean});
    mean.add(batch);
    EXPECT_EQ(mean.values(), (std::vector<float>{2, 2}));
    EXPECT_EQ(mean.origins()[0].position, -1);

    PointCloud max({PointPooling::Max});
    max.add(batch);
    EXPECT_EQ(max.values(), (std::vector<float>{5, 8}));

    PointCloud last({PointPooling::Last});
    last.add(batch);
    EXPECT_EQ(last.values(), (std::vector<float>{0, 8}));
    EXPECT_EQ(last.origins()[0].position, 2);
}

TEST_F(PointCloudTest, ARankTwoBatchHasOnePositionPerSampleUnderEveryPooling) {
    for (PointPooling pooling : {PointPooling::Flatten, PointPooling::PerPosition, PointPooling::Mean,
                                 PointPooling::Max, PointPooling::Last}) {
        PointCloud cloud({pooling});
        cloud.add(Iota({3, 4}));
        EXPECT_EQ(cloud.size(), 3);
        EXPECT_EQ(cloud.dimension(), 4);
        EXPECT_EQ(cloud.values()[5], 5.0f);
    }
}

TEST_F(PointCloudTest, BatchesOfDifferentLengthsConcatenateAndKeepCountingSamples) {
    PointCloud cloud({PointPooling::PerPosition});
    cloud.add(Iota({1, 3, 2}));  // a 3-token text
    cloud.add(Iota({2, 1, 2}));  // two 1-token texts

    EXPECT_EQ(cloud.size(), 5);
    EXPECT_EQ(cloud.samples(), 3);
    EXPECT_EQ(cloud.origins()[3].sample, 1);
    EXPECT_EQ(cloud.origins()[4].sample, 2);
    EXPECT_EQ(cloud.origins()[4].position, 0);
    EXPECT_EQ(cloud.values()[8], 2.0f);  // the second batch's second sample starts at its element 2
}

TEST_F(PointCloudTest, ToTensorGivesTheRowsAsAnNByDimensionTensor) {
    PointCloud cloud({PointPooling::PerPosition});
    cloud.add(Iota({2, 2, 3}));
    const Tensor t = cloud.ToTensor(&backend);

    ASSERT_EQ(t.rank(), 2);
    EXPECT_EQ(t.shape().dim(0), 4);
    EXPECT_EQ(t.shape().dim(1), 3);
    EXPECT_EQ(t.to_host_vector(), cloud.values());
}

TEST_F(PointCloudTest, SnapshotsFromSeveralForwardPassesGiveOneCloud) {
    LinearModule linear(3, 4, &backend);
    ReluModule relu(&backend);
    ExplainerContext ctx({&linear, &relu});
    const NodeId relu_output = 2;  // node 0 is the input, node i + 1 module i's output

    std::vector<ActivationSnapshot> snapshots;
    (void)ctx.forward_pass(Tensor(Shape({2, 3}), &backend, {0.5f, -1.0f, 2.0f, 1.0f, 0.0f, -0.5f}));
    snapshots.push_back(ctx.activation_snapshot());
    (void)ctx.forward_pass(Tensor(Shape({1, 3}), &backend, {-2.0f, 1.5f, 0.25f}));
    snapshots.push_back(ctx.activation_snapshot());

    const PointCloud cloud = MakePointCloud(snapshots, relu_output);
    EXPECT_EQ(cloud.size(), 3);
    EXPECT_EQ(cloud.dimension(), 4);
    std::vector<float> expected = snapshots[0].activation(relu_output).to_host_vector();
    const std::vector<float> second = snapshots[1].activation(relu_output).to_host_vector();
    expected.insert(expected.end(), second.begin(), second.end());
    EXPECT_EQ(cloud.values(), expected);
}

TEST_F(PointCloudTest, ANodeMissingFromTheSnapshotThrowsInsteadOfAsserting) {
    LinearModule linear(3, 4, &backend);
    ExplainerContext ctx({&linear});
    (void)ctx.forward_pass(Tensor(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f}));
    PointCloud cloud;
    EXPECT_THROW(cloud.add(ctx.activation_snapshot(), 99), std::invalid_argument);
    EXPECT_THROW((void)MakePointCloud({}, 0), std::invalid_argument);
}

TEST_F(PointCloudTest, ABatchOfAnotherDimensionThrowsAndLeavesTheCloudUnchanged) {
    PointCloud cloud({PointPooling::PerPosition});
    cloud.add(Iota({1, 2, 3}));
    EXPECT_THROW(cloud.add(Iota({1, 2, 4})), std::invalid_argument);
    EXPECT_EQ(cloud.size(), 2);
    EXPECT_EQ(cloud.samples(), 1);
    EXPECT_EQ(cloud.values().size(), 6u);
}

TEST_F(PointCloudTest, BadBatchesAndOptionsThrow) {
    PointCloud flat;
    EXPECT_THROW(flat.add(Iota({4})), std::invalid_argument);  // no batch axis and features
    EXPECT_THROW(flat.add(Tensor(Shape({1, 2}), &backend, {1.0f, std::numeric_limits<float>::quiet_NaN()})),
                 std::invalid_argument);
    EXPECT_THROW(flat.add(Tensor(Shape({1, 2}), &backend, {1.0f, std::numeric_limits<float>::infinity()})),
                 std::invalid_argument);
    EXPECT_EQ(flat.size(), 0);
    EXPECT_EQ(flat.dimension(), 0);
    EXPECT_THROW((void)flat.ToTensor(&backend), std::invalid_argument);

    PointCloud on_batch_axis({PointPooling::PerPosition, /*feature_axis=*/0});
    EXPECT_THROW(on_batch_axis.add(Iota({2, 3})), std::invalid_argument);
    PointCloud past_the_end({PointPooling::Mean, /*feature_axis=*/3});
    EXPECT_THROW(past_the_end.add(Iota({2, 3, 4})), std::invalid_argument);
    PointCloud before_the_start({PointPooling::Mean, /*feature_axis=*/-4});
    EXPECT_THROW(before_the_start.add(Iota({2, 3, 4})), std::invalid_argument);

    EXPECT_THROW(PointCloud({static_cast<PointPooling>(42)}), std::invalid_argument);
}

TEST_F(PointCloudTest, MeanPoolingAccumulatesInDoublePrecision) {
    // A float32 running sum rounds 2^24 + 1 back to 2^24 twice, for a mean of 5592405.5 instead of
    // (2^24 + 2) / 3 = 5592406.
    PointCloud big({PointPooling::Mean});
    big.add(Tensor(Shape({1, 3, 1}), &backend, {16777216.0f, 1.0f, 1.0f}));
    EXPECT_EQ(big.values()[0], 5592406.0f);
}

}  // namespace
}  // namespace pulsatrix
