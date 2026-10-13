// XAI-4: prototypes and contrasts by nearest neighbors, and ProtoDash, on data whose answers are
// known.

#include "pulsatrix/prototypes.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

class PrototypesTest : public ::testing::Test {
protected:
    CPUBackend backend;

    Tensor Row(float a, float b) { return Tensor(Shape({1, 2}), &backend, {a, b}); }

    // Class 1 when x0 > 0.
    std::function<Tensor(const Tensor&)> sign = [this](const Tensor& in) {
        const bool one = in.to_host_vector()[0] > 0.0f;
        return Tensor(Shape({1, 2}), &backend, {one ? 0.0f : 1.0f, one ? 1.0f : 0.0f});
    };

    // A 9 x 9 grid on [-1, 1]^2.
    std::vector<Tensor> Grid() {
        std::vector<Tensor> out;
        for (int i = 0; i < 9; ++i) {
            for (int j = 0; j < 9; ++j) out.push_back(Row(-1.0f + 0.25f * static_cast<float>(i), -1.0f + 0.25f * static_cast<float>(j)));
        }
        return out;
    }

    // Three clusters of 60, 30 and 10 points around (0, 0), (5, 0) and (0, 5).
    std::vector<Tensor> Clusters() {
        std::vector<Tensor> out;
        const float centers[3][2] = {{0, 0}, {5, 0}, {0, 5}};
        const int sizes[3] = {60, 30, 10};
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < sizes[c]; ++i) {
                const float t = static_cast<float>(i);
                out.push_back(Row(centers[c][0] + 0.3f * std::sin(1.7f * t + static_cast<float>(c)),
                                  centers[c][1] + 0.3f * std::cos(2.3f * t)));
            }
        }
        return out;
    }
};

TEST_F(PrototypesTest, PrototypesShareTheClassAndContrastsDont) {
    const std::vector<Tensor> grid = Grid();
    PrototypeOptions options;
    options.k = 4;
    const PrototypeResult r = FindPrototypes(sign, Row(0.3f, 0.1f), grid, options);
    EXPECT_EQ(r.predicted_class, 1);
    ASSERT_EQ(r.prototypes.size(), 4u);
    ASSERT_EQ(r.contrasts.size(), 4u);
    for (const Example& e : r.prototypes) EXPECT_EQ(e.predicted_class, 1);
    for (const Example& e : r.contrasts) EXPECT_EQ(e.predicted_class, 0);
    // Nearest first, and the very nearest point of each class.
    for (size_t i = 1; i < 4; ++i) {
        EXPECT_LE(r.prototypes[i - 1].distance, r.prototypes[i].distance);
        EXPECT_LE(r.contrasts[i - 1].distance, r.contrasts[i].distance);
    }
    const std::vector<float> p = grid[static_cast<size_t>(r.prototypes[0].index)].to_host_vector();
    EXPECT_FLOAT_EQ(p[0], 0.25f);
    EXPECT_FLOAT_EQ(p[1], 0.0f);
    const std::vector<float> c = grid[static_cast<size_t>(r.contrasts[0].index)].to_host_vector();
    EXPECT_FLOAT_EQ(c[0], 0.0f);  // x0 = 0 is class 0 here: the boundary itself
    EXPECT_FLOAT_EQ(c[1], 0.0f);
    EXPECT_NEAR(r.contrasts[0].distance, std::sqrt(0.3 * 0.3 + 0.1 * 0.1), 1e-6);
}

TEST_F(PrototypesTest, ScaleAndEmbeddingChangeWhatNearMeans) {
    const std::vector<Tensor> grid = Grid();
    PrototypeOptions options;
    options.k = 1;
    // Feature 1 measured in units of 100: only feature 0 matters.
    options.scale = {1.0f, 100.0f};
    PrototypeResult r = FindPrototypes(sign, Row(0.6f, 0.9f), grid, options);
    EXPECT_FLOAT_EQ(grid[static_cast<size_t>(r.contrasts[0].index)].to_host_vector()[0], 0.0f);
    // Only feature 1, through an embedding: the nearest point shares x1 = 1.
    options.scale.clear();
    options.embed = [](const Tensor& t) { return std::vector<float>{t.to_host_vector()[1]}; };
    r = FindPrototypes(sign, Row(0.6f, 0.95f), grid, options);
    EXPECT_FLOAT_EQ(grid[static_cast<size_t>(r.prototypes[0].index)].to_host_vector()[1], 1.0f);
    EXPECT_NEAR(r.prototypes[0].distance, 0.05, 1e-6);

    const JsonValue doc = ParseJson(WriteJson(ToJson(r)));
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.prototypes.v1");
    EXPECT_EQ(doc.find("prototypes")->as_array().size(), 1u);
    EXPECT_EQ(doc.find("contrasts")->as_array()[0].find("predicted_class")->as_int64(), 0);
}

TEST_F(PrototypesTest, FindPrototypesValidates) {
    const std::vector<Tensor> grid = Grid();
    EXPECT_THROW((void)FindPrototypes(sign, Row(0, 0), {}), std::invalid_argument);
    PrototypeOptions options;
    options.k = 0;
    EXPECT_THROW((void)FindPrototypes(sign, Row(0, 0), grid, options), std::invalid_argument);
    options.k = 1;
    options.scale = {1.0f};
    EXPECT_THROW((void)FindPrototypes(sign, Row(0, 0), grid, options), std::invalid_argument);
    options.scale = {1.0f, 0.0f};
    EXPECT_THROW((void)FindPrototypes(sign, Row(0, 0), grid, options), std::invalid_argument);
    EXPECT_THROW((void)FindPrototypes(sign, Row(0, 0), {Tensor(Shape({1, 3}), &backend)}), std::invalid_argument);
}

// One prototype per cluster, the biggest first, weighted by cluster size.
TEST_F(PrototypesTest, ProtoDashPicksOnePrototypePerClusterWeightedBySize) {
    const std::vector<Tensor> data = Clusters();
    ProtoDashOptions options;
    options.count = 3;
    options.sigma = 1.0;
    const ProtoDashResult r = ProtoDash(data, data, options);
    ASSERT_EQ(r.indices.size(), 3u);
    const auto cluster = [](int64_t i) { return i < 60 ? 0 : (i < 90 ? 1 : 2); };
    EXPECT_EQ(cluster(r.indices[0]), 0);
    EXPECT_EQ(cluster(r.indices[1]), 1);
    EXPECT_EQ(cluster(r.indices[2]), 2);
    EXPECT_NEAR(r.weights[0], 0.6, 0.05);
    EXPECT_NEAR(r.weights[1], 0.3, 0.05);
    EXPECT_NEAR(r.weights[2], 0.1, 0.05);
    EXPECT_NEAR(r.weights[0] + r.weights[1] + r.weights[2], 1.0, 1e-12);
    for (size_t i = 1; i < r.objective.size(); ++i) EXPECT_GE(r.objective[i], r.objective[i - 1]);
}

// A target of one cluster, a source of everything: the prototypes come from that cluster.
TEST_F(PrototypesTest, ProtoDashRepresentsTheTarget) {
    const std::vector<Tensor> data = Clusters();
    const std::vector<Tensor> small(data.begin() + 90, data.end());
    ProtoDashOptions options;
    options.count = 2;
    options.sigma = 1.0;
    const ProtoDashResult r = ProtoDash(data, small, options);
    ASSERT_FALSE(r.indices.empty());
    for (int64_t i : r.indices) EXPECT_GE(i, 90);
}

TEST_F(PrototypesTest, ProtoDashMedianHeuristicAndValidation) {
    const std::vector<Tensor> data = Clusters();
    const ProtoDashResult r = ProtoDash(data, data);  // sigma: the median distance
    EXPECT_GT(r.sigma, 0.3);
    EXPECT_LT(r.sigma, 7.1);
    EXPECT_LE(r.indices.size(), 5u);
    // One point: it is its own prototype.
    const ProtoDashResult one = ProtoDash({Row(1, 2)}, {Row(1, 2)});
    ASSERT_EQ(one.indices.size(), 1u);
    EXPECT_DOUBLE_EQ(one.weights[0], 1.0);

    EXPECT_THROW((void)ProtoDash({}, data), std::invalid_argument);
    ProtoDashOptions bad;
    bad.count = 0;
    EXPECT_THROW((void)ProtoDash(data, data, bad), std::invalid_argument);
    bad = ProtoDashOptions();
    bad.sigma = -1.0;
    EXPECT_THROW((void)ProtoDash(data, data, bad), std::invalid_argument);
    EXPECT_THROW((void)ProtoDash(data, {Tensor(Shape({1, 3}), &backend)}), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
