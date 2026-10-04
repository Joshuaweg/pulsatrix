#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/dataset.hpp"
#include "pulsatrix/determinism.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/linear_probe.hpp"
#include "pulsatrix/sampler.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

// Global seed state is process-wide: every test starts from, and restores, the default.
class DeterminismTest : public ::testing::Test {
protected:
    CPUBackend backend;
    void SetUp() override {
        set_seed(0);
        set_deterministic(true);
    }
    void TearDown() override {
        set_seed(0);
        set_deterministic(true);
    }

    std::vector<float> dropout_mask(DropoutModule& d) {
        return values_of(d.forward(Tensor(Shape({64}), &backend, std::vector<float>(64, 1.0f))));
    }
};

TEST_F(DeterminismTest, NextSeedIsAReproducibleStreamOfDistinctSeeds) {
    set_seed(7);
    const uint64_t a = next_seed(), b = next_seed(), c = next_seed();
    EXPECT_NE(a, b);
    EXPECT_NE(b, c);
    EXPECT_NE(a, c);
    EXPECT_EQ(global_seed(), 7u);
    set_seed(7);
    EXPECT_EQ(next_seed(), a);
    EXPECT_EQ(next_seed(), b);
    set_seed(8);
    EXPECT_NE(next_seed(), a);
}

// Before FND-7 every DropoutModule defaulted to seed 42, so two layers drew identical masks.
TEST_F(DeterminismTest, DefaultSeededDropoutLayersDrawDifferentButReproducibleMasks) {
    set_seed(1);
    DropoutModule first(0.5f, &backend), second(0.5f, &backend);
    const std::vector<float> m1 = dropout_mask(first), m2 = dropout_mask(second);
    EXPECT_NE(m1, m2);

    set_seed(1);
    DropoutModule again_first(0.5f, &backend), again_second(0.5f, &backend);
    EXPECT_EQ(dropout_mask(again_first), m1);
    EXPECT_EQ(dropout_mask(again_second), m2);

    set_seed(2);
    DropoutModule other(0.5f, &backend);
    EXPECT_NE(dropout_mask(other), m1);
}

TEST_F(DeterminismTest, ExplicitSeedsIgnoreTheGlobalSeed) {
    set_seed(1);
    DropoutModule a(0.5f, &backend, 7);
    set_seed(99);
    (void)next_seed();
    DropoutModule b(0.5f, &backend, 7);
    EXPECT_EQ(dropout_mask(a), dropout_mask(b));
}

TEST_F(DeterminismTest, DefaultSeededInitializationFollowsTheGlobalSeed) {
    auto probe_weights = [&](uint64_t seed) {
        set_seed(seed);
        LinearProbe probe(6, &backend);
        return values_of(probe.classifier().weight());
    };
    EXPECT_EQ(probe_weights(3), probe_weights(3));
    EXPECT_NE(probe_weights(3), probe_weights(4));

    auto sae_weights = [&](uint64_t seed) {
        set_seed(seed);
        SparseAutoencoder sae(4, 8, 0.1f, &backend);
        return values_of(sae.encoder().weight());
    };
    EXPECT_EQ(sae_weights(3), sae_weights(3));
    EXPECT_NE(sae_weights(3), sae_weights(4));
}

class RangeDataset : public Dataset {
public:
    explicit RangeDataset(DeviceBackend* backend) : backend_(backend) {}
    [[nodiscard]] int64_t size() const override { return 32; }
    [[nodiscard]] Sample get(int64_t index) const override {
        return Sample{{Tensor(Shape({1}), backend_, {static_cast<float>(index)})}};
    }

private:
    DeviceBackend* backend_;
};

std::vector<float> epoch_order(DataLoader& loader) {
    std::vector<float> order;
    while (auto batch = loader.next_batch()) {
        for (float v : values_of(batch->fields[0])) order.push_back(v);
    }
    return order;
}

TEST_F(DeterminismTest, DataLoaderShuffleFollowsTheGlobalSeedUnlessGivenOne) {
    auto dataset = std::make_shared<RangeDataset>(&backend);
    DataLoaderOptions options;
    options.shuffle = true;
    options.batch_size = 4;

    set_seed(5);
    DataLoader a(dataset, &backend, options);
    set_seed(5);
    DataLoader b(dataset, &backend, options);
    set_seed(6);
    DataLoader c(dataset, &backend, options);
    const std::vector<float> order_a = epoch_order(a);
    EXPECT_EQ(epoch_order(b), order_a);
    EXPECT_NE(epoch_order(c), order_a);

    options.shuffle_seed = 99u;  // an explicit seed is used exactly, as before
    set_seed(5);
    DataLoader explicit_a(dataset, &backend, options);
    set_seed(6);
    DataLoader explicit_b(dataset, &backend, options);
    EXPECT_EQ(epoch_order(explicit_a), epoch_order(explicit_b));
}

TEST_F(DeterminismTest, DeterministicModeIsOnByDefaultAndForbidsNondeterministicPaths) {
    EXPECT_TRUE(deterministic());
    EXPECT_THROW(check_deterministic_allowed("a nondeterministic kernel"), std::logic_error);
    set_deterministic(false);
    EXPECT_FALSE(deterministic());
    EXPECT_NO_THROW(check_deterministic_allowed("a nondeterministic kernel"));
}

}  // namespace
}  // namespace pulsatrix
