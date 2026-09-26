#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dataset.hpp"
#include "pulsatrix/tensor.hpp"
#include "pulsatrix/transform.hpp"

namespace pulsatrix {
namespace {

// Adds a constant to every element of the sample's first field --
// campaign_exai_dl_library_data_pipeline, Mission 1, Objective 1.
class AddConstantTransform : public Transform {
public:
    explicit AddConstantTransform(float value, DeviceBackend* backend) : value_(value), backend_(backend) {}

    [[nodiscard]] Sample apply(Sample sample) const override {
        Tensor& field = sample.fields[0];
        for (int64_t i = 0; i < field.numel(); ++i) {
            field.data()[i] += value_;
        }
        return sample;
    }

private:
    float value_;
    DeviceBackend* backend_;
};

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

class TransformTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(TransformTest, ComposeAppliesStepsInOrder) {
    Compose compose({std::make_shared<AddConstantTransform>(1.0f, &backend),
                      std::make_shared<AddConstantTransform>(2.0f, &backend)});
    Sample sample{{Tensor(Shape({1}), &backend, {10.0f})}};

    Sample result = compose.apply(std::move(sample));

    EXPECT_FLOAT_EQ(result.fields[0].data()[0], 13.0f);
}

TEST_F(TransformTest, ComposeWithNoStepsIsIdentity) {
    Compose compose({});
    Sample sample{{Tensor(Shape({1}), &backend, {5.0f})}};

    Sample result = compose.apply(std::move(sample));

    EXPECT_FLOAT_EQ(result.fields[0].data()[0], 5.0f);
}

TEST_F(TransformTest, TransformDatasetAppliesTransformToEverySample) {
    auto base = std::make_shared<InMemoryDataset>(std::vector<Sample>{
        Sample{{Tensor(Shape({1}), &backend, {1.0f})}},
        Sample{{Tensor(Shape({1}), &backend, {2.0f})}},
    });
    auto transform = std::make_shared<AddConstantTransform>(10.0f, &backend);
    TransformDataset dataset(base, transform);

    EXPECT_EQ(dataset.size(), 2);
    EXPECT_FLOAT_EQ(dataset.get(0).fields[0].data()[0], 11.0f);
    EXPECT_FLOAT_EQ(dataset.get(1).fields[0].data()[0], 12.0f);
}

TEST_F(TransformTest, TransformDatasetPropagatesBaseDatasetThrow) {
    auto base = std::make_shared<InMemoryDataset>(std::vector<Sample>{});
    auto transform = std::make_shared<AddConstantTransform>(1.0f, &backend);
    TransformDataset dataset(base, transform);

    EXPECT_THROW(dataset.get(0), std::out_of_range);
}

}  // namespace
}  // namespace pulsatrix
