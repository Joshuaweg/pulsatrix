#include <gtest/gtest.h>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/cpu_backend.hpp"

// Attribution is a first-class result type (charter Part 2 SS4) carrying an explanation's
// raw values, the method that produced them, and any relevant metadata (baseline used for
// IG, kernel width for LIME, etc.) -- so downstream code doesn't have to guess what
// produced a number. Missions 2-3 (Saliency/IG/Grad-CAM) populate this; this mission ships
// the type itself.
namespace pulsatrix {
namespace {

class AttributionTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(AttributionTest, ConstructionStoresMethodValuesAndMetadata) {
    Tensor values(Shape({3}), &backend, {0.1f, 0.2f, 0.3f});
    Attribution attr{"saliency", Tensor(values), {{"note", "raw gradient"}}};

    EXPECT_EQ(attr.method, "saliency");
    EXPECT_FLOAT_EQ(attr.values.data()[0], 0.1f);
    EXPECT_FLOAT_EQ(attr.values.data()[1], 0.2f);
    EXPECT_FLOAT_EQ(attr.values.data()[2], 0.3f);
    EXPECT_EQ(attr.metadata.at("note"), "raw gradient");
}

TEST_F(AttributionTest, MetadataDefaultsEmptyWhenNotProvided) {
    Tensor values(Shape({1}), &backend, {1.0f});
    Attribution attr{"grad_cam", Tensor(values), {}};

    EXPECT_TRUE(attr.metadata.empty());
}

}  // namespace
}  // namespace pulsatrix
