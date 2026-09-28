/** @file dataset_preview_demo.cpp
 *  @brief Standalone GUI demo for Phase C's data-loading touchpoints -- DatasetStatisticsView
 *         (per-field histograms + DatasetValidator issue count) and ImageGridView (texture-
 *         cached thumbnail grid) -- run against small synthetic datasets (no external file/
 *         download dependency, unlike a real ImageFolderDataset/CsvDataset would need).
 *  @note Not a test -- see explanation_dashboard_demo.cpp's note on this module's testing
 *        strategy.
 */
#include <cmath>
#include <cstdint>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dataset.hpp"
#include "pulsatrix/viz/dataset_statistics_view.hpp"
#include "pulsatrix/viz/image_grid_view.hpp"
#include "pulsatrix/viz/window.hpp"

#include <imgui.h>

namespace {

using pulsatrix::DeviceBackend;
using pulsatrix::Sample;
using pulsatrix::Shape;
using pulsatrix::Tensor;

// Two numeric fields: a smooth sine wave, and a mostly-small field with a handful of large
// outliers -- gives DatasetValidator's outlier detection something real to flag.
class SyntheticTabularDataset : public pulsatrix::Dataset {
public:
    explicit SyntheticTabularDataset(DeviceBackend* backend) : backend_(backend) {}

    [[nodiscard]] int64_t size() const override { return 200; }

    [[nodiscard]] Sample get(int64_t index) const override {
        float feature0 = std::sin(static_cast<float>(index) * 0.1f) * 5.0f;
        float feature1 = (index % 13 == 0) ? 100.0f : static_cast<float>(index % 20);
        return Sample{{Tensor(Shape({1}), backend_, {feature0}), Tensor(Shape({1}), backend_, {feature1})}};
    }

private:
    DeviceBackend* backend_;
};

// Small procedurally-generated RGB images -- exercises ImageGridView/TextureCache without
// needing a real image file on disk.
class SyntheticImageDataset : public pulsatrix::Dataset {
public:
    explicit SyntheticImageDataset(DeviceBackend* backend) : backend_(backend) {}

    [[nodiscard]] int64_t size() const override { return 12; }

    [[nodiscard]] Sample get(int64_t index) const override {
        constexpr int64_t kSide = 16;
        std::vector<float> pixels(static_cast<size_t>(3 * kSide * kSide));
        for (int64_t c = 0; c < 3; ++c) {
            for (int64_t y = 0; y < kSide; ++y) {
                for (int64_t x = 0; x < kSide; ++x) {
                    float value = 0.5f + 0.5f * std::sin(0.3f * static_cast<float>(x + y + index) +
                                                          static_cast<float>(c));
                    pixels[static_cast<size_t>(c * kSide * kSide + y * kSide + x)] = value;
                }
            }
        }
        return Sample{{Tensor(Shape({3, kSide, kSide}), backend_, pixels)}};
    }

private:
    DeviceBackend* backend_;
};

}  // namespace

int main() {
    using namespace pulsatrix;

    CPUBackend backend;
    SyntheticTabularDataset tabular(&backend);
    SyntheticImageDataset images(&backend);
    TextureCache texture_cache;

    VizWindow window("pulsatrix -- Dataset Preview Demo", 1100, 750);
    window.run([&]() {
        ImGui::SetNextWindowSize(ImVec2(1080, 730), ImGuiCond_FirstUseEver);
        ImGui::Begin("Dataset Preview");

        if (ImGui::CollapsingHeader("Field Statistics", ImGuiTreeNodeFlags_DefaultOpen)) {
            DatasetStatisticsView::Draw("tabular_stats", tabular);
        }

        if (ImGui::CollapsingHeader("Image Grid", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImageGridView::Draw("image_grid", images, texture_cache, 0, images.size(), 4, 96.0f);
        }

        ImGui::End();
    });

    return 0;
}
