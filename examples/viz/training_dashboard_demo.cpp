/** @file training_dashboard_demo.cpp
 *  @brief Standalone GUI demo for Phase B's ImPlotMetricsSink/TrainingDashboard -- trains
 *         XorNetwork (no external data dependency, unlike mnist_training_demo.cpp) live,
 *         a few steps per frame, so the loss curve visibly animates.
 *  @note Not a test -- see explanation_dashboard_demo.cpp's note on this module's testing
 *        strategy. Acceptance check for ImPlotMetricsSink::Draw()/TrainingDashboard::Draw(),
 *        the two GUI-calling pieces this phase's GoogleTest suite deliberately doesn't cover.
 */
#include <array>
#include <utility>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/training_dashboard.hpp"
#include "pulsatrix/viz/window.hpp"
#include "pulsatrix/xor_training_example.hpp"

#include <imgui.h>

int main() {
    using namespace pulsatrix;

    CPUBackend backend;
    XorNetwork net(&backend);
    AdamOptimizer optimizer(0.05f, &backend);
    ImPlotMetricsSink sink;

    const std::array<std::pair<std::array<float, 2>, float>, 4> examples = {{
        {{0.0f, 0.0f}, 0.0f},
        {{0.0f, 1.0f}, 1.0f},
        {{1.0f, 0.0f}, 1.0f},
        {{1.0f, 1.0f}, 0.0f},
    }};

    int step = 0;
    constexpr int kMaxSteps = 4000;

    VizWindow window("pulsatrix -- Training Dashboard Demo", 1000, 700);
    window.run([&]() {
        // A handful of steps per frame so the loss curve visibly animates instead of
        // finishing before the first frame renders or needing thousands of frames to
        // converge.
        for (int i = 0; i < 4 && step < kMaxSteps; ++i, ++step) {
            const auto& [xy, target] = examples[static_cast<size_t>(step) % examples.size()];
            // Batch-1, not bare (2,)/(1,): every module is always-batched post
            // campaign_exai_dl_library_batch_dimension_support (see
            // xor_training_example_test.cpp's own fixtures -- XorNetwork's header doc
            // comment predates that migration and is stale).
            Tensor input(Shape({1, 2}), &backend, {xy[0], xy[1]});
            Tensor target_tensor(Shape({1, 1}), &backend, {target});
            net.train_step(input, target_tensor, optimizer, sink, step);
        }

        ImGui::SetNextWindowSize(ImVec2(980, 680), ImGuiCond_FirstUseEver);
        ImGui::Begin("Training Dashboard");
        TrainingDashboard::Draw(sink);
        ImGui::End();
    });

    return 0;
}
