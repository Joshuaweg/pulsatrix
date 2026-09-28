/** @file live_inference_demo.cpp
 *  @brief Standalone GUI demo for Phase C's "inference touchpoint" -- reusing Phase A's
 *         ExplanationScoreCard, not a new widget (pulsatrix has no separate inference
 *         abstraction; Module::forward() *is* inference). Cycles through the XOR inputs,
 *         recomputing Saliency + LRP conservation for whichever one is live.
 *  @note Not a test -- see explanation_dashboard_demo.cpp's note on this module's testing
 *        strategy.
 */
#include <array>
#include <cmath>
#include <string>
#include <utility>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_conservation.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/saliency.hpp"
#include "pulsatrix/viz/explanation_score_card.hpp"
#include "pulsatrix/viz/window.hpp"

#include <imgui.h>

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    LinearModule linear1(2, 4, &backend);
    linear1.set_weight({0.6f, -0.3f, 0.4f, -0.7f, 0.2f, 0.5f, -0.6f, 0.1f});
    linear1.set_bias({0.0f, 0.0f, 0.0f, 0.0f});
    ReluModule relu(&backend);
    LinearModule linear2(4, 1, &backend);
    linear2.set_weight({0.5f, -0.4f, 0.3f, 0.6f});
    linear2.set_bias({0.0f});

    ExplainerContext ctx({&linear1, &relu, &linear2});
    Saliency saliency;
    LRPRuleConfig lrp_config;

    const std::array<std::pair<float, float>, 4> inputs = {{{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}}};

    ScoreCardScaleContext shared_scale;
    int64_t frame = 0;

    VizWindow window("pulsatrix -- Live Inference Demo", 900, 650);
    window.run([&]() {
        // Cycle through inputs every ~90 frames (~1.5s at 60fps) so the score card visibly
        // updates for a new "live" prediction without needing user interaction.
        int input_index = static_cast<int>((frame++ / 90) % static_cast<int64_t>(inputs.size()));
        auto [a, b] = inputs[static_cast<size_t>(input_index)];

        Tensor input(Shape({1, 2}), &backend, {a, b});
        Tensor output = ctx.forward_pass(input);
        float confidence = 1.0f / (1.0f + std::exp(-output.data()[0]));

        Attribution sal_attr = saliency.explain(ctx, input, /*target_index=*/0, &backend);

        Tensor relevance_seed(Shape({1, 1}), &backend, {output.data()[0]});
        Tensor relevance_h2 = linear2.propagate_relevance(relevance_seed, lrp_config);
        Tensor relevance_h1 = relu.propagate_relevance(relevance_h2, lrp_config);
        Tensor relevance_input = linear1.propagate_relevance(relevance_h1, lrp_config);
        ConservationResult conservation = ComputeConservation(relevance_input, relevance_seed);

        ImGui::SetNextWindowSize(ImVec2(880, 630), ImGuiCond_FirstUseEver);
        ImGui::Begin("Live Inference");
        ImGui::Text("Streaming prediction %d/%d", input_index + 1, static_cast<int>(inputs.size()));

        ExplanationScoreCard::Input card_input{
            "Input: (" + std::to_string(a) + ", " + std::to_string(b) + ")",
            confidence,
            sal_attr,
            conservation,
            false,  // Saliency is deterministic by construction
            StabilityResult{0.0f, true},
        };
        ExplanationScoreCard::Draw("live_score_card", card_input, &shared_scale);
        ImGui::End();
    });

    return 0;
}
