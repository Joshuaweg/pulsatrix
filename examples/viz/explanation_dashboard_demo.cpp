/** @file explanation_dashboard_demo.cpp
 *  @brief Standalone GUI demo for Phase A's native visualization widgets -- feature-importance
 *         bar chart, waterfall, saliency heatmap, circuit graph, and the ExplanationScoreCard --
 *         run against the same small Linear->ReLU->Linear XOR network explainer_demo.cpp uses.
 *  @note Not a test -- see plans/okay-we-have-now-buzzing-moth.md's Testing Strategy. GoogleTest
 *        cannot meaningfully exercise ImGui/ImPlot draw calls; this exists so a human can watch
 *        the charts render, and is the acceptance check for everything in src/viz/ that calls
 *        into ImGui or ImPlot directly (mirrors explainer_demo.cpp's own relationship to
 *        saliency_test.cpp/integrated_gradients_test.cpp).
 */
#include <cmath>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/integrated_gradients.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_conservation.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/saliency.hpp"
#include "pulsatrix/viz/attribution_bar_chart.hpp"
#include "pulsatrix/viz/attribution_waterfall.hpp"
#include "pulsatrix/viz/circuit_graph_view.hpp"
#include "pulsatrix/viz/explanation_score_card.hpp"
#include "pulsatrix/viz/saliency_heatmap_view.hpp"
#include "pulsatrix/viz/window.hpp"

#include <imgui.h>

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    // Same small MLP shape as explainer_demo.cpp's -- Linear(2,4) -> ReLU -> Linear(4,1).
    LinearModule linear1(2, 4, &backend);
    linear1.set_weight({0.6f, -0.3f, 0.4f, -0.7f, 0.2f, 0.5f, -0.6f, 0.1f});
    linear1.set_bias({0.0f, 0.0f, 0.0f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 1, &backend);
    linear2.set_weight({0.5f, -0.4f, 0.3f, 0.6f});
    linear2.set_bias({0.0f});

    ExplainerContext ctx({&linear1, &relu, &linear2});

    Tensor input(Shape({1, 2}), &backend, {1.0f, 0.0f});
    Tensor baseline(Shape({1, 2}), &backend, {0.0f, 0.0f});

    Tensor output = ctx.forward_pass(input);
    float confidence = 1.0f / (1.0f + std::exp(-output.data()[0]));  // sigmoid, for display only

    Saliency saliency;
    Attribution sal_attr = saliency.explain(ctx, input, /*target_index=*/0, &backend);

    IntegratedGradients ig;
    Attribution ig_attr = ig.explain(ctx, input, baseline, /*target_index=*/0, /*steps=*/200, &backend);
    float baseline_output = ctx.forward_pass(baseline).data()[0];

    CircuitGraph circuit = ctx.build_circuit_graph(input);

    // LRP conservation for the score card's trustworthiness tile: propagate relevance
    // backward through the same three modules the forward pass just used.
    LRPRuleConfig lrp_config;
    Tensor relevance_seed(Shape({1, 1}), &backend, {output.data()[0]});
    Tensor relevance_h2 = linear2.propagate_relevance(relevance_seed, lrp_config);
    Tensor relevance_h1 = relu.propagate_relevance(relevance_h2, lrp_config);
    Tensor relevance_input = linear1.propagate_relevance(relevance_h1, lrp_config);
    ConservationResult conservation = ComputeConservation(relevance_input, relevance_seed);

    VizWindow window("pulsatrix -- Explanation Dashboard Demo", 1280, 800);
    window.run([&]() {
        ImGui::SetNextWindowSize(ImVec2(1260, 780), ImGuiCond_FirstUseEver);
        ImGui::Begin("Explanation Dashboard");

        if (ImGui::CollapsingHeader("Feature Importance (Saliency)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginChild("bar_chart", ImVec2(0, 220));
            AttributionBarChart::Draw("Saliency", sal_attr, 2);
            ImGui::EndChild();
        }

        if (ImGui::CollapsingHeader("Waterfall (Integrated Gradients)", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginChild("waterfall", ImVec2(0, 220));
            AttributionWaterfallChart::Draw("Integrated Gradients", ig_attr, baseline_output);
            ImGui::EndChild();
        }

        if (ImGui::CollapsingHeader("Saliency Heatmap", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginChild("heatmap", ImVec2(0, 160));
            SaliencyHeatmapView::Draw("##saliency_heatmap", sal_attr);
            ImGui::EndChild();
        }

        if (ImGui::CollapsingHeader("Circuit Graph", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::BeginChild("circuit", ImVec2(0, 200));
            CircuitGraphView::Draw("##circuit", circuit);
            ImGui::EndChild();
        }

        if (ImGui::CollapsingHeader("Explanation Score Card", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Aggregate-initialized (not default-constructed then assigned): Attribution
            // holds a Tensor, which has no default constructor.
            ExplanationScoreCard::Input card_input{
                "Input: (1.0, 0.0)",
                confidence,
                sal_attr,
                conservation,
                false,  // Saliency is deterministic by construction -- no stability run needed
                StabilityResult{0.0f, true},
            };
            ExplanationScoreCard::Draw("score_card", card_input);
        }

        ImGui::End();
    });

    return 0;
}
