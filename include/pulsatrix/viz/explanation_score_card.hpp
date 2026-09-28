/** @file explanation_score_card.hpp
 *  @brief Aggregated per-prediction "what/why/how confident/how trustworthy" panel.
 *  @ingroup visualization
 */
#pragma once

#include <string>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/explainer_stability.hpp"
#include "pulsatrix/lrp_conservation.hpp"

namespace pulsatrix {

/**
 * @brief Shared axis/color-scale state across several ExplanationScoreCard instances shown
 *        together (small multiples -- hc_information_visualization.md SS6), so each card's
 *        attribution bar chart uses a comparable scale rather than independently
 *        auto-scaling and silently making cross-card comparison invalid.
 */
struct ScoreCardScaleContext {
    float max_abs_attribution = 0.0f;
};

/**
 * @brief Aggregates, per prediction, a 2x2 panel: top row is "what was shown, how confident
 *        was the model" (input label + ConfidenceMeter), bottom row is "how trustworthy is
 *        this explanation" (LRP conservation delta + explainer stability), visually
 *        separated by a rule (Gestalt proximity grouping -- these are two different
 *        semantic questions, not one undifferentiated block of numbers).
 */
class ExplanationScoreCard {
public:
    struct Input {
        /** @brief Short caption identifying the input (e.g. "Sample #3"). */
        std::string input_label;

        /** @brief Model output confidence, expected in [0, 1]. */
        float confidence;

        /** @brief The explanation to visualize (top-k feature-importance bars). */
        Attribution attribution;

        /** @brief LRP conservation delta for this prediction's relevance propagation. */
        ConservationResult conservation;

        /** @brief True iff stability was actually measured (skip for deterministic
         *         methods rather than running redundant repeats -- see
         *         ComputeAttributionStability's doc comment on known-deterministic
         *         explainers). When false, the card shows "0.0 (deterministic)". */
        bool has_stability = false;

        /** @brief Only meaningful when has_stability is true. */
        StabilityResult stability;
    };

    /**
     * @param title A unique ImGui window/child id for this card.
     * @param input The data to display.
     * @param shared_scale When non-null, this card's attribution bar chart shares (and
     *        updates) this scale instead of auto-scaling to just its own attribution.
     */
    static void Draw(const char* title, const Input& input, ScoreCardScaleContext* shared_scale = nullptr);
};

}  // namespace pulsatrix
