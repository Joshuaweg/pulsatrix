/** @file confidence_meter.hpp
 *  @brief Length-encoded confidence/uncertainty meter with an always-visible numeric label.
 *  @ingroup visualization
 */
#pragma once

namespace pulsatrix {

/**
 * @brief Draws a filled bar whose fill length encodes confidence (position/length, never
 *        color alone -- hc_information_visualization.md SS1/SS5: "confidence bar/meter...
 *        fastest to process"). The exact percentage is always rendered as text alongside
 *        the fill -- never a color-only encoding (also satisfies WCAG: a colorblind user
 *        must be able to read the value without relying on hue).
 */
class ConfidenceMeter {
public:
    /**
     * @param label A short caption drawn above the meter (e.g. "Confidence").
     * @param confidence_0_to_1 Confidence value; clamped to [0, 1] before display.
     */
    static void Draw(const char* label, float confidence_0_to_1);
};

}  // namespace pulsatrix
