/** @file colormap.hpp
 *  @brief Pure, ImGui/ImPlot-independent colormap and normalization functions for viz.
 *  @ingroup visualization
 */
#pragma once

namespace pulsatrix {

/** @brief An RGB color, each channel in [0, 1]. */
struct RgbColor {
    float r;
    float g;
    float b;
};

/**
 * @brief Normalizes value into [0, 1] against a known maximum magnitude, for unsigned
 *        (magnitude-only) quantities such as saliency intensity or ablation effect.
 * @param value The raw value to normalize. Values outside [0, max_abs] are clamped.
 * @param max_abs The value that should map to 1.0. If 0 (a degenerate all-zero series),
 *        returns 0.0 rather than dividing by zero.
 */
[[nodiscard]] float NormalizeUnsigned(float value, float max_abs);

/**
 * @brief Normalizes value into [-1, 1] against a known maximum absolute magnitude, for
 *        signed quantities such as attribution direction (positive/negative contribution).
 * @param value The raw value to normalize. Values outside [-max_abs, max_abs] are clamped.
 * @param max_abs The absolute value that should map to +/-1.0. If 0, returns 0.0 rather
 *        than dividing by zero.
 */
[[nodiscard]] float NormalizeSigned(float value, float max_abs);

/**
 * @brief Viridis colormap -- perceptually uniform, colorblind-safe -- for sequential/unsigned
 *        magnitude (hc_information_visualization.md SS4: "Recommended Color Systems").
 * @param normalized_value Position along the colormap, expected in [0, 1]; values outside
 *        this range are clamped to the nearest end anchor.
 * @note Deliberately never a rainbow/jet map -- see the design doc's SS4 "Rainbow (Jet)
 *       Colormap Problem". Approximates matplotlib's Viridis via a small set of anchor
 *       colors with linear interpolation between them; exact per-channel fidelity against
 *       ImPlot's own built-in ImPlotColormap_Viridis is an open verification item (plan
 *       Open Risk 4), not assumed here.
 */
[[nodiscard]] RgbColor ViridisColormap(float normalized_value);

/**
 * @brief Blue-White-Red diverging colormap for signed attribution (positive/negative
 *        contribution) -- hue encodes direction only, never magnitude
 *        (hc_information_visualization.md SS1's Cleveland-McGill rule: color hue is
 *        categorical/directional, magnitude must be position or length).
 * @param signed_normalized_value Position along the colormap, expected in [-1, 1]; values
 *        outside this range are clamped to the nearest end anchor.
 */
[[nodiscard]] RgbColor DivergingColormap(float signed_normalized_value);

}  // namespace pulsatrix
