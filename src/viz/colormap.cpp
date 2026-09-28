#include "pulsatrix/viz/colormap.hpp"

#include <algorithm>
#include <array>

namespace pulsatrix {
namespace {

struct ColormapAnchor {
    float position;
    RgbColor color;
};

RgbColor Lerp(const RgbColor& a, const RgbColor& b, float t) {
    return RgbColor{a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t};
}

// 9-anchor approximation of matplotlib's Viridis, linearly interpolated between anchors.
// A full 256-entry lookup table is unnecessary for chart-quality rendering; this preserves
// Viridis's defining property (monotonically increasing perceived luminance) with far less
// data than the reference table.
template <size_t N>
RgbColor SampleAnchors(const std::array<ColormapAnchor, N>& anchors, float t) {
    t = std::clamp(t, anchors.front().position, anchors.back().position);
    for (size_t i = 0; i + 1 < anchors.size(); ++i) {
        const ColormapAnchor& lo = anchors[i];
        const ColormapAnchor& hi = anchors[i + 1];
        if (t >= lo.position && t <= hi.position) {
            float span = hi.position - lo.position;
            float local_t = span > 0.0f ? (t - lo.position) / span : 0.0f;
            return Lerp(lo.color, hi.color, local_t);
        }
    }
    return anchors.back().color;
}

}  // namespace

float NormalizeUnsigned(float value, float max_abs) {
    if (max_abs <= 0.0f) return 0.0f;
    return std::clamp(value / max_abs, 0.0f, 1.0f);
}

float NormalizeSigned(float value, float max_abs) {
    if (max_abs <= 0.0f) return 0.0f;
    return std::clamp(value / max_abs, -1.0f, 1.0f);
}

RgbColor ViridisColormap(float normalized_value) {
    static const std::array<ColormapAnchor, 9> kAnchors = {{
        {0.00f, {0.267f, 0.004f, 0.329f}},
        {0.13f, {0.282f, 0.157f, 0.471f}},
        {0.25f, {0.243f, 0.290f, 0.537f}},
        {0.38f, {0.193f, 0.408f, 0.557f}},
        {0.50f, {0.149f, 0.510f, 0.557f}},
        {0.63f, {0.122f, 0.620f, 0.537f}},
        {0.75f, {0.209f, 0.718f, 0.475f}},
        {0.88f, {0.427f, 0.804f, 0.349f}},
        {1.00f, {0.993f, 0.906f, 0.145f}},
    }};
    return SampleAnchors(kAnchors, normalized_value);
}

RgbColor DivergingColormap(float signed_normalized_value) {
    // Moreland-style Blue-White-Red diverging map (hc_information_visualization.md SS4):
    // perceptually balanced, avoids pure white's washout at the midpoint.
    static const std::array<ColormapAnchor, 3> kAnchors = {{
        {-1.0f, {0.230f, 0.299f, 0.754f}},
        {0.0f, {0.865f, 0.865f, 0.865f}},
        {1.0f, {0.706f, 0.016f, 0.150f}},
    }};
    return SampleAnchors(kAnchors, signed_normalized_value);
}

}  // namespace pulsatrix
