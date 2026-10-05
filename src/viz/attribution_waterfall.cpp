#include "pulsatrix/viz/attribution_waterfall.hpp"

#include <algorithm>
#include <vector>

#include <imgui.h>
#include <implot.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/viz/colormap.hpp"
#include "pulsatrix/viz/plot_data.hpp"

namespace pulsatrix {

void AttributionWaterfallChart::Draw(const char* title, const Attribution& attr, float baseline_value) {
    std::vector<WaterfallStep> steps = ToWaterfallSteps(attr, baseline_value);
    if (steps.empty()) {
        return;
    }
    // Floating bars (ToWaterfallBars), not stacked bar groups: ImPlot stacks positive and
    // negative segments separately from zero, which mis-draws any cascade whose running total
    // is below zero (e.g. a negative baseline or a negative logit being explained).
    std::vector<WaterfallBar> bars = ToWaterfallBars(steps, baseline_value);

    int n = static_cast<int>(steps.size());
    std::vector<double> positions(static_cast<size_t>(n));
    std::vector<const char*> label_ptrs(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        positions[static_cast<size_t>(i)] = static_cast<double>(i);
        label_ptrs[static_cast<size_t>(i)] = steps[static_cast<size_t>(i)].label.c_str();
    }

    const RgbColor pos = DivergingColormap(1.0f);
    const RgbColor neg = DivergingColormap(-1.0f);
    constexpr double kHalfWidth = 0.335;

    // Explicit padded limits: AutoFit would put the cascade's extreme bar (and the baseline,
    // when it is the extreme) exactly on the plot border.
    float lo = baseline_value;
    float hi = baseline_value;
    for (const WaterfallBar& bar : bars) {
        lo = std::min(lo, bar.bottom);
        hi = std::max(hi, bar.top);
    }
    double pad = 0.06 * std::max(static_cast<double>(hi - lo), 1e-6);

    if (ImPlot::BeginPlot(title, ImVec2(-1, 0))) {
        ImPlot::SetupAxes(nullptr, "Value", 0, 0);
        ImPlot::SetupAxesLimits(-0.5, static_cast<double>(n) - 0.5, static_cast<double>(lo) - pad,
                                static_cast<double>(hi) + pad, ImPlotCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_X1, positions.data(), n, label_ptrs.data());

        // Reference line at the baseline the cascade starts from.
        double ref_x[2] = {-0.5, static_cast<double>(n) - 0.5};
        double ref_y[2] = {static_cast<double>(baseline_value), static_cast<double>(baseline_value)};
        ImPlot::SetNextLineStyle(ImVec4(0.6f, 0.6f, 0.6f, 0.8f), 1.0f);
        ImPlot::PlotLine("Baseline", ref_x, ref_y, 2);

        for (int i = 0; i < n; ++i) {
            const WaterfallBar& bar = bars[static_cast<size_t>(i)];
            double xs[2] = {i - kHalfWidth, i + kHalfWidth};
            double lo[2] = {static_cast<double>(bar.bottom), static_cast<double>(bar.bottom)};
            double hi[2] = {static_cast<double>(bar.top), static_cast<double>(bar.top)};
            const RgbColor& c = bar.increase ? pos : neg;
            ImPlot::SetNextFillStyle(ImVec4(c.r, c.g, c.b, 1.0f));
            ImPlot::PlotShaded(bar.increase ? "Increase" : "Decrease", xs, lo, hi, 2);
        }
        ImPlot::EndPlot();
    }
}

void AttributionWaterfallChart::Draw(const char* title, const AttributionDocument& doc, float baseline_value) {
    static CPUBackend backend;
    Draw(title, ToAttribution(doc, &backend), baseline_value);
}

}  // namespace pulsatrix
