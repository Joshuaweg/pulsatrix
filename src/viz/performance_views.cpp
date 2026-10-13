// The reliability diagram and the per-slice error figure, and the readers of the documents they draw
// (KS-3).

#include "pulsatrix/viz/performance_views.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"
#include "document_io.hpp"
#include "svg_detail.hpp"

namespace pulsatrix {

using namespace document_io;
using namespace svg_detail;

namespace {

constexpr int64_t kMaxCount = int64_t{1} << 50;

// Tableau 10, as the other views use; one color per series.
constexpr const char* kSeriesColors[] = {"#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
                                         "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"};
constexpr const char* kWorseColor = "#e15759";
constexpr const char* kSliceColor = "#4e79a7";

bool In(double v, double lo, double hi) { return v >= lo && v <= hi; }

// A small tolerance for values a reader computed in floating point.
constexpr double kSlack = 1e-9;

}  // namespace

ClassificationReport ParseClassificationReport(std::string_view json) {
    DocReader r(json, "classification_report");
    const JsonValue& root = r.root();
    ClassificationReport rep;
    auto score = [&](const char* key, double lo, double hi) {
        const std::string p = std::string("/") + key;
        const double v = r.number_double(r.member(root, "", key), p);
        if (!In(v, lo, hi)) r.fail(p, "must be in [" + ValueText(lo) + ", " + ValueText(hi) + "]");
        return v;
    };
    rep.n = r.integer(r.member(root, "", "n"), "/n", 1, kMaxCount);
    rep.classes = r.integer(r.member(root, "", "classes"), "/classes", 2, 1 << 20);
    rep.accuracy = score("accuracy", 0, 1);
    rep.log_loss = score("log_loss", 0, 1e6);
    rep.brier = score("brier", 0, 2);
    rep.ece = score("ece", 0, 1);
    rep.mce = score("mce", 0, 1);
    rep.macro_f1 = score("macro_f1", 0, 1);

    const JsonValue::Array& bins = r.array(r.member(root, "", "reliability"), "/reliability");
    if (bins.empty()) r.fail("/reliability", "needs at least one bin");
    int64_t total = 0;
    for (size_t i = 0; i < bins.size(); ++i) {
        const std::string p = Index("/reliability", i);
        (void)r.object(bins[i], p);
        auto unit = [&](const char* key) {
            const double v = r.number_double(r.member(bins[i], p, key), p + "/" + key);
            if (!In(v, 0, 1)) r.fail(p + "/" + key, "must be in [0, 1]");
            return v;
        };
        ReliabilityBin b;
        b.lower = unit("lower");
        b.upper = unit("upper");
        b.count = r.integer(r.member(bins[i], p, "count"), p + "/count", 0, kMaxCount);
        b.confidence = unit("confidence");
        b.accuracy = unit("accuracy");
        if (b.upper < b.lower) r.fail(p + "/upper", "is below the bin's lower edge");
        if (i > 0 && b.lower < rep.bins.back().upper - kSlack) r.fail(p + "/lower", "overlaps the bin before it");
        total += b.count;
        if (total > kMaxCount) r.fail(p + "/count", "the counts are too large");
        rep.bins.push_back(b);
    }
    if (total != rep.n) r.fail("/reliability", "the counts sum to " + std::to_string(total) + ", not n = " + std::to_string(rep.n));

    const JsonValue::Array& confusion = r.array(r.member(root, "", "confusion"), "/confusion");
    if (static_cast<int64_t>(confusion.size()) != rep.classes) r.fail("/confusion", "needs one row per class");
    total = 0;
    for (size_t i = 0; i < confusion.size(); ++i) {
        const std::string p = Index("/confusion", i);
        rep.confusion.push_back(r.ints(confusion[i], p, 0, kMaxCount));
        if (static_cast<int64_t>(rep.confusion.back().size()) != rep.classes) r.fail(p, "needs one entry per class");
        for (int64_t v : rep.confusion.back()) {
            total += v;
            if (total > kMaxCount) r.fail(p, "the counts are too large");
        }
    }
    if (total != rep.n) r.fail("/confusion", "the counts sum to " + std::to_string(total) + ", not n = " + std::to_string(rep.n));

    const JsonValue::Array& per_class = r.array(r.member(root, "", "per_class"), "/per_class");
    if (static_cast<int64_t>(per_class.size()) != rep.classes) r.fail("/per_class", "needs one entry per class");
    for (size_t c = 0; c < per_class.size(); ++c) {
        const std::string p = Index("/per_class", c);
        (void)r.object(per_class[c], p);
        auto unit = [&](const char* key) {
            const double v = r.number_double(r.member(per_class[c], p, key), p + "/" + key);
            if (!In(v, 0, 1)) r.fail(p + "/" + key, "must be in [0, 1]");
            return v;
        };
        rep.precision.push_back(unit("precision"));
        rep.recall.push_back(unit("recall"));
        rep.f1.push_back(unit("f1"));
        rep.support.push_back(r.integer(r.member(per_class[c], p, "support"), p + "/support", 0, rep.n));
    }
    r.finish();
    return rep;
}

SliceErrorReport ParseSliceErrorReport(std::string_view json) {
    DocReader r(json, "slice_error");
    const JsonValue& root = r.root();
    SliceErrorReport rep;
    auto finite = [&](const JsonValue& object, const std::string& p, const char* key) {
        const double v = r.number_double(r.member(object, p, key), p + "/" + key);
        if (!std::isfinite(v)) r.fail(p + "/" + key, "must be finite");
        return v;
    };
    rep.n = r.integer(r.member(root, "", "n"), "/n", 1, kMaxCount);
    rep.error = finite(root, "", "error");
    rep.test = r.string(r.member(root, "", "test"), "/test");
    rep.alpha = finite(root, "", "alpha");
    if (!(rep.alpha > 0 && rep.alpha < 1)) r.fail("/alpha", "must be in (0, 1)");
    const JsonValue::Array& slices = r.array(r.member(root, "", "slices"), "/slices");
    if (slices.empty()) r.fail("/slices", "needs at least one slice");
    int64_t total = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
        const std::string p = Index("/slices", i);
        (void)r.object(slices[i], p);
        SliceError s;
        s.name = r.string(r.member(slices[i], p, "name"), p + "/name");
        s.count = r.integer(r.member(slices[i], p, "count"), p + "/count", 1, rep.n);
        s.error = finite(slices[i], p, "error");
        s.lower = finite(slices[i], p, "lower");
        s.upper = finite(slices[i], p, "upper");
        s.rest_error = finite(slices[i], p, "rest_error");
        s.p_value = finite(slices[i], p, "p_value");
        s.worse = r.boolean(r.member(slices[i], p, "worse"), p + "/worse");
        const double slack = kSlack * std::max(1.0, std::abs(s.error));
        if (!(s.lower <= s.error + slack && s.error <= s.upper + slack)) r.fail(p + "/error", "is outside its interval");
        if (!In(s.p_value, 0, 1)) r.fail(p + "/p_value", "must be in [0, 1]");
        total += s.count;
        if (total > rep.n) r.fail(p + "/count", "the counts sum to more than n = " + std::to_string(rep.n));
        rep.slices.push_back(std::move(s));
    }
    if (total != rep.n) r.fail("/slices", "the counts sum to " + std::to_string(total) + ", not n = " + std::to_string(rep.n));
    r.finish();
    return rep;
}

std::string RenderReliabilityDiagramSvg(const ClassificationReport& report, const SvgOptions& options) {
    return RenderReliabilityDiagramSvg(std::vector<ReliabilitySeries>{{"", report}}, options);
}

std::string RenderReliabilityDiagramSvg(const std::vector<ReliabilitySeries>& series, const SvgOptions& options) {
    CheckOptions(options);
    if (series.empty()) throw std::invalid_argument("RenderReliabilityDiagramSvg: needs at least one classifier");
    if (series.size() > 10) throw std::invalid_argument("RenderReliabilityDiagramSvg: draws at most 10 classifiers");
    // The domain, the same on both axes so the diagonal is at 45 degrees: from the lowest edge of a
    // bin that holds inputs (a classifier of K classes is never below 1/K confident), or the lowest
    // accuracy, down to a tenth.
    double lo = 1.0;
    double largest_share = 0.0;
    for (const ReliabilitySeries& s : series) {
        if (s.report.bins.empty()) throw std::invalid_argument("RenderReliabilityDiagramSvg: \"" + s.name + "\" has no reliability bins");
        if (s.report.n < 1) throw std::invalid_argument("RenderReliabilityDiagramSvg: \"" + s.name + "\" has no inputs");
        for (const ReliabilityBin& b : s.report.bins) {
            if (b.count == 0) continue;
            lo = std::min({lo, b.lower, b.accuracy});
            largest_share = std::max(largest_share, static_cast<double>(b.count) / static_cast<double>(s.report.n));
        }
    }
    lo = std::max(0.0, std::floor(lo * 10 + kSlack) / 10);
    if (lo >= 1.0) lo = 0.9;

    const bool named = series.size() > 1 || !series.front().name.empty();
    Figure f(options, named ? "Reliability diagram" : "Reliability diagram, n = " + std::to_string(series.front().report.n));
    const double fs = f.fs();
    double label_w = 0;
    for (double t : NiceTicks(lo, 1.0)) label_w = std::max(label_w, TextWidth(ValueText(t), fs * 0.85));
    std::vector<std::string> legend;
    double legend_w = 0;
    for (const ReliabilitySeries& s : series) {
        legend.push_back((s.name.empty() ? std::string() : s.name + ": ") + "ECE " + ValueText(s.report.ece));
        legend_w = std::max(legend_w, fs * 1.6 + TextWidth(legend.back(), fs * 0.9));
    }
    const double x0 = fs * 2.2 + label_w + fs * 0.6;
    const double side = std::max(fs * 8, std::min(f.width() - x0 - fs * 2.5 - legend_w, 400.0));
    const double y_top = f.top() + fs;
    const double y_bottom = y_top + side;
    auto X = [&](double v) { return x0 + (v - lo) / (1.0 - lo) * side; };
    auto Y = [&](double v) { return y_bottom - (std::max(v, lo) - lo) / (1.0 - lo) * side; };

    f.line("diagonal", X(lo), Y(lo), X(1), Y(1), kAxisColor, 1.0, "4 3");
    f.text("diagonal-label", X(1) - fs * 0.3, Y(1) + fs * 1.2, "calibrated", "end", kMutedColor,
           " font-size=\"" + Num(fs * 0.8) + "\" transform=\"rotate(-45 " + Num(X(1) - fs * 0.3) + " " + Num(Y(1) + fs * 1.2) + ")\"");
    const double r_max = fs * 0.55, r_min = fs * 0.18;
    for (size_t k = 0; k < series.size(); ++k) {
        const ReliabilitySeries& s = series[k];
        const char* color = kSeriesColors[k];
        std::string points;
        std::string marks;
        for (const ReliabilityBin& b : s.report.bins) {
            if (b.count == 0) continue;
            const double share = static_cast<double>(b.count) / static_cast<double>(s.report.n);
            const double radius = r_min + (r_max - r_min) * std::sqrt(share / largest_share);
            const double cx = X(b.confidence), cy = Y(b.accuracy);
            points += Num(cx) + "," + Num(cy) + " ";
            const std::string tip = (s.name.empty() ? std::string() : s.name + ", ") + "confidence (" + ValueText(b.lower) + ", " +
                                    ValueText(b.upper) + "]: " + std::to_string(b.count) + " inputs, mean confidence " +
                                    ValueText(b.confidence) + ", accuracy " + ValueText(b.accuracy);
            marks += "<circle class=\"bin s" + std::to_string(k) + "\" cx=\"" + Num(cx) + "\" cy=\"" + Num(cy) + "\" r=\"" + Num(radius) +
                     "\" fill=\"" + color + "\" fill-opacity=\"0.8\" stroke=\"#ffffff\" stroke-width=\"0.8\"><title>" + Escape(tip) +
                     "</title></circle>\n";
        }
        if (!points.empty()) {
            points.pop_back();
            f.body() += "<polyline class=\"curve s" + std::to_string(k) + "\" points=\"" + points + "\" fill=\"none\" stroke=\"" + color +
                        "\" stroke-width=\"1.6\"/>\n";
        }
        f.body() += marks;
    }
    f.y_axis(x0, y_top, y_bottom, lo, 1.0, "accuracy", label_w);
    f.x_axis(y_bottom, x0, X(1), lo, 1.0, "");

    // The share of inputs in each bin, one thin bar per classifier, under the x axis.
    const double strip_top = y_bottom + fs * 1.9;
    const double strip_h = fs * 3.0;
    const double strip_bottom = strip_top + strip_h;
    for (size_t k = 0; k < series.size(); ++k) {
        const ReliabilitySeries& s = series[k];
        for (const ReliabilityBin& b : s.report.bins) {
            if (b.count == 0 || b.upper <= lo) continue;
            const double share = static_cast<double>(b.count) / static_cast<double>(s.report.n);
            const double left = X(std::max(b.lower, lo)), right = X(b.upper);
            const double w = (right - left) / static_cast<double>(series.size());
            const double h = std::max(0.5, share / largest_share * strip_h);
            f.body() += "<rect class=\"share s" + std::to_string(k) + "\" x=\"" + Num(left + w * static_cast<double>(k)) + "\" y=\"" +
                        Num(strip_bottom - h) + "\" width=\"" + Num(std::max(w - 0.5, 0.5)) + "\" height=\"" + Num(h) + "\" fill=\"" +
                        kSeriesColors[k] + "\" fill-opacity=\"0.6\"><title>" +
                        Escape(ValueText(100 * share) + "% of the inputs") + "</title></rect>\n";
        }
    }
    f.line("axis", x0, strip_bottom, X(1), strip_bottom, kAxisColor);
    f.text("strip-label", x0 - fs * 0.5, strip_top + strip_h / 2 + fs * 0.3, "inputs", "end", kMutedColor,
           " font-size=\"" + Num(fs * 0.85) + "\"");
    f.text("axis-label", (x0 + X(1)) / 2, strip_bottom + fs * 1.4, "confidence (the largest probability)", "middle", kMutedColor);

    const double lx = X(1) + fs * 1.5;
    double right = X(1);
    for (size_t k = 0; k < series.size(); ++k) {
        const double ly = y_top + fs * 1.5 * static_cast<double>(k);
        f.body() += "<circle class=\"legend-swatch\" cx=\"" + Num(lx + fs * 0.4) + "\" cy=\"" + Num(ly + fs * 0.5) + "\" r=\"" +
                    Num(fs * 0.4) + "\" fill=\"" + kSeriesColors[k] + "\"/>\n";
        f.text("legend-label", lx + fs * 1.2, ly + fs * 0.85, legend[k], "start", kTextColor, " font-size=\"" + Num(fs * 0.9) + "\"");
        right = std::max(right, lx + fs * 1.2 + TextWidth(legend[k], fs * 0.9));
    }
    const double note_y = y_top + fs * 1.5 * static_cast<double>(series.size()) + fs * 0.6;
    f.text("legend-note", lx, note_y, "below the line: overconfident", "start", kMutedColor, " font-size=\"" + Num(fs * 0.8) + "\"");
    f.text("legend-note", lx, note_y + fs * 1.1, "above: underconfident", "start", kMutedColor, " font-size=\"" + Num(fs * 0.8) + "\"");
    right = std::max(right, lx + TextWidth("below the line: overconfident", fs * 0.8));
    return f.finish(strip_bottom + fs * 2.4, right + fs);
}

std::string RenderSliceErrorSvg(const SliceErrorReport& report, const SliceErrorViewOptions& view, const SvgOptions& options) {
    CheckOptions(options);
    if (report.slices.empty()) throw std::invalid_argument("RenderSliceErrorSvg: the report has no slices");
    if (view.max_slices < 1) throw std::invalid_argument("RenderSliceErrorSvg: max_slices must be at least 1");
    const bool has_reference = !std::isnan(view.reference);
    if (has_reference && !(std::isfinite(view.reference) && view.reference >= 0)) {
        throw std::invalid_argument("RenderSliceErrorSvg: the reference must be finite and not negative, or NaN");
    }
    const size_t shown = std::min(report.slices.size(), static_cast<size_t>(view.max_slices));

    double lo = 0.0, hi = 0.0;
    auto widen = [&](double v) {
        if (std::isfinite(v)) lo = std::min(lo, v), hi = std::max(hi, v);
    };
    widen(report.error);
    if (has_reference) widen(view.reference);
    for (size_t i = 0; i < shown; ++i) {
        widen(report.slices[i].lower);
        widen(report.slices[i].upper);
    }
    if (!(hi > lo)) hi = lo + 1;
    hi += 0.04 * (hi - lo);

    Figure f(options, "Error by slice");
    const double fs = f.fs();
    std::vector<std::string> labels;
    double label_w = 0;
    for (size_t i = 0; i < shown; ++i) {
        const SliceError& s = report.slices[i];
        labels.push_back(Truncate(s.name, 28) + " (" + std::to_string(s.count) + ")");
        label_w = std::max(label_w, TextWidth(labels.back(), fs * 0.9));
    }
    const double p_w = TextWidth("p = 0.000e-00 *", fs * 0.85);
    const double x0 = fs + label_w + fs * 0.8;
    const double x1 = std::max(x0 + fs * 10, f.width() - fs - p_w - fs * 0.8);
    auto X = [&](double v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); };
    const double row = fs * 1.6;
    const double y_top = f.top() + fs * 1.6;
    const double y_bottom = y_top + row * static_cast<double>(shown);

    auto marker_line = [&](const char* cls, double v, const char* dash, const std::string& label, double label_y) {
        f.line(cls, X(v), y_top - fs * 0.4, X(v), y_bottom, kAxisColor, 1.0, dash);
        f.text(cls, X(v), label_y, label, "middle", kMutedColor, " font-size=\"" + Num(fs * 0.8) + "\"");
    };
    marker_line("overall", report.error, "4 3", "all: " + ValueText(report.error), y_top - fs * 0.7);
    if (has_reference) {
        marker_line("reference", view.reference, "1 3", view.reference_label + ": " + ValueText(view.reference), y_top - fs * 1.7);
    }
    for (size_t i = 0; i < shown; ++i) {
        const SliceError& s = report.slices[i];
        const double cy = y_top + row * (static_cast<double>(i) + 0.5);
        const char* color = s.worse ? kWorseColor : kSliceColor;
        f.text("slice-label", x0 - fs * 0.8, cy + fs * 0.3, labels[i], "end", kTextColor, " font-size=\"" + Num(fs * 0.9) + "\"");
        f.line("interval", X(s.lower), cy, X(s.upper), cy, color, 1.6);
        f.line("interval-end", X(s.lower), cy - fs * 0.3, X(s.lower), cy + fs * 0.3, color, 1.2);
        f.line("interval-end", X(s.upper), cy - fs * 0.3, X(s.upper), cy + fs * 0.3, color, 1.2);
        const std::string tip = s.name + ": " + view.metric + " " + ValueText(s.error) + " (" + ValueText(s.lower) + " to " +
                                ValueText(s.upper) + ") on " + std::to_string(s.count) + " inputs; the rest " + ValueText(s.rest_error) +
                                "; p = " + ValueText(s.p_value) + (s.worse ? ", worse" : "");
        f.body() += std::string("<circle class=\"slice") + (s.worse ? " worse" : "") + "\" cx=\"" + Num(X(s.error)) + "\" cy=\"" + Num(cy) +
                    "\" r=\"" + Num(fs * 0.38) + "\" fill=\"" + color + "\"><title>" + Escape(tip) + "</title></circle>\n";
        f.text("p-value", x1 + fs * 0.8, cy + fs * 0.3, "p = " + ValueText(s.p_value) + (s.worse ? " *" : ""), "start",
               s.worse ? kWorseColor : kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
    }
    f.x_axis(y_bottom, x0, x1, lo, hi, view.metric);
    double bottom = y_bottom + fs * 4.2;
    auto note = [&](const std::string& s) {
        f.text("slice-note", x0, bottom, s, "start", kMutedColor, " font-size=\"" + Num(fs * 0.85) + "\"");
        bottom += fs * 1.2;
    };
    if (report.slices.size() > shown) note("+" + std::to_string(report.slices.size() - shown) + " more slices not shown");
    const std::string level = ValueText(100 * (1 - report.alpha)) + "%";
    const std::string test = report.test == "fisher_exact" ? "Fisher's exact test"
                             : report.test == "permutation" ? "a permutation test"
                                                            : report.test;
    note("Bars: " + level + " intervals. Red, *: worse than the rest (" + test + ", Bonferroni at " + ValueText(report.alpha) + ")");
    return f.finish(bottom);
}

}  // namespace pulsatrix
