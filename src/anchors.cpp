// XAI-4: Anchors (Ribeiro, Singh and Guestrin 2018), with KL-LUCB as in the reference
// implementation (anchor_base.py of marcotcr/anchor, and alibi's AnchorBaseBeam).

#include "pulsatrix/anchors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <tuple>
#include <numeric>
#include <stdexcept>

#include "portable_random.hpp"

namespace pulsatrix {

namespace {

// numpy's default percentile: linear interpolation between the closest ranks.
double Percentile(std::vector<float> values, double percent) {
    std::sort(values.begin(), values.end());
    const double rank = percent / 100.0 * static_cast<double>(values.size() - 1);
    const auto lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = std::min(lo + 1, values.size() - 1);
    const double frac = rank - static_cast<double>(lo);
    return static_cast<double>(values[lo]) + frac * (static_cast<double>(values[hi]) - values[lo]);
}

// The reference code's beta for KL-LUCB at round t over n arms.
double Beta(int64_t arms, int64_t t, double delta) {
    const double temp = std::log(405.5 * static_cast<double>(arms) * std::pow(static_cast<double>(t), 1.1) / delta);
    return temp + std::log(temp);
}

int64_t ArgMax(const std::vector<float>& v) {
    return static_cast<int64_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

struct Candidate {
    std::vector<int64_t> features;  // indices of the chosen predicates, in the order added
    double coverage = 0;
    int64_t positives = 0;
    int64_t samples = 0;

    [[nodiscard]] double mean() const { return samples > 0 ? static_cast<double>(positives) / static_cast<double>(samples) : 0; }
};

class Search {
public:
    Search(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input, const std::vector<Tensor>& background,
           const AnchorOptions& options)
        : predict_(predict), input_(input), options_(options), rng_{options.seed} {
        x_ = input.to_host_vector();
        for (const Tensor& b : background) rows_.push_back(b.to_host_vector());
        target_ = ArgMax(predict_(input).to_host_vector());
        ++calls_;
        const size_t d = x_.size();
        columns_.resize(d);
        for (size_t f = 0; f < d; ++f) {
            for (const auto& r : rows_) columns_[f].push_back(r[f]);
            std::vector<double> edges;
            for (double p : options.percentiles) {
                const double e = Percentile(columns_[f], p);
                if (edges.empty() || e > edges.back()) edges.push_back(e);
            }
            // The reference implementation's predicates for an ordinal feature: every half-line at a
            // percentile edge that contains the input, x > e below it and x <= e above it.
            for (double e : edges) {
                AnchorPredicate pred;
                pred.feature = static_cast<int64_t>(f);
                (static_cast<double>(x_[f]) > e ? pred.lower : pred.upper) = e;
                predicates_.push_back(pred);
            }
        }
    }

    int64_t target() const { return target_; }
    int64_t calls() const { return calls_; }
    const std::vector<AnchorPredicate>& predicates() const { return predicates_; }
    size_t features() const { return x_.size(); }
    int64_t candidates() const { return static_cast<int64_t>(predicates_.size()); }

    // The candidate's predicates merged into one interval per feature, in the order added.
    std::vector<AnchorPredicate> Merged(const std::vector<int64_t>& chosen) const {
        std::vector<AnchorPredicate> out;
        for (int64_t i : chosen) {
            const AnchorPredicate& p = predicates_[static_cast<size_t>(i)];
            auto same = std::find_if(out.begin(), out.end(), [&](const AnchorPredicate& q) { return q.feature == p.feature; });
            if (same == out.end()) {
                out.push_back(p);
            } else {
                same->lower = std::max(same->lower, p.lower);
                same->upper = std::min(same->upper, p.upper);
            }
        }
        return out;
    }

    double Coverage(const std::vector<int64_t>& chosen) const {
        const std::vector<AnchorPredicate> merged = Merged(chosen);
        int64_t covered = 0;
        for (const auto& r : rows_) {
            bool all = true;
            for (const AnchorPredicate& p : merged) all = all && p.holds(r[static_cast<size_t>(p.feature)]);
            covered += all ? 1 : 0;
        }
        return static_cast<double>(covered) / static_cast<double>(rows_.size());
    }

    // Background values of a feature inside an interval; the input's own value when there are none.
    const std::vector<float>& Values(const AnchorPredicate& p) {
        const auto key = std::make_tuple(p.feature, p.lower, p.upper);
        auto it = values_.find(key);
        if (it == values_.end()) {
            std::vector<float> v;
            for (float x : columns_[static_cast<size_t>(p.feature)]) {
                if (p.holds(x)) v.push_back(x);
            }
            if (v.empty()) v.push_back(x_[static_cast<size_t>(p.feature)]);
            it = values_.emplace(key, std::move(v)).first;
        }
        return it->second;
    }

    // Draws @p count inputs satisfying the candidate and counts those that keep the class.
    void Sample(Candidate& c, int64_t count) {
        const std::vector<AnchorPredicate> merged = Merged(c.features);
        for (int64_t s = 0; s < count; ++s) {
            std::vector<float> z = rows_[rng_.next() % rows_.size()];
            for (const AnchorPredicate& p : merged) {
                const std::vector<float>& in = Values(p);
                z[static_cast<size_t>(p.feature)] = in[rng_.next() % in.size()];
            }
            const Tensor out = predict_(Tensor(input_.shape(), input_.backend(), z, input_.device()));
            ++calls_;
            c.positives += ArgMax(out.to_host_vector()) == target_ ? 1 : 0;
            ++c.samples;
        }
    }

    // KL-LUCB: the indices of the (up to) `top` candidates with the highest precision.
    std::vector<size_t> Best(std::vector<Candidate>& cands, size_t top) {
        for (Candidate& c : cands) {
            if (c.samples == 0) Sample(c, options_.batch_size);
        }
        const auto arms = static_cast<int64_t>(cands.size());
        std::vector<size_t> order(cands.size());
        for (int64_t t = 1;; ++t) {
            std::iota(order.begin(), order.end(), size_t{0});
            std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return cands[a].mean() > cands[b].mean(); });
            if (cands.size() <= top) break;
            const double beta = Beta(arms, t, options_.delta);
            size_t lowest = order[0], highest = order[top];
            double lb = 2, ub = -1;
            for (size_t i = 0; i < top; ++i) {
                const Candidate& c = cands[order[i]];
                const double b = KlLowerBound(c.mean(), beta / static_cast<double>(c.samples));
                if (b < lb) lb = b, lowest = order[i];
            }
            for (size_t i = top; i < order.size(); ++i) {
                const Candidate& c = cands[order[i]];
                const double b = KlUpperBound(c.mean(), beta / static_cast<double>(c.samples));
                if (b > ub) ub = b, highest = order[i];
            }
            if (ub - lb <= options_.epsilon) break;
            if (cands[lowest].samples >= options_.max_samples_per_candidate &&
                cands[highest].samples >= options_.max_samples_per_candidate) {
                break;
            }
            if (cands[lowest].samples < options_.max_samples_per_candidate) Sample(cands[lowest], options_.batch_size);
            if (cands[highest].samples < options_.max_samples_per_candidate) Sample(cands[highest], options_.batch_size);
        }
        order.resize(std::min(top, order.size()));
        return order;
    }

    // Samples until the candidate's precision is decided either way; true if it clears the threshold.
    bool Verify(Candidate& c, double& lower, double& upper) {
        const double tau = options_.precision_threshold, stop = options_.epsilon_stop;
        const double beta =
            std::log(1.0 / (options_.delta / (1.0 + static_cast<double>(options_.beam_size - 1) * static_cast<double>(candidates()))));
        for (;;) {
            const double mean = c.mean(), level = beta / static_cast<double>(c.samples);
            lower = KlLowerBound(mean, level);
            upper = KlUpperBound(mean, level);
            const bool undecided = (mean >= tau && lower < tau - stop) || (mean < tau && upper >= tau + stop);
            if (!undecided || c.samples >= options_.max_samples_per_candidate) break;
            Sample(c, options_.batch_size);
        }
        return c.mean() >= tau && lower > tau - stop;
    }

private:
    const std::function<Tensor(const Tensor&)>& predict_;
    const Tensor& input_;
    const AnchorOptions& options_;
    PortableRng rng_;
    std::vector<float> x_;
    std::vector<std::vector<float>> rows_;
    std::vector<std::vector<float>> columns_;
    std::vector<AnchorPredicate> predicates_;  // every half-line that contains the input
    std::map<std::tuple<int64_t, double, double>, std::vector<float>> values_;
    int64_t target_ = 0;
    int64_t calls_ = 0;
};

bool SameIntervals(const std::vector<AnchorPredicate>& a, const std::vector<AnchorPredicate>& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].feature != b[i].feature || a[i].lower != b[i].lower || a[i].upper != b[i].upper) return false;
    }
    return true;
}

Anchor Make(const Search& search, const Candidate& c, double lower, double upper, bool meets) {
    Anchor a;
    a.predicted_class = search.target();
    a.predicates = search.Merged(c.features);
    a.precision = c.mean();
    a.precision_lower = lower;
    a.precision_upper = upper;
    a.samples = c.samples;
    a.coverage = c.coverage;
    a.meets_threshold = meets;
    return a;
}

std::string Fmt(double v, const char* format = "%.4g") {
    char buf[40];
    std::snprintf(buf, sizeof buf, format, v);
    return buf;
}

std::string FeatureName(int64_t f, const std::vector<std::string>& names) {
    if (f >= 0 && static_cast<size_t>(f) < names.size() && !names[static_cast<size_t>(f)].empty()) {
        return names[static_cast<size_t>(f)];
    }
    return "x[" + std::to_string(f) + "]";
}

JsonValue Bound(double v) { return std::isinf(v) ? JsonValue(nullptr) : JsonValue(v); }

}  // namespace

double BernoulliKl(double p, double q) {
    p = std::min(0.9999999999999999, std::max(0.0000001, p));
    q = std::min(0.9999999999999999, std::max(0.0000001, q));
    return p * std::log(p / q) + (1 - p) * std::log((1 - p) / (1 - q));
}

double KlUpperBound(double p, double level) {
    double lm = p, um = std::min(1.0, p + std::sqrt(level / 2.0));
    for (int j = 1; j < 17; ++j) {
        const double qm = (um + lm) / 2.0;
        (BernoulliKl(p, qm) > level ? um : lm) = qm;
    }
    return um;
}

double KlLowerBound(double p, double level) {
    double um = p, lm = std::max(0.0, p - std::sqrt(level / 2.0));
    for (int j = 1; j < 17; ++j) {
        const double qm = (um + lm) / 2.0;
        (BernoulliKl(p, qm) > level ? lm : um) = qm;
    }
    return lm;
}

bool Anchor::applies_to(const std::vector<float>& x) const {
    return std::all_of(predicates.begin(), predicates.end(), [&](const AnchorPredicate& p) {
        if (p.feature < 0 || static_cast<size_t>(p.feature) >= x.size()) {
            throw std::invalid_argument("Anchor::applies_to: the input has no feature " + std::to_string(p.feature));
        }
        return p.holds(x[static_cast<size_t>(p.feature)]);
    });
}

Anchor FindAnchor(const std::function<Tensor(const Tensor&)>& predict, const Tensor& input, const std::vector<Tensor>& background,
                  const AnchorOptions& options) {
    if (background.empty()) throw std::invalid_argument("FindAnchor: needs background inputs");
    for (const Tensor& b : background) {
        if (b.numel() != input.numel()) throw std::invalid_argument("FindAnchor: a background input isn't the input's size");
    }
    if (!(options.precision_threshold > 0 && options.precision_threshold <= 1) || !(options.delta > 0 && options.delta < 1) ||
        options.epsilon < 0 || options.epsilon_stop < 0 || options.beam_size < 1 || options.batch_size < 1 ||
        options.max_samples_per_candidate < options.batch_size || options.max_anchor_size < 0) {
        throw std::invalid_argument("FindAnchor: an option is out of range");
    }
    for (size_t i = 0; i < options.percentiles.size(); ++i) {
        const double p = options.percentiles[i];
        if (!(p > 0 && p < 100) || (i > 0 && p <= options.percentiles[i - 1])) {
            throw std::invalid_argument("FindAnchor: percentiles must increase within (0, 100)");
        }
    }

    Search search(predict, input, background, options);
    double lower = 0, upper = 1;

    // The empty anchor: the model may give this class almost everywhere.
    Candidate empty;
    empty.coverage = 1.0;
    search.Sample(empty, options.batch_size);
    if (search.Verify(empty, lower, upper)) {
        Anchor a = Make(search, empty, lower, upper, true);
        a.model_calls = search.calls();
        return a;
    }

    const int64_t d = search.candidates();
    const int64_t max_size = options.max_anchor_size > 0 ? std::min(options.max_anchor_size, d) : d;
    std::vector<Candidate> beam{empty};
    bool found = false;
    Anchor best, fallback = Make(search, empty, lower, upper, false);
    for (int64_t size = 1; size <= max_size; ++size) {
        std::vector<Candidate> cands;
        std::map<std::vector<int64_t>, bool> seen;
        for (const Candidate& b : beam) {
            for (int64_t f = 0; f < d; ++f) {
                if (std::find(b.features.begin(), b.features.end(), f) != b.features.end()) continue;
                Candidate c;
                c.features = b.features;
                c.features.push_back(f);
                std::vector<int64_t> key = c.features;
                std::sort(key.begin(), key.end());
                if (seen[key]) continue;
                seen[key] = true;
                // A predicate the others already imply (x > e beside x > e' with e' > e) adds nothing.
                if (search.Merged(c.features).size() == search.Merged(b.features).size() &&
                    SameIntervals(search.Merged(c.features), search.Merged(b.features))) {
                    continue;
                }
                c.coverage = search.Coverage(c.features);
                // Adding a predicate never raises coverage, so one below the best can't win.
                if (found && c.coverage <= best.coverage) continue;
                cands.push_back(std::move(c));
            }
        }
        if (cands.empty()) break;
        const std::vector<size_t> top = search.Best(cands, static_cast<size_t>(options.beam_size));
        std::vector<Candidate> next;
        for (size_t i : top) {
            Candidate& c = cands[i];
            const bool valid = search.Verify(c, lower, upper);
            if (valid && (!found || c.coverage > best.coverage)) {
                best = Make(search, c, lower, upper, true);
                found = true;
            }
            if (!found && c.mean() > fallback.precision) fallback = Make(search, c, lower, upper, false);
            next.push_back(c);
        }
        beam = std::move(next);
    }
    Anchor result = found ? best : fallback;
    result.model_calls = search.calls();
    return result;
}

std::string DescribeAnchor(const Anchor& anchor, const std::vector<std::string>& feature_names) {
    std::string rule;
    for (const AnchorPredicate& p : anchor.predicates) {
        const std::string name = FeatureName(p.feature, feature_names);
        std::string text;
        if (!std::isinf(p.lower) && !std::isinf(p.upper)) {
            text = Fmt(p.lower) + " < " + name + " <= " + Fmt(p.upper);
        } else if (!std::isinf(p.lower)) {
            text = name + " > " + Fmt(p.lower);
        } else if (!std::isinf(p.upper)) {
            text = name + " <= " + Fmt(p.upper);
        } else {
            text = name + " is anything";
        }
        rule += (rule.empty() ? "" : " AND ") + text;
    }
    return (rule.empty() ? std::string("ALWAYS") : "IF " + rule) + " THEN class " + std::to_string(anchor.predicted_class) +
           " (precision " + Fmt(anchor.precision, "%.3g") + ", coverage " + Fmt(anchor.coverage, "%.3g") +
           (anchor.meets_threshold ? "" : "; below the precision threshold") + ")";
}

JsonValue ToJson(const Anchor& anchor, const std::vector<std::string>& feature_names) {
    JsonValue::Array predicates;
    for (const AnchorPredicate& p : anchor.predicates) {
        predicates.push_back(JsonValue(JsonValue::Object{
            {"feature", JsonValue(p.feature)},
            {"name", FeatureName(p.feature, feature_names)},
            {"lower", Bound(p.lower)},
            {"upper", Bound(p.upper)},
        }));
    }
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.anchor.v1"},
        {"predicted_class", JsonValue(anchor.predicted_class)},
        {"predicates", JsonValue(std::move(predicates))},
        {"precision", JsonValue(anchor.precision)},
        {"precision_lower", JsonValue(anchor.precision_lower)},
        {"precision_upper", JsonValue(anchor.precision_upper)},
        {"samples", JsonValue(anchor.samples)},
        {"coverage", JsonValue(anchor.coverage)},
        {"meets_threshold", anchor.meets_threshold},
        {"model_calls", JsonValue(anchor.model_calls)},
        {"rule", DescribeAnchor(anchor, feature_names)},
    });
}

}  // namespace pulsatrix
