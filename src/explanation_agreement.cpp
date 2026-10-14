// Teacher-student explanation agreement (KD-2): per-input Spearman, top-k overlap and cosine, CKA
// across inputs, and a permutation test against the student's maps of other inputs.

#include "pulsatrix/explanation_agreement.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "portable_random.hpp"
#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::vector<double> Values(const Attribution& a, bool magnitudes) {
    const std::vector<float> v = a.values.to_host_vector();
    std::vector<double> out(v.size());
    for (size_t k = 0; k < v.size(); ++k) out[k] = magnitudes ? std::abs(static_cast<double>(v[k])) : v[k];
    return out;
}

// Ranks with ties given their average, then centered and scaled to unit norm, so the Spearman
// correlation of two maps is one dot product. All zeros when the map is constant.
std::vector<double> StandardizedRanks(const std::vector<double>& v) {
    const size_t n = v.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> rank(n);
    for (size_t i = 0; i < n;) {
        size_t j = i;
        while (j + 1 < n && v[order[j + 1]] == v[order[i]]) ++j;
        const double average = (static_cast<double>(i) + static_cast<double>(j)) / 2.0 + 1.0;
        for (size_t k = i; k <= j; ++k) rank[order[k]] = average;
        i = j + 1;
    }
    const double mean = (static_cast<double>(n) + 1.0) / 2.0;
    double norm = 0;
    for (double& r : rank) r -= mean, norm += r * r;
    norm = std::sqrt(norm);
    for (double& r : rank) r = norm > 1e-12 ? r / norm : 0.0;
    return rank;
}

double Dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t k = 0; k < a.size(); ++k) s += a[k] * b[k];
    return s;
}

// The k largest entries' indices, ties to the lower index.
std::vector<size_t> TopK(const std::vector<double>& v, size_t k) {
    std::vector<size_t> order(v.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return v[a] > v[b]; });
    order.resize(k);
    std::sort(order.begin(), order.end());
    return order;
}

JsonValue NumberOrNull(double v) { return std::isfinite(v) ? JsonValue(v) : JsonValue(); }

}  // namespace

AgreementReport CompareExplanations(const std::vector<Attribution>& teacher, const std::vector<Attribution>& student,
                                    const AgreementOptions& options) {
    if (teacher.empty() || teacher.size() != student.size()) {
        throw std::invalid_argument("CompareExplanations: the teacher's and student's attributions must be non-empty lists of the same length");
    }
    if (!(options.top_fraction > 0 && options.top_fraction <= 1)) {
        throw std::invalid_argument("CompareExplanations: top_fraction must be in (0, 1]");
    }
    if (options.permutations < 0) throw std::invalid_argument("CompareExplanations: permutations must not be negative");
    const size_t n = teacher.size();
    std::vector<std::vector<double>> t(n), s(n);
    for (size_t i = 0; i < n; ++i) {
        t[i] = Values(teacher[i], options.magnitudes);
        s[i] = Values(student[i], options.magnitudes);
    }
    const size_t features = t[0].size();
    if (features < 2) throw std::invalid_argument("CompareExplanations: the attributions need at least 2 features");
    for (size_t i = 0; i < n; ++i) {
        if (t[i].size() != features || s[i].size() != features) {
            throw std::invalid_argument("CompareExplanations: attribution " + std::to_string(i) + " has " +
                                        std::to_string(t[i].size() != features ? t[i].size() : s[i].size()) +
                                        " values, not " + std::to_string(features));
        }
    }

    AgreementReport r;
    r.inputs = static_cast<int64_t>(n);
    r.features = static_cast<int64_t>(features);
    const auto k = static_cast<size_t>(std::max(1.0, std::round(options.top_fraction * static_cast<double>(features))));
    r.top_k = static_cast<int64_t>(k);

    std::vector<std::vector<double>> zt(n), zs(n);
    for (size_t i = 0; i < n; ++i) zt[i] = StandardizedRanks(t[i]), zs[i] = StandardizedRanks(s[i]);
    for (size_t i = 0; i < n; ++i) {
        InputAgreement a;
        a.spearman = Dot(zt[i], zs[i]);
        const std::vector<size_t> top_t = TopK(t[i], k), top_s = TopK(s[i], k);
        std::vector<size_t> shared;
        std::set_intersection(top_t.begin(), top_t.end(), top_s.begin(), top_s.end(), std::back_inserter(shared));
        a.top_k_overlap = static_cast<double>(shared.size()) / static_cast<double>(k);
        const double norms = std::sqrt(Dot(t[i], t[i]) * Dot(s[i], s[i]));
        a.cosine = norms > 0 ? Dot(t[i], s[i]) / norms : 0.0;
        r.spearman += a.spearman;
        r.top_k_overlap += a.top_k_overlap;
        r.cosine += a.cosine;
        r.per_input.push_back(a);
    }
    r.spearman /= static_cast<double>(n);
    r.top_k_overlap /= static_cast<double>(n);
    r.cosine /= static_cast<double>(n);

    const size_t needed = options.estimator == CkaEstimator::Debiased ? 4 : 2;
    r.cka = kNaN;
    if (n >= needed) {
        static CPUBackend cpu;
        std::vector<float> a, b;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < features; ++j) a.push_back(static_cast<float>(t[i][j])), b.push_back(static_cast<float>(s[i][j]));
        }
        const Shape shape({static_cast<int64_t>(n), static_cast<int64_t>(features)});
        r.cka = LinearCka(Tensor(shape, &cpu, a), Tensor(shape, &cpu, b), options.estimator);
    }

    r.mismatched_spearman = kNaN;
    r.p_value = kNaN;
    if (n < 2) return r;
    // Every teacher map against every student map: matched pairs on the diagonal.
    std::vector<double> c(n * n);
    double off = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            c[i * n + j] = Dot(zt[i], zs[j]);
            if (i != j) off += c[i * n + j];
        }
    }
    r.mismatched_spearman = off / static_cast<double>(n * (n - 1));
    if (options.permutations == 0) return r;
    r.permutations = options.permutations;
    std::vector<size_t> label(n);
    std::iota(label.begin(), label.end(), size_t{0});
    auto score = [&] {
        double sum = 0;
        for (size_t i = 0; i < n; ++i) sum += c[i * n + label[i]];
        return sum / static_cast<double>(n);
    };
    // Relabelings that tie with the real one count as reaching it, whatever the rounding.
    const double reach = score() - 1e-12;
    PortableRng rng{options.seed};
    int64_t at_least = 1;
    for (int64_t p = 0; p < options.permutations; ++p) {
        for (size_t i = n; i > 1; --i) std::swap(label[i - 1], label[static_cast<size_t>(rng.next() % i)]);
        if (score() >= reach) ++at_least;
    }
    r.p_value = static_cast<double>(at_least) / static_cast<double>(options.permutations + 1);
    return r;
}

AgreementReport ExplanationAgreement(const std::vector<Tensor>& inputs, const ExplainFn& teacher, const ExplainFn& student,
                                     const AgreementOptions& options) {
    std::vector<Attribution> a, b;
    for (const Tensor& x : inputs) {
        a.push_back(teacher(x));
        b.push_back(student(x));
    }
    return CompareExplanations(a, b, options);
}

JsonValue ToJson(const AgreementReport& r) {
    JsonValue::Array per_input;
    for (const InputAgreement& a : r.per_input) {
        per_input.push_back(JsonValue(JsonValue::Object{
            {"spearman", JsonValue(a.spearman)},
            {"top_k_overlap", JsonValue(a.top_k_overlap)},
            {"cosine", JsonValue(a.cosine)},
        }));
    }
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.explanation_agreement.v1"},
        {"inputs", JsonValue(r.inputs)},
        {"features", JsonValue(r.features)},
        {"top_k", JsonValue(r.top_k)},
        {"spearman", JsonValue(r.spearman)},
        {"top_k_overlap", JsonValue(r.top_k_overlap)},
        {"cosine", JsonValue(r.cosine)},
        {"cka", NumberOrNull(r.cka)},
        {"mismatched_spearman", NumberOrNull(r.mismatched_spearman)},
        {"p_value", NumberOrNull(r.p_value)},
        {"permutations", JsonValue(r.permutations)},
        {"per_input", JsonValue(std::move(per_input))},
    });
}

}  // namespace pulsatrix
