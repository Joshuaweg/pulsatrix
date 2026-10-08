#include "pulsatrix/featurizer_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>

#include "portable_random.hpp"

namespace pulsatrix {

namespace {

void CheckBatch(const Featurizer& f, const Tensor& x, const char* who) {
    if (x.rank() != 2 || x.shape().dim(1) != f.input_dim() || x.shape().dim(0) < 1) {
        throw std::invalid_argument(std::string(who) + ": the batch must be (N, input_dim) with N > 0");
    }
}

/** @brief Every row's active codes (above zero), encoded in batches: row r's are entries
 *         [start[r], start[r + 1]). Real featurizers are sparse, and dense codes for many inputs
 *         would not fit in memory. */
struct Sparse {
    std::vector<int64_t> start;
    std::vector<int64_t> feature;
    std::vector<float> value;
};

Sparse EncodeAll(Featurizer& f, const std::vector<float>& x, int64_t rows, DeviceBackend* backend, DeviceType device, int64_t batch) {
    const int64_t d = f.input_dim(), m = f.num_features();
    Sparse out;
    out.start.push_back(0);
    for (int64_t r0 = 0; r0 < rows; r0 += batch) {
        const int64_t n = std::min(batch, rows - r0);
        const Tensor xb(Shape({n, d}), backend, std::vector<float>(x.begin() + r0 * d, x.begin() + (r0 + n) * d), device);
        const std::vector<float> c = f.encode(xb).to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            for (int64_t j = 0; j < m; ++j) {
                const float v = c[static_cast<size_t>(r * m + j)];
                if (v <= 0.0f) continue;
                out.feature.push_back(j);
                out.value.push_back(v);
            }
            out.start.push_back(static_cast<int64_t>(out.feature.size()));
        }
    }
    return out;
}

/** @brief F1 of @p predicted against @p labels over the rows in @p use. */
double F1(const std::vector<bool>& predicted, const std::vector<int>& labels, const std::vector<bool>& use) {
    int64_t tp = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (!use[i]) continue;
        tp += predicted[i] && labels[i] == 1 ? 1 : 0;
        fp += predicted[i] && labels[i] == 0 ? 1 : 0;
        fn += !predicted[i] && labels[i] == 1 ? 1 : 0;
    }
    return tp == 0 ? 0.0 : 2.0 * static_cast<double>(tp) / static_cast<double>(2 * tp + fp + fn);
}

/** @brief A logistic probe in the raw input space, fit by Adam on standardized inputs. */
struct Probe {
    std::vector<double> w;  ///< raw-space weights
    double b = 0;
    [[nodiscard]] double Score(const float* row) const {
        double s = b;
        for (size_t j = 0; j < w.size(); ++j) s += w[j] * row[j];
        return s;
    }
};

Probe FitProbe(const std::vector<float>& x, int64_t d, const std::vector<int>& labels, const std::vector<bool>& train,
               const AbsorptionOptions& o) {
    std::vector<size_t> rows;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (train[i]) rows.push_back(i);
    }
    std::vector<double> mean(static_cast<size_t>(d), 0.0), sd(static_cast<size_t>(d), 0.0);
    for (size_t r : rows) {
        for (int64_t j = 0; j < d; ++j) mean[static_cast<size_t>(j)] += x[r * static_cast<size_t>(d) + static_cast<size_t>(j)];
    }
    for (double& v : mean) v /= static_cast<double>(rows.size());
    for (size_t r : rows) {
        for (int64_t j = 0; j < d; ++j) {
            const double v = x[r * static_cast<size_t>(d) + static_cast<size_t>(j)] - mean[static_cast<size_t>(j)];
            sd[static_cast<size_t>(j)] += v * v;
        }
    }
    for (double& v : sd) v = std::max(std::sqrt(v / static_cast<double>(rows.size())), 1e-8);
    // Positives are weighted up to balance the classes, as a rare concept would otherwise be
    // ignored.
    int64_t pos = 0;
    for (size_t r : rows) pos += labels[r];
    const double w_pos = pos > 0 ? static_cast<double>(rows.size() - static_cast<size_t>(pos)) / static_cast<double>(pos) : 1.0;
    std::vector<double> w(static_cast<size_t>(d) + 1, 0.0), m1(w.size(), 0.0), m2(w.size(), 0.0), g(w.size());
    PortableRng rng{o.seed + 1};
    const size_t batch = 256;
    int64_t t = 0;
    for (int64_t epoch = 0; epoch < o.probe_epochs; ++epoch) {
        for (size_t i = rows.size(); i > 1; --i) std::swap(rows[i - 1], rows[rng.next() % i]);
        for (size_t s = 0; s < rows.size(); s += batch) {
            std::fill(g.begin(), g.end(), 0.0);
            double weight_sum = 0;
            for (size_t q = s; q < std::min(rows.size(), s + batch); ++q) {
                const float* row = x.data() + rows[q] * static_cast<size_t>(d);
                double z = w[static_cast<size_t>(d)];
                for (int64_t j = 0; j < d; ++j) z += w[static_cast<size_t>(j)] * (row[j] - mean[static_cast<size_t>(j)]) / sd[static_cast<size_t>(j)];
                const int y = labels[rows[q]];
                const double weight = y == 1 ? w_pos : 1.0, err = weight * (1.0 / (1.0 + std::exp(-z)) - y);
                weight_sum += weight;
                for (int64_t j = 0; j < d; ++j) g[static_cast<size_t>(j)] += err * (row[j] - mean[static_cast<size_t>(j)]) / sd[static_cast<size_t>(j)];
                g[static_cast<size_t>(d)] += err;
            }
            ++t;
            const double lr = o.probe_learning_rate, b1 = 0.9, b2 = 0.999;
            for (size_t j = 0; j < w.size(); ++j) {
                const double gj = g[j] / weight_sum + (j < static_cast<size_t>(d) ? 1e-4 * w[j] : 0.0);
                m1[j] = b1 * m1[j] + (1 - b1) * gj;
                m2[j] = b2 * m2[j] + (1 - b2) * gj * gj;
                const double mh = m1[j] / (1 - std::pow(b1, static_cast<double>(t))), vh = m2[j] / (1 - std::pow(b2, static_cast<double>(t)));
                w[j] -= lr * mh / (std::sqrt(vh) + 1e-8);
            }
        }
    }
    Probe p;
    p.w.resize(static_cast<size_t>(d));
    p.b = w[static_cast<size_t>(d)];
    for (int64_t j = 0; j < d; ++j) {
        p.w[static_cast<size_t>(j)] = w[static_cast<size_t>(j)] / sd[static_cast<size_t>(j)];
        p.b -= p.w[static_cast<size_t>(j)] * mean[static_cast<size_t>(j)];
    }
    return p;
}

std::vector<bool> RowMask(const std::vector<bool>& rows, int64_t n, const char* who) {
    if (rows.empty()) return std::vector<bool>(static_cast<size_t>(n), true);
    if (static_cast<int64_t>(rows.size()) != n) throw std::invalid_argument(std::string(who) + ": rows must have one entry per (n, l)");
    return rows;
}

}  // namespace

ReconstructionMetrics EvaluateReconstruction(Featurizer& f, const Tensor& x, double dense_rate, int64_t batch) {
    CheckBatch(f, x, "EvaluateReconstruction");
    return EvaluatePrediction(f, x, x, dense_rate, batch);
}

ReconstructionMetrics EvaluatePrediction(Featurizer& f, const Tensor& x, const Tensor& target, double dense_rate, int64_t batch) {
    CheckBatch(f, x, "EvaluatePrediction");
    if (batch < 1) throw std::invalid_argument("EvaluatePrediction: batch must be positive");
    const int64_t N = x.shape().dim(0), in = f.input_dim(), d = f.output_dim(), m = f.num_features();
    if (target.rank() != 2 || target.shape().dim(0) != N || target.shape().dim(1) != d) {
        throw std::invalid_argument("EvaluatePrediction: the targets must be (N, output_dim)");
    }
    const std::vector<float> xin = x.to_host_vector(), xv = target.to_host_vector();
    std::vector<double> mean(static_cast<size_t>(d), 0.0);
    for (int64_t r = 0; r < N; ++r) {
        for (int64_t j = 0; j < d; ++j) mean[static_cast<size_t>(j)] += xv[static_cast<size_t>(r * d + j)];
    }
    for (double& v : mean) v /= static_cast<double>(N);
    ReconstructionMetrics out;
    out.inputs = N;
    std::vector<int64_t> fired(static_cast<size_t>(m), 0);
    double err = 0, var = 0, cos = 0, ratio = 0, active = 0;
    for (int64_t r0 = 0; r0 < N; r0 += batch) {
        const int64_t n = std::min(batch, N - r0);
        const Tensor xb(Shape({n, in}), x.backend(), std::vector<float>(xin.begin() + r0 * in, xin.begin() + (r0 + n) * in), x.device());
        const std::vector<float> c = f.encode(xb).to_host_vector(), xh = f.predict(xb).to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            double dot = 0, a2 = 0, b2 = 0;
            for (int64_t j = 0; j < d; ++j) {
                const double a = xv[static_cast<size_t>((r0 + r) * d + j)], h = xh[static_cast<size_t>(r * d + j)];
                err += (a - h) * (a - h);
                var += (a - mean[static_cast<size_t>(j)]) * (a - mean[static_cast<size_t>(j)]);
                dot += a * h;
                a2 += a * a;
                b2 += h * h;
            }
            cos += a2 > 0 && b2 > 0 ? dot / std::sqrt(a2 * b2) : 0.0;
            ratio += a2 > 0 ? std::sqrt(b2 / a2) : 0.0;
            for (int64_t i = 0; i < m; ++i) {
                if (c[static_cast<size_t>(r * m + i)] > 0.0f) {
                    ++fired[static_cast<size_t>(i)];
                    active += 1;
                }
            }
        }
    }
    out.mse = err / static_cast<double>(N * d);
    out.explained_variance = var > 0 ? 1.0 - err / var : std::numeric_limits<double>::quiet_NaN();
    out.cosine = cos / static_cast<double>(N);
    out.norm_ratio = ratio / static_cast<double>(N);
    out.l0 = active / static_cast<double>(N);
    out.firing_rates.resize(static_cast<size_t>(m));
    int64_t dead = 0, dense = 0;
    for (int64_t i = 0; i < m; ++i) {
        const double rate = static_cast<double>(fired[static_cast<size_t>(i)]) / static_cast<double>(N);
        out.firing_rates[static_cast<size_t>(i)] = rate;
        dead += fired[static_cast<size_t>(i)] == 0 ? 1 : 0;
        dense += rate > dense_rate ? 1 : 0;
    }
    out.dead_fraction = static_cast<double>(dead) / static_cast<double>(m);
    out.dense_fraction = static_cast<double>(dense) / static_cast<double>(m);
    return out;
}

HiddenStateHook SpliceHook(Featurizer& featurizer, int64_t position, std::vector<bool> rows) {
    Featurizer* f = &featurizer;
    return [f, position, rows = std::move(rows)](int64_t at, const Tensor& hidden) -> Tensor {
        if (at != position) return hidden;
        const int64_t d = f->input_dim();
        if (hidden.rank() < 2 || hidden.shape().dim(hidden.rank() - 1) != d) {
            throw std::invalid_argument("SpliceHook: the hidden size isn't the featurizer's input_dim");
        }
        const int64_t n = hidden.numel() / d;
        const std::vector<bool> use = RowMask(rows, n, "SpliceHook");
        std::vector<float> h = hidden.to_host_vector();
        std::vector<float> picked;
        for (int64_t r = 0; r < n; ++r) {
            if (use[static_cast<size_t>(r)]) picked.insert(picked.end(), h.begin() + r * d, h.begin() + (r + 1) * d);
        }
        if (picked.empty()) return hidden;
        const auto k = static_cast<int64_t>(picked.size()) / d;
        const std::vector<float> xh =
            f->decode(f->encode(Tensor(Shape({k, d}), hidden.backend(), picked, hidden.device()))).to_host_vector();
        for (int64_t r = 0, q = 0; r < n; ++r) {
            if (!use[static_cast<size_t>(r)]) continue;
            std::copy(xh.begin() + q * d, xh.begin() + (q + 1) * d, h.begin() + r * d);
            ++q;
        }
        return Tensor(hidden.shape(), hidden.backend(), h, hidden.device());
    };
}

HiddenStateHook AblationHook(int64_t position, std::vector<bool> rows) {
    return [position, rows = std::move(rows)](int64_t at, const Tensor& hidden) -> Tensor {
        if (at != position) return hidden;
        const int64_t d = hidden.shape().dim(hidden.rank() - 1), n = hidden.numel() / d;
        const std::vector<bool> use = RowMask(rows, n, "AblationHook");
        std::vector<float> h = hidden.to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            if (use[static_cast<size_t>(r)]) std::fill_n(h.begin() + r * d, d, 0.0f);
        }
        return Tensor(hidden.shape(), hidden.backend(), h, hidden.device());
    };
}

MlpHook MlpSpliceHook(Featurizer& transcoder, std::vector<bool> rows) {
    Featurizer* f = &transcoder;
    return [f, rows = std::move(rows)](const Tensor& input, const Tensor& output) -> Tensor {
        const int64_t in = f->input_dim(), out = f->output_dim();
        if (input.shape().dim(input.rank() - 1) != in || output.shape().dim(output.rank() - 1) != out) {
            throw std::invalid_argument("MlpSpliceHook: the MLP's sizes aren't the transcoder's");
        }
        const int64_t n = input.numel() / in;
        const std::vector<bool> use = RowMask(rows, n, "MlpSpliceHook");
        const std::vector<float> xin = input.to_host_vector();
        std::vector<float> y = output.to_host_vector(), picked;
        for (int64_t r = 0; r < n; ++r) {
            if (use[static_cast<size_t>(r)]) picked.insert(picked.end(), xin.begin() + r * in, xin.begin() + (r + 1) * in);
        }
        if (picked.empty()) return output;
        const auto k = static_cast<int64_t>(picked.size()) / in;
        const std::vector<float> yh = f->predict(Tensor(Shape({k, in}), input.backend(), picked, input.device())).to_host_vector();
        for (int64_t r = 0, q = 0; r < n; ++r) {
            if (!use[static_cast<size_t>(r)]) continue;
            std::copy(yh.begin() + q * out, yh.begin() + (q + 1) * out, y.begin() + r * out);
            ++q;
        }
        return Tensor(output.shape(), output.backend(), y, output.device());
    };
}

MlpHook MlpAblationHook(std::vector<bool> rows) {
    return [rows = std::move(rows)](const Tensor&, const Tensor& output) -> Tensor {
        const int64_t d = output.shape().dim(output.rank() - 1), n = output.numel() / d;
        const std::vector<bool> use = RowMask(rows, n, "MlpAblationHook");
        std::vector<float> y = output.to_host_vector();
        for (int64_t r = 0; r < n; ++r) {
            if (use[static_cast<size_t>(r)]) std::fill_n(y.begin() + r * d, d, 0.0f);
        }
        return Tensor(output.shape(), output.backend(), y, output.device());
    };
}

LossRecovered MeasureMlpLossRecovered(Featurizer& transcoder, const std::function<double(const MlpHook&)>& loss, const std::vector<bool>& rows) {
    LossRecovered out;
    out.clean = loss(MlpHook{});
    out.spliced = loss(MlpSpliceHook(transcoder, rows));
    out.ablated = loss(MlpAblationHook(rows));
    const double gap = out.ablated - out.clean;
    out.recovered = gap > 0 ? (out.ablated - out.spliced) / gap : std::numeric_limits<double>::quiet_NaN();
    return out;
}

LossRecovered MeasureLossRecovered(Featurizer& featurizer, int64_t position, const std::function<double(const HiddenStateHook&)>& loss,
                                   const std::vector<bool>& rows) {
    LossRecovered out;
    out.clean = loss(HiddenStateHook{});
    out.spliced = loss(SpliceHook(featurizer, position, rows));
    out.ablated = loss(AblationHook(position, rows));
    const double gap = out.ablated - out.clean;
    out.recovered = gap > 0 ? (out.ablated - out.spliced) / gap : std::numeric_limits<double>::quiet_NaN();
    return out;
}

AbsorptionResult FeatureAbsorption(Featurizer& f, const Tensor& x, const std::vector<int>& labels, const AbsorptionOptions& o) {
    CheckBatch(f, x, "FeatureAbsorption");
    if (f.output_dim() != f.input_dim()) throw std::invalid_argument("FeatureAbsorption: needs an autoencoder (output_dim == input_dim)");
    const int64_t N = x.shape().dim(0), d = f.input_dim(), m = f.num_features();
    if (static_cast<int64_t>(labels.size()) != N) throw std::invalid_argument("FeatureAbsorption: labels must have one value per input");
    if (std::any_of(labels.begin(), labels.end(), [](int y) { return y != 0 && y != 1; })) {
        throw std::invalid_argument("FeatureAbsorption: labels must be 0 or 1");
    }
    // A seeded split, the same for the probe and the main features.
    PortableRng rng{o.seed};
    std::vector<bool> train(static_cast<size_t>(N)), test(static_cast<size_t>(N));
    int64_t classes[2][2] = {{0, 0}, {0, 0}};
    for (int64_t i = 0; i < N; ++i) {
        const bool t = rng.uniform() < o.train_fraction;
        train[static_cast<size_t>(i)] = t;
        test[static_cast<size_t>(i)] = !t;
        ++classes[t ? 0 : 1][labels[static_cast<size_t>(i)]];
    }
    if (classes[0][0] == 0 || classes[0][1] == 0 || classes[1][0] == 0 || classes[1][1] == 0) {
        throw std::invalid_argument("FeatureAbsorption: the train and test splits each need both classes");
    }
    const std::vector<float> xv = x.to_host_vector();
    const Sparse codes = EncodeAll(f, xv, N, x.backend(), x.device(), std::max<int64_t>(1, o.batch));

    // 1. The probe, and its F1 on the test inputs.
    const Probe probe = FitProbe(xv, d, labels, train, o);
    std::vector<bool> probe_says(static_cast<size_t>(N));
    for (int64_t i = 0; i < N; ++i) probe_says[static_cast<size_t>(i)] = probe.Score(xv.data() + i * d) > 0;
    AbsorptionResult out;
    out.probe_f1 = F1(probe_says, labels, test);

    // 2. Main features: ranked by mean code on positives minus negatives (train inputs).
    std::vector<double> diff(static_cast<size_t>(m), 0.0);
    for (int64_t i = 0; i < N; ++i) {
        if (!train[static_cast<size_t>(i)]) continue;
        const int y = labels[static_cast<size_t>(i)];
        const double w = 1.0 / static_cast<double>(classes[0][y]);
        for (int64_t e = codes.start[static_cast<size_t>(i)]; e < codes.start[static_cast<size_t>(i) + 1]; ++e) {
            diff[static_cast<size_t>(codes.feature[static_cast<size_t>(e)])] += (y == 1 ? w : -w) * codes.value[static_cast<size_t>(e)];
        }
    }
    std::vector<int64_t> order(static_cast<size_t>(m));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return diff[static_cast<size_t>(a)] > diff[static_cast<size_t>(b)]; });
    std::vector<bool> any(static_cast<size_t>(N), false);
    double f1 = 0;
    for (int64_t rank = 0; rank < std::min<int64_t>(o.max_main_features, m) && static_cast<int64_t>(out.main_features.size()) < o.max_main_features;
         ++rank) {
        const int64_t j = order[static_cast<size_t>(rank)];
        if (diff[static_cast<size_t>(j)] <= 0) break;
        std::vector<bool> with = any;
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t e = codes.start[static_cast<size_t>(i)]; e < codes.start[static_cast<size_t>(i) + 1]; ++e) {
                if (codes.feature[static_cast<size_t>(e)] == j) with[static_cast<size_t>(i)] = true;
            }
        }
        const double f1_with = F1(with, labels, train);
        if (out.main_features.empty() || f1_with - f1 >= o.min_f1_gain) {
            out.main_features.push_back(j);
            any = std::move(with);
            f1 = f1_with;
        }
    }
    out.main_f1 = F1(any, labels, test);

    // 3. Absorption on the held-out positives the probe finds and the main features miss.
    std::vector<double> unit(probe.w);
    double norm = 0;
    for (double v : unit) norm += v * v;
    norm = std::sqrt(norm);
    for (double& v : unit) v /= norm;
    std::vector<double> neg_mean(static_cast<size_t>(d), 0.0);
    for (int64_t i = 0; i < N; ++i) {
        if (!train[static_cast<size_t>(i)] || labels[static_cast<size_t>(i)] != 0) continue;
        for (int64_t j = 0; j < d; ++j) neg_mean[static_cast<size_t>(j)] += xv[static_cast<size_t>(i * d + j)] / static_cast<double>(classes[0][0]);
    }
    std::vector<double> along(static_cast<size_t>(m)), cosine(static_cast<size_t>(m));  // D_j · p̂, cos(D_j, p̂)
    for (int64_t j = 0; j < m; ++j) {
        const std::vector<float> dir = f.decoder_direction(j);
        double dot = 0, n2 = 0;
        for (int64_t q = 0; q < d; ++q) {
            dot += dir[static_cast<size_t>(q)] * unit[static_cast<size_t>(q)];
            n2 += static_cast<double>(dir[static_cast<size_t>(q)]) * dir[static_cast<size_t>(q)];
        }
        along[static_cast<size_t>(j)] = dot;
        cosine[static_cast<size_t>(j)] = n2 > 0 ? dot / std::sqrt(n2) : 0.0;
    }
    std::vector<bool> is_main(static_cast<size_t>(m), false);
    for (int64_t j : out.main_features) is_main[static_cast<size_t>(j)] = true;
    std::map<int64_t, int64_t> absorbing;
    for (int64_t i = 0; i < N; ++i) {
        if (!test[static_cast<size_t>(i)]) continue;
        ++out.test_inputs;
        if (labels[static_cast<size_t>(i)] != 1) continue;
        ++out.positives;
        if (!probe_says[static_cast<size_t>(i)]) continue;
        ++out.probe_hits;
        if (any[static_cast<size_t>(i)]) continue;
        ++out.missed;
        double projection = 0;
        for (int64_t q = 0; q < d; ++q) projection += (xv[static_cast<size_t>(i * d + q)] - neg_mean[static_cast<size_t>(q)]) * unit[static_cast<size_t>(q)];
        if (projection <= 0) continue;
        double carried = 0;
        std::vector<int64_t> carriers;
        for (int64_t e = codes.start[static_cast<size_t>(i)]; e < codes.start[static_cast<size_t>(i) + 1]; ++e) {
            const int64_t j = codes.feature[static_cast<size_t>(e)];
            const float c = codes.value[static_cast<size_t>(e)];
            if (is_main[static_cast<size_t>(j)] || cosine[static_cast<size_t>(j)] < o.min_cosine) continue;
            carried += c * along[static_cast<size_t>(j)];
            carriers.push_back(j);
        }
        if (carried / projection >= o.min_share) {
            ++out.absorbed;
            for (int64_t j : carriers) ++absorbing[j];
        }
    }
    out.miss_rate = out.probe_hits > 0 ? static_cast<double>(out.missed) / static_cast<double>(out.probe_hits) : 0.0;
    out.absorption_rate = out.probe_hits > 0 ? static_cast<double>(out.absorbed) / static_cast<double>(out.probe_hits) : 0.0;
    out.absorbing_features.assign(absorbing.begin(), absorbing.end());
    std::stable_sort(out.absorbing_features.begin(), out.absorbing_features.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
    return out;
}

}  // namespace pulsatrix
