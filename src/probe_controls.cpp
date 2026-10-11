#include "pulsatrix/probe_controls.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

#include "portable_random.hpp"
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/fitness_metrics.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {

ProbeResult TrainLinearProbe(const std::vector<float>& features, int64_t dim, const std::vector<int>& labels, int classes,
                             const std::vector<bool>& train, const std::vector<bool>& test, DeviceBackend* backend, const ProbeOptions& o) {
    const auto N = static_cast<int64_t>(labels.size());
    if (dim < 1 || features.size() != static_cast<size_t>(N * dim) || train.size() != labels.size() || test.size() != labels.size() || classes < 2) {
        throw std::invalid_argument("TrainLinearProbe: features must be rows x dim with one label and flag per row, and classes >= 2");
    }
    std::vector<int64_t> train_rows, test_rows;
    for (int64_t i = 0; i < N; ++i) {
        if (labels[static_cast<size_t>(i)] < 0 || labels[static_cast<size_t>(i)] >= classes) throw std::invalid_argument("TrainLinearProbe: a label is outside [0, classes)");
        if (train[static_cast<size_t>(i)]) train_rows.push_back(i);
        if (test[static_cast<size_t>(i)]) test_rows.push_back(i);
    }
    if (train_rows.empty() || test_rows.empty()) throw std::invalid_argument("TrainLinearProbe: the train and test sets must not be empty");
    // Standardize by the training rows.
    std::vector<double> mean(static_cast<size_t>(dim), 0.0), sd(static_cast<size_t>(dim), 0.0);
    for (int64_t r : train_rows) {
        for (int64_t j = 0; j < dim; ++j) mean[static_cast<size_t>(j)] += features[static_cast<size_t>(r * dim + j)];
    }
    for (double& m : mean) m /= static_cast<double>(train_rows.size());
    for (int64_t r : train_rows) {
        for (int64_t j = 0; j < dim; ++j) {
            const double d = features[static_cast<size_t>(r * dim + j)] - mean[static_cast<size_t>(j)];
            sd[static_cast<size_t>(j)] += d * d;
        }
    }
    for (double& s : sd) s = std::sqrt(s / static_cast<double>(train_rows.size())) + 1e-6;
    auto batch_of = [&](const std::vector<int64_t>& rows, size_t from, size_t to, std::vector<float>& x, std::vector<float>& y) {
        x.resize((to - from) * static_cast<size_t>(dim));
        y.resize(to - from);
        for (size_t k = from; k < to; ++k) {
            const int64_t r = rows[k];
            for (int64_t j = 0; j < dim; ++j) {
                x[(k - from) * static_cast<size_t>(dim) + static_cast<size_t>(j)] =
                    static_cast<float>((features[static_cast<size_t>(r * dim + j)] - mean[static_cast<size_t>(j)]) / sd[static_cast<size_t>(j)]);
            }
            y[k - from] = static_cast<float>(labels[static_cast<size_t>(r)]);
        }
    };
    LinearModule probe(dim, classes, backend);
    AdamWOptimizer opt(o.learning_rate, backend, o.weight_decay);
    TokenCrossEntropyLoss loss(backend);
    PortableRng rng{o.seed};
    std::vector<float> x, y;
    for (int64_t epoch = 0; epoch < o.epochs; ++epoch) {
        for (size_t i = train_rows.size(); i > 1; --i) std::swap(train_rows[i - 1], train_rows[static_cast<size_t>(rng.next() % i)]);
        for (size_t from = 0; from < train_rows.size(); from += static_cast<size_t>(o.batch)) {
            const size_t to = std::min(train_rows.size(), from + static_cast<size_t>(o.batch));
            batch_of(train_rows, from, to, x, y);
            const auto B = static_cast<int64_t>(to - from);
            opt.zero_grad(probe);
            const Tensor logits = probe.forward(Tensor(Shape({B, dim}), backend, x, backend->device()));
            (void)loss.forward(logits, Tensor(Shape({B}), backend, y, backend->device()));
            (void)probe.backward(loss.backward());
            opt.step(probe);
        }
    }
    // Score the held-out rows.
    std::vector<int64_t> correct(static_cast<size_t>(classes), 0), count(static_cast<size_t>(classes), 0);
    std::vector<int> binary_labels;
    std::vector<double> binary_scores;
    for (size_t from = 0; from < test_rows.size(); from += static_cast<size_t>(o.batch)) {
        const size_t to = std::min(test_rows.size(), from + static_cast<size_t>(o.batch));
        batch_of(test_rows, from, to, x, y);
        const auto B = static_cast<int64_t>(to - from);
        const std::vector<float> logits = probe.forward(Tensor(Shape({B, dim}), backend, x, backend->device())).to_host_vector();
        for (int64_t b = 0; b < B; ++b) {
            const float* row = logits.data() + b * classes;
            const int pred = static_cast<int>(std::max_element(row, row + classes) - row);
            const int truth = static_cast<int>(y[static_cast<size_t>(b)]);
            ++count[static_cast<size_t>(truth)];
            correct[static_cast<size_t>(truth)] += pred == truth ? 1 : 0;
            if (classes == 2) {
                binary_labels.push_back(truth);
                binary_scores.push_back(static_cast<double>(row[1]) - row[0]);  // the log-odds of class 1
            }
        }
    }
    ProbeResult r;
    r.train = static_cast<int64_t>(train_rows.size());
    r.test = static_cast<int64_t>(test_rows.size());
    r.accuracy = static_cast<double>(std::accumulate(correct.begin(), correct.end(), int64_t{0})) / static_cast<double>(r.test);
    double recall_sum = 0;
    int present = 0;
    for (int c = 0; c < classes; ++c) {
        if (count[static_cast<size_t>(c)] == 0) continue;
        recall_sum += static_cast<double>(correct[static_cast<size_t>(c)]) / static_cast<double>(count[static_cast<size_t>(c)]);
        ++present;
    }
    r.balanced_accuracy = present > 0 ? recall_sum / present : 0.0;
    r.auc = classes == 2 ? RocAuc(binary_labels, binary_scores) : std::numeric_limits<double>::quiet_NaN();
    return r;
}

std::vector<int> ControlTaskLabels(const std::vector<int64_t>& token_types, const std::vector<int>& labels, int classes, uint64_t seed) {
    if (token_types.size() != labels.size() || labels.empty() || classes < 2) {
        throw std::invalid_argument("ControlTaskLabels: token_types and labels must be the same nonzero length, and classes >= 2");
    }
    std::vector<int64_t> count(static_cast<size_t>(classes), 0);
    for (int y : labels) {
        if (y < 0 || y >= classes) throw std::invalid_argument("ControlTaskLabels: a label is outside [0, classes)");
        ++count[static_cast<size_t>(y)];
    }
    // Each type, in increasing order, draws a label from the labels' distribution: a uniform
    // position among the rows, mapped to the class whose share of the rows it falls in.
    std::map<int64_t, int> control;
    for (int64_t t : token_types) control.emplace(t, 0);
    PortableRng rng{seed};
    const auto n = static_cast<double>(labels.size());
    for (auto& [type, label] : control) {
        (void)type;
        const auto position = std::min(static_cast<int64_t>(rng.uniform() * n), static_cast<int64_t>(labels.size()) - 1);
        int64_t below = 0;
        label = 0;
        while (position >= below + count[static_cast<size_t>(label)]) below += count[static_cast<size_t>(label++)];
    }
    std::vector<int> out(labels.size());
    for (size_t i = 0; i < token_types.size(); ++i) out[i] = control.at(token_types[i]);
    return out;
}

ProbeSelectivity MeasureSelectivity(const std::vector<float>& features, int64_t dim, const std::vector<int>& labels, int classes,
                                    const std::vector<int64_t>& token_types, const std::vector<bool>& train, const std::vector<bool>& test,
                                    DeviceBackend* backend, const ProbeOptions& options) {
    if (token_types.size() != labels.size()) throw std::invalid_argument("MeasureSelectivity: token_types must have one type per row");
    const std::vector<int> control = ControlTaskLabels(token_types, labels, classes, options.seed);
    ProbeSelectivity s;
    s.task = TrainLinearProbe(features, dim, labels, classes, train, test, backend, options);
    s.control = TrainLinearProbe(features, dim, control, classes, train, test, backend, options);
    s.selectivity = s.task.accuracy - s.control.accuracy;
    s.balanced_selectivity = s.task.balanced_accuracy - s.control.balanced_accuracy;
    std::set<int64_t> types;
    for (size_t i = 0; i < token_types.size(); ++i) {
        if (train[i]) types.insert(token_types[i]);
    }
    s.train_types = static_cast<int64_t>(types.size());
    return s;
}

ProbeGain CompareToBaseline(const std::vector<float>& features, int64_t dim, const std::vector<float>& baseline, int64_t baseline_dim,
                            const std::vector<int>& labels, int classes, const std::vector<bool>& train, const std::vector<bool>& test,
                            DeviceBackend* backend, const ProbeOptions& options) {
    if (baseline_dim < 1 || baseline.size() != labels.size() * static_cast<size_t>(baseline_dim)) {
        throw std::invalid_argument("CompareToBaseline: baseline must be rows x baseline_dim, the same rows as features");
    }
    ProbeGain g;
    g.probe = TrainLinearProbe(features, dim, labels, classes, train, test, backend, options);
    g.baseline = TrainLinearProbe(baseline, baseline_dim, labels, classes, train, test, backend, options);
    g.gain = g.probe.accuracy - g.baseline.accuracy;
    g.balanced_gain = g.probe.balanced_accuracy - g.baseline.balanced_accuracy;
    g.auc_gain = classes == 2 ? g.probe.auc - g.baseline.auc : std::numeric_limits<double>::quiet_NaN();
    return g;
}

}  // namespace pulsatrix
