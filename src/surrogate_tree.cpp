// XAI-4: a global surrogate decision tree (CART, as scikit-learn grows it) and its fidelity.

#include "pulsatrix/surrogate_tree.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "portable_random.hpp"

namespace pulsatrix {

namespace {

// scikit-learn's FEATURE_THRESHOLD: values closer than this aren't split apart.
constexpr double kFeatureThreshold = 1e-7;
constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

struct Builder {
    const std::vector<std::vector<float>>& x;
    const std::vector<double>& y;
    SurrogateTask task;
    int64_t num_classes;
    const SurrogateTreeOptions& options;
    std::vector<SurrogateNode> nodes;

    // Gini impurity from class counts, or the variance from a sum and a sum of squares.
    static double Gini(const std::vector<int64_t>& counts, int64_t n) {
        double sum = 0;
        for (int64_t c : counts) sum += static_cast<double>(c) * static_cast<double>(c);
        return 1.0 - sum / (static_cast<double>(n) * static_cast<double>(n));
    }
    static double Variance(double sum, double sum_sq, int64_t n) {
        const double mean = sum / static_cast<double>(n);
        return sum_sq / static_cast<double>(n) - mean * mean;
    }

    void Describe(SurrogateNode& node, const std::vector<int>& idx) const {
        node.samples = static_cast<int64_t>(idx.size());
        if (task == SurrogateTask::Classification) {
            node.class_counts.assign(static_cast<size_t>(num_classes), 0);
            for (int i : idx) ++node.class_counts[static_cast<size_t>(y[static_cast<size_t>(i)])];
            node.impurity = Gini(node.class_counts, node.samples);
            node.value = static_cast<double>(std::max_element(node.class_counts.begin(), node.class_counts.end()) -
                                             node.class_counts.begin());
        } else {
            double sum = 0, sum_sq = 0;
            for (int i : idx) {
                const double v = y[static_cast<size_t>(i)];
                sum += v;
                sum_sq += v * v;
            }
            node.impurity = Variance(sum, sum_sq, node.samples);
            node.value = sum / static_cast<double>(node.samples);
        }
    }

    struct Split {
        int64_t feature = -1;
        double threshold = 0;
        double improvement = -std::numeric_limits<double>::infinity();
    };

    Split Best(const std::vector<int>& idx, double impurity) const {
        Split best;
        const auto n = static_cast<int64_t>(idx.size());
        const int64_t leaf = options.min_samples_leaf;
        const auto d = static_cast<int64_t>(x.front().size());
        std::vector<int> sorted = idx;
        for (int64_t f = 0; f < d; ++f) {
            const auto value = [&](int i) { return static_cast<double>(x[static_cast<size_t>(i)][static_cast<size_t>(f)]); };
            std::stable_sort(sorted.begin(), sorted.end(), [&](int a, int b) { return value(a) < value(b); });
            if (value(sorted.back()) <= value(sorted.front()) + kFeatureThreshold) continue;  // constant here

            std::vector<int64_t> left_counts(static_cast<size_t>(num_classes), 0), right_counts;
            double left_sum = 0, left_sq = 0, right_sum = 0, right_sq = 0;
            if (task == SurrogateTask::Classification) {
                right_counts.assign(static_cast<size_t>(num_classes), 0);
                for (int i : sorted) ++right_counts[static_cast<size_t>(y[static_cast<size_t>(i)])];
            } else {
                for (int i : sorted) {
                    right_sum += y[static_cast<size_t>(i)];
                    right_sq += y[static_cast<size_t>(i)] * y[static_cast<size_t>(i)];
                }
            }
            for (int64_t p = 1; p < n; ++p) {
                const int moved = sorted[static_cast<size_t>(p - 1)];
                const double t = y[static_cast<size_t>(moved)];
                if (task == SurrogateTask::Classification) {
                    ++left_counts[static_cast<size_t>(t)];
                    --right_counts[static_cast<size_t>(t)];
                } else {
                    left_sum += t;
                    left_sq += t * t;
                    right_sum -= t;
                    right_sq -= t * t;
                }
                const double lo = value(sorted[static_cast<size_t>(p - 1)]);
                const double hi = value(sorted[static_cast<size_t>(p)]);
                if (hi <= lo + kFeatureThreshold) continue;
                if (p < leaf || n - p < leaf) continue;
                const double left = task == SurrogateTask::Classification ? Gini(left_counts, p) : Variance(left_sum, left_sq, p);
                const double right = task == SurrogateTask::Classification ? Gini(right_counts, n - p)
                                                                           : Variance(right_sum, right_sq, n - p);
                const double improvement = impurity - (static_cast<double>(p) * left + static_cast<double>(n - p) * right) /
                                                          static_cast<double>(n);
                if (improvement > best.improvement) {
                    best.feature = f;
                    best.improvement = improvement;
                    best.threshold = lo / 2.0 + hi / 2.0;
                    if (best.threshold == hi || std::isinf(best.threshold)) best.threshold = lo;
                }
            }
        }
        return best;
    }

    // Depth first, left before right, so node indices follow scikit-learn's.
    int Grow(const std::vector<int>& idx, int depth) {
        const int id = static_cast<int>(nodes.size());
        nodes.emplace_back();
        SurrogateNode node;
        Describe(node, idx);
        const auto n = static_cast<int64_t>(idx.size());
        bool leaf = depth >= options.max_depth || n < 2 || n < 2 * options.min_samples_leaf || node.impurity <= kEpsilon;
        Split split;
        if (!leaf) {
            split = Best(idx, node.impurity);
            leaf = split.feature < 0;
        }
        if (!leaf) {
            std::vector<int> left, right;
            for (int i : idx) {
                const double v = x[static_cast<size_t>(i)][static_cast<size_t>(split.feature)];
                (v <= split.threshold ? left : right).push_back(i);
            }
            node.feature = split.feature;
            node.threshold = split.threshold;
            nodes[static_cast<size_t>(id)] = node;
            const int l = Grow(left, depth + 1);
            const int r = Grow(right, depth + 1);
            nodes[static_cast<size_t>(id)].left = l;
            nodes[static_cast<size_t>(id)].right = r;
        } else {
            nodes[static_cast<size_t>(id)] = node;
        }
        return id;
    }
};

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

std::string LeafText(const SurrogateTree& tree, const SurrogateNode& n) {
    if (tree.task == SurrogateTask::Regression) {
        return "value " + Fmt(n.value) + " (" + std::to_string(n.samples) + " inputs)";
    }
    const int64_t majority = n.class_counts[static_cast<size_t>(n.value)];
    return "class " + std::to_string(static_cast<int64_t>(n.value)) + " (" + std::to_string(n.samples) + " inputs, " +
           Fmt(100.0 * static_cast<double>(majority) / static_cast<double>(n.samples), "%.0f") + "%)";
}

void DescribeNode(const SurrogateTree& tree, int id, int indent, const std::vector<std::string>& names,
                  std::ostringstream& out) {
    const SurrogateNode& n = tree.nodes[static_cast<size_t>(id)];
    const std::string pad(static_cast<size_t>(2 * indent), ' ');
    if (n.is_leaf()) {
        out << pad << LeafText(tree, n) << "\n";
        return;
    }
    out << pad << "if " << FeatureName(n.feature, names) << " <= " << Fmt(n.threshold) << ":\n";
    DescribeNode(tree, n.left, indent + 1, names, out);
    out << pad << "else:\n";
    DescribeNode(tree, n.right, indent + 1, names, out);
}

double Fidelity(const SurrogateTree& tree, const std::vector<std::vector<float>>& x, const std::vector<double>& y,
                const std::vector<int>& idx) {
    if (idx.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (tree.task == SurrogateTask::Classification) {
        int64_t agree = 0;
        for (int i : idx) agree += tree.predict(x[static_cast<size_t>(i)]) == y[static_cast<size_t>(i)] ? 1 : 0;
        return static_cast<double>(agree) / static_cast<double>(idx.size());
    }
    // R², as scikit-learn's r2_score: 1 for a perfect fit of a constant target, else 0 for one.
    double mean = 0;
    for (int i : idx) mean += y[static_cast<size_t>(i)];
    mean /= static_cast<double>(idx.size());
    double sse = 0, sst = 0;
    for (int i : idx) {
        const double e = y[static_cast<size_t>(i)] - tree.predict(x[static_cast<size_t>(i)]);
        sse += e * e;
        sst += (y[static_cast<size_t>(i)] - mean) * (y[static_cast<size_t>(i)] - mean);
    }
    if (sst == 0) return sse == 0 ? 1.0 : 0.0;
    return 1.0 - sse / sst;
}

}  // namespace

int SurrogateTree::leaf(const std::vector<float>& x) const {
    if (static_cast<int64_t>(x.size()) != num_features) {
        throw std::invalid_argument("SurrogateTree: the input has " + std::to_string(x.size()) + " features, not " +
                                    std::to_string(num_features));
    }
    int id = 0;
    while (!nodes[static_cast<size_t>(id)].is_leaf()) {
        const SurrogateNode& n = nodes[static_cast<size_t>(id)];
        id = static_cast<double>(x[static_cast<size_t>(n.feature)]) <= n.threshold ? n.left : n.right;
    }
    return id;
}

double SurrogateTree::predict(const std::vector<float>& x) const { return nodes[static_cast<size_t>(leaf(x))].value; }

int SurrogateTree::depth() const {
    // Preorder storage: a node's depth is its parent's plus one, and parents come first.
    std::vector<int> d(nodes.size(), 0);
    int deepest = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const SurrogateNode& n = nodes[i];
        if (n.is_leaf()) continue;
        d[static_cast<size_t>(n.left)] = d[static_cast<size_t>(n.right)] = d[i] + 1;
        deepest = std::max(deepest, d[i] + 1);
    }
    return deepest;
}

int64_t SurrogateTree::leaves() const {
    return std::count_if(nodes.begin(), nodes.end(), [](const SurrogateNode& n) { return n.is_leaf(); });
}

SurrogateTree FitDecisionTree(const std::vector<std::vector<float>>& x, const std::vector<double>& y, SurrogateTask task,
                              int64_t num_classes, const SurrogateTreeOptions& options) {
    if (x.empty() || x.front().empty()) throw std::invalid_argument("FitDecisionTree: needs at least one row with features");
    if (y.size() != x.size()) throw std::invalid_argument("FitDecisionTree: x and y have different lengths");
    for (const auto& row : x) {
        if (row.size() != x.front().size()) throw std::invalid_argument("FitDecisionTree: rows differ in length");
    }
    if (options.max_depth < 0) throw std::invalid_argument("FitDecisionTree: max_depth must not be negative");
    if (options.min_samples_leaf < 1) throw std::invalid_argument("FitDecisionTree: min_samples_leaf must be at least 1");
    if (task == SurrogateTask::Classification) {
        if (num_classes < 1) throw std::invalid_argument("FitDecisionTree: num_classes must be at least 1");
        for (double c : y) {
            if (c < 0 || c >= static_cast<double>(num_classes) || c != std::floor(c)) {
                throw std::invalid_argument("FitDecisionTree: a class label isn't in [0, num_classes)");
            }
        }
    }
    Builder b{x, y, task, task == SurrogateTask::Classification ? num_classes : 0, options, {}};
    std::vector<int> all(x.size());
    std::iota(all.begin(), all.end(), 0);
    b.Grow(all, 0);
    SurrogateTree tree;
    tree.task = task;
    tree.num_features = static_cast<int64_t>(x.front().size());
    tree.num_classes = b.num_classes;
    tree.nodes = std::move(b.nodes);
    return tree;
}

std::vector<double> FeatureImportance(const SurrogateTree& tree) {
    std::vector<double> importance(static_cast<size_t>(tree.num_features), 0.0);
    for (const SurrogateNode& n : tree.nodes) {
        if (n.is_leaf()) continue;
        const SurrogateNode& l = tree.nodes[static_cast<size_t>(n.left)];
        const SurrogateNode& r = tree.nodes[static_cast<size_t>(n.right)];
        importance[static_cast<size_t>(n.feature)] += static_cast<double>(n.samples) * n.impurity -
                                                      static_cast<double>(l.samples) * l.impurity -
                                                      static_cast<double>(r.samples) * r.impurity;
    }
    const double total = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (total > 0) {
        for (double& v : importance) v /= total;
    }
    return importance;
}

SurrogateTreeResult FitSurrogateTree(const std::function<Tensor(const Tensor&)>& predict, const std::vector<Tensor>& instances,
                                     const SurrogateTreeOptions& options) {
    if (instances.empty()) throw std::invalid_argument("FitSurrogateTree: needs at least one instance");
    if (!(options.test_fraction >= 0.0 && options.test_fraction < 1.0)) {
        throw std::invalid_argument("FitSurrogateTree: test_fraction must be in [0, 1)");
    }
    const int64_t n = static_cast<int64_t>(instances.size());
    std::vector<std::vector<float>> x;
    std::vector<double> y;
    int64_t outputs = 0;
    for (const Tensor& instance : instances) {
        if (instance.numel() != instances.front().numel()) {
            throw std::invalid_argument("FitSurrogateTree: instances have different sizes");
        }
        x.push_back(instance.to_host_vector());
        const std::vector<float> out = predict(instance).to_host_vector();
        if (outputs == 0) outputs = static_cast<int64_t>(out.size());
        if (static_cast<int64_t>(out.size()) != outputs || out.empty()) {
            throw std::invalid_argument("FitSurrogateTree: the model's outputs differ in size between instances");
        }
        if (options.task == SurrogateTask::Classification) {
            y.push_back(static_cast<double>(std::max_element(out.begin(), out.end()) - out.begin()));
        } else {
            if (options.target < 0 || options.target >= outputs) {
                throw std::invalid_argument("FitSurrogateTree: target is outside the model's output");
            }
            y.push_back(out[static_cast<size_t>(options.target)]);
        }
    }

    // A seeded shuffle; the first ceil(n * test_fraction) are held out, as train_test_split counts them.
    std::vector<int> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    PortableRng rng{options.seed};
    for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.next() % i]);
    const auto n_test = static_cast<int64_t>(std::ceil(options.test_fraction * static_cast<double>(n)));
    if (n - n_test < 1) throw std::invalid_argument("FitSurrogateTree: no instances would be left to fit");
    const std::vector<int> test(order.begin(), order.begin() + n_test);
    std::vector<int> train(order.begin() + n_test, order.end());
    std::sort(train.begin(), train.end());

    std::vector<std::vector<float>> train_x;
    std::vector<double> train_y;
    for (int i : train) {
        train_x.push_back(x[static_cast<size_t>(i)]);
        train_y.push_back(y[static_cast<size_t>(i)]);
    }
    SurrogateTreeResult result;
    result.tree = FitDecisionTree(train_x, train_y, options.task, outputs, options);
    result.fidelity.train = Fidelity(result.tree, x, y, train);
    result.fidelity.test = Fidelity(result.tree, x, y, test);
    result.fidelity.train_size = static_cast<int64_t>(train.size());
    result.fidelity.test_size = n_test;
    result.feature_importance = FeatureImportance(result.tree);
    return result;
}

std::string DescribeSurrogateTree(const SurrogateTree& tree, const std::vector<std::string>& feature_names) {
    std::ostringstream out;
    if (!tree.nodes.empty()) DescribeNode(tree, 0, 0, feature_names, out);
    return out.str();
}

JsonValue ToJson(const SurrogateTreeResult& result, const std::vector<std::string>& feature_names) {
    const SurrogateTree& tree = result.tree;
    JsonValue::Array nodes, importance, names;
    for (const SurrogateNode& n : tree.nodes) {
        JsonValue node(JsonValue::Object{
            {"feature", JsonValue(n.feature)},
            {"threshold", JsonValue(n.threshold)},
            {"left", JsonValue(static_cast<int64_t>(n.left))},
            {"right", JsonValue(static_cast<int64_t>(n.right))},
            {"samples", JsonValue(n.samples)},
            {"impurity", JsonValue(n.impurity)},
            {"value", JsonValue(n.value)},
        });
        if (tree.task == SurrogateTask::Classification) {
            JsonValue::Array counts;
            for (int64_t c : n.class_counts) counts.push_back(JsonValue(c));
            node.add("class_counts", JsonValue(std::move(counts)));
        }
        nodes.push_back(std::move(node));
    }
    for (double v : result.feature_importance) importance.push_back(JsonValue(v));
    for (int64_t f = 0; f < tree.num_features; ++f) names.push_back(JsonValue(FeatureName(f, feature_names)));
    const SurrogateFidelity& fid = result.fidelity;
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.surrogate_tree.v1"},
        {"task", std::string(to_string(tree.task))},
        {"num_features", JsonValue(tree.num_features)},
        {"num_classes", JsonValue(tree.num_classes)},
        {"feature_names", JsonValue(std::move(names))},
        {"depth", JsonValue(static_cast<int64_t>(tree.depth()))},
        {"leaves", JsonValue(tree.leaves())},
        {"nodes", JsonValue(std::move(nodes))},
        {"fidelity", JsonValue(JsonValue::Object{
                         {"metric", tree.task == SurrogateTask::Classification ? "agreement" : "r2"},
                         {"train", JsonValue(fid.train)},
                         {"test", fid.test_size > 0 ? JsonValue(fid.test) : JsonValue(nullptr)},
                         {"train_size", JsonValue(fid.train_size)},
                         {"test_size", JsonValue(fid.test_size)},
                     })},
        {"feature_importance", JsonValue(std::move(importance))},
    });
}

std::string_view to_string(SurrogateTask task) {
    switch (task) {
        case SurrogateTask::Classification: return "classification";
        case SurrogateTask::Regression: return "regression";
    }
    throw std::invalid_argument("to_string: unknown SurrogateTask");
}

}  // namespace pulsatrix
