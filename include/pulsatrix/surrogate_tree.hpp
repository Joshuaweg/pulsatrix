/** @file surrogate_tree.hpp
 *  @brief A global surrogate: a shallow decision tree fitted to a model's predictions, with its
 *         fidelity to the model on held-out inputs (XAI-4; TREPAN, Craven and Shavlik 1996).
 *  @ingroup interpretability_agnostic
 *
 * A surrogate answers "how does the model work overall?" with rules a person can read, and its
 * fidelity says how far to trust them: a tree that agrees with the model on 70% of new inputs
 * describes a different model 30% of the time. Below about 85% at a readable depth, the tree isn't
 * a description of the model (the falsifier in the roadmap's research notes).
 *
 * The tree is CART as scikit-learn builds it (DecisionTreeClassifier and DecisionTreeRegressor
 * with splitter="best"): Gini or squared-error impurity, thresholds halfway between consecutive
 * distinct values, `x <= threshold` going left, nodes numbered depth first. Ties between features
 * go to the lower index, where scikit-learn breaks them at random.
 *
 * @code
 * SurrogateTreeOptions options;
 * options.max_depth = 3;
 * SurrogateTreeResult s = FitSurrogateTree(predict, background, options);
 * std::cout << DescribeSurrogateTree(s.tree, feature_names);
 * // s.fidelity.test: the share of held-out inputs where the tree's class is the model's
 * @endcode
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "pulsatrix/explainer_capabilities.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief What the surrogate imitates. */
enum class SurrogateTask {
    /// The model's class: the index of its largest output. Fidelity is agreement.
    Classification,
    /// One output's value. Fidelity is R².
    Regression
};

/** @brief How the tree is grown. */
struct SurrogateTreeOptions {
    SurrogateTask task = SurrogateTask::Classification;
    /** @brief Regression: the output (flat index) imitated. */
    int64_t target = 0;
    /** @brief The deepest a leaf may be; the root is depth 0. Three or four levels stay readable. */
    int max_depth = 3;
    /** @brief The fewest inputs a leaf may hold. */
    int64_t min_samples_leaf = 1;
    /** @brief FitSurrogateTree: the share of the inputs held out to measure fidelity, in [0, 1). */
    double test_fraction = 0.3;
    /** @brief FitSurrogateTree: seeds the held-out split. */
    uint64_t seed = 0;
};

/** @brief One node. A leaf has feature -1. */
struct SurrogateNode {
    int64_t feature = -1;
    /** @brief Inputs with `x[feature] <= threshold` go to left. */
    double threshold = 0.0;
    int left = -1;
    int right = -1;
    /** @brief Training inputs that reached the node. */
    int64_t samples = 0;
    /** @brief Gini impurity, or the variance of the target. */
    double impurity = 0.0;
    /** @brief The mean target, or the majority class (lowest index on a tie). */
    double value = 0.0;
    /** @brief Classification: how many training inputs of each class reached the node. */
    std::vector<int64_t> class_counts;

    [[nodiscard]] bool is_leaf() const { return feature < 0; }
};

/** @brief A fitted tree. nodes[0] is the root. */
struct SurrogateTree {
    SurrogateTask task = SurrogateTask::Classification;
    int64_t num_features = 0;
    /** @brief Classification: the number of classes (the model's output size). */
    int64_t num_classes = 0;
    std::vector<SurrogateNode> nodes;

    /** @brief The leaf @p x falls in. @throws std::invalid_argument if x has the wrong size. */
    [[nodiscard]] int leaf(const std::vector<float>& x) const;
    /** @brief The leaf's value: a class index, or the predicted target. */
    [[nodiscard]] double predict(const std::vector<float>& x) const;
    [[nodiscard]] int depth() const;
    [[nodiscard]] int64_t leaves() const;
};

/** @brief How well the tree imitates the model. */
struct SurrogateFidelity {
    /** @brief On the inputs the tree was fitted to. */
    double train = 0.0;
    /** @brief On the held-out inputs; the number to report. NaN when nothing was held out. */
    double test = 0.0;
    int64_t train_size = 0;
    int64_t test_size = 0;
};

/** @brief FitSurrogateTree()'s result. */
struct SurrogateTreeResult {
    SurrogateTree tree;
    SurrogateFidelity fidelity;
    /** @brief Each feature's total impurity decrease, weighted by the inputs it split, summing to 1
     *         (scikit-learn's feature_importances_). All zero for a tree with no split. */
    std::vector<double> feature_importance;
};

/**
 * @brief Fits CART to rows @p x and targets @p y: class indices for classification, values for
 *        regression. FitSurrogateTree() calls this on the model's predictions; it is public so the
 *        tree can be checked on its own.
 * @param num_classes Classification: the number of classes; every y must be in [0, num_classes).
 * @throws std::invalid_argument if x is empty, rows differ in length or don't match y, a class is
 *         out of range, max_depth < 0 or min_samples_leaf < 1.
 */
[[nodiscard]] SurrogateTree FitDecisionTree(const std::vector<std::vector<float>>& x, const std::vector<double>& y,
                                            SurrogateTask task, int64_t num_classes, const SurrogateTreeOptions& options);

/** @brief Each feature's share of the tree's impurity decrease (scikit-learn's feature_importances_). */
[[nodiscard]] std::vector<double> FeatureImportance(const SurrogateTree& tree);

/**
 * @brief Labels @p instances with the model, holds out options.test_fraction of them, fits a tree to
 *        the rest and measures its fidelity on both parts.
 * @param predict The model as a function of one instance (a leading batch dimension of 1).
 * @param instances The inputs to imitate the model on, each flattened to one row of features.
 * @throws std::invalid_argument if instances is empty, their sizes differ, test_fraction isn't in
 *         [0, 1), fewer than one input would be left to fit, or target is outside the output.
 */
[[nodiscard]] SurrogateTreeResult FitSurrogateTree(const std::function<Tensor(const Tensor&)>& predict,
                                                   const std::vector<Tensor>& instances,
                                                   const SurrogateTreeOptions& options = {});

/**
 * @brief The tree as nested if/else rules, one node per line:
 *        `if income <= 41.5:` ... `class 1 (40 inputs, 95%)`.
 * @param feature_names A name per feature; `x[3]` where missing.
 */
[[nodiscard]] std::string DescribeSurrogateTree(const SurrogateTree& tree,
                                                const std::vector<std::string>& feature_names = {});

/** @brief The result as a `pulsatrix.surrogate_tree.v1` document. */
[[nodiscard]] JsonValue ToJson(const SurrogateTreeResult& result, const std::vector<std::string>& feature_names = {});

[[nodiscard]] std::string_view to_string(SurrogateTask task);

}  // namespace pulsatrix
