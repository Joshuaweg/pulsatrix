/** @file rule_extraction.hpp
 *  @brief Rules from trees, run by the Datalog engine (roadmap KD-3): a surrogate or distilled
 *         tree becomes a Datalog program that classifies the same way.
 *  @ingroup interpretability_agnostic
 *
 * Every root-to-leaf path of a tree is one rule. A crisp tree (FitDecisionTree's, or
 * FitSurrogateTree's, distilled from a model) gives
 *
 *   predict(S, versicolor) :- sample(S), gt(S, petal_length, 2.45), le(S, petal_width, 1.75).
 *
 * with the conditions on one feature merged to the tightest (two `le` on a feature keep the
 * smaller threshold), and facts `le(s, f, t)` or `gt(s, f, t)` for each sample and each threshold
 * the rules use. The Datalog engine's fixpoint then holds `predict(s, c)` for exactly the class the
 * tree predicts.
 *
 * A SoftDecisionTree gives one rule per leaf over weighted facts: `right(s, n)` with the
 * probability of going right at node n, `left(s, n)` with its complement, and `leaf(l, c)` with
 * leaf l's probability of class c:
 *
 *   predict(S, C) :- left(S, n0), right(S, n1), leaf(l1, C).
 *
 * Evaluated over the real semiring, which sums over derivations the product of their facts'
 * weights, `predict(s, c)` is Σ_ℓ P_ℓ(s) Q_ℓc: exactly the tree's class probability.
 *
 * Datalog constants are strings, so thresholds are written with 9 significant digits, enough to
 * tell any two floats apart; the facts are written the same way, so rules and facts match.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/datalog_fact_database.hpp"
#include "pulsatrix/datalog_rule.hpp"
#include "pulsatrix/datalog_weighted_fact_database.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/soft_decision_tree.hpp"
#include "pulsatrix/surrogate_tree.hpp"

namespace pulsatrix {

/** @brief One test on a path: x[feature] > threshold when `greater`, x[feature] <= threshold
 *         otherwise. */
struct RuleCondition {
    int64_t feature = 0;
    bool greater = false;
    double threshold = 0.0;
};

/** @brief One leaf of a crisp tree as a rule. */
struct ExtractedRule {
    /** @brief The path's tests, merged per feature and direction, in the order they're first met. */
    std::vector<RuleCondition> conditions;
    int64_t predicted_class = 0;
    /** @brief The leaf's node index in the tree. */
    int64_t leaf = 0;
    /** @brief Training samples that reached the leaf, and the share of them in its class. */
    int64_t support = 0;
    double confidence = 0.0;
    /** @brief `predict(S, <class>) :- sample(S), <conditions>.` */
    datalog::Rule rule;
};

/** @brief A crisp tree's rules, one per leaf. */
struct TreeRules {
    std::vector<ExtractedRule> rules;
    std::vector<std::string> feature_names;
    std::vector<std::string> class_names;

    /** @brief The rules alone, for the engine. */
    [[nodiscard]] std::vector<datalog::Rule> Program() const;
    /** @brief `sample(<id>)` and an `le` or `gt` fact for every (feature, threshold) the rules
     *         test. @throws std::invalid_argument if x has fewer values than the tree's features. */
    [[nodiscard]] datalog::FactDatabase Facts(const std::string& sample, const std::vector<float>& x) const;
};

/**
 * @brief The rules of a classification tree, one per leaf, in the tree's node order.
 * @param feature_names Names for the features, or empty for x0, x1, ...
 * @param class_names Names for the classes, or empty for class0, class1, ...
 * @throws std::invalid_argument for a regression tree, or names of the wrong count or that aren't
 *         distinct.
 */
[[nodiscard]] TreeRules ExtractRules(const SurrogateTree& tree, const std::vector<std::string>& feature_names = {},
                                     const std::vector<std::string>& class_names = {});

/**
 * @brief Runs the rules on each input with the Datalog engine and returns the class whose
 *        `predict` fact it derives, or -1 if none.
 */
[[nodiscard]] std::vector<int64_t> ClassifyWithRules(const TreeRules& rules, const std::vector<std::vector<float>>& inputs);

/** @brief A soft decision tree's rules, one per leaf, over weighted facts. */
struct SoftTreeRules {
    std::vector<datalog::Rule> rules;
    std::vector<std::string> class_names;
    int64_t inner = 0;
    int64_t leaves = 0;

    /** @brief The weighted facts for row @p row of @p tree's last forward(): right and left with
     *         its gate probabilities, and leaf with the leaves' distributions.
     *  @throws std::invalid_argument if the tree's shape isn't the rules' or row is out of range;
     *          std::logic_error before the tree's first forward(). */
    [[nodiscard]] datalog::WeightedFactDatabase<double> Facts(const SoftDecisionTree& tree, int64_t row,
                                                              const std::string& sample) const;
};

/** @brief The rules of a soft decision tree. @throws std::invalid_argument if the class names'
 *         count is wrong or they aren't distinct. */
[[nodiscard]] SoftTreeRules ExtractRules(const SoftDecisionTree& tree, const std::vector<std::string>& class_names = {});

/**
 * @brief Each class's probability for row @p row of @p tree's last forward(), from the rules
 *        evaluated over the real semiring.
 */
[[nodiscard]] std::vector<double> ClassProbabilitiesFromRules(const SoftTreeRules& rules, const SoftDecisionTree& tree, int64_t row);

/** @brief A rule in Datalog's usual text: `head(A, b) :- body1(A), body2(A, c).` */
[[nodiscard]] std::string FormatRule(const datalog::Rule& rule);

/** @brief The rules as a `pulsatrix.tree_rules.v1` document: each rule's text, conditions,
 *         class, support and confidence. */
[[nodiscard]] JsonValue ToJson(const TreeRules& rules);

}  // namespace pulsatrix
