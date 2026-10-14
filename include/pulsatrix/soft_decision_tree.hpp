/** @file soft_decision_tree.hpp
 *  @brief A soft decision tree (Frosst and Hinton, "Distilling a Neural Network Into a Soft
 *         Decision Tree", 2017): an interpretable student for a network teacher (roadmap KD-3).
 *  @ingroup dl_modules
 *
 * A complete binary tree of depth d. Inner node i sends an input right with probability
 * p_i(x) = σ(β (x · w_i + b_i)) and left with 1 − p_i(x); each leaf ℓ holds a distribution
 * Q_ℓ = softmax(φ_ℓ) over the classes. An input reaches leaf ℓ with probability P_ℓ(x), the
 * product of the choices on its path, and the tree's output is the mixture
 *
 *   y(x) = log Σ_ℓ P_ℓ(x) Q_ℓ,
 *
 * log-probabilities, so it trains with DistillationLoss on a teacher's logits (softmax(y) is the
 * mixture itself) or with CrossEntropyLoss on labels. Every decision is one learned linear filter,
 * which can be read directly: Frosst and Hinton show them as images. To explain one prediction,
 * follow the most probable path, MostProbableLeaf(), and read its filters and its leaf.
 *
 * Frosst and Hinton's penalty keeps every inner node from sending nearly everything one way:
 * with α_i the share of the probability reaching node i that goes right over a batch,
 *
 *   C = −λ Σ_i 2^(−depth(i)) · ½ (log α_i + log(1 − α_i)),
 *
 * added to the loss when `penalty` is positive. Unlike the paper, α is the current batch's alone,
 * not a moving average over batches.
 *
 * Nodes are numbered as a heap: node i's children are 2i + 1 (left) and 2i + 2 (right); the
 * 2^d − 1 inner nodes come first and leaf ℓ is node 2^d − 1 + ℓ.
 *
 * tests/soft_decision_tree_test.cpp checks the output, the gradients and the penalty against
 * tools/generate_soft_decision_tree_reference_values.py (numpy, gradients by central differences).
 */
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/module.hpp"

namespace pulsatrix {

/** @brief SoftDecisionTree's settings. */
struct SoftDecisionTreeOptions {
    /** @brief Levels of inner nodes: 2^depth leaves. Between 1 and 16. */
    int depth = 4;
    /** @brief The gates' inverse temperature β: larger is closer to a hard tree. */
    float beta = 1.0f;
    /** @brief The balance penalty's λ; 0 turns it off. Frosst and Hinton use 0.1 or so. */
    float penalty = 0.0f;
    /** @brief The gates' weights and biases start uniform in ±1/√inputs, as nn.Linear's do; the
     *         leaves start at zero, the uniform distribution. */
    uint64_t seed = 0;
};

/**
 * @brief Frosst and Hinton's soft decision tree, (N, inputs) to (N, classes) log-probabilities.
 * @note The routing and the leaves are computed on the host in double precision; the gates' linear
 *       layer runs on the backend's device, and the output comes back on the input's device.
 * @note LRP isn't defined for it: propagate_relevance() throws. Explain it with its paths, or with
 *       gradient explainers.
 */
class SoftDecisionTree : public Module {
public:
    SoftDecisionTree(int64_t inputs, int64_t classes, DeviceBackend* backend, SoftDecisionTreeOptions options = {});

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] bool supports_lrp_rule(LRPRule) const override { return false; }
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    /** @brief `gates.weight`, `gates.bias` and `leaves`. */
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedModuleRef> named_children() override;

    [[nodiscard]] int depth() const { return options_.depth; }
    [[nodiscard]] int64_t num_inner() const { return (int64_t{1} << options_.depth) - 1; }
    [[nodiscard]] int64_t num_leaves() const { return int64_t{1} << options_.depth; }
    [[nodiscard]] int64_t num_inputs() const { return inputs_; }
    [[nodiscard]] int64_t num_classes() const { return classes_; }
    [[nodiscard]] const SoftDecisionTreeOptions& options() const { return options_; }

    /** @brief The gates: column i of its (inputs, inner) weight and entry i of its bias are node i's
     *         w_i and b_i. */
    [[nodiscard]] LinearModule& gates() { return gates_; }
    /** @brief The leaves' logits φ, (leaves, classes). */
    [[nodiscard]] Tensor& leaf_logits() { return leaves_; }
    /** @brief The leaves' distributions softmax(φ), row-major (leaves, classes). */
    [[nodiscard]] std::vector<double> LeafDistributions() const;

    /** @brief From the last forward(): each input's probability of going right at each inner node,
     *         (N, inner), and of reaching each leaf, (N, leaves), row-major. */
    [[nodiscard]] const std::vector<double>& right_probabilities() const { return p_; }
    [[nodiscard]] const std::vector<double>& leaf_probabilities() const { return reach_leaf_; }
    /** @brief The last forward()'s balance penalty C, λ included; 0 when λ is 0. Add it to the loss
     *         you report: backward() adds its gradient. */
    [[nodiscard]] double penalty() const { return penalty_; }

    /** @brief For each input of the last forward(), the leaf its most probable path ends at: the
     *         greedy path, taking the likelier branch at every node, as Frosst and Hinton do. */
    [[nodiscard]] std::vector<int64_t> MostProbableLeaf() const;
    /** @brief The inner nodes on the path from the root to @p leaf, root first, and whether the path
     *         goes right at each. */
    [[nodiscard]] std::vector<std::pair<int64_t, bool>> PathTo(int64_t leaf) const;

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    DeviceBackend* backend_;
    int64_t inputs_;
    int64_t classes_;
    SoftDecisionTreeOptions options_;
    LinearModule gates_;
    Tensor leaves_;
    Tensor leaves_grad_;
    // The last forward()'s values, row-major.
    std::vector<double> p_;           // (N, inner)
    std::vector<double> reach_;       // (N, inner): probability of reaching each inner node
    std::vector<double> reach_leaf_;  // (N, leaves)
    std::vector<double> q_;           // (leaves, classes)
    std::vector<double> mixture_;     // (N, classes)
    double penalty_ = 0.0;
    int64_t batch_ = 0;
    DeviceType input_device_ = DeviceType::Cpu;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
