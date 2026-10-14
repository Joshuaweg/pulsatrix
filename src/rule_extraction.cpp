// Rule extraction (KD-3): a tree's root-to-leaf paths as Datalog rules that the engine runs.

#include "pulsatrix/rule_extraction.hpp"

#include <algorithm>
#include <cstdio>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/datalog_engine.hpp"
#include "pulsatrix/datalog_semiring.hpp"
#include "pulsatrix/datalog_weighted_engine.hpp"

namespace pulsatrix {

namespace {

using datalog::Atom;
using datalog::Rule;
using datalog::Term;

Term C(const std::string& v) { return Term::make_constant(v); }
Term V(const std::string& v) { return Term::make_variable(v); }

// 9 significant digits tell any two floats apart.
std::string Number(double v) {
    char buffer[32];
    std::snprintf(buffer, sizeof buffer, "%.9g", v);
    return buffer;
}

std::vector<std::string> Names(const std::vector<std::string>& given, int64_t count, const std::string& prefix, const char* what) {
    if (given.empty()) {
        std::vector<std::string> out;
        for (int64_t k = 0; k < count; ++k) out.push_back(prefix + std::to_string(k));
        return out;
    }
    if (static_cast<int64_t>(given.size()) != count) {
        throw std::invalid_argument(std::string("ExtractRules: expected ") + std::to_string(count) + " " + what + " names, got " +
                                    std::to_string(given.size()));
    }
    if (std::set<std::string>(given.begin(), given.end()).size() != given.size()) {
        throw std::invalid_argument(std::string("ExtractRules: the ") + what + " names must be distinct");
    }
    return given;
}

Atom Condition(const std::string& sample, const RuleCondition& c, const std::vector<std::string>& features) {
    return Atom(c.greater ? "gt" : "le", {V(sample), C(features[static_cast<size_t>(c.feature)]), C(Number(c.threshold))});
}

// Adds a test to a path, keeping the tightest bound per feature and direction.
void Merge(std::vector<RuleCondition>& path, RuleCondition c) {
    for (RuleCondition& kept : path) {
        if (kept.feature == c.feature && kept.greater == c.greater) {
            kept.threshold = c.greater ? std::max(kept.threshold, c.threshold) : std::min(kept.threshold, c.threshold);
            return;
        }
    }
    path.push_back(c);
}

void Walk(const SurrogateTree& tree, int node, std::vector<RuleCondition> path, const std::vector<std::string>& features,
          const std::vector<std::string>& classes, std::vector<ExtractedRule>& out) {
    const SurrogateNode& n = tree.nodes[static_cast<size_t>(node)];
    if (n.is_leaf()) {
        const auto cls = static_cast<int64_t>(n.value);
        std::vector<Atom> body{Atom("sample", {V("S")})};
        for (const RuleCondition& c : path) body.push_back(Condition("S", c, features));
        const int64_t in_class = cls < static_cast<int64_t>(n.class_counts.size()) ? n.class_counts[static_cast<size_t>(cls)] : 0;
        out.push_back(ExtractedRule{path, cls, node, n.samples, n.samples > 0 ? static_cast<double>(in_class) / static_cast<double>(n.samples) : 0.0,
                                    Rule(Atom("predict", {V("S"), C(classes[static_cast<size_t>(cls)])}), std::move(body))});
        return;
    }
    std::vector<RuleCondition> left = path;
    Merge(left, RuleCondition{n.feature, false, n.threshold});
    Walk(tree, n.left, std::move(left), features, classes, out);
    Merge(path, RuleCondition{n.feature, true, n.threshold});
    Walk(tree, n.right, std::move(path), features, classes, out);
}

std::string FormatAtom(const Atom& a) {
    std::string s = a.predicate_name() + "(";
    for (size_t k = 0; k < a.terms().size(); ++k) s += (k ? ", " : "") + a.terms()[k].value();
    return s + ")";
}

}  // namespace

std::vector<datalog::Rule> TreeRules::Program() const {
    std::vector<Rule> out;
    for (const ExtractedRule& r : rules) out.push_back(r.rule);
    return out;
}

datalog::FactDatabase TreeRules::Facts(const std::string& sample, const std::vector<float>& x) const {
    if (x.size() < feature_names.size()) {
        throw std::invalid_argument("TreeRules::Facts: the input has " + std::to_string(x.size()) + " values, not " +
                                    std::to_string(feature_names.size()));
    }
    datalog::FactDatabase db;
    db.insert(Atom("sample", {C(sample)}));
    for (const ExtractedRule& r : rules) {
        for (const RuleCondition& c : r.conditions) {
            // The same test SurrogateTree::leaf() makes: x <= threshold goes left.
            const bool greater = !(x[static_cast<size_t>(c.feature)] <= c.threshold);
            db.insert(Atom(greater ? "gt" : "le", {C(sample), C(feature_names[static_cast<size_t>(c.feature)]), C(Number(c.threshold))}));
        }
    }
    return db;
}

TreeRules ExtractRules(const SurrogateTree& tree, const std::vector<std::string>& feature_names, const std::vector<std::string>& class_names) {
    if (tree.task != SurrogateTask::Classification) throw std::invalid_argument("ExtractRules: the tree must be a classification tree");
    if (tree.nodes.empty()) throw std::invalid_argument("ExtractRules: the tree has no nodes");
    TreeRules out;
    out.feature_names = Names(feature_names, tree.num_features, "x", "feature");
    out.class_names = Names(class_names, tree.num_classes, "class", "class");
    Walk(tree, 0, {}, out.feature_names, out.class_names, out.rules);
    return out;
}

std::vector<int64_t> ClassifyWithRules(const TreeRules& rules, const std::vector<std::vector<float>>& inputs) {
    const std::vector<Rule> program = rules.Program();
    std::vector<int64_t> out;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const std::string sample = "s" + std::to_string(i);
        const datalog::FactDatabase fixpoint = datalog::semi_naive_evaluate(program, rules.Facts(sample, inputs[i]));
        int64_t found = -1;
        for (size_t k = 0; k < rules.class_names.size(); ++k) {
            if (fixpoint.contains(Atom("predict", {C(sample), C(rules.class_names[k])}))) {
                found = static_cast<int64_t>(k);
                break;
            }
        }
        out.push_back(found);
    }
    return out;
}

SoftTreeRules ExtractRules(const SoftDecisionTree& tree, const std::vector<std::string>& class_names) {
    SoftTreeRules out;
    out.class_names = Names(class_names, tree.num_classes(), "class", "class");
    out.inner = tree.num_inner();
    out.leaves = tree.num_leaves();
    for (int64_t l = 0; l < out.leaves; ++l) {
        std::vector<Atom> body;
        for (const auto& [node, right] : tree.PathTo(l)) body.push_back(Atom(right ? "right" : "left", {V("S"), C("n" + std::to_string(node))}));
        body.push_back(Atom("leaf", {C("l" + std::to_string(l)), V("C")}));
        out.rules.emplace_back(Atom("predict", {V("S"), V("C")}), std::move(body));
    }
    return out;
}

datalog::WeightedFactDatabase<double> SoftTreeRules::Facts(const SoftDecisionTree& tree, int64_t row, const std::string& sample) const {
    if (tree.num_inner() != inner || tree.num_leaves() != leaves || tree.num_classes() != static_cast<int64_t>(class_names.size())) {
        throw std::invalid_argument("SoftTreeRules::Facts: the tree's shape isn't the one the rules came from");
    }
    const std::vector<double>& p = tree.right_probabilities();
    if (p.empty()) throw std::logic_error("SoftTreeRules::Facts: the tree hasn't run forward()");
    const auto rows = static_cast<int64_t>(p.size()) / inner;
    if (row < 0 || row >= rows) throw std::invalid_argument("SoftTreeRules::Facts: row is out of range");
    datalog::WeightedFactDatabase<double> db;
    for (int64_t i = 0; i < inner; ++i) {
        const double right = p[static_cast<size_t>(row * inner + i)];
        db.set(Atom("right", {C(sample), C("n" + std::to_string(i))}), right);
        db.set(Atom("left", {C(sample), C("n" + std::to_string(i))}), 1.0 - right);
    }
    const std::vector<double> q = tree.LeafDistributions();
    const auto classes = static_cast<int64_t>(class_names.size());
    for (int64_t l = 0; l < leaves; ++l) {
        for (int64_t k = 0; k < classes; ++k) {
            db.set(Atom("leaf", {C("l" + std::to_string(l)), C(class_names[static_cast<size_t>(k)])}), q[static_cast<size_t>(l * classes + k)]);
        }
    }
    return db;
}

std::vector<double> ClassProbabilitiesFromRules(const SoftTreeRules& rules, const SoftDecisionTree& tree, int64_t row) {
    const std::string sample = "s" + std::to_string(row);
    const auto fixpoint = datalog::semi_naive_evaluate_weighted<datalog::RealSemiring<double>>(rules.rules, rules.Facts(tree, row, sample));
    std::vector<double> out;
    for (const std::string& c : rules.class_names) out.push_back(fixpoint.weight_of(Atom("predict", {C(sample), C(c)}), 0.0));
    return out;
}

std::string FormatRule(const datalog::Rule& rule) {
    std::string s = FormatAtom(rule.head()) + " :- ";
    for (size_t k = 0; k < rule.body().size(); ++k) s += (k ? ", " : "") + FormatAtom(rule.body()[k]);
    return s + ".";
}

JsonValue ToJson(const TreeRules& rules) {
    JsonValue::Array list;
    for (const ExtractedRule& r : rules.rules) {
        JsonValue::Array conditions;
        for (const RuleCondition& c : r.conditions) {
            conditions.push_back(JsonValue(JsonValue::Object{
                {"feature", JsonValue(rules.feature_names[static_cast<size_t>(c.feature)])},
                {"op", JsonValue(c.greater ? ">" : "<=")},
                {"threshold", JsonValue(c.threshold)},
            }));
        }
        list.push_back(JsonValue(JsonValue::Object{
            {"rule", JsonValue(FormatRule(r.rule))},
            {"class", JsonValue(rules.class_names[static_cast<size_t>(r.predicted_class)])},
            {"conditions", JsonValue(std::move(conditions))},
            {"support", JsonValue(r.support)},
            {"confidence", JsonValue(r.confidence)},
        }));
    }
    JsonValue::Array features, classes;
    for (const std::string& f : rules.feature_names) features.push_back(JsonValue(f));
    for (const std::string& c : rules.class_names) classes.push_back(JsonValue(c));
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.tree_rules.v1"},
        {"features", JsonValue(std::move(features))},
        {"classes", JsonValue(std::move(classes))},
        {"rules", JsonValue(std::move(list))},
    });
}

}  // namespace pulsatrix
