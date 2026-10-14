// KD-3: a tree's paths as Datalog rules. The engine's fixpoint classifies as the crisp tree does, and
// the soft tree's weighted rules give its class probabilities.
#include "pulsatrix/rule_extraction.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/datalog_engine.hpp"

namespace pulsatrix {
namespace {

SurrogateNode Split(int64_t feature, double threshold, int left, int right) {
    SurrogateNode n;
    n.feature = feature;
    n.threshold = threshold;
    n.left = left;
    n.right = right;
    return n;
}

SurrogateNode Leaf(int64_t cls, std::vector<int64_t> counts) {
    SurrogateNode n;
    n.value = static_cast<double>(cls);
    n.class_counts = counts;
    for (int64_t c : counts) n.samples += c;
    return n;
}

// x0 <= 0.5 ? (x0 <= 0.25 ? a : (x1 <= 0.3 ? b : a)) : c, numbered depth-first.
SurrogateTree HandTree() {
    SurrogateTree t;
    t.num_features = 2;
    t.num_classes = 3;
    t.nodes = {Split(0, 0.5, 1, 6), Split(0, 0.25, 2, 3), Leaf(0, {8, 1, 0}), Split(1, 0.3, 4, 5),
               Leaf(1, {1, 5, 0}), Leaf(0, {3, 0, 1}), Leaf(2, {0, 2, 6})};
    return t;
}

TEST(RuleExtractionTest, OneRulePerLeafWithTheTightestBounds) {
    const TreeRules r = ExtractRules(HandTree(), {"length", "width"}, {"a", "b", "c"});
    ASSERT_EQ(r.rules.size(), 4u);
    EXPECT_EQ(FormatRule(r.rules[0].rule), "predict(S, a) :- sample(S), le(S, length, 0.25).");
    EXPECT_EQ(FormatRule(r.rules[1].rule), "predict(S, b) :- sample(S), le(S, length, 0.5), gt(S, length, 0.25), le(S, width, 0.3).");
    EXPECT_EQ(FormatRule(r.rules[2].rule), "predict(S, a) :- sample(S), le(S, length, 0.5), gt(S, length, 0.25), gt(S, width, 0.3).");
    EXPECT_EQ(FormatRule(r.rules[3].rule), "predict(S, c) :- sample(S), gt(S, length, 0.5).");
    EXPECT_EQ(r.rules[0].leaf, 2);
    EXPECT_EQ(r.rules[0].support, 9);
    EXPECT_NEAR(r.rules[0].confidence, 8.0 / 9.0, 1e-12);
    EXPECT_EQ(r.rules[3].predicted_class, 2);
    const std::string json = WriteJsonCompact(ToJson(r));
    EXPECT_NE(json.find("pulsatrix.tree_rules.v1"), std::string::npos);
    EXPECT_NE(json.find("\"op\":\"<=\""), std::string::npos);
}

TEST(RuleExtractionTest, TheEngineClassifiesAsTheTreeDoes) {
    // Three classes from four features: regions of the first two, the others noise.
    std::vector<std::vector<float>> x;
    std::vector<double> y;
    for (int k = 0; k < 300; ++k) {
        const float a = static_cast<float>(std::sin(1.7 * k)), b = static_cast<float>(std::cos(2.3 * k + 0.4));
        x.push_back({a, b, static_cast<float>(std::sin(0.31 * k)), static_cast<float>(std::cos(0.53 * k))});
        y.push_back(a + b > 0.5 ? 2.0 : a - b > 0.2 ? 1.0 : 0.0);
    }
    SurrogateTreeOptions o;
    o.max_depth = 5;
    const SurrogateTree tree = FitDecisionTree(x, y, SurrogateTask::Classification, 3, o);
    const TreeRules rules = ExtractRules(tree);
    EXPECT_EQ(static_cast<int64_t>(rules.rules.size()), tree.leaves());
    int64_t support = 0;
    for (const ExtractedRule& r : rules.rules) support += r.support;
    EXPECT_EQ(support, 300);
    const std::vector<int64_t> classes = ClassifyWithRules(rules, x);
    ASSERT_EQ(classes.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) EXPECT_EQ(classes[i], static_cast<int64_t>(tree.predict(x[i]))) << i;
    // And exactly one rule fires for each input.
    const std::vector<datalog::Rule> program = rules.Program();
    for (size_t i = 0; i < 20; ++i) {
        const datalog::FactDatabase fix = datalog::semi_naive_evaluate(program, rules.Facts("s", x[i]));
        int fired = 0;
        for (const std::string& c : rules.class_names) {
            fired += fix.contains(datalog::Atom("predict", {datalog::Term::make_constant("s"), datalog::Term::make_constant(c)}));
        }
        EXPECT_EQ(fired, 1) << i;
    }
}

TEST(RuleExtractionTest, TheSoftTreesRulesGiveItsProbabilities) {
    CPUBackend backend;
    SoftDecisionTreeOptions o;
    o.depth = 3;
    o.seed = 4;
    SoftDecisionTree tree(3, 4, &backend, o);
    std::vector<float> phi;
    for (int k = 0; k < 32; ++k) phi.push_back(static_cast<float>(1.2 * std::sin(0.9 * k + 0.3)));
    tree.leaf_logits() = Tensor(Shape({8, 4}), &backend, phi);
    std::vector<float> x;
    for (int k = 0; k < 18; ++k) x.push_back(static_cast<float>(std::sin(0.61 * k + 0.2)));
    const std::vector<float> y = tree.forward(Tensor(Shape({6, 3}), &backend, x)).to_host_vector();
    const SoftTreeRules rules = ExtractRules(tree, {"w", "x", "y", "z"});
    ASSERT_EQ(rules.rules.size(), 8u);
    EXPECT_EQ(FormatRule(rules.rules[5]), "predict(S, C) :- right(S, n0), left(S, n2), right(S, n5), leaf(l5, C).");
    const std::vector<double> q = tree.LeafDistributions();
    for (int64_t r = 0; r < 6; ++r) {
        const std::vector<double> p = ClassProbabilitiesFromRules(rules, tree, r);
        ASSERT_EQ(p.size(), 4u);
        double total = 0;
        for (size_t c = 0; c < 4; ++c) {
            double mixture = 0;
            for (size_t l = 0; l < 8; ++l) mixture += tree.leaf_probabilities()[static_cast<size_t>(r) * 8 + l] * q[l * 4 + c];
            EXPECT_NEAR(p[c], mixture, 1e-12) << r << " " << c;
            EXPECT_NEAR(std::log(p[c]), y[static_cast<size_t>(r) * 4 + c], 1e-6);
            total += p[c];
        }
        EXPECT_NEAR(total, 1.0, 1e-12);
    }
    EXPECT_THROW((void)rules.Facts(tree, 6, "s"), std::invalid_argument);
}

TEST(RuleExtractionTest, RejectsWhatItCantTranslate) {
    SurrogateTree regression = HandTree();
    regression.task = SurrogateTask::Regression;
    EXPECT_THROW((void)ExtractRules(regression), std::invalid_argument);
    EXPECT_THROW((void)ExtractRules(HandTree(), {"only_one"}), std::invalid_argument);
    EXPECT_THROW((void)ExtractRules(HandTree(), {"same", "same"}), std::invalid_argument);
    EXPECT_THROW((void)ExtractRules(HandTree(), {}, {"a", "b"}), std::invalid_argument);
    EXPECT_THROW((void)ExtractRules(HandTree()).Facts("s", {0.1f}), std::invalid_argument);
    CPUBackend backend;
    SoftDecisionTreeOptions o;
    o.depth = 2;
    SoftDecisionTree tree(3, 2, &backend, o);
    const SoftTreeRules rules = ExtractRules(tree);
    EXPECT_THROW((void)rules.Facts(tree, 0, "s"), std::logic_error);
}

}  // namespace
}  // namespace pulsatrix
