// XAI-4: the global surrogate tree. FitDecisionTree is checked node by node against scikit-learn
// 1.9.1's DecisionTreeClassifier and DecisionTreeRegressor
// (tools/generate_surrogate_tree_reference_values.py), then FitSurrogateTree against models whose
// best tree is known.

#include "pulsatrix/surrogate_tree.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

// scikit-learn's tree_ arrays; a leaf's feature is -2 there.
struct Reference {
    std::vector<int64_t> feature;
    std::vector<double> threshold;
    std::vector<int64_t> samples;
    std::vector<double> impurity;
    std::vector<double> value;
    std::vector<double> importance;
};

// ---- Reference values: tools/generate_surrogate_tree_reference_values.py output, verbatim ----
const Reference kClassifier3 = {
    {0, 2, 3, -2, -2, 3, -2, -2, 3, 2, -2, -2, 1, -2, -2},
    {0.0638927910476923, -0.06579831801354885, -0.9266597032546997, -2.0, -2.0, 0.47266535460948944, -2.0, -2.0, -0.1670682057738304, -0.5823209285736084, -2.0, -2.0, -0.742924839258194, -2.0, -2.0},
    {200, 96, 49, 1, 48, 47, 37, 10, 104, 50, 10, 40, 54, 19, 35},
    {0.62915, 0.5010850694444444, 0.07830070803831735, 0.0, 0.04079861111111116, 0.4481665912177456, 0.19722425127830534, 0.0, 0.5850591715976332, 0.4952, 0.0, 0.38625, 0.46707818930041156, 0.5263157894736843, 0.25469387755102035},
    {1, 0, 0, 1, 0, 1, 1, 0, 1, 2, 1, 2, 1, 0, 1},
    {0.20532371135880337, 0.07672947438421011, 0.39549408953246484, 0.3224527247245217},
};
const Reference kClassifier4Leaf5 = {
    {0, 2, 3, -2, -2, 3, 0, -2, -2, -2, 3, 2, -2, 0, -2, -2, 1, 0, -2, -2, 1, -2, -2},
    {0.0638927910476923, -0.06579831801354885, -0.8438889980316162, -2.0, -2.0, 0.47266535460948944, -0.17726711928844452, -2.0, -2.0, -2.0, -0.1670682057738304, -0.5823209285736084, -2.0, 0.6741629540920258, -2.0, -2.0, -0.742924839258194, 0.6655350923538208, -2.0, -2.0, 0.7241766452789307, -2.0, -2.0},
    {200, 96, 49, 5, 44, 47, 37, 30, 7, 10, 104, 50, 10, 40, 18, 22, 54, 19, 12, 7, 35, 30, 5},
    {0.62915, 0.5010850694444444, 0.07830070803831735, 0.48, 0.0, 0.4481665912177456, 0.19722425127830534, 0.06444444444444442, 0.4897959183673469, 0.0, 0.5850591715976332, 0.4952, 0.0, 0.38625, 0.5493827160493827, 0.0, 0.46707818930041156, 0.5263157894736843, 0.1527777777777778, 0.24489795918367352, 0.25469387755102035, 0.12444444444444447, 0.48},
    {1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 2, 1, 2, 1, 2, 1, 0, 0, 1, 1, 1, 2},
    {0.3129858941052918, 0.09227461932287898, 0.33009318313433483, 0.2646463034374944},
};
const Reference kRegressor3 = {
    {2, 2, 1, -2, -2, 1, -2, -2, 3, 1, -2, -2, 3, -2, -2},
    {-0.03526741731911898, -0.8296501040458679, 0.9449009597301483, -2.0, -2.0, 0.6872364580631256, -2.0, -2.0, -0.523756206035614, -0.8867309093475342, -2.0, -2.0, 0.47266535460948944, -2.0, -2.0},
    {200, 100, 30, 28, 2, 70, 45, 25, 100, 41, 9, 32, 59, 37, 22},
    {0.7075494136114929, 0.3249389712170989, 0.12008385261678187, 0.07501643979334324, 0.00020416433814784973, 0.21866868611295665, 0.13543496298336755, 0.13359548412640687, 0.5118884134339377, 0.27814399801397605, 0.048175942939090444, 0.1516513619042606, 0.3155841428437425, 0.2579715326509704, 0.14017712586048425},
    {-0.19252960841894398, -0.7302430277102748, -0.1672391360508524, -0.10744490684184824, -1.0043583449769107, -0.9715304098500271, -0.7556463361949775, -1.360121742429117, 0.3451838108723868, -0.20670052369228617, -0.9350636870000747, -0.001848384011970539, 0.7286966535359732, 0.4829880333637429, 1.1419338783710884},
    {0.0, 0.1203958387413916, 0.6372673088609896, 0.2423368523976187},
};
// ---- end of reference values ----

// The generator's data: x[i][a] = sin(0.3 + 0.37 i + 1.13 a + 0.05 i a + 0.71 i^2) in float32.
std::vector<std::vector<float>> Data(int64_t n = 200, int64_t p = 4) {
    std::vector<std::vector<float>> x(static_cast<size_t>(n), std::vector<float>(static_cast<size_t>(p)));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t a = 0; a < p; ++a) {
            const auto di = static_cast<double>(i), da = static_cast<double>(a);
            x[static_cast<size_t>(i)][static_cast<size_t>(a)] =
                static_cast<float>(std::sin(0.3 + 0.37 * di + 1.13 * da + 0.05 * di * da + 0.71 * di * di));
        }
    }
    return x;
}

double Class(const std::vector<float>& r) {
    const double x0 = r[0], x1 = r[1], x2 = r[2], x3 = r[3];
    return static_cast<double>((x0 + 0.5 * x1 > 0.3 ? 1 : 0) + (x2 - x3 > 0.4 ? 1 : 0));
}

double Target(const std::vector<float>& r) {
    const double x0 = r[0], x1 = r[1], x2 = r[2], x3 = r[3];
    return x0 * x1 + std::sin(3 * x2) + 0.5 * x3;
}

void ExpectMatches(const SurrogateTree& tree, const Reference& ref) {
    ASSERT_EQ(tree.nodes.size(), ref.feature.size());
    for (size_t i = 0; i < ref.feature.size(); ++i) {
        const SurrogateNode& n = tree.nodes[i];
        SCOPED_TRACE("node " + std::to_string(i));
        EXPECT_EQ(n.feature, ref.feature[i] == -2 ? -1 : ref.feature[i]);
        if (!n.is_leaf()) EXPECT_DOUBLE_EQ(n.threshold, ref.threshold[i]);
        EXPECT_EQ(n.samples, ref.samples[i]);
        EXPECT_NEAR(n.impurity, ref.impurity[i], 1e-12);
        EXPECT_NEAR(n.value, ref.value[i], 1e-12);
    }
    const std::vector<double> importance = FeatureImportance(tree);
    ASSERT_EQ(importance.size(), ref.importance.size());
    for (size_t f = 0; f < importance.size(); ++f) EXPECT_NEAR(importance[f], ref.importance[f], 1e-12) << "feature " << f;
}

TEST(SurrogateTree, ClassifierMatchesScikitLearn) {
    const auto x = Data();
    std::vector<double> y;
    for (const auto& r : x) y.push_back(Class(r));
    SurrogateTreeOptions options;
    options.max_depth = 3;
    const SurrogateTree tree = FitDecisionTree(x, y, SurrogateTask::Classification, 3, options);
    ExpectMatches(tree, kClassifier3);
    EXPECT_EQ(tree.depth(), 3);
    EXPECT_EQ(tree.leaves(), 8);

    options.max_depth = 4;
    options.min_samples_leaf = 5;
    ExpectMatches(FitDecisionTree(x, y, SurrogateTask::Classification, 3, options), kClassifier4Leaf5);
}

TEST(SurrogateTree, RegressorMatchesScikitLearn) {
    const auto x = Data();
    std::vector<double> y;
    for (const auto& r : x) y.push_back(Target(r));
    SurrogateTreeOptions options;
    options.max_depth = 3;
    const SurrogateTree tree = FitDecisionTree(x, y, SurrogateTask::Regression, 0, options);
    ExpectMatches(tree, kRegressor3);
    // A leaf predicts its mean: route a row and compare with the leaf's value.
    EXPECT_DOUBLE_EQ(tree.predict(x[0]), tree.nodes[static_cast<size_t>(tree.leaf(x[0]))].value);
}

TEST(SurrogateTree, StopsWhereScikitLearnStops) {
    const auto x = Data(10, 2);
    // A pure node isn't split, whatever the depth allows.
    const SurrogateTree pure = FitDecisionTree(x, std::vector<double>(10, 1.0), SurrogateTask::Classification, 2, {});
    ASSERT_EQ(pure.nodes.size(), 1u);
    EXPECT_EQ(pure.nodes[0].value, 1.0);
    EXPECT_EQ(pure.depth(), 0);
    for (double v : FeatureImportance(pure)) EXPECT_EQ(v, 0.0);

    // Depth 0 is a single leaf; min_samples_leaf above half the rows allows no split.
    std::vector<double> y;
    for (const auto& r : x) y.push_back(r[0] > 0 ? 1.0 : 0.0);
    SurrogateTreeOptions options;
    options.max_depth = 0;
    EXPECT_EQ(FitDecisionTree(x, y, SurrogateTask::Classification, 2, options).nodes.size(), 1u);
    options.max_depth = 3;
    options.min_samples_leaf = 6;
    EXPECT_EQ(FitDecisionTree(x, y, SurrogateTask::Classification, 2, options).nodes.size(), 1u);

    // A constant feature is never split on.
    std::vector<std::vector<float>> constant = x;
    for (auto& r : constant) r[0] = 0.5f;
    options.min_samples_leaf = 1;
    for (const SurrogateNode& n : FitDecisionTree(constant, y, SurrogateTask::Classification, 2, options).nodes) {
        EXPECT_NE(n.feature, 0);
    }
}

TEST(SurrogateTree, RejectsBadInput) {
    const auto x = Data(10, 2);
    const std::vector<double> y(10, 0.0);
    EXPECT_THROW((void)FitDecisionTree({}, {}, SurrogateTask::Regression, 0, {}), std::invalid_argument);
    EXPECT_THROW((void)FitDecisionTree(x, std::vector<double>(9, 0.0), SurrogateTask::Regression, 0, {}), std::invalid_argument);
    EXPECT_THROW((void)FitDecisionTree(x, std::vector<double>(10, 2.0), SurrogateTask::Classification, 2, {}),
                 std::invalid_argument);
    EXPECT_THROW((void)FitDecisionTree(x, std::vector<double>(10, 0.5), SurrogateTask::Classification, 2, {}),
                 std::invalid_argument);
    SurrogateTreeOptions options;
    options.min_samples_leaf = 0;
    EXPECT_THROW((void)FitDecisionTree(x, y, SurrogateTask::Regression, 0, options), std::invalid_argument);
    const SurrogateTree tree = FitDecisionTree(x, y, SurrogateTask::Regression, 0, {});
    EXPECT_THROW((void)tree.predict({1.0f}), std::invalid_argument);
}

class SurrogateModelTest : public ::testing::Test {
protected:
    CPUBackend backend;

    std::vector<Tensor> Instances(int64_t n, int64_t p) {
        std::vector<Tensor> out;
        for (const auto& r : Data(n, p)) out.emplace_back(Shape({1, p}), &backend, r);
        return out;
    }
};

// A model a depth-2 tree describes exactly: the class is which quadrant (x0 > 0.2, x1 > -0.1) is in.
TEST_F(SurrogateModelTest, AModelATreeCanDescribeHasFidelityOne) {
    auto predict = [this](const Tensor& in) {
        const std::vector<float> v = in.to_host_vector();
        const int c = (v[0] > 0.2f ? 1 : 0) + (v[1] > -0.1f ? 2 : 0);
        std::vector<float> out(4, 0.0f);
        out[static_cast<size_t>(c)] = 1.0f;
        return Tensor(Shape({1, 4}), &backend, out);
    };
    SurrogateTreeOptions options;
    options.max_depth = 2;
    const SurrogateTreeResult r = FitSurrogateTree(predict, Instances(300, 3), options);
    EXPECT_EQ(r.fidelity.test_size, 90);  // ceil(0.3 * 300)
    EXPECT_EQ(r.fidelity.train_size, 210);
    EXPECT_DOUBLE_EQ(r.fidelity.train, 1.0);
    EXPECT_DOUBLE_EQ(r.fidelity.test, 1.0);
    EXPECT_EQ(r.tree.num_classes, 4);
    EXPECT_EQ(r.feature_importance[2], 0.0);  // the model never reads x2
    EXPECT_NEAR(r.feature_importance[0] + r.feature_importance[1], 1.0, 1e-12);
    // The root splits near one of the model's thresholds.
    const SurrogateNode& root = r.tree.nodes[0];
    EXPECT_NEAR(root.threshold, root.feature == 0 ? 0.2 : -0.1, 0.05);
}

// The falsifier from the research notes: a model with more structure than a shallow tree can hold
// gets low fidelity, and a deeper tree recovers it.
TEST_F(SurrogateModelTest, FidelityExposesATreeTooShallowForTheModel) {
    auto predict = [this](const Tensor& in) {
        const std::vector<float> v = in.to_host_vector();
        return Tensor(Shape({1, 1}), &backend, {std::sin(4.0f * v[0]) + std::cos(3.0f * v[1]) * v[2]});
    };
    SurrogateTreeOptions shallow;
    shallow.task = SurrogateTask::Regression;
    shallow.max_depth = 1;
    shallow.seed = 4;
    const SurrogateTreeResult s = FitSurrogateTree(predict, Instances(400, 3), shallow);
    SurrogateTreeOptions deep = shallow;
    deep.max_depth = 6;
    deep.min_samples_leaf = 3;
    const SurrogateTreeResult d = FitSurrogateTree(predict, Instances(400, 3), deep);
    EXPECT_LT(s.fidelity.test, 0.85);
    EXPECT_GT(d.fidelity.test, s.fidelity.test + 0.2);
    EXPECT_GE(d.fidelity.train, d.fidelity.test - 0.05);  // fitted rows are no harder than new ones
}

TEST_F(SurrogateModelTest, SplitIsSeededAndValidated) {
    auto predict = [this](const Tensor& in) {
        const std::vector<float> v = in.to_host_vector();
        return Tensor(Shape({1, 2}), &backend, {v[0], v[1]});
    };
    const auto instances = Instances(50, 2);
    SurrogateTreeOptions options;
    const SurrogateTreeResult a = FitSurrogateTree(predict, instances, options);
    const SurrogateTreeResult b = FitSurrogateTree(predict, instances, options);
    EXPECT_EQ(DescribeSurrogateTree(a.tree), DescribeSurrogateTree(b.tree));

    options.test_fraction = 0.0;
    const SurrogateTreeResult all = FitSurrogateTree(predict, instances, options);
    EXPECT_EQ(all.fidelity.test_size, 0);
    EXPECT_TRUE(std::isnan(all.fidelity.test));

    options.test_fraction = 1.0;
    EXPECT_THROW((void)FitSurrogateTree(predict, instances, options), std::invalid_argument);
    options.test_fraction = 0.3;
    options.task = SurrogateTask::Regression;
    options.target = 2;
    EXPECT_THROW((void)FitSurrogateTree(predict, instances, options), std::invalid_argument);
    EXPECT_THROW((void)FitSurrogateTree(predict, {}, {}), std::invalid_argument);
}

TEST(SurrogateTree, DescribesTheTreeAsRules) {
    const std::vector<std::vector<float>> x = {{0.0f, 1.0f}, {1.0f, 1.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}};
    const std::vector<double> y = {0, 0, 1, 1};
    SurrogateTreeResult r;
    r.tree = FitDecisionTree(x, y, SurrogateTask::Classification, 2, {});
    r.feature_importance = FeatureImportance(r.tree);
    // Feature 0 and feature 1 separate the classes equally well; the lower index wins.
    EXPECT_EQ(DescribeSurrogateTree(r.tree, {"income", "debt"}),
              "if income <= 1.5:\n"
              "  class 0 (2 inputs, 100%)\n"
              "else:\n"
              "  class 1 (2 inputs, 100%)\n");
    EXPECT_EQ(DescribeSurrogateTree(r.tree), "if x[0] <= 1.5:\n  class 0 (2 inputs, 100%)\nelse:\n  class 1 (2 inputs, 100%)\n");

    const JsonValue doc = ParseJson(WriteJson(ToJson(r, {"income"})));
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.surrogate_tree.v1");
    EXPECT_EQ(doc.find("task")->as_string(), "classification");
    EXPECT_EQ(doc.find("nodes")->as_array().size(), 3u);
    EXPECT_EQ(doc.find("feature_names")->as_array()[1].as_string(), "x[1]");
    EXPECT_EQ(doc.find("fidelity")->find("metric")->as_string(), "agreement");
    EXPECT_TRUE(doc.find("fidelity")->find("test")->is_null());
    EXPECT_DOUBLE_EQ(doc.find("feature_importance")->as_array()[0].as_double(), 1.0);
}

}  // namespace
}  // namespace pulsatrix
