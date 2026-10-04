#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explanation_metrics.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

class ExplanationMetricsTest : public ::testing::Test {
protected:
    CPUBackend backend;
    Attribution attribution(const Shape& shape, std::vector<float> values) {
        return Attribution{"test", Tensor(shape, &backend, values), {}};
    }
};

// --- Complexity family -------------------------------------------------------------------

TEST_F(ExplanationMetricsTest, SparsenessIsTheGiniIndexOfMagnitudes) {
    EXPECT_NEAR(Sparseness(attribution(Shape({4}), {1, 1, 1, 1})), 0.0f, 1e-6f);  // evenly spread
    EXPECT_NEAR(Sparseness(attribution(Shape({4}), {0, 0, -5, 0})), 0.75f, 1e-6f);  // (n - 1) / n
    // Chalasani et al.: G = sum_i (2i - n - 1) a_(i) / (n sum a), magnitudes sorted ascending.
    const float a[3] = {1, 2, 3};
    const float expected = ((2 * 1 - 4) * a[0] + (2 * 2 - 4) * a[1] + (2 * 3 - 4) * a[2]) / (3.0f * 6.0f);
    EXPECT_NEAR(Sparseness(attribution(Shape({3}), {3, -1, 2})), expected, 1e-6f);
}

TEST_F(ExplanationMetricsTest, ComplexityIsTheEntropyOfFractionalContributions) {
    EXPECT_NEAR(Complexity(attribution(Shape({4}), {2, -2, 2, 2})), std::log(4.0f), 1e-6f);
    EXPECT_NEAR(Complexity(attribution(Shape({4}), {0, 7, 0, 0})), 0.0f, 1e-6f);
}

TEST_F(ExplanationMetricsTest, AllZeroAttributionsAreScoredZeroNotNaN) {
    EXPECT_EQ(Sparseness(attribution(Shape({3}), {0, 0, 0})), 0.0f);
    EXPECT_EQ(Complexity(attribution(Shape({3}), {0, 0, 0})), 0.0f);
}

TEST_F(ExplanationMetricsTest, SpearmanRankCorrelationHandlesTies) {
    EXPECT_NEAR(SpearmanRankCorrelation({1, 2, 3, 4}, {10, 20, 30, 40}), 1.0f, 1e-6f);
    EXPECT_NEAR(SpearmanRankCorrelation({1, 2, 3, 4}, {4, 3, 2, 1}), -1.0f, 1e-6f);
    // Ties take their average rank: x ranks (1, 2.5, 2.5, 4), y ranks (1, 2, 3, 4).
    const double rho = (1 * 1 + 2.5 * 2 + 2.5 * 3 + 4 * 4 - 4 * 2.5 * 2.5) /
                       std::sqrt((1 + 6.25 + 6.25 + 16 - 25.0) * (1 + 4 + 9 + 16 - 25.0));
    EXPECT_NEAR(SpearmanRankCorrelation({1, 5, 5, 9}, {1, 2, 3, 4}), rho, 1e-6);
    EXPECT_THROW((void)SpearmanRankCorrelation({1, 2}, {1, 2, 3}), std::invalid_argument);
}

// --- Faithfulness: deletion and insertion -------------------------------------------------

// A linear model f(x) = w . x, so input * weight is each feature's exact contribution.
class PerturbationTest : public ExplanationMetricsTest {
protected:
    std::vector<float> w = {4, -1, 2, 0.5f};
    Tensor x{Shape({1, 4}), &backend, {1, 2, 3, 4}};  // contributions 4, -2, 6, 2
    PredictFn predict = [this](const Tensor& in) {
        std::vector<float> v = values_of(in);
        float s = 0;
        for (size_t i = 0; i < 4; ++i) s += w[i] * v[i];
        return Tensor(Shape({1, 1}), &backend, {s});
    };
};

TEST_F(PerturbationTest, DeletionRemovesTheMostRelevantFeaturesFirst) {
    PerturbationOptions options;
    options.steps = 4;
    PerturbationCurve curve = DeletionCurve(predict, x, attribution(Shape({1, 4}), {4, -2, 6, 2}), options);
    ASSERT_EQ(curve.scores.size(), 5u);
    // Removal order by attribution: x2 (6), x0 (4), x3 (2), x1 (-2).
    EXPECT_EQ(curve.scores, (std::vector<float>{10, 4, 0, -2, 0}));
    EXPECT_EQ(curve.fractions, (std::vector<float>{0, 0.25f, 0.5f, 0.75f, 1.0f}));
    // Trapezoid rule over the fractions.
    EXPECT_NEAR(curve.auc, 0.25f * ((10 + 4) / 2.0f + (4 + 0) / 2.0f + (0 - 2) / 2.0f + (-2 + 0) / 2.0f), 1e-6f);
}

TEST_F(PerturbationTest, AFaithfulAttributionDeletesFasterThanAnUnfaithfulOne) {
    PerturbationOptions options;
    options.steps = 4;
    const float faithful = DeletionCurve(predict, x, attribution(Shape({1, 4}), {4, -2, 6, 2}), options).auc;
    const float reversed = DeletionCurve(predict, x, attribution(Shape({1, 4}), {-4, 2, -6, -2}), options).auc;
    EXPECT_LT(faithful, reversed);
    const float faithful_in = InsertionCurve(predict, x, attribution(Shape({1, 4}), {4, -2, 6, 2}), options).auc;
    const float reversed_in = InsertionCurve(predict, x, attribution(Shape({1, 4}), {-4, 2, -6, -2}), options).auc;
    EXPECT_GT(faithful_in, reversed_in);
}

TEST_F(PerturbationTest, InsertionStartsFromTheImputedInput) {
    PerturbationOptions options;
    options.steps = 4;
    options.imputation = Imputation::Constant(1.0f);  // a baseline of all ones scores 5.5
    PerturbationCurve curve = InsertionCurve(predict, x, attribution(Shape({1, 4}), {4, -2, 6, 2}), options);
    EXPECT_EQ(curve.scores.front(), 5.5f);
    EXPECT_EQ(curve.scores.back(), 10.0f);
}

TEST_F(PerturbationTest, TargetPicksTheOutputScored) {
    PredictFn two_outputs = [this](const Tensor& in) {
        std::vector<float> v = values_of(in);
        return Tensor(Shape({1, 2}), &backend, {v[0], v[1] + v[2]});
    };
    PerturbationOptions options;
    options.steps = 2;
    options.target = 1;
    EXPECT_EQ(DeletionCurve(two_outputs, x, attribution(Shape({1, 4}), {0, 1, 2, 0}), options).scores.front(), 5.0f);
}

TEST_F(PerturbationTest, RejectsMismatchedOrInvalidArguments) {
    PerturbationOptions options;
    EXPECT_THROW((void)DeletionCurve(predict, x, attribution(Shape({4}), {1, 2, 3, 4}), options), std::invalid_argument);
    options.steps = 0;
    EXPECT_THROW((void)DeletionCurve(predict, x, attribution(Shape({1, 4}), {1, 2, 3, 4}), options), std::invalid_argument);
    options.steps = 4;
    options.target = 3;
    EXPECT_THROW((void)DeletionCurve(predict, x, attribution(Shape({1, 4}), {1, 2, 3, 4}), options), std::invalid_argument);
}

// --- ROAD's noisy linear imputation ---------------------------------------------------------

// On a linear ramp, the 1/6 (direct) and 1/12 (diagonal) neighbor average reproduces the
// missing values exactly, so without noise the imputation is the ramp itself.
TEST_F(ExplanationMetricsTest, NoisyLinearImputationFillsHolesFromNeighbors) {
    std::vector<float> ramp(5 * 6);
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 6; ++j) ramp[i * 6 + j] = 0.5f * i - 0.25f * j + 1.0f;
    std::vector<bool> removed(ramp.size(), false);
    for (int idx : {7, 8, 14, 20, 21, 22}) removed[static_cast<size_t>(idx)] = true;  // an interior blob
    Tensor image(Shape({1, 1, 5, 6}), &backend, ramp);
    Tensor filled = Impute(image, removed, Imputation::NoisyLinear(/*noise_std=*/0.0f));
    std::vector<float> got = values_of(filled);
    for (size_t i = 0; i < ramp.size(); ++i) EXPECT_NEAR(got[i], ramp[i], 1e-4f) << i;

    // With noise, removed pixels move (reproducibly for a seed); kept ones don't.
    Tensor a = Impute(image, removed, Imputation::NoisyLinear(0.01f, 7));
    Tensor b = Impute(image, removed, Imputation::NoisyLinear(0.01f, 7));
    EXPECT_EQ(values_of(a), values_of(b));
    EXPECT_NE(values_of(a)[7], got[7]);
    EXPECT_EQ(values_of(a)[0], ramp[0]);
}

TEST_F(ExplanationMetricsTest, ConstantImputationReplacesOnlyRemovedValues) {
    Tensor t(Shape({1, 3}), &backend, {1, 2, 3});
    EXPECT_EQ(values_of(Impute(t, {false, true, false}, Imputation::Constant(-1.0f))),
              (std::vector<float>{1, -1, 3}));
}

// --- Randomization: the model-parameter randomization test ----------------------------------

class RandomizationTest : public ExplanationMetricsTest {
protected:
    LinearModule l0{4, 6, &backend};
    ReluModule relu{&backend};
    LinearModule l1{6, 2, &backend};
    SequentialModule model{{&l0, &relu, &l1}};
    Tensor x{Shape({1, 4}), &backend, {0.5f, -1, 2, 1}};

    void SetUp() override {
        l0.set_weight({0.3f, -0.2f, 0.5f, 0.1f, 0.4f, -0.6f, 0.2f, 0.7f, -0.1f, 0.3f, 0.5f, 0.2f,
                       -0.4f, 0.6f, 0.1f, 0.2f, -0.3f, 0.4f, 0.5f, 0.1f, 0.3f, -0.2f, 0.6f, 0.4f});
        l1.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.6f, 0.1f, 0.3f, 0.7f, -0.2f, 0.5f, 0.4f, -0.1f});
    }

    // Gradient x input for output 0: depends on the weights.
    ExplainFn gradient_times_input = [this](const Tensor& in) {
        for (ParamRef p : model.parameters()) p.grad->fill(0.0f);
        Tensor out = model.forward(in);
        Tensor seed(out.shape(), &backend, {1.0f, 0.0f});
        Tensor g = model.backward(seed);
        std::vector<float> gv = values_of(g), xv = values_of(in);
        for (size_t i = 0; i < gv.size(); ++i) gv[i] *= xv[i];
        return Attribution{"gxi", Tensor(in.shape(), &backend, gv), {}};
    };
    // A "method" that ignores the model entirely: it should fail the test (Adebayo et al.).
    ExplainFn input_itself = [this](const Tensor& in) { return Attribution{"input", Tensor(in), {}}; };
};

TEST_F(RandomizationTest, AModelIndependentExplanationIsExposed) {
    RandomizationResult r = ModelParameterRandomizationTest(model, input_itself, x, /*seed=*/1);
    ASSERT_EQ(r.layers, (std::vector<std::string>{"2", "0"}));  // cascading from the top
    for (float s : r.similarity) EXPECT_NEAR(s, 1.0f, 1e-6f);
}

TEST_F(RandomizationTest, AModelDependentExplanationChanges) {
    RandomizationResult r = ModelParameterRandomizationTest(model, gradient_times_input, x, 1);
    ASSERT_EQ(r.similarity.size(), 2u);
    EXPECT_LT(r.similarity.back(), 0.9f);
}

TEST_F(RandomizationTest, TheModelIsRestoredExactly) {
    std::vector<std::vector<float>> before;
    for (ParamRef p : model.parameters()) before.push_back(values_of(*p.value));
    (void)ModelParameterRandomizationTest(model, gradient_times_input, x, 3);
    std::vector<ParamRef> after = model.parameters();
    for (size_t i = 0; i < after.size(); ++i) EXPECT_EQ(values_of(*after[i].value), before[i]);
}

TEST_F(RandomizationTest, IsReproducibleForASeed) {
    RandomizationResult a = ModelParameterRandomizationTest(model, gradient_times_input, x, 5);
    RandomizationResult b = ModelParameterRandomizationTest(model, gradient_times_input, x, 5);
    EXPECT_EQ(a.similarity, b.similarity);
}

}  // namespace
}  // namespace pulsatrix
