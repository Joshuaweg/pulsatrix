#include "pulsatrix/explanation_report.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/kernel_shap.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return t.to_host_vector(); }

bool HasCaveat(const Explanation& e, const std::string& id, CaveatSeverity severity) {
    for (const Caveat& c : e.caveats) {
        if (c.id == id) return c.severity == severity;
    }
    return false;
}

const JsonValue& At(const JsonValue& object, const char* key) {
    const JsonValue* v = object.find(key);
    if (v == nullptr) throw std::runtime_error(std::string("no member ") + key);
    return *v;
}

bool HasCaveat(const Explanation& e, const std::string& id) {
    for (const Caveat& c : e.caveats) {
        if (c.id == id) return true;
    }
    return false;
}

// A 12-feature MLP with uneven weights, explained exactly by KernelSHAP (2^12 coalitions).
class ExplanationReportTest : public ::testing::Test {
protected:
    static constexpr int64_t kFeatures = 12;
    CPUBackend backend;
    LinearModule l0{kFeatures, 16, &backend};
    ReluModule relu{&backend};
    LinearModule l1{16, 2, &backend};
    SequentialModule model{{&l0, &relu, &l1}};
    Tensor x{Shape({1, kFeatures}), &backend, {1.5f, -0.4f, 0.9f, 2.0f, -1.2f, 0.3f, 0.7f, -2.2f, 1.1f, 0.05f, -0.6f, 1.8f}};
    Tensor zeros{Shape({1, kFeatures}), &backend};

    void SetUp() override {
        std::vector<float> w0(kFeatures * 16), b0(16), w1(16 * 2);
        for (size_t i = 0; i < w0.size(); ++i) w0[i] = 0.6f * std::sin(0.73f * static_cast<float>(i) + 0.2f);
        for (size_t i = 0; i < b0.size(); ++i) b0[i] = 0.05f * std::cos(static_cast<float>(i));
        for (size_t i = 0; i < w1.size(); ++i) w1[i] = 0.5f * std::cos(1.31f * static_cast<float>(i));
        l0.set_weight(w0);
        l0.set_bias(b0);
        l1.set_weight(w1);
    }

    PredictFn predict = [this](const Tensor& in) { return model.forward(in); };
    ExplainFn shap = [this](const Tensor& in) { return KernelSHAP().explain(predict, in, zeros, 0, &backend); };

    ExplanationRequest Request(XaiReason reason, std::string method, ModelAccess access) const {
        ExplanationRequest r;
        r.question = XaiQuestion::Why;
        r.reason = reason;
        r.method = std::move(method);
        r.traits.access = access;
        r.traits.input = InputKind::Tabular;
        r.traits.can_reinitialize = true;
        r.traits.features = kFeatures;
        return r;
    }

    std::vector<std::vector<float>> snapshot() {
        std::vector<std::vector<float>> out;
        for (ParamRef p : model.parameters()) out.push_back(values_of(*p.value));
        return out;
    }
};

TEST_F(ExplanationReportTest, KernelShapOnAnMlpPassesTheJustifyChecks) {
    ExplanationRequest request = Request(XaiReason::Justify, "KernelSHAP", ModelAccess::BlackBox);
    request.baseline = zeros;
    const auto before = snapshot();
    const Explanation e = BuildExplanation(request, x, shap, predict, &model);
    EXPECT_EQ(snapshot(), before);  // the randomization test restored the model

    EXPECT_EQ(e.method, "KernelSHAP");
    EXPECT_EQ(e.plan_rank, 1);
    ASSERT_TRUE(e.attribution.has_value());
    EXPECT_EQ(e.attribution->method, "kernel_shap");
    EXPECT_EQ(e.setting("target_index"), "0");
    EXPECT_EQ(e.setting("check.random_orderings"), "19");

    const CheckResult* faithfulness = e.check("faithfulness");
    ASSERT_NE(faithfulness, nullptr);
    EXPECT_EQ(faithfulness->priority, CheckPriority::Required);
    EXPECT_EQ(faithfulness->outcome, CheckOutcome::Passed) << faithfulness->summary;
    EXPECT_DOUBLE_EQ(*faithfulness->value("p_value"), 0.05);
    EXPECT_GT(*faithfulness->value("insertion_auc") - *faithfulness->value("deletion_auc"),
              *faithfulness->value("random_insertion_auc") - *faithfulness->value("random_deletion_auc"));

    const CheckResult* stability = e.check("stability");
    ASSERT_NE(stability, nullptr);
    EXPECT_EQ(stability->outcome, CheckOutcome::Passed) << stability->summary;
    EXPECT_EQ(stability->value("deterministic"), 1.0);
    EXPECT_GT(*stability->value("perturbed_similarity"), 0.8);

    // Shapley efficiency: the values add up to f(x) - f(0).
    const CheckResult* conservation = e.check("conservation");
    ASSERT_NE(conservation, nullptr);
    EXPECT_EQ(conservation->outcome, CheckOutcome::Passed) << conservation->summary;
    EXPECT_NEAR(*conservation->value("relative_gap"), 0.0, 1e-4);

    const CheckResult* randomization = e.check("randomization");
    ASSERT_NE(randomization, nullptr);
    EXPECT_EQ(randomization->priority, CheckPriority::Recommended);
    EXPECT_NE(randomization->outcome, CheckOutcome::NotRun);
    EXPECT_EQ(*randomization->value("layers"), 2.0);

    EXPECT_TRUE(e.supports_reason());
    EXPECT_TRUE(HasCaveat(e, "method.kernel_shap_baseline", CaveatSeverity::Note));
    EXPECT_TRUE(HasCaveat(e, "reason.relied_on", CaveatSeverity::Note));
    EXPECT_EQ(e.metrics.size(), 2u);
}

// The same values in reverse order rank the least relevant features first: deletion then removes
// what matters last, and no random ordering does worse.
TEST_F(ExplanationReportTest, AReversedRankingFailsFaithfulnessAndBlocksJustify) {
    ExplanationRequest request = Request(XaiReason::Justify, "KernelSHAP", ModelAccess::BlackBox);
    request.baseline = zeros;
    ExplainFn reversed = [this](const Tensor& in) {
        Attribution a = shap(in);
        std::vector<float> v = values_of(a.values);
        for (float& f : v) f = -f;
        a.values = Tensor(a.values.shape(), &backend, v);
        return a;
    };
    const Explanation e = BuildExplanation(request, x, reversed, predict, &model);
    const CheckResult* faithfulness = e.check("faithfulness");
    ASSERT_NE(faithfulness, nullptr);
    EXPECT_EQ(faithfulness->outcome, CheckOutcome::Failed);
    EXPECT_GT(*faithfulness->value("p_value"), 0.5);
    EXPECT_TRUE(HasCaveat(e, "check.faithfulness", CaveatSeverity::Blocking));
    EXPECT_FALSE(e.supports_reason());
    EXPECT_EQ(e.caveats.front().severity, CaveatSeverity::Blocking);  // blocking first
}

// Adebayo et al.'s finding: a map that reads only the input stays the same however the model is
// randomized. Control requires the randomization test, so it blocks; it also needs per-slice error,
// which pulsatrix doesn't have yet.
TEST_F(ExplanationReportTest, AnInputOnlyMapFailsRandomizationUnderControl) {
    const ExplanationRequest request = Request(XaiReason::Control, "Saliency", ModelAccess::Gradients);
    ExplainFn input_only = [this](const Tensor& in) {
        std::vector<float> v = values_of(in);
        for (float& f : v) f = std::fabs(f);
        return Attribution{"edges", Tensor(in.shape(), &backend, v), {}};
    };
    const auto before = snapshot();
    const Explanation e = BuildExplanation(request, x, input_only, predict, &model);
    EXPECT_EQ(snapshot(), before);
    const CheckResult* randomization = e.check("randomization");
    ASSERT_NE(randomization, nullptr);
    EXPECT_EQ(randomization->priority, CheckPriority::Required);
    EXPECT_EQ(randomization->outcome, CheckOutcome::Failed);
    EXPECT_DOUBLE_EQ(*randomization->value("similarity_all_layers"), 1.0);
    EXPECT_TRUE(HasCaveat(e, "check.randomization", CaveatSeverity::Blocking));

    const CheckResult* slices = e.check("per_slice_error");
    ASSERT_NE(slices, nullptr);
    EXPECT_EQ(slices->outcome, CheckOutcome::NotRun);
    EXPECT_NE(slices->summary.find("KS-3"), std::string::npos);
    EXPECT_TRUE(HasCaveat(e, "check.per_slice_error", CaveatSeverity::Blocking));
    EXPECT_FALSE(e.supports_reason());
    EXPECT_TRUE(HasCaveat(e, "method.saliency_sensitivity", CaveatSeverity::Note));
}

TEST_F(ExplanationReportTest, ChecksThatCantRunSayWhy) {
    // No baseline: conservation can't be checked, and Justify requires it.
    ExplanationRequest request = Request(XaiReason::Justify, "KernelSHAP", ModelAccess::BlackBox);
    Explanation e = BuildExplanation(request, x, shap, predict, &model);
    ASSERT_NE(e.check("conservation"), nullptr);
    EXPECT_EQ(e.check("conservation")->outcome, CheckOutcome::NotRun);
    EXPECT_NE(e.check("conservation")->summary.find("ExplanationRequest::baseline"), std::string::npos);
    EXPECT_TRUE(HasCaveat(e, "check.conservation", CaveatSeverity::Blocking));
    EXPECT_FALSE(e.supports_reason());

    // A model that can't be re-initialized: the randomization test is planned unavailable, and a
    // recommended check that doesn't run is a note.
    request.baseline = zeros;
    request.traits.can_reinitialize = false;
    e = BuildExplanation(request, x, shap, predict, nullptr);
    EXPECT_EQ(e.check("randomization")->outcome, CheckOutcome::NotRun);
    EXPECT_TRUE(HasCaveat(e, "check.randomization", CaveatSeverity::Note));
    EXPECT_TRUE(e.supports_reason());

    // Only the required checks.
    request.traits.can_reinitialize = true;
    ExplanationCheckOptions options;
    options.required_only = true;
    e = BuildExplanation(request, x, shap, predict, &model, options);
    EXPECT_EQ(e.check("randomization")->outcome, CheckOutcome::NotRun);
    EXPECT_NE(e.check("randomization")->summary.find("required_only"), std::string::npos);
    EXPECT_EQ(e.check("faithfulness")->outcome, CheckOutcome::Passed);
}

TEST_F(ExplanationReportTest, LrpConservationReadsItsSeededTotal) {
    ExplanationRequest request = Request(XaiReason::Justify, "LRP", ModelAccess::Relevance);
    // Half of the seeded relevance went missing on the way to the input.
    ExplainFn leaky = [this](const Tensor& in) {
        Attribution a = shap(in);
        float sum = 0;
        for (float v : values_of(a.values)) sum += v;
        a.method = "lrp";
        a.metadata = {{"relevance_out_sum", std::to_string(2 * sum)}, {"rule", "epsilon"}};
        return a;
    };
    const Explanation e = BuildExplanation(request, x, leaky, predict, &model);
    const CheckResult* conservation = e.check("conservation");
    ASSERT_NE(conservation, nullptr);
    EXPECT_EQ(conservation->outcome, CheckOutcome::Failed);
    EXPECT_NEAR(*conservation->value("relative_gap"), 0.5, 1e-3);
    EXPECT_TRUE(HasCaveat(e, "check.conservation", CaveatSeverity::Blocking));
    EXPECT_TRUE(HasCaveat(e, "method.lrp_rules", CaveatSeverity::Note));
    EXPECT_EQ(e.setting("rule"), "epsilon");
}

TEST_F(ExplanationReportTest, AnExcludedMethodBlocks) {
    // Saliency needs gradients; a black box doesn't give them.
    ExplanationRequest request = Request(XaiReason::Justify, "Saliency", ModelAccess::BlackBox);
    const Explanation e = BuildExplanation(request, x, shap, predict, &model);
    EXPECT_EQ(e.plan_rank, 0);
    EXPECT_NE(e.excluded_reason.find("needs gradients"), std::string::npos);
    EXPECT_TRUE(HasCaveat(e, "plan.excluded", CaveatSeverity::Blocking));
    EXPECT_FALSE(e.supports_reason());
}

TEST_F(ExplanationReportTest, RejectsMethodsItCantReport) {
    ExplanationRequest request = Request(XaiReason::Justify, "PCA", ModelAccess::Activations);
    EXPECT_THROW((void)BuildExplanation(request, x, shap, predict, &model), std::invalid_argument);  // not a Why explainer
    request.question = XaiQuestion::HowGlobal;
    EXPECT_THROW((void)BuildExplanation(request, x, shap, predict, &model), std::invalid_argument);  // no attribution
    request.method = "NoSuchExplainer";
    EXPECT_THROW((void)BuildExplanation(request, x, shap, predict, &model), std::invalid_argument);
    ExplanationCheckOptions options;
    options.random_orderings = 0;
    request = Request(XaiReason::Justify, "KernelSHAP", ModelAccess::BlackBox);
    EXPECT_THROW((void)BuildExplanation(request, x, shap, predict, &model, options), std::invalid_argument);
}

TEST_F(ExplanationReportTest, ImproveAggregatesOverTheDataset) {
    ExplanationRequest request = Request(XaiReason::Improve, "KernelSHAP", ModelAccess::BlackBox);
    request.traits.has_dataset = true;
    request.baseline = zeros;
    for (int k = 0; k < 4; ++k) {
        std::vector<float> v = values_of(x);
        for (size_t i = 0; i < v.size(); ++i) v[i] *= 0.5f + 0.25f * static_cast<float>((k + i) % 3);
        request.dataset.emplace_back(x.shape(), &backend, v);
    }
    const Explanation e = BuildExplanation(request, x, shap, predict, &model);
    const CheckResult* aggregate = e.check("aggregate");
    ASSERT_NE(aggregate, nullptr);
    EXPECT_EQ(aggregate->priority, CheckPriority::Required);
    EXPECT_EQ(aggregate->outcome, CheckOutcome::Measured);
    ASSERT_TRUE(e.aggregate.has_value());
    EXPECT_EQ(e.aggregate->method, "mean_abs_kernel_shap");

    // The aggregate is the mean magnitude, and the top feature is its largest entry.
    std::vector<float> expected(kFeatures, 0.0f);
    for (const Tensor& in : request.dataset) {
        const std::vector<float> v = values_of(shap(in).values);
        for (size_t i = 0; i < v.size(); ++i) expected[i] += std::fabs(v[i]) / 4.0f;
    }
    const std::vector<float> got = values_of(e.aggregate->values);
    size_t top = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(got[i], expected[i], 1e-5f);
        if (expected[i] > expected[top]) top = i;
    }
    EXPECT_EQ(*aggregate->value("top_feature"), static_cast<double>(top));
    EXPECT_FALSE(HasCaveat(e, "check.aggregate"));
}

// Integrated Gradients on a linear image model, which it explains exactly: the warnings come from
// the settings and the input, not from a failed check.
TEST_F(ExplanationReportTest, ImageInputAndFewStepsWarn) {
    const Shape shape({1, 1, 4, 4});
    std::vector<float> w(16), xv(16);
    for (size_t i = 0; i < 16; ++i) {
        w[i] = std::sin(1.7f * static_cast<float>(i));
        xv[i] = 0.25f + 0.05f * static_cast<float>(i);
    }
    PredictFn linear = [&](const Tensor& in) {
        const std::vector<float> v = values_of(in);
        float y = 0;
        for (size_t i = 0; i < v.size(); ++i) y += w[i] * v[i];
        return Tensor(Shape({1, 1}), &backend, {y});
    };
    ExplainFn ig = [&](const Tensor& in) {
        std::vector<float> v = values_of(in);
        for (size_t i = 0; i < v.size(); ++i) v[i] *= w[i];
        return Attribution{"integrated_gradients", Tensor(in.shape(), &backend, v), {{"steps", "10"}, {"target_index", "0"}}};
    };
    ExplanationRequest request;
    request.question = XaiQuestion::Why;
    request.reason = XaiReason::Justify;
    request.method = "IntegratedGradients";
    request.traits.access = ModelAccess::Gradients;
    request.traits.input = InputKind::Image;
    request.baseline = Tensor(shape, &backend);
    const Explanation e = BuildExplanation(request, Tensor(shape, &backend, xv), ig, linear, nullptr);
    EXPECT_EQ(e.check("faithfulness")->outcome, CheckOutcome::Passed) << e.check("faithfulness")->summary;
    EXPECT_EQ(e.check("conservation")->outcome, CheckOutcome::Passed);
    EXPECT_TRUE(HasCaveat(e, "input.constant_fill", CaveatSeverity::Warning));
    EXPECT_TRUE(HasCaveat(e, "input.per_channel", CaveatSeverity::Note));
    EXPECT_TRUE(HasCaveat(e, "method.integrated_gradients_steps", CaveatSeverity::Warning));
    EXPECT_TRUE(HasCaveat(e, "method.integrated_gradients_baseline", CaveatSeverity::Note));
    EXPECT_TRUE(e.supports_reason());

    // ROAD's fill removes the constant-fill warning.
    ExplanationCheckOptions road;
    road.imputation = Imputation::NoisyLinear(0.01f, 3);
    const Explanation r = BuildExplanation(request, Tensor(shape, &backend, xv), ig, linear, nullptr, road);
    EXPECT_FALSE(HasCaveat(r, "input.constant_fill"));
    EXPECT_EQ(r.setting("check.imputation"), "noisy_linear:0.01");
}

TEST_F(ExplanationReportTest, DiscreteInputIsNotPerturbed) {
    ExplanationRequest request = Request(XaiReason::Justify, "KernelSHAP", ModelAccess::BlackBox);
    request.traits.input = InputKind::Tokens;
    request.baseline = zeros;
    const Explanation e = BuildExplanation(request, x, shap, predict, &model);
    const CheckResult* stability = e.check("stability");
    ASSERT_NE(stability, nullptr);
    EXPECT_FALSE(stability->value("perturbed_similarity").has_value());
    EXPECT_NE(stability->summary.find("discrete"), std::string::npos);
    EXPECT_TRUE(HasCaveat(e, "input.discrete", CaveatSeverity::Warning));
}

// Explainers that don't return an attribution report checks they ran themselves.
TEST(ExplanationReport, ReportExplanationMatchesChecksByName) {
    ExplanationRequest request;
    request.question = XaiQuestion::HowGlobal;
    request.reason = XaiReason::Discover;
    request.method = "PersistentHomology";
    request.traits.access = ModelAccess::Activations;
    request.traits.input = InputKind::Representation;
    request.traits.can_reinitialize = true;
    request.settings = {{"layer", "12"}};

    CheckResult significance;
    significance.name = "topology_significance";
    significance.outcome = CheckOutcome::Passed;
    significance.summary = "One H1 feature beats the universal null (p = 0.001).";
    significance.values = {{"p_value", 0.001}};
    CheckResult extra;
    extra.name = "cyclic_order";
    extra.outcome = CheckOutcome::Passed;
    extra.summary = "The days are in calendar order.";

    const Explanation e = ReportExplanation(request, std::nullopt, {significance, extra});
    EXPECT_FALSE(e.attribution.has_value());
    EXPECT_TRUE(e.metrics.empty());
    EXPECT_GT(e.plan_rank, 0);
    EXPECT_EQ(e.setting("layer"), "12");
    ASSERT_NE(e.check("topology_significance"), nullptr);
    EXPECT_EQ(e.check("topology_significance")->priority, CheckPriority::Required);  // the plan's priority
    EXPECT_EQ(e.check("topology_significance")->outcome, CheckOutcome::Passed);
    ASSERT_NE(e.check("cyclic_order"), nullptr);  // kept, though not planned
    ASSERT_NE(e.check("null_model"), nullptr);
    EXPECT_EQ(e.check("null_model")->outcome, CheckOutcome::NotRun);
    EXPECT_TRUE(HasCaveat(e, "check.null_model", CaveatSeverity::Blocking));  // Discover requires it
    EXPECT_FALSE(HasCaveat(e, "reason.relied_on"));                           // no attribution
    EXPECT_FALSE(e.supports_reason());

    CheckResult null_model;
    null_model.name = "null_model";
    null_model.outcome = CheckOutcome::Passed;
    null_model.summary = "Random networks give no such loop.";
    EXPECT_TRUE(ReportExplanation(request, std::nullopt, {significance, null_model}).supports_reason());

    request.method = "LRP";  // not a How (global) explainer
    EXPECT_THROW((void)ReportExplanation(request, std::nullopt, {}), std::invalid_argument);
}

TEST_F(ExplanationReportTest, JsonAndText) {
    ExplanationRequest request = Request(XaiReason::Justify, "KernelSHAP", ModelAccess::BlackBox);
    const Explanation e = BuildExplanation(request, x, shap, predict, &model);  // no baseline: blocked

    const JsonValue doc = ParseJson(WriteJson(ToJson(e)));
    EXPECT_EQ(At(doc, "schema").as_string(), "pulsatrix.explanation.v1");
    EXPECT_EQ(At(doc, "question").as_string(), "why");
    EXPECT_EQ(At(doc, "reason").as_string(), "justify");
    EXPECT_EQ(At(doc, "method").as_string(), "KernelSHAP");
    EXPECT_EQ(At(doc, "header").as_string(), "pulsatrix/kernel_shap.hpp");
    EXPECT_FALSE(At(doc, "supports_reason").as_bool());
    EXPECT_EQ(At(At(doc, "attribution"), "values").as_array().size(), static_cast<size_t>(kFeatures));
    EXPECT_EQ(At(At(doc, "attribution"), "shape").as_array().size(), 2u);
    EXPECT_EQ(At(doc, "checks").as_array().size(), e.checks.size());
    EXPECT_EQ(At(doc, "caveats").as_array().size(), e.caveats.size());
    EXPECT_EQ(At(At(doc, "caveats").as_array()[0], "severity").as_string(), "blocking");
    EXPECT_EQ(At(At(At(doc, "checks").as_array()[0], "values"), "random_orderings").as_double(), 19.0);

    const std::string text = DescribeExplanation(e);
    EXPECT_NE(text.find("Explanation: KernelSHAP, for the why question."), std::string::npos);
    EXPECT_NE(text.find("Supports the reason: no, 1 blocking caveat\n"), std::string::npos);
    EXPECT_NE(text.find("Largest attributions (kernel_shap, 12 features):"), std::string::npos);
    EXPECT_NE(text.find("[required] faithfulness: passed."), std::string::npos);
    EXPECT_NE(text.find("[blocking] Not checked for conservation"), std::string::npos);
}

}  // namespace
}  // namespace pulsatrix
