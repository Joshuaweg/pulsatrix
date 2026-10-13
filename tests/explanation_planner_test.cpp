// XAI-2: the explanation planner, from a question and a reason to ranked explainers and checks.
#include "pulsatrix/explanation_planner.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace pulsatrix {
namespace {

std::vector<std::string> Names(const ExplanationPlan& plan) {
    std::vector<std::string> names;
    for (const PlannedExplainer& p : plan.explainers) names.push_back(p.capabilities->name);
    return names;
}

const PlannedCheck* FindCheck(const ExplanationPlan& plan, const std::string& name) {
    for (const PlannedCheck& c : plan.checks) {
        if (c.name == name) return &c;
    }
    return nullptr;
}

const ExcludedExplainer* FindExcluded(const ExplanationPlan& plan, const std::string& name) {
    for (const ExcludedExplainer& e : plan.excluded) {
        if (e.capabilities->name == name) return &e;
    }
    return nullptr;
}

bool Has(const std::vector<std::string>& items, const std::string& item) {
    return std::find(items.begin(), items.end(), item) != items.end();
}

ModelTraits Traits(ModelAccess access, InputKind input, bool reinit = false, bool dataset = false) {
    ModelTraits t;
    t.access = access;
    t.input = input;
    t.can_reinitialize = reinit;
    t.has_dataset = dataset;
    return t;
}

// Justifying an image classifier with LRP rules: complete attributions first, best-read first.
TEST(ExplanationPlanner, JustifyingWhyPutsCompleteAttributionsFirst) {
    const ExplanationPlan plan =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Justify, Traits(ModelAccess::Relevance, InputKind::Image, true));
    const std::vector<std::string> names = Names(plan);
    ASSERT_GE(names.size(), 8u);
    EXPECT_EQ((std::vector<std::string>(names.begin(), names.begin() + 5)),
              (std::vector<std::string>{"LRP", "IntegratedGradients", "KernelSHAP", "ConceptRelevance", "GradCAM"}));
    for (size_t i = 0; i < plan.explainers.size(); ++i) EXPECT_EQ(plan.explainers[i].rank, static_cast<int>(i) + 1);
    EXPECT_TRUE(Has(plan.explainers.front().reasons, "its attributions add up to the output, which can be checked"));
    // Not for this model: token, protein and Datalog explainers, and the fine-tune diff without a base model.
    EXPECT_FALSE(Has(names, "RelevanceGraph"));
    EXPECT_NE(FindExcluded(plan, "ProteinRelevance"), nullptr);
    EXPECT_NE(FindExcluded(plan, "FineTuneRelevanceDiff")->reason.find("fine-tuned"), std::string::npos);

    // Justify requires faithfulness, stability and conservation; all can run, so the plan is complete.
    for (const char* name : {"faithfulness", "stability", "conservation"}) {
        const PlannedCheck* c = FindCheck(plan, name);
        ASSERT_NE(c, nullptr) << name;
        EXPECT_EQ(c->priority, CheckPriority::Required) << name;
        EXPECT_TRUE(c->available) << name;
    }
    EXPECT_TRUE(Has(FindCheck(plan, "conservation")->applies_to, "ConceptRelevance"));
    EXPECT_FALSE(Has(FindCheck(plan, "faithfulness")->applies_to, "ConceptRelevance"));  // not an Attribution
    EXPECT_EQ(FindCheck(plan, "randomization")->priority, CheckPriority::Recommended);
    EXPECT_EQ(FindCheck(plan, "per_slice_error"), nullptr);
    EXPECT_EQ(plan.checks.front().priority, CheckPriority::Required);
    EXPECT_TRUE(plan.complete());
    EXPECT_TRUE(plan.warnings.empty());
}

TEST(ExplanationPlanner, ABlackBoxGetsTheModelAgnosticExplainersAndSaysWhyNotTheRest) {
    const ExplanationPlan plan =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Justify, Traits(ModelAccess::BlackBox, InputKind::Tabular));
    EXPECT_EQ(Names(plan), (std::vector<std::string>{"KernelSHAP", "LIME", "Occlusion", "Anchors", "Prototypes"}));
    EXPECT_EQ(FindExcluded(plan, "LRP")->reason, "needs an LRP rule at every layer; the model gives a prediction function");
    EXPECT_EQ(FindExcluded(plan, "IntegratedGradients")->reason, "needs gradients; the model gives a prediction function");
    // Access is checked before input, so Grad-CAM is excluded for its gradients.
    EXPECT_NE(FindExcluded(plan, "GradCAM")->reason.find("gradients"), std::string::npos);
    EXPECT_EQ(FindCheck(plan, "conservation")->applies_to, (std::vector<std::string>{"KernelSHAP"}));
    // KernelSHAP enumerates every coalition: with the feature count unknown that's a condition, and
    // above 20 features it drops out.
    EXPECT_EQ(plan.explainers.front().conditions, (std::vector<std::string>{"at most 20 input features"}));
    ModelTraits wide = Traits(ModelAccess::BlackBox, InputKind::Tabular);
    wide.features = 21;
    const ExplanationPlan many = ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Justify, wide);
    EXPECT_EQ(Names(many), (std::vector<std::string>{"LIME", "Occlusion", "Anchors", "Prototypes"}));
    EXPECT_EQ(FindExcluded(many, "KernelSHAP")->reason, "takes at most 20 features; the input has 21");
    wide.features = 20;
    const ExplanationPlan fits = ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Justify, wide);
    EXPECT_EQ(Names(fits).front(), "KernelSHAP");
    EXPECT_TRUE(fits.explainers.front().conditions.empty());
}

TEST(ExplanationPlanner, InputKindsMustMatchAndUnknownInputBecomesACondition) {
    const ExplanationPlan tokens =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Control, Traits(ModelAccess::Relevance, InputKind::Tokens));
    EXPECT_TRUE(Has(Names(tokens), "RelevanceGraph"));
    EXPECT_EQ(FindExcluded(tokens, "GradCAM")->reason, "takes image input; the model takes tokens");
    // Made for tokens, so ahead of the general explainers with the same access.
    const std::vector<std::string> names = Names(tokens);
    EXPECT_LT(std::find(names.begin(), names.end(), "RelevanceGraph"), std::find(names.begin(), names.end(), "LRP"));

    const ExplanationPlan unknown =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Justify, Traits(ModelAccess::Relevance, InputKind::Any));
    for (const PlannedExplainer& p : unknown.explainers) {
        if (p.capabilities->name == "GradCAM") EXPECT_EQ(p.conditions, (std::vector<std::string>{"needs image input"}));
        if (p.capabilities->name == "LRP") EXPECT_TRUE(p.conditions.empty());
    }
    EXPECT_TRUE(Has(Names(unknown), "GradCAM"));
}

TEST(ExplanationPlanner, ControlRequiresRandomizationAndPerSliceErrorAndSaysWhatCantRun) {
    const ExplanationPlan plan =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Control, Traits(ModelAccess::Gradients, InputKind::Image));
    const PlannedCheck* randomization = FindCheck(plan, "randomization");
    ASSERT_NE(randomization, nullptr);
    EXPECT_EQ(randomization->priority, CheckPriority::Required);
    EXPECT_FALSE(randomization->available);
    const PlannedCheck* slices = FindCheck(plan, "per_slice_error");
    ASSERT_NE(slices, nullptr);
    EXPECT_FALSE(slices->available);
    EXPECT_EQ(slices->header, "pulsatrix/performance.hpp");
    EXPECT_NE(slices->unavailable_reason.find("labeled dataset"), std::string::npos);
    EXPECT_FALSE(plan.complete());
    ASSERT_EQ(plan.warnings.size(), 3u);  // randomization, the null model (for LogitLens and the rest), slices
    EXPECT_EQ(plan.warnings.front().rfind("control requires the randomization check", 0), 0u);

    const ExplanationPlan reinit =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Control, Traits(ModelAccess::Gradients, InputKind::Image, true));
    EXPECT_TRUE(FindCheck(reinit, "randomization")->available);
    EXPECT_EQ(reinit.warnings.size(), 1u);  // per-slice error is still missing
    EXPECT_FALSE(reinit.complete());

    // KS-3: with labels and slices, per-slice error can run and the plan is complete.
    ModelTraits labeled = Traits(ModelAccess::Gradients, InputKind::Image, true, true);
    labeled.has_labels = true;
    const ExplanationPlan complete = ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Control, labeled);
    EXPECT_TRUE(FindCheck(complete, "per_slice_error")->available);
    EXPECT_TRUE(complete.warnings.empty());
    EXPECT_TRUE(complete.complete());
}

TEST(ExplanationPlanner, GlobalExplainersNeedADatasetButWeightsDoNot) {
    const ModelTraits one = Traits(ModelAccess::BlackBox, InputKind::Tabular);
    const ExplanationPlan without = ExplanationPlanner().plan(XaiQuestion::HowGlobal, XaiReason::Improve, one);
    EXPECT_TRUE(without.explainers.empty());
    EXPECT_EQ(FindExcluded(without, "PDP")->reason, "looks across a dataset, and there is only the one input");
    ASSERT_EQ(without.warnings.size(), 1u);
    EXPECT_NE(without.warnings.front().find("excluded"), std::string::npos);

    const ExplanationPlan with =
        ExplanationPlanner().plan(XaiQuestion::HowGlobal, XaiReason::Improve, Traits(ModelAccess::BlackBox, InputKind::Tabular, false, true));
    EXPECT_EQ(Names(with), (std::vector<std::string>{"SurrogateTree", "PDP", "PartialDependence2D", "ALE", "Morris", "Sobol", "ICE", "ProtoDash"}));

    ModelTraits fine_tuned = Traits(ModelAccess::Weights, InputKind::Any);
    fine_tuned.has_base_model = true;
    EXPECT_EQ(Names(ExplanationPlanner().plan(XaiQuestion::HowGlobal, XaiReason::Discover, fine_tuned)),
              (std::vector<std::string>{"IntruderDimensions"}));
}

// XAI-4: the surrogate tree answers How (global) for any tabular model, and its fidelity is its
// own control.
TEST(ExplanationPlanner, TheSurrogateTreeIsCheckedByItsFidelity) {
    const ExplanationPlan plan = ExplanationPlanner().plan(XaiQuestion::HowGlobal, XaiReason::Discover,
                                                           Traits(ModelAccess::BlackBox, InputKind::Tabular, true, true));
    ASSERT_FALSE(plan.explainers.empty());
    EXPECT_EQ(plan.explainers.front().capabilities->name, "SurrogateTree");
    const PlannedCheck* fidelity = FindCheck(plan, "surrogate_fidelity");
    ASSERT_NE(fidelity, nullptr);
    EXPECT_EQ(fidelity->priority, CheckPriority::Required);
    EXPECT_EQ(fidelity->applies_to, (std::vector<std::string>{"SurrogateTree"}));
    EXPECT_FALSE(Has(Names(ExplanationPlanner().plan(XaiQuestion::HowGlobal, XaiReason::Discover,
                                                     Traits(ModelAccess::BlackBox, InputKind::Image, true, true))),
                     "SurrogateTree"));  // rules over pixels aren't readable
}

// Discovering from representations the caller already has: no model access needed, and every
// result is held against a random model's and against the explainer's own controls.
TEST(ExplanationPlanner, DiscoverRequiresTheNullModelAndEachExplainersOwnControls) {
    const ExplanationPlan plan = ExplanationPlanner().plan(XaiQuestion::HowGlobal, XaiReason::Discover,
                                                           Traits(ModelAccess::BlackBox, InputKind::Representation, true));
    const std::vector<std::string> names = Names(plan);
    for (const char* name : {"LinearProbe", "PCA", "PersistentHomology", "Featurizers", "RepresentationSimilarity"}) {
        EXPECT_TRUE(Has(names, name)) << name;
    }
    EXPECT_FALSE(Has(names, "TCAV"));  // needs gradients
    const PlannedCheck* null_model = FindCheck(plan, "null_model");
    ASSERT_NE(null_model, nullptr);
    EXPECT_EQ(null_model->priority, CheckPriority::Required);
    EXPECT_TRUE(null_model->applies_to.empty());  // every explainer
    for (const char* name : {"probe_controls", "topology_significance", "split_half", "featurizer_metrics"}) {
        const PlannedCheck* c = FindCheck(plan, name);
        ASSERT_NE(c, nullptr) << name;
        EXPECT_EQ(c->priority, CheckPriority::Required) << name;
        EXPECT_EQ(c->applies_to.size(), 1u) << name;
    }
    EXPECT_EQ(FindCheck(plan, "topology_significance")->applies_to.front(), "PersistentHomology");
    EXPECT_TRUE(plan.complete());

    // The same explainers asked for another reason: the controls are only recommended.
    const ExplanationPlan improve = ExplanationPlanner().plan(XaiQuestion::HowGlobal, XaiReason::Improve,
                                                              Traits(ModelAccess::BlackBox, InputKind::Representation, true));
    EXPECT_EQ(FindCheck(improve, "probe_controls")->priority, CheckPriority::Recommended);
}

TEST(ExplanationPlanner, ImproveAggregatesLocalExplanationsOverTheData) {
    const ExplanationPlan one =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Improve, Traits(ModelAccess::Gradients, InputKind::Tabular, true));
    const PlannedCheck* aggregate = FindCheck(one, "aggregate");
    ASSERT_NE(aggregate, nullptr);
    EXPECT_EQ(aggregate->priority, CheckPriority::Required);
    EXPECT_FALSE(aggregate->available);
    EXPECT_FALSE(one.complete());
    const ExplanationPlan many =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Improve, Traits(ModelAccess::Gradients, InputKind::Tabular, true, true));
    EXPECT_TRUE(FindCheck(many, "aggregate")->available);
    EXPECT_TRUE(many.complete());
}

TEST(ExplanationPlanner, QuestionsWithNoExplainerSaySo) {
    for (XaiQuestion q : {XaiQuestion::Output}) {
        const ExplanationPlan plan = ExplanationPlanner().plan(q, XaiReason::Justify, Traits(ModelAccess::Relevance, InputKind::Any, true, true));
        EXPECT_TRUE(plan.explainers.empty());
        EXPECT_TRUE(plan.excluded.empty());
        EXPECT_TRUE(plan.checks.empty());
        EXPECT_FALSE(plan.complete());
        ASSERT_EQ(plan.warnings.size(), 1u);
        EXPECT_EQ(plan.warnings.front().rfind("pulsatrix has no explainer yet", 0), 0u);
    }
}

// KS-3: the Performance question is answered by the metrics, which look across a labeled dataset.
TEST(ExplanationPlanner, PerformanceIsAnsweredByTheMetrics) {
    const ExplanationPlan many = ExplanationPlanner().plan(XaiQuestion::Performance, XaiReason::Control,
                                                          Traits(ModelAccess::BlackBox, InputKind::Any, false, true));
    EXPECT_EQ(Names(many), (std::vector<std::string>{"ClassifierMetrics", "RegressorMetrics", "PerSliceError"}));
    const ExplanationPlan one = ExplanationPlanner().plan(XaiQuestion::Performance, XaiReason::Control,
                                                         Traits(ModelAccess::BlackBox, InputKind::Any));
    EXPECT_TRUE(one.explainers.empty());
    ASSERT_EQ(one.excluded.size(), 3u);
    EXPECT_EQ(one.excluded.front().reason, "looks across a dataset, and there is only the one input");
    // A re-initialized model is no baseline for a metric: Control needs per-slice error only.
    ModelTraits labeled = Traits(ModelAccess::BlackBox, InputKind::Any, false, true);
    labeled.has_labels = true;
    const ExplanationPlan control = ExplanationPlanner().plan(XaiQuestion::Performance, XaiReason::Control, labeled);
    EXPECT_EQ(FindCheck(control, "null_model"), nullptr);
    ASSERT_EQ(control.checks.size(), 1u);
    EXPECT_EQ(control.checks.front().name, "per_slice_error");
    EXPECT_TRUE(control.complete());
}

// XAI-4: Anchors answer How to still be this, for a black box, and carry their own check.
TEST(ExplanationPlanner, AnchorsAnswerHowToStillBeThis) {
    const ExplanationPlan plan = ExplanationPlanner().plan(XaiQuestion::HowToStillBeThis, XaiReason::Justify,
                                                           Traits(ModelAccess::BlackBox, InputKind::Tabular, false, true));
    EXPECT_EQ(Names(plan), (std::vector<std::string>{"Anchors"}));
    const PlannedCheck* precision = FindCheck(plan, "anchor_precision");
    ASSERT_NE(precision, nullptr);
    EXPECT_EQ(precision->priority, CheckPriority::Recommended);  // required only under Discover
    EXPECT_TRUE(plan.complete());
    // Anchors also answer Why, after the attributions.
    const std::vector<std::string> why = Names(ExplanationPlanner().plan(
        XaiQuestion::Why, XaiReason::Justify, Traits(ModelAccess::BlackBox, InputKind::Tabular, false, true)));
    ASSERT_TRUE(Has(why, "Anchors"));
    EXPECT_GT(std::find(why.begin(), why.end(), "Anchors") - why.begin(),
              std::find(why.begin(), why.end(), "Occlusion") - why.begin());
}

TEST(ExplanationPlanner, CounterfactualsAnswerHowToBeThat) {
    const ExplanationPlan plan =
        ExplanationPlanner().plan(XaiQuestion::HowToBeThat, XaiReason::Justify, Traits(ModelAccess::Gradients, InputKind::Tabular, true));
    // None is complete or an attribution; the two that use gradients come first.
    EXPECT_EQ(Names(plan), (std::vector<std::string>{"GradientCounterfactual", "DiverseCounterfactuals", "GrowingSpheres"}));
    EXPECT_EQ(FindCheck(plan, "faithfulness"), nullptr);
    EXPECT_EQ(FindCheck(plan, "null_model")->priority, CheckPriority::Recommended);
}

// A custom catalog: the ranking rules, one at a time, and catalog order on ties.
TEST(ExplanationPlanner, RanksByQuestionThenReasonThenInputThenAccessThenCatalogOrder) {
    auto entry = [](const std::string& name, std::vector<XaiQuestion> q, ExplanationScope scope, ModelAccess access, InputKind input) {
        ExplainerCapabilities c;
        c.name = name;
        c.header = "pulsatrix/" + name + ".hpp";
        c.entry_points = {name};
        c.summary = name;
        c.questions = std::move(q);
        c.scope = scope;
        c.access = access;
        c.input = input;
        return c;
    };
    using Q = XaiQuestion;
    using S = ExplanationScope;
    using A = ModelAccess;
    using I = InputKind;
    std::vector<ExplainerCapabilities> catalog = {
        entry("Second", {Q::WhatIf, Q::Why}, S::Local, A::Relevance, I::Any),
        entry("Plain", {Q::Why}, S::Local, A::BlackBox, I::Any),
        entry("Tied", {Q::Why}, S::Local, A::BlackBox, I::Any),
        entry("Deep", {Q::Why}, S::Local, A::Gradients, I::Any),
        entry("ForImages", {Q::Why}, S::Local, A::BlackBox, I::Image),
        entry("Global", {Q::Why}, S::Global, A::BlackBox, I::Any),
    };
    catalog[1].complete = true;
    catalog[1].attribution = true;
    const ExplanationPlanner planner(catalog);
    const ModelTraits t = Traits(A::Relevance, I::Image, true, true);
    // Justify: complete Plain first; then local before global; images before Any; deeper access first.
    EXPECT_EQ(Names(planner.plan(Q::Why, XaiReason::Justify, t)),
              (std::vector<std::string>{"Plain", "ForImages", "Deep", "Tied", "Global", "Second"}));
    // Control: global first; the rest by input, then access, then catalog order.
    EXPECT_EQ(Names(planner.plan(Q::Why, XaiReason::Control, t)),
              (std::vector<std::string>{"Global", "ForImages", "Deep", "Plain", "Tied", "Second"}));

    std::vector<ExplainerCapabilities> broken = {entry("Nothing", {}, S::Local, A::BlackBox, I::Any)};
    EXPECT_THROW(ExplanationPlanner{broken}, std::invalid_argument);
}

TEST(ExplanationPlanner, ThePlanAsJsonAndAsText) {
    const ExplanationPlan plan =
        ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Control, Traits(ModelAccess::BlackBox, InputKind::Tabular, true));
    const JsonValue doc = ParseJson(WriteJson(ToJson(plan)));
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.explanation_plan.v1");
    EXPECT_EQ(doc.find("question")->as_string(), "why");
    EXPECT_EQ(doc.find("reason")->as_string(), "control");
    EXPECT_EQ(doc.find("traits")->find("access")->as_string(), "black_box");
    EXPECT_TRUE(doc.find("traits")->find("can_reinitialize")->as_bool());
    const JsonValue::Array& explainers = doc.find("explainers")->as_array();
    ASSERT_EQ(explainers.size(), plan.explainers.size());
    EXPECT_EQ(explainers.front().find("name")->as_string(), "KernelSHAP");
    EXPECT_EQ(explainers.front().find("header")->as_string(), "pulsatrix/kernel_shap.hpp");
    EXPECT_EQ(explainers.front().find("rank")->as_int64(), 1);
    const JsonValue::Array& checks = doc.find("checks")->as_array();
    ASSERT_EQ(checks.size(), plan.checks.size());
    bool saw_unavailable = false;
    for (const JsonValue& c : checks) {
        if (!c.find("available")->as_bool()) {
            saw_unavailable = true;
            EXPECT_NE(c.find("unavailable_reason"), nullptr);
        }
    }
    EXPECT_TRUE(saw_unavailable);
    EXPECT_FALSE(doc.find("complete")->as_bool());
    EXPECT_EQ(doc.find("warnings")->as_array().size(), plan.warnings.size());
    EXPECT_EQ(doc.find("excluded")->as_array().size(), plan.excluded.size());

    const std::string text = DescribePlan(plan);
    EXPECT_NE(text.find("Question: why. Why/how is this instance given this prediction?"), std::string::npos);
    EXPECT_NE(text.find("  1. KernelSHAP (pulsatrix/kernel_shap.hpp: KernelSHAP::explain)"), std::string::npos);
    EXPECT_NE(text.find("[required, can't run] per_slice_error"), std::string::npos);
    EXPECT_NE(text.find("Excluded:\n  LRP: needs an LRP rule at every layer"), std::string::npos);
    EXPECT_NE(text.find("Warnings:\n"), std::string::npos);
}

}  // namespace
}  // namespace pulsatrix
