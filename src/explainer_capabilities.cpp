#include "pulsatrix/explainer_capabilities.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace pulsatrix {

bool ExplainerCapabilities::answers(XaiQuestion question) const {
    return std::find(questions.begin(), questions.end(), question) != questions.end();
}

std::string_view to_string(XaiQuestion question) {
    switch (question) {
        case XaiQuestion::Input: return "input";
        case XaiQuestion::Output: return "output";
        case XaiQuestion::Performance: return "performance";
        case XaiQuestion::HowGlobal: return "how_global";
        case XaiQuestion::Why: return "why";
        case XaiQuestion::WhyNot: return "why_not";
        case XaiQuestion::WhatIf: return "what_if";
        case XaiQuestion::HowToBeThat: return "how_to_be_that";
        case XaiQuestion::HowToStillBeThis: return "how_to_still_be_this";
    }
    throw std::invalid_argument("to_string: unknown XaiQuestion");
}

std::string_view to_string(XaiReason reason) {
    switch (reason) {
        case XaiReason::Justify: return "justify";
        case XaiReason::Control: return "control";
        case XaiReason::Improve: return "improve";
        case XaiReason::Discover: return "discover";
    }
    throw std::invalid_argument("to_string: unknown XaiReason");
}

std::string_view to_string(ExplanationScope scope) {
    switch (scope) {
        case ExplanationScope::Local: return "local";
        case ExplanationScope::Global: return "global";
    }
    throw std::invalid_argument("to_string: unknown ExplanationScope");
}

std::string_view to_string(ModelAccess access) {
    switch (access) {
        case ModelAccess::BlackBox: return "black_box";
        case ModelAccess::Weights: return "weights";
        case ModelAccess::Activations: return "activations";
        case ModelAccess::Gradients: return "gradients";
        case ModelAccess::Relevance: return "relevance";
    }
    throw std::invalid_argument("to_string: unknown ModelAccess");
}

std::string_view to_string(InputKind input) {
    switch (input) {
        case InputKind::Any: return "any";
        case InputKind::Tabular: return "tabular";
        case InputKind::Image: return "image";
        case InputKind::Tokens: return "tokens";
        case InputKind::Protein: return "protein";
        case InputKind::Symbolic: return "symbolic";
        case InputKind::Representation: return "representation";
    }
    throw std::invalid_argument("to_string: unknown InputKind");
}

const std::vector<XaiQuestion>& AllXaiQuestions() {
    static const std::vector<XaiQuestion> all = {
        XaiQuestion::Input, XaiQuestion::Output,      XaiQuestion::Performance,
        XaiQuestion::HowGlobal, XaiQuestion::Why,     XaiQuestion::WhyNot,
        XaiQuestion::WhatIf, XaiQuestion::HowToBeThat, XaiQuestion::HowToStillBeThis};
    return all;
}

const std::vector<XaiReason>& AllXaiReasons() {
    static const std::vector<XaiReason> all = {XaiReason::Justify, XaiReason::Control, XaiReason::Improve,
                                               XaiReason::Discover};
    return all;
}

std::optional<XaiQuestion> ParseXaiQuestion(std::string_view text) {
    for (XaiQuestion q : AllXaiQuestions()) {
        if (to_string(q) == text) {
            return q;
        }
    }
    return std::nullopt;
}

std::optional<XaiReason> ParseXaiReason(std::string_view text) {
    for (XaiReason r : AllXaiReasons()) {
        if (to_string(r) == text) {
            return r;
        }
    }
    return std::nullopt;
}

// The first question of each category in Liao, Gruen and Miller's question bank (CHI 2020,
// Figure 1), with the question mark the figure leaves off.
std::string_view QuestionText(XaiQuestion question) {
    switch (question) {
        case XaiQuestion::Input: return "What kind of data was the system trained on?";
        case XaiQuestion::Output: return "What kind of output does the system give?";
        case XaiQuestion::Performance: return "How accurate/precise/reliable are the predictions?";
        case XaiQuestion::HowGlobal: return "How does the system make predictions?";
        case XaiQuestion::Why: return "Why/how is this instance given this prediction?";
        case XaiQuestion::WhyNot: return "Why is this instance NOT predicted to be [a different outcome Q]?";
        case XaiQuestion::WhatIf: return "What would the system predict if this instance changes to...?";
        case XaiQuestion::HowToBeThat: return "How should this instance change to get a different prediction Q?";
        case XaiQuestion::HowToStillBeThis:
            return "What is the scope of change permitted for this instance to still get the same prediction?";
    }
    throw std::invalid_argument("QuestionText: unknown XaiQuestion");
}

std::string_view ReasonText(XaiReason reason) {
    switch (reason) {
        case XaiReason::Justify: return "Justify a decision to the people it affects or who must sign it off.";
        case XaiReason::Control: return "Control the model: find its flaws and failure modes.";
        case XaiReason::Improve: return "Improve the model: find what to change in its data, features or training.";
        case XaiReason::Discover: return "Discover something new from what the model learned.";
    }
    throw std::invalid_argument("ReasonText: unknown XaiReason");
}

namespace {

using Q = XaiQuestion;
using S = ExplanationScope;
using A = ModelAccess;
using I = InputKind;

ExplainerCapabilities Entry(std::string name, std::string header, std::vector<std::string> entry_points,
                            std::string summary, std::vector<XaiQuestion> questions, S scope, A access, I input) {
    ExplainerCapabilities c;
    c.name = std::move(name);
    c.header = "pulsatrix/" + std::move(header);
    c.entry_points = std::move(entry_points);
    c.summary = std::move(summary);
    c.questions = std::move(questions);
    c.scope = scope;
    c.access = access;
    c.input = input;
    return c;
}

std::vector<ExplainerCapabilities> BuildCatalog() {
    std::vector<ExplainerCapabilities> c;
    // Attribution: which input features drove this prediction.
    c.push_back(Entry("LRP", "lrp.hpp", {"LRP::explain"},
                      "Layer-wise Relevance Propagation; with an LRPTarget contrast, the relevance of logit P "
                      "minus logit Q.",
                      {Q::Why, Q::WhyNot}, S::Local, A::Relevance, I::Any));
    c.push_back(Entry("Saliency", "saliency.hpp", {"Saliency::explain"},
                      "The gradient of the target output with respect to the input.", {Q::Why}, S::Local,
                      A::Gradients, I::Any));
    c.push_back(Entry("IntegratedGradients", "integrated_gradients.hpp", {"IntegratedGradients::explain"},
                      "Gradients integrated along the straight path from a baseline to the input.", {Q::Why}, S::Local,
                      A::Gradients, I::Any));
    c.push_back(Entry("GradCAM", "grad_cam.hpp", {"GradCAM::explain"},
                      "The last convolution's activations weighted by their pooled gradients: a coarse heatmap.",
                      {Q::Why}, S::Local, A::Gradients, I::Image));
    c.push_back(Entry("KernelSHAP", "kernel_shap.hpp", {"KernelSHAP::explain"},
                      "Exact Shapley values over every coalition of features, against a baseline.", {Q::Why},
                      S::Local, A::BlackBox, I::Any));
    c.push_back(Entry("LIME", "lime.hpp", {"LIME::explain"},
                      "A weighted linear surrogate fitted to the model around the input, with its local fidelity.", {Q::Why}, S::Local,
                      A::BlackBox, I::Any));
    c.push_back(Entry("Occlusion", "sensitivity.hpp", {"Occlusion"},
                      "The change in the output when a sliding window of the input is replaced by a baseline.",
                      {Q::Why, Q::WhatIf}, S::Local, A::BlackBox, I::Any));
    c.push_back(Entry("ConceptRelevance", "concept_relevance.hpp",
                      {"LayerRelevance", "ChannelRelevance", "ConditionalHeatmaps", "ConceptReferenceSamples"},
                      "Concept Relevance Propagation: how much each channel's concept contributes, where in the input "
                      "it is, and the samples that show it.",
                      {Q::Why, Q::HowGlobal}, S::Local, A::Relevance, I::Image));
    c.push_back(Entry("LogitLens", "explainer_context.hpp", {"ExplainerContext::logit_lens"},
                      "What the output layer would predict from an intermediate layer's activations.", {Q::Why},
                      S::Local, A::Activations, I::Any));
    c.push_back(Entry("RelevanceGraph", "relevance_graph.hpp", {"BuildRelevanceGraph"},
                      "An attribution graph over tokens and layers built from AttnLRP relevance.", {Q::Why}, S::Local,
                      A::Relevance, I::Tokens));
    c.push_back(Entry("CircuitTracing", "circuit_tracing.hpp", {"TraceCircuit", "ScoreCircuit"},
                      "An attribution graph through a cross-layer transcoder replacement model.", {Q::Why}, S::Local,
                      A::Activations, I::Tokens));
    c.push_back(Entry("ProteinRelevance", "protein_explanations.hpp", {"EncoderExplainer::explain"},
                      "AttnLRP relevance per residue for a protein encoder's prediction.", {Q::Why}, S::Local,
                      A::Relevance, I::Protein));
    c.push_back(Entry("DatalogRelevance", "datalog_lrp.hpp", {"propagate_relevance_weighted"},
                      "Relevance propagated from a query back through a weighted Datalog proof.", {Q::Why}, S::Local,
                      A::Relevance, I::Symbolic));
    c.push_back(Entry("FineTuneRelevanceDiff", "fine_tune_diff.hpp",
                      {"DiffRelevance", "DiffAdapterRelevance", "TokenRelevanceShift"},
                      "The fine-tuned model's relevance minus the base model's for the same prediction.", {Q::Why},
                      S::Local, A::Relevance, I::Any));
    // What if: the prediction as the input changes.
    c.push_back(Entry("LocalSensitivity", "sensitivity.hpp", {"ComputeLocalSensitivity"},
                      "The output as each feature in turn moves to its low and high bound.", {Q::WhatIf}, S::Local,
                      A::BlackBox, I::Tabular));
    c.push_back(Entry("ActivationPatching", "explainer_context.hpp", {"ExplainerContext::forward_pass_with_patch"},
                      "The output when one layer's activation is replaced, for example by another input's.",
                      {Q::WhatIf}, S::Local, A::Activations, I::Any));
    c.push_back(Entry("ICE", "ice.hpp", {"ComputeIce"},
                      "Individual conditional expectation: each instance's output as one feature moves over a grid.",
                      {Q::WhatIf, Q::HowGlobal}, S::Global, A::BlackBox, I::Tabular));
    c.push_back(Entry("SurrogateTree", "surrogate_tree.hpp", {"FitSurrogateTree", "DescribeSurrogateTree"},
                      "A shallow decision tree fitted to the model's predictions, as readable rules, with its "
                      "fidelity to the model on held-out inputs.",
                      {Q::HowGlobal}, S::Global, A::BlackBox, I::Tabular));
    c.push_back(Entry("PDP", "pdp.hpp", {"PDP::explain"},
                      "Partial dependence: the mean output as one feature moves over a grid.",
                      {Q::HowGlobal, Q::WhatIf}, S::Global, A::BlackBox, I::Tabular));
    c.push_back(Entry("PartialDependence2D", "ice.hpp", {"ComputePartialDependence2D"},
                      "Partial dependence on two features at once, which shows how they interact.",
                      {Q::HowGlobal, Q::WhatIf}, S::Global, A::BlackBox, I::Tabular));
    c.push_back(Entry("ALE", "ice.hpp", {"ComputeAle"},
                      "Accumulated local effects: partial dependence that stays valid when features are correlated.",
                      {Q::HowGlobal, Q::WhatIf}, S::Global, A::BlackBox, I::Tabular));
    c.push_back(Entry("Morris", "global_sensitivity.hpp", {"ComputeMorris"},
                      "Morris elementary effects: which features matter, and which act non-linearly or interact.",
                      {Q::HowGlobal, Q::WhatIf}, S::Global, A::BlackBox, I::Tabular));
    c.push_back(Entry("Sobol", "global_sensitivity.hpp", {"ComputeSobol"},
                      "Sobol indices: the share of the output's variance due to each feature, alone and in total.",
                      {Q::HowGlobal, Q::WhatIf}, S::Global, A::BlackBox, I::Tabular));
    c.push_back(Entry("Steering", "steering.hpp", {"MeasureSteering", "DifferenceOfMeans"},
                      "A behavior's response as a direction is added to the residual stream at increasing strength.",
                      {Q::WhatIf, Q::HowGlobal}, S::Global, A::Activations, I::Tokens));
    c.push_back(Entry("VariantScoring", "variant_scoring.hpp", {"VariantScorer::score", "VariantScorer::single_mutant_scan"},
                      "The predicted effect of each mutation, from masked marginals.", {Q::WhatIf}, S::Local,
                      A::BlackBox, I::Protein));
    // How to be that: the change that gives another prediction.
    c.push_back(Entry("GradientCounterfactual", "counterfactual.hpp", {"FindCounterfactual"},
                      "The nearest input, by a sparse weighted L1 distance, that reaches the target class or range.",
                      {Q::HowToBeThat, Q::WhyNot}, S::Local, A::Gradients, I::Tabular));
    c.push_back(Entry("GrowingSpheres", "counterfactual.hpp", {"GrowingSpheresCounterfactual"},
                      "A counterfactual found without gradients, by sampling spheres of growing radius.",
                      {Q::HowToBeThat, Q::WhyNot}, S::Local, A::BlackBox, I::Tabular));
    c.push_back(Entry("DiverseCounterfactuals", "counterfactual.hpp", {"FindDiverseCounterfactuals"},
                      "A set of counterfactuals that differ from each other (DiCE).", {Q::HowToBeThat, Q::WhyNot},
                      S::Local, A::Gradients, I::Tabular));
    // Input: the data the model sees.
    c.push_back(Entry("ContrastiveExplanations", "contrastive.hpp", {"FindPertinentNegative", "FindPertinentPositive"},
                      "CEM: what is absent and would change the class if present (a pertinent negative), and the "
                      "least of the input that keeps it (a pertinent positive).",
                      {Q::WhyNot, Q::Why}, S::Local, A::Gradients, I::Any));
    c.push_back(Entry("Anchors", "anchors.hpp", {"FindAnchor", "DescribeAnchor"},
                      "An if-then rule around the input that keeps the prediction with high estimated precision, "
                      "and the share of the data it covers.",
                      {Q::HowToStillBeThis, Q::Why}, S::Local, A::BlackBox, I::Tabular));
    c.push_back(Entry("NearestNeighbors", "nearest_neighbors.hpp", {"NearestNeighbors", "MutualNearestNeighbors"},
                      "The examples nearest an input in a representation space.", {Q::Input, Q::Why}, S::Local,
                      A::Activations, I::Representation));
    // How (global): what the model represents and how it works.
    c.push_back(Entry("Prototypes", "prototypes.hpp", {"FindPrototypes"},
                      "The background examples nearest the input that the model classes the same way, and the "
                      "nearest it classes differently.",
                      {Q::Input, Q::Why}, S::Local, A::BlackBox, I::Any));
    c.push_back(Entry("ProtoDash", "prototypes.hpp", {"ProtoDash"},
                      "Weighted prototypes whose kernel mean matches a dataset's or a class's (MMD).",
                      {Q::Input, Q::HowGlobal}, S::Global, A::BlackBox, I::Any));
    c.push_back(Entry("TCAV", "tcav.hpp", {"RunTcav"},
                      "How sensitive a class is to a concept defined by examples, tested against random concepts.",
                      {Q::HowGlobal}, S::Global, A::Gradients, I::Any));
    c.push_back(Entry("LinearProbe", "probe_controls.hpp", {"TrainLinearProbe", "MeasureSelectivity", "CompareToBaseline"},
                      "Whether a property can be read linearly from a representation, with control tasks.",
                      {Q::HowGlobal}, S::Global, A::Activations, I::Representation));
    c.push_back(Entry("RepresentationSimilarity", "representation_similarity.hpp", {"LinearCka", "Rsa", "Svcca"},
                      "How alike two representations of the same inputs are: CKA, RSA and SVCCA.", {Q::HowGlobal},
                      S::Global, A::Activations, I::Representation));
    c.push_back(Entry("RepresentationGeometry", "representation_geometry.hpp",
                      {"Anisotropy", "MeasureEffectiveRank", "TwoNNDimension"},
                      "A representation's anisotropy, effective rank and intrinsic dimension.", {Q::HowGlobal},
                      S::Global, A::Activations, I::Representation));
    c.push_back(Entry("PCA", "pca.hpp", {"FitPca", "PcaSplitHalfOverlap"},
                      "The principal directions of a representation, and how stable they are.", {Q::HowGlobal},
                      S::Global, A::Activations, I::Representation));
    c.push_back(Entry("PersistentHomology", "tda/rips_persistence.hpp", {"ComputeRipsPersistence"},
                      "The persistent homology of a representation: its clusters, loops and voids across scales.",
                      {Q::HowGlobal}, S::Global, A::Activations, I::Representation));
    c.push_back(Entry("RepresentationTopologyDivergence", "tda/rtd.hpp", {"CompareRepresentations", "Rtd"},
                      "How far apart two representations of the same inputs are in topology.", {Q::HowGlobal},
                      S::Global, A::Activations, I::Representation));
    c.push_back(Entry("Featurizers", "featurizer.hpp", {"TrainFeaturizer", "Featurizer::encode"},
                      "Sparse dictionaries of features on activations: SAEs, transcoders, block-sparse featurizers.",
                      {Q::HowGlobal}, S::Global, A::Activations, I::Representation));
    c.push_back(Entry("Crosscoder", "crosscoder.hpp", {"Crosscoder", "CrosscoderLatents", "MeasureLatentScaling"},
                      "One dictionary across two models or layers: which features are shared and which belong to "
                      "one.",
                      {Q::HowGlobal}, S::Global, A::Activations, I::Representation));
    c.push_back(Entry("ParameterDecomposition", "parameter_decomposition.hpp", {"TrainDecomposition", "MeasureImportance"},
                      "The weights split into rank-one components, each with a causal importance per input.",
                      {Q::HowGlobal}, S::Global, A::Gradients, I::Any));
    c.push_back(Entry("IntruderDimensions", "fine_tune_diff.hpp", {"IntruderDimensions", "LoraIntruderDimensions"},
                      "Top singular directions of fine-tuned weights that no base direction resembles.",
                      {Q::HowGlobal}, S::Global, A::Weights, I::Any));
    c.push_back(Entry("ContactPrediction", "protein_contacts.hpp", {"RankContactHeads", "ContactPredictor::predict"},
                      "Which attention heads track residue contacts, and how well.", {Q::HowGlobal}, S::Global,
                      A::Activations, I::Protein));
    // Performance: how well the model does (KS-3). Each needs labeled predictions over a dataset.
    c.push_back(Entry("ClassifierMetrics", "performance.hpp", {"EvaluateClassifier"},
                      "Accuracy, log loss, the Brier score, the expected calibration error with its reliability "
                      "diagram, and each class's precision, recall and F1.",
                      {Q::Performance}, S::Global, A::BlackBox, I::Any));
    c.push_back(Entry("RegressorMetrics", "performance.hpp", {"EvaluateRegressor"},
                      "Mean squared and absolute error, and R squared.", {Q::Performance}, S::Global, A::BlackBox,
                      I::Any));
    c.push_back(Entry("PerSliceError", "performance.hpp", {"PerSliceError", "DescribeSliceError"},
                      "The error on each slice of the data, each tested against the rest with a Bonferroni "
                      "correction, so a failure confined to one group isn't averaged away.",
                      {Q::Performance}, S::Global, A::BlackBox, I::Any));
    for (ExplainerCapabilities& e : c) {
        e.attribution = e.name == "LRP" || e.name == "Saliency" || e.name == "IntegratedGradients" || e.name == "GradCAM" ||
                        e.name == "KernelSHAP" || e.name == "LIME" || e.name == "Occlusion";
        e.complete = e.name == "LRP" || e.name == "IntegratedGradients" || e.name == "KernelSHAP" ||
                     e.name == "ConceptRelevance" || e.name == "ProteinRelevance" || e.name == "FineTuneRelevanceDiff";
        e.needs_base_model = e.name == "FineTuneRelevanceDiff" || e.name == "IntruderDimensions";
        e.max_features = e.name == "KernelSHAP" ? 20 : 0;
    }
    return c;
}

}  // namespace

const std::vector<ExplainerCapabilities>& ExplainerCatalog() {
    static const std::vector<ExplainerCapabilities> catalog = BuildCatalog();
    return catalog;
}

const ExplainerCapabilities& ExplainerCapabilitiesOf(std::string_view name) {
    for (const ExplainerCapabilities& c : ExplainerCatalog()) {
        if (c.name == name) {
            return c;
        }
    }
    throw std::invalid_argument("ExplainerCapabilitiesOf: no explainer named \"" + std::string(name) + "\"");
}

std::vector<const ExplainerCapabilities*> ExplainersFor(XaiQuestion question) {
    std::vector<const ExplainerCapabilities*> primary, secondary;
    for (const ExplainerCapabilities& c : ExplainerCatalog()) {
        if (!c.questions.empty() && c.questions.front() == question) {
            primary.push_back(&c);
        } else if (c.answers(question)) {
            secondary.push_back(&c);
        }
    }
    primary.insert(primary.end(), secondary.begin(), secondary.end());
    return primary;
}

JsonValue ToJson(const ExplainerCapabilities& capabilities) {
    JsonValue::Array entry_points, questions;
    for (const std::string& e : capabilities.entry_points) {
        entry_points.emplace_back(e);
    }
    for (XaiQuestion q : capabilities.questions) {
        questions.emplace_back(std::string(to_string(q)));
    }
    return JsonValue(JsonValue::Object{
        {"name", capabilities.name},
        {"header", capabilities.header},
        {"entry_points", std::move(entry_points)},
        {"summary", capabilities.summary},
        {"questions", std::move(questions)},
        {"scope", std::string(to_string(capabilities.scope))},
        {"access", std::string(to_string(capabilities.access))},
        {"input", std::string(to_string(capabilities.input))},
        {"attribution", capabilities.attribution},
        {"complete", capabilities.complete},
        {"needs_base_model", capabilities.needs_base_model},
        {"max_features", JsonValue(capabilities.max_features)},
    });
}

JsonValue ExplainerCatalogDocument() {
    JsonValue::Array explainers;
    for (const ExplainerCapabilities& c : ExplainerCatalog()) {
        explainers.push_back(ToJson(c));
    }
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.explainer_catalog.v1"},
        {"explainers", std::move(explainers)},
    });
}

}  // namespace pulsatrix
