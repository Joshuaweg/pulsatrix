// XAI-2: from a question and a reason to ranked explainers and the checks they need.

#include "pulsatrix/explanation_planner.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace pulsatrix {

namespace {

int Level(ModelAccess access) { return static_cast<int>(access); }

std::string AccessText(ModelAccess access) {
    switch (access) {
        case ModelAccess::BlackBox: return "a prediction function";
        case ModelAccess::Weights: return "the weights";
        case ModelAccess::Activations: return "activations";
        case ModelAccess::Gradients: return "gradients";
        case ModelAccess::Relevance: return "an LRP rule at every layer";
    }
    throw std::invalid_argument("AccessText: unknown ModelAccess");
}

// Why @p c can't run on a model with @p t, or "" if it can.
std::string Unusable(const ExplainerCapabilities& c, const ModelTraits& t) {
    if (c.needs_base_model && !t.has_base_model) return "compares the model with the one it was fine-tuned from, which isn't given";
    // Representations the caller already has need no access to the model to collect them.
    const bool has_representations = c.input == InputKind::Representation && t.input == InputKind::Representation;
    if (!has_representations && Level(c.access) > Level(t.access)) {
        return "needs " + AccessText(c.access) + "; the model gives " + AccessText(t.access);
    }
    if (c.input != InputKind::Any && c.input != InputKind::Representation && t.input != InputKind::Any && c.input != t.input) {
        return "takes " + std::string(to_string(c.input)) + " input; the model takes " + std::string(to_string(t.input));
    }
    if (c.max_features > 0 && t.features > c.max_features) {
        return "takes at most " + std::to_string(c.max_features) + " features; the input has " + std::to_string(t.features);
    }
    // A matrix of representations is many inputs already.
    const bool many = t.has_dataset || t.input == InputKind::Representation;
    if (c.scope == ExplanationScope::Global && c.access != ModelAccess::Weights && !many) {
        return "looks across a dataset, and there is only the one input";
    }
    return "";
}

bool ReadsInternals(const ExplainerCapabilities& c) { return Level(c.access) >= Level(ModelAccess::Activations); }

// The ranking key: smaller is better, compared in order (the class comment lists the steps).
std::tuple<int, int, int, int, int> Key(const ExplainerCapabilities& c, XaiQuestion q, XaiReason r, const ModelTraits& t) {
    const int primary = c.questions.front() == q ? 0 : 1;
    const bool global = c.scope == ExplanationScope::Global;
    int first = 0, second = 0;
    switch (r) {
        case XaiReason::Justify:
            first = c.complete ? 0 : 1;
            second = c.attribution ? 0 : global ? 2 : 1;
            break;
        case XaiReason::Control:
        case XaiReason::Improve:
            first = global ? 0 : 1;
            break;
        case XaiReason::Discover:
            first = global ? 0 : 1;
            second = ReadsInternals(c) ? 0 : 1;
            break;
    }
    const int specific = c.input != InputKind::Any && c.input == t.input ? 0 : 1;
    return {primary, first, second, specific, -Level(c.access)};
}

std::vector<std::string> Reasons(const ExplainerCapabilities& c, XaiQuestion q, XaiReason r, const ModelTraits& t) {
    std::vector<std::string> out;
    if (c.questions.front() == q) {
        out.push_back("answers " + std::string(to_string(q)) + " as its main question");
    } else {
        out.push_back("answers " + std::string(to_string(q)) + " too; its main question is " + std::string(to_string(c.questions.front())));
    }
    const bool global = c.scope == ExplanationScope::Global;
    if (r == XaiReason::Justify && c.complete) out.push_back("its attributions add up to the output, which can be checked");
    if (r == XaiReason::Justify && c.attribution) out.push_back("returns an attribution that the faithfulness checks can test");
    if (r == XaiReason::Justify && !global) out.push_back("explains the one prediction being justified");
    if (r != XaiReason::Justify && global) out.push_back("looks across the dataset");
    if (r == XaiReason::Discover && ReadsInternals(c)) out.push_back("reads the model's internals");
    if (c.input != InputKind::Any && c.input == t.input) out.push_back("made for " + std::string(to_string(c.input)) + " input");
    out.push_back("needs " + AccessText(c.access));
    return out;
}

// An explainer's own control, beyond what the reason asks for.
struct OwnControl {
    const char* explainer;
    const char* name;
    const char* header;
    std::vector<std::string> entry_points;
    const char* purpose;
};

const std::vector<OwnControl>& OwnControls() {
    static const std::vector<OwnControl> controls = {
        {"LinearProbe", "probe_controls", "pulsatrix/probe_controls.hpp", {"MeasureSelectivity", "CompareToBaseline"},
         "A probe that also learns a control task with random labels, or scores as well on a baseline "
         "representation, shows what the probe can learn, not what the representation holds."},
        {"PersistentHomology", "topology_significance", "pulsatrix/tda/significance.hpp",
         {"UniversalNullTest", "BootstrapBand", "GaussianBaseline", "BaselinePValue"},
         "Noise has loops too: test each feature against the universal null, a bootstrap band or "
         "Gaussian clouds with the same covariance."},
        {"PCA", "split_half", "pulsatrix/pca.hpp", {"PcaSplitHalfOverlap"},
         "Principal directions that change between two halves of the data aren't a property of the "
         "representation."},
        {"SurrogateTree", "surrogate_fidelity", "pulsatrix/surrogate_tree.hpp", {"FitSurrogateTree"},
         "A tree that agrees with the model on too few held-out inputs (under about 85% at a readable depth) "
         "describes a different model; read its rules only when SurrogateFidelity::test is high."},
        {"Anchors", "anchor_precision", "pulsatrix/anchors.hpp", {"FindAnchor"},
         "Precision is estimated from samples: check that the rule met the threshold, its lower bound "
         "(Anchor::precision_lower), and its coverage; a rule that holds on a sliver of the data says little."},
        {"Featurizers", "featurizer_metrics", "pulsatrix/featurizer_metrics.hpp", {"MeasureLossRecovered"},
         "Features that don't recover the model's loss when they replace its activations describe "
         "something other than what the model computes."},
    };
    return controls;
}

PlannedCheck Check(std::string name, std::string header, std::vector<std::string> entry_points, std::string purpose,
                   CheckPriority priority, std::vector<std::string> applies_to) {
    PlannedCheck c;
    c.name = std::move(name);
    c.header = std::move(header);
    c.entry_points = std::move(entry_points);
    c.purpose = std::move(purpose);
    c.priority = priority;
    c.applies_to = std::move(applies_to);
    return c;
}

std::vector<PlannedCheck> Checks(const std::vector<PlannedExplainer>& planned, XaiReason r, const ModelTraits& t) {
    // studied: the explainers a re-initialized model is a baseline for. A metric describes how the
    // model behaves, not what it learned, so the Performance explainers get no null-model check.
    std::vector<std::string> attribution, complete, local, other, all, studied;
    for (const PlannedExplainer& p : planned) {
        const ExplainerCapabilities& c = *p.capabilities;
        all.push_back(c.name);
        if (c.questions.front() != XaiQuestion::Performance) {
            studied.push_back(c.name);
            (c.attribution ? attribution : other).push_back(c.name);
        }
        if (c.complete) complete.push_back(c.name);
        if (c.scope == ExplanationScope::Local) local.push_back(c.name);
    }
    const auto need = [r](XaiReason when) { return r == when ? CheckPriority::Required : CheckPriority::Recommended; };
    const std::string no_reinit = "needs a pulsatrix Module whose weights can be redrawn and restored";

    std::vector<PlannedCheck> checks;
    if (!attribution.empty()) {
        checks.push_back(Check("faithfulness", "pulsatrix/explanation_metrics.hpp", {"DeletionCurve", "InsertionCurve"},
                               "Removing the features the explanation ranks highest should change the prediction most "
                               "(Imputation::NoisyLinear, ROAD, for images).",
                               need(XaiReason::Justify), attribution));
        checks.push_back(Check("stability", "pulsatrix/explainer_stability.hpp", {"ComputeAttributionStability"},
                               "Repeated runs, and slightly perturbed inputs, should give the same explanation.",
                               need(XaiReason::Justify), attribution));
    }
    if (!complete.empty()) {
        checks.push_back(Check("conservation", "pulsatrix/lrp_conservation.hpp", {"ComputeConservation"},
                               "The attributions should add up to the output; a large gap means the rules or the "
                               "baseline lost part of it.",
                               need(XaiReason::Justify), complete));
    }
    if (!attribution.empty()) {
        PlannedCheck c = Check("randomization", "pulsatrix/explanation_metrics.hpp", {"ModelParameterRandomizationTest"},
                               "The explanation should change when the model's layers are randomized; one that doesn't "
                               "isn't explaining what the model learned.",
                               need(XaiReason::Control), attribution);
        if (!t.can_reinitialize) c.available = false, c.unavailable_reason = no_reinit;
        checks.push_back(std::move(c));
    }
    // Discover needs every result compared with a random model's; the others need it for whatever the
    // randomization test doesn't cover.
    const std::vector<std::string> null_targets =
        r != XaiReason::Discover ? other : studied.size() == all.size() ? std::vector<std::string>{} : studied;
    if (r == XaiReason::Discover ? !studied.empty() : !other.empty()) {
        const CheckPriority p = r == XaiReason::Discover || r == XaiReason::Control ? CheckPriority::Required : CheckPriority::Recommended;
        PlannedCheck c = Check("null_model", "pulsatrix/null_model_baseline.hpp", {"NullModelBaseline"},
                               "The same analysis of a randomly re-initialized model: a result it also gives isn't "
                               "evidence about what the model learned.",
                               p, null_targets);
        if (!t.can_reinitialize) c.available = false, c.unavailable_reason = no_reinit;
        checks.push_back(std::move(c));
    }
    for (const OwnControl& o : OwnControls()) {
        if (std::find(all.begin(), all.end(), o.explainer) == all.end()) continue;
        checks.push_back(Check(o.name, o.header, o.entry_points, o.purpose, need(XaiReason::Discover), {o.explainer}));
    }
    if (r == XaiReason::Improve && !local.empty()) {
        PlannedCheck c = Check("aggregate", "", {},
                               "Explain every input in the dataset and average each feature's attribution, so features "
                               "the model relies on everywhere, spurious ones included, stand out.",
                               CheckPriority::Required, local);
        if (!t.has_dataset) c.available = false, c.unavailable_reason = "needs a dataset; there is only the one input";
        checks.push_back(std::move(c));
    }
    if (r == XaiReason::Control) {
        PlannedCheck c = Check("per_slice_error", "pulsatrix/performance.hpp", {"PerSliceError"},
                               "The model's error on each slice of the data, each tested against the rest, so a "
                               "failure confined to one group isn't averaged away.",
                               CheckPriority::Required, {});
        if (!t.has_labels) c.available = false, c.unavailable_reason = "needs a labeled dataset with a slice for each input";
        checks.push_back(std::move(c));
    }
    std::stable_sort(checks.begin(), checks.end(), [](const PlannedCheck& a, const PlannedCheck& b) {
        return a.priority == CheckPriority::Required && b.priority != CheckPriority::Required;
    });
    return checks;
}

std::string Join(const std::vector<std::string>& items, const char* separator) {
    std::string out;
    for (size_t i = 0; i < items.size(); ++i) out += (i ? separator : "") + items[i];
    return out;
}

JsonValue Strings(const std::vector<std::string>& items) {
    JsonValue::Array a;
    for (const std::string& s : items) a.emplace_back(s);
    return JsonValue(std::move(a));
}

}  // namespace

bool ExplanationPlan::complete() const {
    if (explainers.empty()) return false;
    for (const PlannedCheck& c : checks) {
        if (c.priority == CheckPriority::Required && !c.available) return false;
    }
    return true;
}

ExplanationPlanner::ExplanationPlanner() : catalog_(&ExplainerCatalog()) {}

ExplanationPlanner::ExplanationPlanner(const std::vector<ExplainerCapabilities>& catalog) : catalog_(&catalog) {
    for (const ExplainerCapabilities& c : catalog) {
        if (c.questions.empty()) throw std::invalid_argument("ExplanationPlanner: " + c.name + " answers no question");
    }
}

ExplanationPlan ExplanationPlanner::plan(XaiQuestion question, XaiReason reason, const ModelTraits& traits) const {
    ExplanationPlan plan;
    plan.question = question;
    plan.reason = reason;
    plan.traits = traits;
    struct Candidate {
        std::tuple<int, int, int, int, int> key;
        size_t index;
    };
    std::vector<Candidate> usable;
    for (size_t i = 0; i < catalog_->size(); ++i) {
        const ExplainerCapabilities& c = (*catalog_)[i];
        if (!c.answers(question)) continue;
        const std::string why_not = Unusable(c, traits);
        if (!why_not.empty()) {
            plan.excluded.push_back({&c, why_not});
            continue;
        }
        usable.push_back({Key(c, question, reason, traits), i});
    }
    std::stable_sort(usable.begin(), usable.end(), [](const Candidate& a, const Candidate& b) { return a.key < b.key; });
    for (const Candidate& u : usable) {
        const ExplainerCapabilities& c = (*catalog_)[u.index];
        PlannedExplainer p;
        p.capabilities = &c;
        p.rank = static_cast<int>(plan.explainers.size()) + 1;
        p.reasons = Reasons(c, question, reason, traits);
        if (traits.input == InputKind::Any && c.input != InputKind::Any && c.input != InputKind::Representation) {
            p.conditions.push_back("needs " + std::string(to_string(c.input)) + " input");
        }
        if (c.max_features > 0 && traits.features == 0) {
            p.conditions.push_back("at most " + std::to_string(c.max_features) + " input features");
        }
        plan.explainers.push_back(std::move(p));
    }
    plan.checks = Checks(plan.explainers, reason, traits);

    if (plan.explainers.empty()) {
        if (plan.excluded.empty()) {
            plan.warnings.push_back("pulsatrix has no explainer yet for \"" + std::string(QuestionText(question)) + "\"");
        } else {
            plan.warnings.push_back("every explainer that answers " + std::string(to_string(question)) +
                                    " needs more than this model gives; see the excluded ones");
        }
    }
    for (const PlannedCheck& c : plan.checks) {
        if (c.priority == CheckPriority::Required && !c.available) {
            plan.warnings.push_back(std::string(to_string(reason)) + " requires the " + c.name + " check, which can't run: " +
                                    c.unavailable_reason);
        }
    }
    return plan;
}

std::string_view to_string(CheckPriority priority) {
    switch (priority) {
        case CheckPriority::Required: return "required";
        case CheckPriority::Recommended: return "recommended";
    }
    throw std::invalid_argument("to_string: unknown CheckPriority");
}

JsonValue ToJson(const ExplanationPlan& plan) {
    JsonValue::Array explainers, checks, excluded;
    for (const PlannedExplainer& p : plan.explainers) {
        explainers.push_back(JsonValue(JsonValue::Object{
            {"rank", JsonValue(static_cast<int64_t>(p.rank))},
            {"name", p.capabilities->name},
            {"header", p.capabilities->header},
            {"entry_points", Strings(p.capabilities->entry_points)},
            {"reasons", Strings(p.reasons)},
            {"conditions", Strings(p.conditions)},
        }));
    }
    for (const PlannedCheck& c : plan.checks) {
        JsonValue check(JsonValue::Object{
            {"name", c.name},
            {"header", c.header},
            {"entry_points", Strings(c.entry_points)},
            {"purpose", c.purpose},
            {"priority", std::string(to_string(c.priority))},
            {"applies_to", Strings(c.applies_to)},
            {"available", c.available},
        });
        if (!c.available) check.add("unavailable_reason", c.unavailable_reason);
        checks.push_back(std::move(check));
    }
    for (const ExcludedExplainer& e : plan.excluded) {
        excluded.push_back(JsonValue(JsonValue::Object{{"name", e.capabilities->name}, {"reason", e.reason}}));
    }
    return JsonValue(JsonValue::Object{
        {"schema", "pulsatrix.explanation_plan.v1"},
        {"question", std::string(to_string(plan.question))},
        {"reason", std::string(to_string(plan.reason))},
        {"traits", JsonValue(JsonValue::Object{
                       {"access", std::string(to_string(plan.traits.access))},
                       {"input", std::string(to_string(plan.traits.input))},
                       {"can_reinitialize", plan.traits.can_reinitialize},
                       {"has_dataset", plan.traits.has_dataset},
                       {"has_labels", plan.traits.has_labels},
                       {"has_base_model", plan.traits.has_base_model},
                       {"features", JsonValue(plan.traits.features)},
                   })},
        {"explainers", JsonValue(std::move(explainers))},
        {"checks", JsonValue(std::move(checks))},
        {"excluded", JsonValue(std::move(excluded))},
        {"warnings", Strings(plan.warnings)},
        {"complete", plan.complete()},
    });
}

std::string DescribePlan(const ExplanationPlan& plan) {
    std::ostringstream out;
    out << "Question: " << to_string(plan.question) << ". " << QuestionText(plan.question) << "\n"
        << "Reason: " << to_string(plan.reason) << ". " << ReasonText(plan.reason) << "\n";
    const ModelTraits& t = plan.traits;
    out << "Model: " << AccessText(t.access) << ", "
        << (t.input == InputKind::Any ? std::string("input of unknown kind") : std::string(to_string(t.input)) + " input")
        << (t.can_reinitialize ? ", can be re-initialized" : "") << (t.has_dataset ? ", a dataset" : ", one input")
        << (t.has_labels ? " with labels and slices" : "")
        << (t.has_base_model ? ", a base model" : "")
        << (t.features > 0 ? ", " + std::to_string(t.features) + " features per input" : std::string()) << "\n";
    out << "\nExplainers, best first:\n";
    if (plan.explainers.empty()) out << "  none\n";
    for (const PlannedExplainer& p : plan.explainers) {
        out << "  " << p.rank << ". " << p.capabilities->name << " (" << p.capabilities->header << ": "
            << Join(p.capabilities->entry_points, ", ") << ")\n"
            << "     " << Join(p.reasons, "; ") << "\n";
        if (!p.conditions.empty()) out << "     if: " << Join(p.conditions, "; ") << "\n";
    }
    if (!plan.checks.empty()) out << "\nChecks:\n";
    for (const PlannedCheck& c : plan.checks) {
        out << "  [" << to_string(c.priority) << (c.available ? "" : ", can't run") << "] " << c.name;
        if (!c.entry_points.empty()) out << ": " << Join(c.entry_points, ", ") << " (" << c.header << ")";
        if (!c.applies_to.empty()) out << ", for " << Join(c.applies_to, ", ");
        out << "\n     " << c.purpose << "\n";
        if (!c.available) out << "     can't run: " << c.unavailable_reason << "\n";
    }
    if (!plan.excluded.empty()) out << "\nExcluded:\n";
    for (const ExcludedExplainer& e : plan.excluded) out << "  " << e.capabilities->name << ": " << e.reason << "\n";
    if (!plan.warnings.empty()) out << "\nWarnings:\n";
    for (const std::string& w : plan.warnings) out << "  " << w << "\n";
    return out.str();
}

}  // namespace pulsatrix
