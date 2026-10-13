/** @file explanation_planner.hpp
 *  @brief Turns a question and a reason into a plan: which explainers can answer the question for
 *         this model, ranked, and which checks must run before the answer is trusted (XAI-2).
 *  @ingroup interpretability
 *
 * The planner only reads the catalog (explainer_capabilities.hpp); it runs nothing. It is optional:
 * every explainer it names is still called directly, through the entry points in its declaration.
 *
 * @code
 * ModelTraits traits;
 * traits.access = ModelAccess::Relevance;   // a pulsatrix network with LRP rules
 * traits.input = InputKind::Image;
 * traits.can_reinitialize = true;
 * const ExplanationPlan plan = ExplanationPlanner().plan(XaiQuestion::Why, XaiReason::Justify, traits);
 * // plan.explainers.front().capabilities->name: "LRP"
 * // plan.checks: faithfulness, stability and conservation, each with the call that runs it
 * std::cout << DescribePlan(plan);
 * @endcode
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/explainer_capabilities.hpp"
#include "pulsatrix/json.hpp"

namespace pulsatrix {

/** @brief What the caller has: the model, its inputs and its data. */
struct ModelTraits {
    /** @brief The most the model gives, from a prediction function only (BlackBox) to an LRP rule
     *         at every layer (Relevance). Each level includes the ones before it. */
    ModelAccess access = ModelAccess::BlackBox;
    /** @brief What the model takes. Any means unknown: explainers for one kind of input are then
     *         kept, with that kind as a condition. */
    InputKind input = InputKind::Any;
    /** @brief A pulsatrix Module whose weights can be redrawn and restored, for the randomization
     *         test and the null-model baseline. */
    bool can_reinitialize = false;
    /** @brief Many inputs, not just the one being explained: global explainers need them, and so
     *         does aggregating explanations over a dataset. */
    bool has_dataset = false;
    /** @brief The model this one was fine-tuned from, for explainers that compare the two. */
    bool has_base_model = false;
    /** @brief How many features each input has, or 0 if unknown; explainers with a limit
     *         (ExplainerCapabilities::max_features) are excluded above it. */
    int64_t features = 0;
};

/** @brief How much a check matters for this reason. */
enum class CheckPriority {
    /// The reason requires it: an explanation without it shouldn't be used for that reason.
    Required,
    /// Worth running; the reason doesn't depend on it.
    Recommended
};

/** @brief One check the plan asks for. */
struct PlannedCheck {
    /** @brief A stable identifier: "faithfulness", "stability", "conservation", "randomization",
     *         "null_model", "aggregate", "per_slice_error", or an explainer's own control such as
     *         "probe_controls". */
    std::string name;
    /** @brief The header and the calls that run it; empty when pulsatrix has none yet. */
    std::string header;
    std::vector<std::string> entry_points;
    /** @brief What it tests, in one sentence. */
    std::string purpose;
    CheckPriority priority = CheckPriority::Recommended;
    /** @brief The explainers it applies to; empty for all of them. */
    std::vector<std::string> applies_to;
    /** @brief Whether it can run with what the caller has. */
    bool available = true;
    /** @brief Why it can't, when it can't. */
    std::string unavailable_reason;
};

/** @brief One explainer the plan recommends. */
struct PlannedExplainer {
    /** @brief Its catalog entry. */
    const ExplainerCapabilities* capabilities = nullptr;
    /** @brief Its place in the ranking, from 1. */
    int rank = 0;
    /** @brief Why it is ranked where it is, one phrase each: "answers why as its main question". */
    std::vector<std::string> reasons;
    /** @brief What must hold for it to apply, when the traits don't say: "needs image input". */
    std::vector<std::string> conditions;
};

/** @brief An explainer that answers the question but can't run on this model. */
struct ExcludedExplainer {
    const ExplainerCapabilities* capabilities = nullptr;
    /** @brief Why not: "needs gradients; the model gives a black box". */
    std::string reason;
};

/** @brief What ExplanationPlanner::plan() returns. */
struct ExplanationPlan {
    XaiQuestion question = XaiQuestion::Why;
    XaiReason reason = XaiReason::Justify;
    ModelTraits traits;
    /** @brief The explainers to use, best first. Empty when none can answer the question here. */
    std::vector<PlannedExplainer> explainers;
    /** @brief The checks to run on their results, required ones first. */
    std::vector<PlannedCheck> checks;
    /** @brief The explainers that answer the question but need more than the caller has. */
    std::vector<ExcludedExplainer> excluded;
    /** @brief What the plan can't deliver: a question with no explainer, a required check that
     *         can't run. A report built from the plan should repeat these. */
    std::vector<std::string> warnings;

    /** @brief Whether every required check can run and some explainer applies. */
    [[nodiscard]] bool complete() const;
};

/**
 * @brief Plans explanations from the explainer catalog.
 *
 * The ranking, best first, compares each of these in turn and keeps catalog order on a tie:
 *  1. explainers whose main question it is, before those that answer it second;
 *  2. the reason's preference:
 *     - Justify: complete explainers first, then those that return an Attribution the
 *       faithfulness checks can test, then other local explainers;
 *     - Control and Improve: global explainers first, which look across the data;
 *     - Discover: global explainers first, then those that read the model's internals;
 *  3. explainers made for the model's kind of input before general ones;
 *  4. more model access before less, since an explainer that reads the computation can be checked
 *     against it.
 *
 * The checks follow the reason (the roadmap's XAI table):
 *  - Justify requires faithfulness (DeletionCurve / InsertionCurve) and stability, and
 *    conservation for complete explainers;
 *  - Control requires the model-parameter randomization test for attributions, the null-model
 *    baseline for the rest, and per-slice error, which pulsatrix doesn't have yet (KS-3);
 *  - Improve requires aggregating the explanations over a dataset;
 *  - Discover requires the null-model baseline and each explainer's own controls (control tasks
 *    for probes, a significance test for persistent homology, split halves for PCA).
 * The same checks are recommended, not required, under the other reasons.
 */
class ExplanationPlanner {
public:
    /** @brief Plans from ExplainerCatalog(). */
    ExplanationPlanner();
    /** @brief Plans from another catalog, which must outlive the planner and its plans. */
    explicit ExplanationPlanner(const std::vector<ExplainerCapabilities>& catalog);

    /** @brief The plan for @p question, asked for @p reason, about a model with @p traits. */
    [[nodiscard]] ExplanationPlan plan(XaiQuestion question, XaiReason reason, const ModelTraits& traits = {}) const;

private:
    const std::vector<ExplainerCapabilities>* catalog_;
};

[[nodiscard]] std::string_view to_string(CheckPriority priority);

/** @brief The plan as a `pulsatrix.explanation_plan.v1` document. Each explainer appears by name,
 *         with its header and entry points, so the document stands alone. */
[[nodiscard]] JsonValue ToJson(const ExplanationPlan& plan);

/** @brief The plan as plain text, for a terminal or a log: the question, the explainers in order
 *         with their reasons, the checks, and the warnings. */
[[nodiscard]] std::string DescribePlan(const ExplanationPlan& plan);

}  // namespace pulsatrix
