/** @file explanation_report.hpp
 *  @brief An explanation that says what it supports (XAI-3): the attribution together with the
 *         question and reason it was made for, the method and its settings, the results of the
 *         checks the plan asks for, and caveats derived from those results and from the method's
 *         known failure modes.
 *  @ingroup interpretability
 *
 * BuildExplanation() runs an attribution explainer and every check ExplanationPlanner (XAI-2) plans
 * for it that pulsatrix can run on one input: faithfulness against random orderings, stability,
 * conservation, the randomization test, the null-model baseline and aggregation over a dataset.
 * ReportExplanation() builds the same report from checks run elsewhere, for explainers that don't
 * return an attribution (a probe, persistent homology). Either way the caveats follow from the
 * report alone (DeriveCaveats()), so a stored report can be re-checked.
 *
 * @code
 * ExplanationRequest request;
 * request.question = XaiQuestion::Why;
 * request.reason = XaiReason::Justify;
 * request.traits.access = ModelAccess::Gradients;
 * request.traits.can_reinitialize = true;
 * request.method = "IntegratedGradients";
 * request.baseline = zeros;  // what IG integrated from, for the conservation check
 * const Explanation e = BuildExplanation(request, x,
 *     [&](const Tensor& in) { return ig.explain(ctx, in, zeros, 0, 64, &backend); },
 *     [&](const Tensor& in) { return model.forward(in); }, &model);
 * if (!e.supports_reason()) std::cout << DescribeExplanation(e);  // the blocking caveats say why
 * @endcode
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/explanation_metrics.hpp"
#include "pulsatrix/explanation_planner.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/performance.hpp"

namespace pulsatrix {

/** @brief What a check found. */
enum class CheckOutcome {
    /// The check ran and the explanation met its criterion.
    Passed,
    /// The check ran and the explanation didn't meet it.
    Failed,
    /// The check ran and produced numbers with no pass or fail, such as an aggregate.
    Measured,
    /// The check didn't run; CheckResult::summary says why.
    NotRun
};

/** @brief One check's result. */
struct CheckResult {
    /** @brief The PlannedCheck's name: "faithfulness", "randomization", ... */
    std::string name;
    CheckPriority priority = CheckPriority::Recommended;
    CheckOutcome outcome = CheckOutcome::NotRun;
    /** @brief One or two sentences: what was found, or why it didn't run. */
    std::string summary;
    /** @brief The numbers behind the outcome, in a fixed order: "p_value", "deletion_auc", ... */
    std::vector<std::pair<std::string, double>> values;

    /** @brief The value named @p key, or std::nullopt. */
    [[nodiscard]] std::optional<double> value(std::string_view key) const;
};

/** @brief How much a caveat limits what the explanation supports. */
enum class CaveatSeverity {
    /// The explanation shouldn't be used for its reason: a required check failed or didn't run, or
    /// the planner excluded the method for this model.
    Blocking,
    /// A recommended check failed, or a known failure mode applies to these settings.
    Warning,
    /// A limitation of the method worth knowing whenever it is read.
    Note
};

/** @brief One thing the explanation doesn't support, and why. */
struct Caveat {
    /** @brief A stable identifier: "check.faithfulness", "method.integrated_gradients_baseline". */
    std::string id;
    CaveatSeverity severity = CaveatSeverity::Note;
    /** @brief Where it came from: "check", "method", "input" or "plan". */
    std::string source;
    std::string text;
};

/** @brief An explanation and what it does and doesn't support (XAI-3). */
struct Explanation {
    XaiQuestion question = XaiQuestion::Why;
    XaiReason reason = XaiReason::Justify;
    ModelTraits traits;
    /** @brief The catalog name of the explainer: "IntegratedGradients". */
    std::string method;
    /** @brief Its place in the plan for this question, reason and model, from 1; 0 if the planner
     *         excluded it, and then excluded_reason says why. */
    int plan_rank = 0;
    std::string excluded_reason;
    /** @brief What the explainer returned; empty for explainers that don't return an attribution. */
    std::optional<Attribution> attribution;
    /** @brief The explainer's metadata, sorted by key, then the caller's settings, then the checks'
     *         own settings under "check.": everything needed to reproduce it. */
    std::vector<std::pair<std::string, std::string>> settings;
    /** @brief Properties of the attribution itself: "sparseness" and "complexity". */
    std::vector<std::pair<std::string, double>> metrics;
    /** @brief One result per check the plan asks for, required first, plus any extra checks the
     *         caller ran. */
    std::vector<CheckResult> checks;
    /** @brief The mean absolute attribution over the dataset, when the aggregate check ran. */
    std::optional<Attribution> aggregate;
    /** @brief DeriveCaveats(*this): blocking first, then warnings, then notes. */
    std::vector<Caveat> caveats;

    /** @brief Whether nothing blocks using it for its reason. */
    [[nodiscard]] bool supports_reason() const;
    /** @brief The result of the check named @p name, or nullptr. */
    [[nodiscard]] const CheckResult* check(std::string_view name) const;
    /** @brief The value of setting @p key, or std::nullopt. */
    [[nodiscard]] std::optional<std::string> setting(std::string_view key) const;
};

/** @brief What to explain, why, and what the caller has. */
struct ExplanationRequest {
    XaiQuestion question = XaiQuestion::Why;
    XaiReason reason = XaiReason::Justify;
    ModelTraits traits;
    /** @brief The catalog name of the explainer the ExplainFn runs. */
    std::string method;
    /** @brief The output (flat index into the prediction) the explanation is for. */
    int64_t target = 0;
    /** @brief The baseline a complete explainer measured from (Integrated Gradients' path start,
     *         KernelSHAP's absent features): its attributions should add up to f(input) - f(baseline).
     *         LRP reports its own totals and needs none. */
    std::optional<Tensor> baseline;
    /** @brief Inputs to aggregate explanations over, each shaped like the one explained. */
    std::vector<Tensor> dataset;
    /** @brief For the per-slice error check: the true class of each dataset input, and the slice it
     *         belongs to. */
    std::vector<int64_t> labels;
    std::vector<std::string> slices;
    /** @brief Anything about how the explanation was made that the explainer's metadata leaves out. */
    std::vector<std::pair<std::string, std::string>> settings;
};

/** @brief How BuildExplanation() runs the checks. The defaults are the published ones where there
 *         is one; each is recorded in Explanation::settings. */
struct ExplanationCheckOptions {
    /** @brief Run only the checks the reason requires, and mark the rest NotRun. */
    bool required_only = false;
    /** @brief Seed for the random orderings, the input noise and the re-initialized weights. */
    uint64_t seed = 0;

    /** @brief Faithfulness: points on the deletion and insertion curves. */
    int64_t faithfulness_steps = 20;
    /** @brief Faithfulness: how removed features are filled. Use Imputation::NoisyLinear (ROAD)
     *         for images. */
    Imputation imputation = Imputation::Constant(0.0f);
    /** @brief Faithfulness: random feature orderings to compare with. With 19, passing means
     *         beating every one of them, p = 0.05. */
    int64_t random_orderings = 19;
    /** @brief Faithfulness: the largest p-value that passes. */
    double alpha = 0.05;

    /** @brief Stability: extra runs on the same input. */
    int64_t stability_repeats = 2;
    /** @brief Stability: runs on the input plus Gaussian noise, skipped for token and symbolic input. */
    int64_t perturbations = 3;
    /** @brief Stability: the noise's standard deviation, as a fraction of the input's. */
    double perturbation_scale = 0.01;
    /** @brief Stability: the least Spearman correlation of attribution magnitudes that passes, both
     *         between repeated runs and under input noise. */
    double min_stability = 0.8;

    /** @brief Robustness: perturbations sampled uniformly within this fraction of the input's spread
     *         (L∞), kept when they keep the predicted class. */
    double robustness_radius = 0.05;
    int64_t robustness_samples = 10;
    /** @brief Robustness: the largest relative change of the explanation that passes, both the
     *         largest over the samples (max-sensitivity) and at FGSM's point. */
    double max_sensitivity = 0.5;

    /** @brief Conservation: the largest gap between the attributions' sum and the output they
     *         explain, as a fraction of that output. */
    double conservation_tolerance = 0.05;

    /** @brief Randomization and null model: the largest Spearman correlation between the trained
     *         and the randomized model's attribution magnitudes that passes. */
    double max_random_similarity = 0.5;
};

/**
 * @brief Explains @p input with @p explain and runs the checks the plan asks for.
 *
 * What each check does:
 *  - faithfulness: DeletionCurve and InsertionCurve, scored as insertion AUC minus deletion AUC and
 *    compared with options.random_orderings random orderings of the features; it passes when the
 *    rank p-value is at most options.alpha. Not run when the attribution's shape isn't the input's
 *    (Grad-CAM).
 *  - stability: repeated runs and runs on a noisy input, compared by the Spearman correlation of the
 *    attribution magnitudes, with ComputeAttributionStability's variance; passes when every
 *    correlation is at least options.min_stability.
 *  - robustness: ExplanationSensitivity() with options.robustness_radius and
 *    robustness_samples, and FGSM's point through @p model when there is one (its parameter
 *    gradients are restored); passes when the max-sensitivity and the sensitivity at FGSM's point
 *    are at most options.max_sensitivity. NotRun when every perturbation changed the prediction.
 *  - conservation: the attributions' sum against LRP's seeded total (its metadata), or
 *    f(input) - f(baseline) when request.baseline is given; passes within
 *    options.conservation_tolerance.
 *  - randomization: ModelParameterRandomizationTest; passes when the explanation of the fully
 *    randomized model correlates at most options.max_random_similarity with the original.
 *  - null_model: NullModelBaseline, with the same threshold.
 *  - aggregate: the mean absolute attribution over request.dataset (Measured).
 *  - per_slice_error: PerSliceError() on request.dataset, with an input wrong when the index of
 *    predict's largest output isn't its label; fails when a slice is worse than the rest at
 *    options.alpha (SliceErrorCheck()).
 * Any other check is NotRun.
 *
 * @param input The input to explain, with a leading batch dimension of 1.
 * @param explain Runs the explainer on its argument, reading the model as it is at that moment:
 *        the randomization test and the null model call it again on a randomized model.
 * @param predict The model as a function, as in DeletionCurve().
 * @param model The Module @p explain and @p predict read, for the randomization test and the null
 *        model; nullptr if there is none, and those checks don't run. It is restored exactly.
 * @throws std::invalid_argument if request.method isn't in the catalog, doesn't answer
 *         request.question, or doesn't return an attribution (use ReportExplanation()).
 */
[[nodiscard]] Explanation BuildExplanation(const ExplanationRequest& request, const Tensor& input,
                                           const ExplainFn& explain, const PredictFn& predict, Module* model,
                                           const ExplanationCheckOptions& options = {});

/**
 * @brief The report for an explanation whose checks ran elsewhere: plans request.question and
 *        request.reason, matches @p checks to the planned ones by name (each takes the planned
 *        priority), marks the planned checks missing from @p checks NotRun, and derives the caveats.
 * @param attribution What the explainer returned; empty for explainers that don't return an
 *        attribution. request.target, baseline and dataset aren't used.
 * @throws std::invalid_argument if request.method isn't in the catalog or doesn't answer
 *         request.question.
 */
[[nodiscard]] Explanation ReportExplanation(const ExplanationRequest& request, std::optional<Attribution> attribution,
                                            std::vector<CheckResult> checks);

/**
 * @brief The per_slice_error check's result for a report PerSliceError() made: Failed when a slice
 *        is worse than the rest, Passed otherwise. For ReportExplanation(), or a model
 *        BuildExplanation() can't score, such as a regressor.
 */
[[nodiscard]] CheckResult SliceErrorCheck(const SliceErrorReport& report);

/**
 * @brief The caveats an explanation carries, from the report alone:
 *  - Blocking: a required check failed or didn't run, or the planner excluded the method;
 *  - Warning: a recommended check failed, or a known failure mode applies to these settings or this input (a constant fill on images, discrete input,
 *    few Integrated Gradients steps);
 *  - Note: a recommended check that didn't run, the method's own known limitations (a baseline, a rule choice, random sampling) and,
 *    for Justify, that an attribution shows what the model used, not whether it should have.
 */
[[nodiscard]] std::vector<Caveat> DeriveCaveats(const Explanation& explanation);

[[nodiscard]] std::string_view to_string(CheckOutcome outcome);
[[nodiscard]] std::string_view to_string(CaveatSeverity severity);

/** @brief The explanation as a `pulsatrix.explanation.v1` document, attribution values included. */
[[nodiscard]] JsonValue ToJson(const Explanation& explanation);

/** @brief The explanation as plain text: the question, whether it supports the reason, the
 *         largest attributions, the checks and the caveats. */
[[nodiscard]] std::string DescribeExplanation(const Explanation& explanation);

}  // namespace pulsatrix
