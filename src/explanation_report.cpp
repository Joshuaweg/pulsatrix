// XAI-3: an explanation together with the checks its plan asks for and the caveats that follow.

#include "pulsatrix/explanation_report.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "portable_random.hpp"
#include "pulsatrix/explainer_stability.hpp"
#include "pulsatrix/null_model_baseline.hpp"

namespace pulsatrix {

namespace {

std::string Fmt(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.3g", v);
    return buf;
}

std::string Percent(double fraction) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.0f%%", 100.0 * fraction);
    return buf;
}

std::vector<float> Magnitudes(const Tensor& t) {
    std::vector<float> v = t.to_host_vector();
    for (float& x : v) x = std::fabs(x);
    return v;
}

// Spearman correlation, with identical vectors (an all-zero map repeated, say) counted as 1, not as
// the 0 SpearmanRankCorrelation gives a constant vector.
double Similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a == b) return 1.0;
    return SpearmanRankCorrelation(a, b);
}

CheckResult Result(const PlannedCheck& planned, CheckOutcome outcome, std::string summary) {
    CheckResult r;
    r.name = planned.name;
    r.priority = planned.priority;
    r.outcome = outcome;
    r.summary = std::move(summary);
    return r;
}

std::string ShapeText(const Shape& s) {
    std::string out = "(";
    for (int64_t i = 0; i < s.rank(); ++i) out += (i ? ", " : "") + std::to_string(s.dim(i));
    return out + ")";
}

// --- The checks BuildExplanation runs ------------------------------------------------------------

struct CheckInputs {
    const ExplanationRequest& request;
    const Tensor& input;
    const Attribution& attribution;
    const ExplainFn& explain;
    const PredictFn& predict;
    Module* model;
    const ExplanationCheckOptions& options;
};

CheckResult Faithfulness(const PlannedCheck& planned, const CheckInputs& in) {
    const Attribution& a = in.attribution;
    if (a.values.shape() != in.input.shape()) {
        return Result(planned, CheckOutcome::NotRun,
                      "The attribution's shape " + ShapeText(a.values.shape()) + " isn't the input's " +
                          ShapeText(in.input.shape()) + ", so its features can't be removed one at a time.");
    }
    const ExplanationCheckOptions& o = in.options;
    PerturbationOptions po;
    po.steps = o.faithfulness_steps;
    po.target = in.request.target;
    po.imputation = o.imputation;
    const PerturbationCurve deletion = DeletionCurve(in.predict, in.input, a, po);
    const PerturbationCurve insertion = InsertionCurve(in.predict, in.input, a, po);
    const double gap = insertion.auc - deletion.auc;

    // Random orderings: the values 0..n-1 shuffled, so there are no ties.
    PortableRng rng{o.seed};
    std::vector<float> order(static_cast<size_t>(in.input.numel()));
    std::iota(order.begin(), order.end(), 0.0f);
    int64_t as_good = 0;
    double random_deletion = 0, random_insertion = 0;
    for (int64_t r = 0; r < o.random_orderings; ++r) {
        for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[rng.next() % i]);
        const Attribution random{"random_order", Tensor(in.input.shape(), in.input.backend(), order, in.input.device()), {}};
        const PerturbationCurve d = DeletionCurve(in.predict, in.input, random, po);
        const PerturbationCurve i = InsertionCurve(in.predict, in.input, random, po);
        random_deletion += d.auc;
        random_insertion += i.auc;
        if (i.auc - d.auc >= gap) ++as_good;
    }
    const auto n = static_cast<double>(o.random_orderings);
    const double p = (1.0 + static_cast<double>(as_good)) / (1.0 + n);
    const bool passed = p <= o.alpha;
    CheckResult r = Result(planned, passed ? CheckOutcome::Passed : CheckOutcome::Failed,
                           "Removing the features it ranks highest, and restoring them, moved the target output "
                           "more than " + std::to_string(o.random_orderings - as_good) + " of " +
                               std::to_string(o.random_orderings) + " random orderings did (insertion minus deletion AUC " +
                               Fmt(gap) + ", p = " + Fmt(p) + ").");
    r.values = {{"p_value", p},
                {"deletion_auc", deletion.auc},
                {"insertion_auc", insertion.auc},
                {"random_deletion_auc", random_deletion / n},
                {"random_insertion_auc", random_insertion / n},
                {"random_orderings", n}};
    return r;
}

CheckResult Stability(const PlannedCheck& planned, const CheckInputs& in) {
    const Attribution& a = in.attribution;
    const ExplanationCheckOptions& o = in.options;
    if (a.values.numel() < 2) return Result(planned, CheckOutcome::NotRun, "The attribution has fewer than two values to rank.");
    const std::vector<float> reference = Magnitudes(a.values);

    std::vector<Attribution> runs{a};
    double repeat = 1.0;
    for (int64_t k = 0; k < o.stability_repeats; ++k) {
        runs.push_back(in.explain(in.input));
        repeat = std::min(repeat, Similarity(reference, Magnitudes(runs.back().values)));
    }
    const StabilityResult variance = ComputeAttributionStability(runs);

    const InputKind kind = in.request.traits.input;
    const bool perturb = o.perturbations > 0 && kind != InputKind::Tokens && kind != InputKind::Symbolic;
    double perturbed = 0;
    if (perturb) {
        const std::vector<float> x = in.input.to_host_vector();
        double mean = 0, var = 0;
        for (float v : x) mean += v;
        mean /= static_cast<double>(x.size());
        for (float v : x) var += (v - mean) * (v - mean);
        const double sd = std::sqrt(var / static_cast<double>(x.size()));
        const double scale = o.perturbation_scale * (sd > 0 ? sd : 1.0);
        PortableRng rng{o.seed ^ 0x5DEECE66DULL};
        for (int64_t k = 0; k < o.perturbations; ++k) {
            std::vector<float> noisy = x;
            for (float& v : noisy) v += static_cast<float>(scale * rng.gaussian());
            const Attribution e = in.explain(Tensor(in.input.shape(), in.input.backend(), noisy, in.input.device()));
            perturbed += Similarity(reference, Magnitudes(e.values));
        }
        perturbed /= static_cast<double>(o.perturbations);
    }

    const bool passed = repeat >= o.min_stability && (!perturb || perturbed >= o.min_stability);
    std::string summary;
    if (o.stability_repeats > 0) {
        summary = variance.is_deterministic ? "Repeated runs were identical"
                                            : "Repeated runs differed (rank correlation at least " + Fmt(repeat) + ")";
    } else {
        summary = "No repeated runs";
    }
    if (perturb) {
        summary += "; with noise of " + Percent(o.perturbation_scale) + " of the input's spread, the ranking correlated " +
                   Fmt(perturbed) + " with the original on average";
    } else if (kind == InputKind::Tokens || kind == InputKind::Symbolic) {
        summary += "; the input is discrete, so it wasn't perturbed";
    }
    CheckResult r = Result(planned, passed ? CheckOutcome::Passed : CheckOutcome::Failed, summary + ".");
    if (o.stability_repeats > 0) r.values.emplace_back("repeat_similarity", repeat);
    if (perturb) r.values.emplace_back("perturbed_similarity", perturbed);
    r.values.emplace_back("mean_variance", variance.mean_variance);
    r.values.emplace_back("deterministic", variance.is_deterministic ? 1.0 : 0.0);
    return r;
}

CheckResult Conservation(const PlannedCheck& planned, const CheckInputs& in) {
    const Attribution& a = in.attribution;
    double expected = 0;
    std::string what;
    const auto seeded = a.metadata.find("relevance_out_sum");
    if (seeded != a.metadata.end()) {
        expected = std::stod(seeded->second);
        what = "the relevance LRP seeded";
    } else if (in.request.baseline) {
        const int64_t t = in.request.target;
        expected = static_cast<double>(in.predict(in.input).read_element(t)) -
                   static_cast<double>(in.predict(*in.request.baseline).read_element(t));
        what = "f(input) - f(baseline)";
    } else {
        return Result(planned, CheckOutcome::NotRun,
                      "Needs the baseline the explainer measured from (ExplanationRequest::baseline).");
    }
    double sum = 0;
    for (float v : a.values.to_host_vector()) sum += v;
    const double gap = std::fabs(sum - expected);
    const double relative = gap / std::max(std::fabs(expected), 1e-12);
    const bool passed = gap <= in.options.conservation_tolerance * std::max(std::fabs(expected), 1e-6);
    CheckResult r = Result(planned, passed ? CheckOutcome::Passed : CheckOutcome::Failed,
                           "The attributions add up to " + Fmt(sum) + " against " + Fmt(expected) + ", " + what + ": a gap of " +
                               Percent(std::min(relative, 99.99)) + ".");
    r.values = {{"attribution_sum", sum}, {"expected", expected}, {"gap", gap}, {"relative_gap", relative}};
    return r;
}

CheckResult Randomization(const PlannedCheck& planned, const CheckInputs& in) {
    if (in.model == nullptr) return Result(planned, CheckOutcome::NotRun, "No Module was given to randomize.");
    const RandomizationResult result = ModelParameterRandomizationTest(*in.model, in.explain, in.input, in.options.seed);
    if (result.similarity.empty()) return Result(planned, CheckOutcome::NotRun, "The model has no parameters to randomize.");
    const double all = result.similarity.back();
    const bool passed = all <= in.options.max_random_similarity;
    CheckResult r = Result(planned, passed ? CheckOutcome::Passed : CheckOutcome::Failed,
                           "With the " + std::to_string(result.similarity.size()) +
                               " layers randomized from the output down, the explanation's ranking correlates " + Fmt(all) +
                               " with the original (" + Fmt(result.similarity.front()) + " after the top layer alone).");
    r.values = {{"layers", static_cast<double>(result.similarity.size())},
                {"similarity_top_layer", result.similarity.front()},
                {"similarity_all_layers", all}};
    return r;
}

CheckResult NullModel(const PlannedCheck& planned, const CheckInputs& in) {
    if (in.model == nullptr) return Result(planned, CheckOutcome::NotRun, "No Module was given to re-initialize.");
    const AttributionNullReport report = NullModelBaseline(*in.model, in.explain, in.input, in.options.seed);
    const bool passed = report.rank_similarity <= in.options.max_random_similarity;
    CheckResult r = Result(planned, passed ? CheckOutcome::Passed : CheckOutcome::Failed,
                           "Explained with a re-initialized copy of the model, the ranking correlates " +
                               Fmt(report.rank_similarity) + " with the trained model's.");
    r.values = {{"rank_similarity", report.rank_similarity}};
    return r;
}

CheckResult Aggregate(const PlannedCheck& planned, const CheckInputs& in, std::optional<Attribution>& aggregate) {
    const std::vector<Tensor>& dataset = in.request.dataset;
    if (dataset.empty()) return Result(planned, CheckOutcome::NotRun, "Needs a dataset (ExplanationRequest::dataset).");
    const Shape& shape = in.attribution.values.shape();
    std::vector<double> total(static_cast<size_t>(shape.numel()), 0.0);
    for (const Tensor& x : dataset) {
        const Attribution e = in.explain(x);
        if (e.values.shape() != shape) {
            throw std::invalid_argument("BuildExplanation: an attribution of a dataset input has shape " +
                                        ShapeText(e.values.shape()) + ", not " + ShapeText(shape));
        }
        const std::vector<float> m = Magnitudes(e.values);
        for (size_t i = 0; i < m.size(); ++i) total[i] += m[i];
    }
    std::vector<float> mean(total.size());
    double sum = 0;
    size_t top = 0;
    for (size_t i = 0; i < total.size(); ++i) {
        mean[i] = static_cast<float>(total[i] / static_cast<double>(dataset.size()));
        sum += mean[i];
        if (mean[i] > mean[top]) top = i;
    }
    const double share = sum > 0 ? mean[top] / sum : 0.0;
    aggregate = Attribution{"mean_abs_" + in.attribution.method,
                            Tensor(shape, in.attribution.values.backend(), mean, in.attribution.values.device()),
                            {{"inputs", std::to_string(dataset.size())}}};
    CheckResult r = Result(planned, CheckOutcome::Measured,
                           "Averaged over " + std::to_string(dataset.size()) + " inputs, feature " + std::to_string(top) +
                               " holds " + Percent(share) + " of the absolute attribution.");
    r.values = {{"inputs", static_cast<double>(dataset.size())}, {"top_feature", static_cast<double>(top)}, {"top_share", share}};
    return r;
}

CheckResult PerSlice(const PlannedCheck& planned, const CheckInputs& in) {
    const ExplanationRequest& q = in.request;
    if (q.dataset.empty() || q.labels.empty() || q.slices.empty()) {
        return Result(planned, CheckOutcome::NotRun,
                      "Needs a labeled dataset: ExplanationRequest::dataset, ::labels and ::slices.");
    }
    if (q.labels.size() != q.dataset.size() || q.slices.size() != q.dataset.size()) {
        throw std::invalid_argument("BuildExplanation: " + std::to_string(q.dataset.size()) + " dataset inputs, " +
                                    std::to_string(q.labels.size()) + " labels and " + std::to_string(q.slices.size()) +
                                    " slices");
    }
    std::vector<double> errors;
    for (size_t i = 0; i < q.dataset.size(); ++i) {
        const std::vector<float> out = in.predict(q.dataset[i]).to_host_vector();
        if (out.empty()) throw std::invalid_argument("BuildExplanation: the model's output for a dataset input is empty");
        const int64_t predicted = static_cast<int64_t>(std::max_element(out.begin(), out.end()) - out.begin());
        errors.push_back(predicted == q.labels[i] ? 0.0 : 1.0);
    }
    SliceErrorOptions options;
    options.alpha = in.options.alpha;
    options.seed = in.options.seed;
    CheckResult r = SliceErrorCheck(PerSliceError(errors, q.slices, options));
    r.priority = planned.priority;
    return r;
}

// --- Assembly ------------------------------------------------------------------------------------

const ExplainerCapabilities& Method(const ExplanationRequest& request) {
    const ExplainerCapabilities& c = ExplainerCapabilitiesOf(request.method);
    if (!c.answers(request.question)) {
        throw std::invalid_argument(request.method + " doesn't answer the " + std::string(to_string(request.question)) +
                                    " question");
    }
    return c;
}

bool AppliesTo(const PlannedCheck& c, const std::string& method) {
    return c.applies_to.empty() || std::find(c.applies_to.begin(), c.applies_to.end(), method) != c.applies_to.end();
}

Explanation Start(const ExplanationRequest& request, const ExplanationPlan& plan) {
    Explanation e;
    e.question = request.question;
    e.reason = request.reason;
    e.traits = request.traits;
    e.method = request.method;
    for (const PlannedExplainer& p : plan.explainers) {
        if (p.capabilities->name == request.method) e.plan_rank = p.rank;
    }
    for (const ExcludedExplainer& x : plan.excluded) {
        if (x.capabilities->name == request.method) e.excluded_reason = x.reason;
    }
    return e;
}

void AddSettings(Explanation& e, const ExplanationRequest& request) {
    if (e.attribution) {
        std::vector<std::pair<std::string, std::string>> metadata(e.attribution->metadata.begin(),
                                                                  e.attribution->metadata.end());
        std::sort(metadata.begin(), metadata.end());
        e.settings = std::move(metadata);
    }
    e.settings.insert(e.settings.end(), request.settings.begin(), request.settings.end());
}

void AddMetrics(Explanation& e) {
    if (!e.attribution || e.attribution->values.numel() == 0) return;
    e.metrics = {{"sparseness", Sparseness(*e.attribution)}, {"complexity", Complexity(*e.attribution)}};
}

std::string ImputationText(const Imputation& i) {
    return i.kind == Imputation::Kind::Constant ? "constant:" + Fmt(i.value) : "noisy_linear:" + Fmt(i.noise_std);
}

void Validate(const ExplanationCheckOptions& o) {
    if (o.faithfulness_steps < 1) throw std::invalid_argument("BuildExplanation: faithfulness_steps must be at least 1");
    if (o.random_orderings < 1) throw std::invalid_argument("BuildExplanation: random_orderings must be at least 1");
    if (o.stability_repeats < 0 || o.perturbations < 0) {
        throw std::invalid_argument("BuildExplanation: stability_repeats and perturbations must not be negative");
    }
}

// --- Caveats ---------------------------------------------------------------------------------------

std::string Use(XaiReason reason) {
    switch (reason) {
        case XaiReason::Justify: return "justify this prediction";
        case XaiReason::Control: return "vet the model";
        case XaiReason::Improve: return "decide what to change in the model";
        case XaiReason::Discover: return "draw conclusions about the domain";
    }
    throw std::invalid_argument("Use: unknown XaiReason");
}

struct MethodCaveat {
    const char* method;
    const char* id;
    const char* text;
};

// Known failure modes of each attribution method, read whenever its explanation is.
const std::vector<MethodCaveat>& MethodCaveats() {
    static const std::vector<MethodCaveat> caveats = {
        {"Saliency", "method.saliency_sensitivity",
         "Gradients show how sensitive the output is to each feature at this input, not how much each "
         "contributed, and they vanish where the output saturates (Sundararajan et al. 2017)."},
        {"IntegratedGradients", "method.integrated_gradients_baseline",
         "The attributions explain f(input) - f(baseline): another baseline gives another explanation "
         "(Sturmfels et al. 2020)."},
        {"LRP", "method.lrp_rules",
         "The relevance depends on the rule at each layer: epsilon absorbs part of it and biases take a share, "
         "so another composite can rank the features differently."},
        {"GradCAM", "method.grad_cam_resolution",
         "The heatmap has the target layer's resolution, upsampled, so it can't localize anything smaller, "
         "and it keeps only positive evidence."},
        {"KernelSHAP", "method.kernel_shap_baseline",
         "An absent feature takes the baseline's value, so the Shapley values compare with the baseline and "
         "treat features as independent; correlated features split credit unpredictably (Kumar et al. 2020)."},
        {"LIME", "method.lime_sampling",
         "The surrogate is fitted to random perturbations: another seed, sample count or kernel width gives "
         "another explanation (Alvarez-Melis and Jaakkola 2018)."},
        {"Occlusion", "method.occlusion_fill",
         "Covered windows are filled with a baseline value the model may never have seen, so the score "
         "includes its reaction to the fill (Hooker et al. 2019)."},
    };
    return caveats;
}

std::string Capitalized(std::string s) {
    if (!s.empty() && s[0] >= 'a' && s[0] <= 'z') s[0] = static_cast<char>(s[0] - 'a' + 'A');
    return s;
}

bool Ran(const CheckResult* c) {
    return c != nullptr && (c->outcome == CheckOutcome::Passed || c->outcome == CheckOutcome::Failed);
}

JsonValue Pairs(const std::vector<std::pair<std::string, double>>& values) {
    JsonValue::Object o;
    for (const auto& [k, v] : values) o.emplace_back(k, JsonValue(v));
    return JsonValue(std::move(o));
}

JsonValue AttributionJson(const Attribution& a) {
    JsonValue::Array shape, values;
    for (int64_t i = 0; i < a.values.rank(); ++i) shape.push_back(JsonValue(a.values.shape().dim(i)));
    for (float v : a.values.to_host_vector()) values.push_back(JsonValue(static_cast<double>(v)));
    std::vector<std::pair<std::string, std::string>> metadata(a.metadata.begin(), a.metadata.end());
    std::sort(metadata.begin(), metadata.end());
    JsonValue::Object m;
    for (auto& [k, v] : metadata) m.emplace_back(k, JsonValue(v));
    return JsonValue(JsonValue::Object{
        {"method", a.method},
        {"shape", JsonValue(std::move(shape))},
        {"values", JsonValue(std::move(values))},
        {"metadata", JsonValue(std::move(m))},
    });
}

}  // namespace

std::optional<double> CheckResult::value(std::string_view key) const {
    for (const auto& [k, v] : values) {
        if (k == key) return v;
    }
    return std::nullopt;
}

bool Explanation::supports_reason() const {
    return std::none_of(caveats.begin(), caveats.end(), [](const Caveat& c) { return c.severity == CaveatSeverity::Blocking; });
}

const CheckResult* Explanation::check(std::string_view name) const {
    for (const CheckResult& c : checks) {
        if (c.name == name) return &c;
    }
    return nullptr;
}

std::optional<std::string> Explanation::setting(std::string_view key) const {
    for (const auto& [k, v] : settings) {
        if (k == key) return v;
    }
    return std::nullopt;
}

Explanation BuildExplanation(const ExplanationRequest& request, const Tensor& input, const ExplainFn& explain,
                             const PredictFn& predict, Module* model, const ExplanationCheckOptions& options) {
    const ExplainerCapabilities& caps = Method(request);
    if (!caps.attribution) {
        throw std::invalid_argument("BuildExplanation: " + request.method +
                                    " doesn't return an attribution; run its checks and use ReportExplanation");
    }
    Validate(options);
    const ExplanationPlan plan = ExplanationPlanner().plan(request.question, request.reason, request.traits);
    Explanation e = Start(request, plan);
    e.attribution = explain(input);
    AddSettings(e, request);
    const ExplanationCheckOptions& o = options;
    e.settings.insert(e.settings.end(), {
                                            {"check.target", std::to_string(request.target)},
                                            {"check.seed", std::to_string(o.seed)},
                                            {"check.imputation", ImputationText(o.imputation)},
                                            {"check.faithfulness_steps", std::to_string(o.faithfulness_steps)},
                                            {"check.random_orderings", std::to_string(o.random_orderings)},
                                            {"check.alpha", Fmt(o.alpha)},
                                            {"check.stability_repeats", std::to_string(o.stability_repeats)},
                                            {"check.perturbations", std::to_string(o.perturbations)},
                                            {"check.perturbation_scale", Fmt(o.perturbation_scale)},
                                            {"check.min_stability", Fmt(o.min_stability)},
                                            {"check.conservation_tolerance", Fmt(o.conservation_tolerance)},
                                            {"check.max_random_similarity", Fmt(o.max_random_similarity)},
                                        });
    AddMetrics(e);

    const CheckInputs in{request, input, *e.attribution, explain, predict, model, options};
    for (const PlannedCheck& planned : plan.checks) {
        if (!AppliesTo(planned, request.method)) continue;
        if (!planned.available) {
            e.checks.push_back(Result(planned, CheckOutcome::NotRun, Capitalized(planned.unavailable_reason) + "."));
        } else if (o.required_only && planned.priority != CheckPriority::Required) {
            e.checks.push_back(Result(planned, CheckOutcome::NotRun,
                                      "Skipped: only the required checks were run (ExplanationCheckOptions::required_only)."));
        } else if (planned.name == "faithfulness") {
            e.checks.push_back(Faithfulness(planned, in));
        } else if (planned.name == "stability") {
            e.checks.push_back(Stability(planned, in));
        } else if (planned.name == "conservation") {
            e.checks.push_back(Conservation(planned, in));
        } else if (planned.name == "randomization") {
            e.checks.push_back(Randomization(planned, in));
        } else if (planned.name == "null_model") {
            e.checks.push_back(NullModel(planned, in));
        } else if (planned.name == "aggregate") {
            e.checks.push_back(Aggregate(planned, in, e.aggregate));
        } else if (planned.name == "per_slice_error") {
            e.checks.push_back(PerSlice(planned, in));
        } else {
            e.checks.push_back(Result(planned, CheckOutcome::NotRun,
                                      "BuildExplanation can't run it; run it and report it with ReportExplanation."));
        }
    }
    e.caveats = DeriveCaveats(e);
    return e;
}

CheckResult SliceErrorCheck(const SliceErrorReport& report) {
    if (report.slices.empty()) throw std::invalid_argument("SliceErrorCheck: the report has no slices");
    const SliceError& worst = report.slices.front();
    CheckResult r;
    r.name = "per_slice_error";
    r.priority = CheckPriority::Required;
    r.outcome = report.any_worse() ? CheckOutcome::Failed : CheckOutcome::Passed;
    std::string worse;
    for (const SliceError& s : report.slices) {
        if (s.worse) worse += (worse.empty() ? "" : ", ") + s.name;
    }
    r.summary = "Error " + Fmt(report.error) + " over " + std::to_string(report.n) + " inputs; the worst of " +
                std::to_string(report.slices.size()) + " slices, " + worst.name + ", has " + Fmt(worst.error) +
                " against " + Fmt(worst.rest_error) + " for the rest (p = " + Fmt(worst.p_value) + "). " +
                (worse.empty() ? "No slice is worse than the rest at alpha " + Fmt(report.alpha) + "."
                               : "Worse than the rest at alpha " + Fmt(report.alpha) + ": " + worse + ".");
    int64_t count = 0;
    for (const SliceError& s : report.slices) count += s.worse ? 1 : 0;
    r.values = {{"error", report.error},
                {"slices", static_cast<double>(report.slices.size())},
                {"worse_slices", static_cast<double>(count)},
                {"worst_slice_error", worst.error},
                {"worst_slice_p_value", worst.p_value}};
    return r;
}

Explanation ReportExplanation(const ExplanationRequest& request, std::optional<Attribution> attribution,
                              std::vector<CheckResult> checks) {
    Method(request);
    const ExplanationPlan plan = ExplanationPlanner().plan(request.question, request.reason, request.traits);
    Explanation e = Start(request, plan);
    e.attribution = std::move(attribution);
    AddSettings(e, request);
    AddMetrics(e);
    for (const PlannedCheck& planned : plan.checks) {
        if (!AppliesTo(planned, request.method)) continue;
        const auto given = std::find_if(checks.begin(), checks.end(), [&](const CheckResult& c) { return c.name == planned.name; });
        if (given == checks.end()) {
            e.checks.push_back(Result(planned, CheckOutcome::NotRun,
                                      planned.available ? "Not reported." : Capitalized(planned.unavailable_reason) + "."));
        } else {
            given->priority = planned.priority;
            e.checks.push_back(std::move(*given));
            checks.erase(given);
        }
    }
    for (CheckResult& extra : checks) e.checks.push_back(std::move(extra));
    e.caveats = DeriveCaveats(e);
    return e;
}

std::vector<Caveat> DeriveCaveats(const Explanation& e) {
    std::vector<Caveat> out;
    const auto add = [&out](std::string id, CaveatSeverity severity, std::string source, std::string text) {
        out.push_back(Caveat{std::move(id), severity, std::move(source), std::move(text)});
    };
    if (e.plan_rank == 0) {
        add("plan.excluded", CaveatSeverity::Blocking, "plan",
            "The planner excluded " + e.method + " for this model" +
                (e.excluded_reason.empty() ? std::string(".") : ": it " + e.excluded_reason + "."));
    }
    for (const CheckResult& c : e.checks) {
        const bool required = c.priority == CheckPriority::Required;
        if (c.outcome == CheckOutcome::Failed) {
            add("check." + c.name, required ? CaveatSeverity::Blocking : CaveatSeverity::Warning, "check",
                Capitalized(c.name) + " check failed. " + c.summary +
                    (required ? " Don't use it to " + Use(e.reason) + "." : std::string()));
        } else if (c.outcome == CheckOutcome::NotRun) {
            add("check." + c.name, required ? CaveatSeverity::Blocking : CaveatSeverity::Note, "check",
                (required ? "Not checked for " + c.name + ", which " + std::string(to_string(e.reason)) + " requires. "
                          : "Not checked for " + c.name + ". ") +
                    c.summary);
        }
    }

    const bool attribution = e.attribution && e.attribution->values.numel() > 0;
    const CheckResult* faithfulness = e.check("faithfulness");
    if (attribution && e.traits.input == InputKind::Image) {
        const std::optional<std::string> fill = e.setting("check.imputation");
        if (Ran(faithfulness) && fill && fill->rfind("constant", 0) == 0) {
            add("input.constant_fill", CaveatSeverity::Warning, "input",
                "Faithfulness filled removed pixels with a constant, which tells the model which pixels were removed "
                "(Rong et al. 2022); Imputation::NoisyLinear (ROAD) doesn't.");
        }
        add("input.per_channel", CaveatSeverity::Note, "input",
            "Each channel of a pixel is ranked as its own feature; sum over channels before reading per-pixel ranks.");
    }
    if (attribution && Ran(faithfulness) && (e.traits.input == InputKind::Tokens || e.traits.input == InputKind::Symbolic)) {
        add("input.discrete", CaveatSeverity::Warning, "input",
            "Faithfulness replaced removed elements of a discrete input with a fill value, which for token ids is "
            "another token; the curves measure the model's reaction to that token too.");
    }

    if (e.method == "IntegratedGradients") {
        const std::optional<std::string> steps = e.setting("steps");
        if (steps && std::stoll(*steps) < 50) {
            add("method.integrated_gradients_steps", CaveatSeverity::Warning, "method",
                "Only " + *steps + " interpolation steps: completeness holds only up to the Riemann sum's error "
                "(50 to 300 is typical).");
        }
    }
    if (e.method == "LIME") {
        const std::optional<std::string> fidelity = e.setting("local_fidelity");
        if (fidelity && std::stod(*fidelity) < 0.8) {
            add("method.lime_local_fidelity", CaveatSeverity::Warning, "method",
                "The linear surrogate fits the model poorly around this input (local fidelity " +
                    Fmt(std::stod(*fidelity)) + ", weighted R²), so its coefficients aren't the model's local "
                    "behaviour; a smaller sigma or more samples may fit better.");
        }
    }
    const CheckResult* stability = e.check("stability");
    if (stability != nullptr && stability->outcome == CheckOutcome::Passed && stability->value("deterministic") == 0.0) {
        add("method.stochastic", CaveatSeverity::Note, "method",
            "Repeated runs differ; report the seed with the explanation.");
    }
    for (const MethodCaveat& m : MethodCaveats()) {
        if (e.method == m.method) add(m.id, CaveatSeverity::Note, "method", m.text);
    }
    if (attribution && e.reason == XaiReason::Justify) {
        add("reason.relied_on", CaveatSeverity::Note, "plan",
            "An attribution shows what the model relied on for this prediction, not whether it should have: passing "
            "the checks means the ranking matches the model, not the world.");
    }
    std::stable_sort(out.begin(), out.end(), [](const Caveat& a, const Caveat& b) { return a.severity < b.severity; });
    return out;
}

std::string_view to_string(CheckOutcome outcome) {
    switch (outcome) {
        case CheckOutcome::Passed: return "passed";
        case CheckOutcome::Failed: return "failed";
        case CheckOutcome::Measured: return "measured";
        case CheckOutcome::NotRun: return "not_run";
    }
    throw std::invalid_argument("to_string: unknown CheckOutcome");
}

std::string_view to_string(CaveatSeverity severity) {
    switch (severity) {
        case CaveatSeverity::Blocking: return "blocking";
        case CaveatSeverity::Warning: return "warning";
        case CaveatSeverity::Note: return "note";
    }
    throw std::invalid_argument("to_string: unknown CaveatSeverity");
}

JsonValue ToJson(const Explanation& e) {
    JsonValue::Object settings;
    for (const auto& [k, v] : e.settings) settings.emplace_back(k, JsonValue(v));
    JsonValue::Array checks, caveats;
    for (const CheckResult& c : e.checks) {
        checks.push_back(JsonValue(JsonValue::Object{
            {"name", c.name},
            {"priority", std::string(to_string(c.priority))},
            {"outcome", std::string(to_string(c.outcome))},
            {"summary", c.summary},
            {"values", Pairs(c.values)},
        }));
    }
    for (const Caveat& c : e.caveats) {
        caveats.push_back(JsonValue(JsonValue::Object{
            {"id", c.id},
            {"severity", std::string(to_string(c.severity))},
            {"source", c.source},
            {"text", c.text},
        }));
    }
    JsonValue doc(JsonValue::Object{
        {"schema", "pulsatrix.explanation.v1"},
        {"question", std::string(to_string(e.question))},
        {"reason", std::string(to_string(e.reason))},
        {"method", e.method},
        {"header", ExplainerCapabilitiesOf(e.method).header},
        {"plan_rank", JsonValue(static_cast<int64_t>(e.plan_rank))},
    });
    if (e.plan_rank == 0) doc.add("excluded_reason", e.excluded_reason);
    doc.add("traits", JsonValue(JsonValue::Object{
                          {"access", std::string(to_string(e.traits.access))},
                          {"input", std::string(to_string(e.traits.input))},
                          {"can_reinitialize", e.traits.can_reinitialize},
                          {"has_dataset", e.traits.has_dataset},
                          {"has_labels", e.traits.has_labels},
                          {"has_base_model", e.traits.has_base_model},
                          {"features", JsonValue(e.traits.features)},
                      }));
    doc.add("settings", JsonValue(std::move(settings)));
    doc.add("attribution", e.attribution ? AttributionJson(*e.attribution) : JsonValue(nullptr));
    doc.add("metrics", Pairs(e.metrics));
    doc.add("checks", JsonValue(std::move(checks)));
    if (e.aggregate) doc.add("aggregate", AttributionJson(*e.aggregate));
    doc.add("caveats", JsonValue(std::move(caveats)));
    doc.add("supports_reason", e.supports_reason());
    return doc;
}

std::string DescribeExplanation(const Explanation& e) {
    std::ostringstream out;
    out << "Explanation: " << e.method << ", for the " << to_string(e.question) << " question. "
        << QuestionText(e.question) << "\n"
        << "Reason: " << to_string(e.reason) << ". " << ReasonText(e.reason) << "\n";
    const auto blocking = std::count_if(e.caveats.begin(), e.caveats.end(),
                                        [](const Caveat& c) { return c.severity == CaveatSeverity::Blocking; });
    out << "Supports the reason: "
        << (blocking == 0 ? std::string("yes")
                          : "no, " + std::to_string(blocking) + " blocking caveat" + (blocking == 1 ? "" : "s"))
        << "\n";
    if (e.plan_rank > 0) {
        out << "Plan: ranked " << e.plan_rank << " for this question, reason and model\n";
    } else {
        out << "Plan: excluded" << (e.excluded_reason.empty() ? "" : ", " + e.excluded_reason) << "\n";
    }

    if (e.attribution && e.attribution->values.numel() > 0) {
        const std::vector<float> v = e.attribution->values.to_host_vector();
        std::vector<size_t> order(v.size());
        std::iota(order.begin(), order.end(), size_t{0});
        const size_t shown = std::min<size_t>(5, v.size());
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(shown), order.end(),
                          [&v](size_t a, size_t b) {
                              return std::fabs(v[a]) > std::fabs(v[b]) || (std::fabs(v[a]) == std::fabs(v[b]) && a < b);
                          });
        out << "\nLargest attributions (" << e.attribution->method << ", " << v.size() << " features):";
        for (size_t i = 0; i < shown; ++i) out << (i ? ", " : " ") << "[" << order[i] << "] " << Fmt(v[order[i]]);
        out << "\n";
    }
    if (!e.metrics.empty()) {
        out << "Metrics:";
        for (size_t i = 0; i < e.metrics.size(); ++i) out << (i ? ", " : " ") << e.metrics[i].first << " " << Fmt(e.metrics[i].second);
        out << "\n";
    }
    if (!e.settings.empty()) {
        out << "Settings:";
        for (size_t i = 0; i < e.settings.size(); ++i) out << (i ? ", " : " ") << e.settings[i].first << "=" << e.settings[i].second;
        out << "\n";
    }
    if (!e.checks.empty()) out << "\nChecks:\n";
    for (const CheckResult& c : e.checks) {
        out << "  [" << to_string(c.priority) << "] " << c.name << ": " << to_string(c.outcome) << ". " << c.summary << "\n";
    }
    if (!e.caveats.empty()) out << "\nCaveats:\n";
    for (const Caveat& c : e.caveats) out << "  [" << to_string(c.severity) << "] " << c.text << "\n";
    return out.str();
}

}  // namespace pulsatrix
