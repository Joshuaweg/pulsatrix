#include "pulsatrix/masked_diffusion_lrp.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include "pulsatrix/explanation_metrics.hpp"  // SpearmanRankCorrelation

namespace pulsatrix {
namespace {

[[noreturn]] void Bad(const char* where, const std::string& what) { throw std::invalid_argument(std::string(where) + ": " + what); }

int64_t Length(const MaskedDiffusionResult& trajectory) { return static_cast<int64_t>(trajectory.tokens.size()); }

/** @brief The index of the last step that wrote each position, and the token it wrote; -1 for
 *         none. */
struct LastWrites {
    std::vector<int64_t> step;
    std::vector<int64_t> token;
};

LastWrites FindLastWrites(const MaskedDiffusionResult& trajectory) {
    const size_t L = trajectory.tokens.size();
    LastWrites last{std::vector<int64_t>(L, -1), std::vector<int64_t>(L, -1)};
    for (size_t s = 0; s < trajectory.steps.size(); ++s) {
        for (const MaskedDiffusionWrite& w : trajectory.steps[s].writes) {
            if (w.position < 0 || w.position >= static_cast<int64_t>(L)) Bad("ExplainCommitments", "a write is outside the sequence");
            last.step[static_cast<size_t>(w.position)] = static_cast<int64_t>(s);
            last.token[static_cast<size_t>(w.position)] = w.token;
        }
    }
    return last;
}

/** @brief Each target's write step and explained token, checked. */
struct ResolvedTarget {
    int64_t position = 0;
    int64_t step = 0;
    int64_t token = 0;
};

std::vector<ResolvedTarget> Resolve(const char* where, const MaskedDiffusionResult& trajectory, const std::vector<CommitmentTarget>& targets) {
    if (targets.empty()) Bad(where, "no targets");
    const int64_t L = Length(trajectory);
    const LastWrites last = FindLastWrites(trajectory);
    std::vector<bool> seen(static_cast<size_t>(L), false);
    std::vector<ResolvedTarget> out;
    for (const CommitmentTarget& t : targets) {
        if (t.position < 0 || t.position >= L) Bad(where, "a target is outside the sequence");
        const size_t p = static_cast<size_t>(t.position);
        if (seen[p]) Bad(where, "a target position is given twice");
        seen[p] = true;
        if (last.step[p] < 0) Bad(where, "a target position wasn't written by the trajectory");
        out.push_back({t.position, last.step[p], t.token ? *t.token : last.token[p]});
    }
    return out;
}

void CheckLogits(const char* where, const std::vector<float>& logits, int64_t L, int64_t* vocab) {
    if (logits.empty() || logits.size() % static_cast<size_t>(L) != 0) Bad(where, "the model must return L * vocab logits");
    const int64_t V = static_cast<int64_t>(logits.size()) / L;
    if (*vocab != 0 && V != *vocab) Bad(where, "the model changed its vocabulary size between steps");
    *vocab = V;
}

/**
 * @brief The replay loop. Calls @p on_step with each step's index and the model's logits for the
 *        step's input whenever @p needs_logits says the step wants them (or the step writes).
 */
MaskedDiffusionResult Replay(const char* where, const DenoiserFn& model, const MaskedDiffusionResult& trajectory,
                             const std::vector<int64_t>& input, const MaskedDiffusionConfig& config, const std::vector<int64_t>& skipped,
                             const std::function<bool(int64_t)>& needs_logits,
                             const std::function<void(int64_t, const std::vector<float>&, int64_t)>& on_step) {
    const int64_t L = Length(trajectory);
    if (static_cast<int64_t>(input.size()) != L) Bad(where, "the input's length differs from the trajectory's");
    if (config.mask_token_id < 0) Bad(where, "mask_token_id must be set");
    std::vector<bool> skip(static_cast<size_t>(L), false);
    for (int64_t p : skipped) {
        if (p < 0 || p >= L) Bad(where, "a skipped position is outside the sequence");
        skip[static_cast<size_t>(p)] = true;
    }
    MaskedDiffusionResult result;
    result.tokens = input;
    result.last_write_step.assign(static_cast<size_t>(L), -1);
    std::vector<float> logits;
    std::vector<int64_t> logits_input;
    int64_t V = 0;
    std::vector<bool> allowed;
    for (size_t s = 0; s < trajectory.steps.size(); ++s) {
        std::vector<int64_t> positions;
        for (const MaskedDiffusionWrite& w : trajectory.steps[s].writes) {
            if (!skip[static_cast<size_t>(w.position)]) positions.push_back(w.position);
        }
        MaskedDiffusionStep record{result.tokens, {}, {}};
        if (!positions.empty() || needs_logits(static_cast<int64_t>(s))) {
            if (logits.empty() || logits_input != result.tokens) {
                logits = model(result.tokens);
                CheckLogits(where, logits, L, &V);
                logits_input = result.tokens;
                if (allowed.empty()) {
                    if (config.mask_token_id >= V) Bad(where, "mask_token_id is outside the vocabulary");
                    allowed.assign(static_cast<size_t>(V), true);
                    allowed[static_cast<size_t>(config.mask_token_id)] = false;
                    for (int64_t t : config.suppressed_token_ids) {
                        if (t < 0 || t >= V) Bad(where, "a suppressed token is outside the vocabulary");
                        allowed[static_cast<size_t>(t)] = false;
                    }
                }
            }
            on_step(static_cast<int64_t>(s), logits, V);
            for (int64_t p : positions) {
                const float* row = logits.data() + p * V;
                int64_t best = -1;
                for (int64_t v = 0; v < V; ++v) {
                    if (allowed[static_cast<size_t>(v)] && (best < 0 || row[v] > row[best])) best = v;
                }
                if (best < 0) Bad(where, "every token is suppressed");
                const double top = *std::max_element(row, row + V);
                double total = 0;
                for (int64_t v = 0; v < V; ++v) total += std::exp(row[v] - top);
                record.writes.push_back({p, best, static_cast<float>(std::exp(row[best] - top) / total)});
            }
        }
        for (const MaskedDiffusionWrite& w : record.writes) {
            result.tokens[static_cast<size_t>(w.position)] = w.token;
            result.last_write_step[static_cast<size_t>(w.position)] = static_cast<int64_t>(s);
        }
        for (int64_t p : trajectory.steps[s].remasked) {
            if (p < 0 || p >= L) Bad(where, "a re-masked position is outside the sequence");
            if (result.tokens[static_cast<size_t>(p)] == config.mask_token_id) continue;  // skipped, so never written
            result.tokens[static_cast<size_t>(p)] = config.mask_token_id;
            record.remasked.push_back(p);
        }
        result.steps.push_back(std::move(record));
    }
    return result;
}

double TrapezoidArea(const std::vector<float>& x, const std::vector<float>& y) {
    double area = 0;
    for (size_t i = 1; i < x.size(); ++i) area += 0.5 * (x[i] - x[i - 1]) * (static_cast<double>(y[i]) + y[i - 1]);
    return area;
}

}  // namespace

EncoderLMStepModel::EncoderLMStepModel(EncoderLM& model, DeviceBackend* backend, LRPRuleConfig config)
    : model_(model), backend_(backend), config_(config) {}

std::vector<float> EncoderLMStepModel::forward(const std::vector<int64_t>& tokens) {
    length_ = static_cast<int64_t>(tokens.size());
    model_.set_keep_activations(true);
    return model_.forward(Tensor(Shape({1, length_}), backend_, std::vector<float>(tokens.begin(), tokens.end()), backend_->device()))
        .to_host_vector();
}

std::vector<float> EncoderLMStepModel::relevance(const std::vector<float>& relevance_logits) {
    const int64_t V = model_.config().vocab_size;
    if (static_cast<int64_t>(relevance_logits.size()) != length_ * V) {
        throw std::invalid_argument("EncoderLMStepModel::relevance: relevance must be L * vocab values for the last forward()");
    }
    return model_
        .propagate_relevance(Tensor(Shape({1, length_, V}), backend_, relevance_logits, backend_->device()), config_)
        .to_host_vector();
}

CausalLMStepModel::CausalLMStepModel(CausalLM& model, DeviceBackend* backend, LRPRuleConfig config)
    : model_(model), backend_(backend), config_(config) {
    if (model.config().causal) throw std::invalid_argument("CausalLMStepModel: the CausalLM is causal; a diffusion step needs bidirectional attention");
}

std::vector<float> CausalLMStepModel::forward(const std::vector<int64_t>& tokens) {
    length_ = static_cast<int64_t>(tokens.size());
    return model_.forward(Tensor(Shape({1, length_}), backend_, std::vector<float>(tokens.begin(), tokens.end()), backend_->device()))
        .to_host_vector();
}

std::vector<float> CausalLMStepModel::relevance(const std::vector<float>& relevance_logits) {
    const int64_t V = model_.config().vocab_size;
    if (static_cast<int64_t>(relevance_logits.size()) != length_ * V) {
        throw std::invalid_argument("CausalLMStepModel::relevance: relevance must be L * vocab values for the last forward()");
    }
    return model_
        .propagate_relevance(Tensor(Shape({1, length_, V}), backend_, relevance_logits, backend_->device()), config_)
        .to_host_vector();
}

double CommitmentRelevance::prompt_total() const { return std::accumulate(prompt.begin(), prompt.end(), 0.0); }
double CommitmentRelevance::relayed_total() const { return std::accumulate(relayed.begin(), relayed.end(), 0.0); }
double CommitmentRelevance::mask_total() const { return std::accumulate(mask.begin(), mask.end(), 0.0); }
double CommitmentRelevance::leak_total() const {
    double total = 0;
    for (const CommitmentHop& h : hops) total += h.leak();
    return total;
}

CommitmentRelevance ExplainCommitments(StepRelevanceModel& model, const MaskedDiffusionResult& trajectory, int64_t mask_token_id,
                                       const std::vector<CommitmentTarget>& targets, CommitmentMode mode) {
    const char* where = "ExplainCommitments";
    if (mask_token_id < 0) Bad(where, "mask_token_id must be set");
    const int64_t L = Length(trajectory);
    const std::vector<ResolvedTarget> resolved = Resolve(where, trajectory, targets);
    const int64_t S = static_cast<int64_t>(trajectory.steps.size());

    // Which earlier step wrote each position of each step's input (-1: never written yet).
    std::vector<std::vector<int64_t>> writer(static_cast<size_t>(S));
    std::vector<int64_t> current(static_cast<size_t>(L), -1);
    for (int64_t s = 0; s < S; ++s) {
        const MaskedDiffusionStep& step = trajectory.steps[static_cast<size_t>(s)];
        if (static_cast<int64_t>(step.input.size()) != L) Bad(where, "a step's input has the wrong length");
        writer[static_cast<size_t>(s)] = current;
        for (const MaskedDiffusionWrite& w : step.writes) current[static_cast<size_t>(w.position)] = s;
    }

    CommitmentRelevance out;
    out.mode = mode;
    out.prompt.assign(static_cast<size_t>(L), 0.0f);
    out.relayed.assign(static_cast<size_t>(L), 0.0f);
    out.mask.assign(static_cast<size_t>(L), 0.0f);
    out.step_relevance.resize(static_cast<size_t>(S));
    std::vector<double> pending(static_cast<size_t>(L), 0.0);  // relevance waiting at each position's write step
    std::vector<double> prompt(static_cast<size_t>(L), 0.0), mask(static_cast<size_t>(L), 0.0), relayed(static_cast<size_t>(L), 0.0);
    int64_t V = 0;
    for (int64_t s = S - 1; s >= 0; --s) {
        const MaskedDiffusionStep& step = trajectory.steps[static_cast<size_t>(s)];
        bool any = false;
        for (const ResolvedTarget& t : resolved) any = any || t.step == s;
        for (const MaskedDiffusionWrite& w : step.writes) any = any || pending[static_cast<size_t>(w.position)] != 0.0;
        if (!any) continue;

        const std::vector<float> logits = model.forward(step.input);
        CheckLogits(where, logits, L, &V);
        std::vector<float> seed(logits.size(), 0.0f);
        CommitmentHop hop;
        hop.step = s;
        for (const ResolvedTarget& t : resolved) {
            if (t.step != s) continue;
            if (t.token < 0 || t.token >= V) Bad(where, "a target token is outside the vocabulary");
            const size_t i = static_cast<size_t>(t.position * V + t.token);
            out.value += logits[i];
            hop.seeded += logits[i];
            seed[i] += logits[i];
        }
        for (const MaskedDiffusionWrite& w : step.writes) {
            const size_t p = static_cast<size_t>(w.position);
            if (pending[p] == 0.0) continue;
            seed[static_cast<size_t>(w.position * V + w.token)] += static_cast<float>(pending[p]);
            hop.seeded += pending[p];
            pending[p] = 0.0;
        }

        std::vector<float> r = model.relevance(seed);
        if (static_cast<int64_t>(r.size()) != L) Bad(where, "the model's relevance must have one value per position");
        const std::vector<int64_t>& from = writer[static_cast<size_t>(s)];
        for (int64_t p = 0; p < L; ++p) {
            const size_t i = static_cast<size_t>(p);
            const double v = r[i];
            if (step.input[i] == mask_token_id) {
                mask[i] += v;
                hop.to_mask += v;
            } else if (from[i] < 0) {
                prompt[i] += v;
                hop.to_prompt += v;
            } else {
                relayed[i] += v;
                hop.to_committed += v;
                if (mode == CommitmentMode::Graph) pending[i] += v;
            }
        }
        out.hops.push_back(hop);
        out.step_relevance[static_cast<size_t>(s)] = std::move(r);
    }
    for (size_t i = 0; i < static_cast<size_t>(L); ++i) {
        out.prompt[i] = static_cast<float>(prompt[i]);
        out.mask[i] = static_cast<float>(mask[i]);
        out.relayed[i] = static_cast<float>(relayed[i]);
    }
    return out;
}

MaskedDiffusionResult ReplayMaskedDiffusion(const DenoiserFn& model, const MaskedDiffusionResult& trajectory, const std::vector<int64_t>& input,
                                            const MaskedDiffusionConfig& config, const std::vector<int64_t>& skipped) {
    return Replay(
        "ReplayMaskedDiffusion", model, trajectory, input, config, skipped, [](int64_t) { return false; },
        [](int64_t, const std::vector<float>&, int64_t) {});
}

double ReplayCommitmentValue(const DenoiserFn& model, const MaskedDiffusionResult& trajectory, const std::vector<int64_t>& input,
                             const MaskedDiffusionConfig& config, const std::vector<CommitmentTarget>& targets,
                             const std::vector<int64_t>& skipped) {
    const char* where = "ReplayCommitmentValue";
    const std::vector<ResolvedTarget> resolved = Resolve(where, trajectory, targets);
    double value = 0;
    (void)Replay(
        where, model, trajectory, input, config, skipped,
        [&](int64_t s) { return std::any_of(resolved.begin(), resolved.end(), [s](const ResolvedTarget& t) { return t.step == s; }); },
        [&](int64_t s, const std::vector<float>& logits, int64_t V) {
            for (const ResolvedTarget& t : resolved) {
                if (t.step != s) continue;
                if (t.token < 0 || t.token >= V) Bad(where, "a target token is outside the vocabulary");
                value += logits[static_cast<size_t>(t.position * V + t.token)];
            }
        });
    return value;
}

CommitmentDeletionCurve CommitmentDeletionCheck(const DenoiserFn& model, const MaskedDiffusionResult& trajectory,
                                                const MaskedDiffusionConfig& config, const std::vector<CommitmentTarget>& targets,
                                                const std::vector<float>& prompt_relevance, int64_t steps, int64_t random_orders,
                                                uint64_t seed) {
    const char* where = "CommitmentDeletionCheck";
    const int64_t L = Length(trajectory);
    if (static_cast<int64_t>(prompt_relevance.size()) != L) Bad(where, "prompt_relevance must have one value per position");
    if (steps < 1 || random_orders < 1) Bad(where, "steps and random_orders must be at least 1");
    if (trajectory.steps.empty()) Bad(where, "the trajectory has no steps");
    const std::vector<int64_t>& input = trajectory.steps.front().input;
    std::vector<int64_t> prompt;
    for (int64_t p = 0; p < L; ++p) {
        if (trajectory.last_write_step[static_cast<size_t>(p)] < 0 && input[static_cast<size_t>(p)] != config.mask_token_id) prompt.push_back(p);
    }
    if (prompt.empty()) Bad(where, "the input has no prompt");

    const auto curve = [&](const std::vector<int64_t>& order) {
        std::vector<float> scores;
        for (int64_t k = 0; k <= steps; ++k) {
            const size_t count = static_cast<size_t>(std::llround(static_cast<double>(k) * static_cast<double>(order.size()) / steps));
            std::vector<int64_t> deleted = input;
            for (size_t i = 0; i < count; ++i) deleted[static_cast<size_t>(order[i])] = config.mask_token_id;
            scores.push_back(static_cast<float>(ReplayCommitmentValue(model, trajectory, deleted, config, targets)));
        }
        return scores;
    };

    CommitmentDeletionCurve out;
    for (int64_t k = 0; k <= steps; ++k) out.fractions.push_back(static_cast<float>(k) / static_cast<float>(steps));
    std::vector<int64_t> order = prompt;
    // Sort by relevance of the value's sign; the sign comes from the first replay, which is the
    // undeleted value.
    const double undeleted = ReplayCommitmentValue(model, trajectory, input, config, targets);
    out.sign = undeleted < 0 ? -1.0f : 1.0f;
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return out.sign * prompt_relevance[static_cast<size_t>(a)] > out.sign * prompt_relevance[static_cast<size_t>(b)];
    });
    out.scores = curve(order);
    out.random_scores.assign(out.fractions.size(), 0.0f);
    std::mt19937_64 engine(seed);
    for (int64_t r = 0; r < random_orders; ++r) {
        std::vector<int64_t> shuffled = prompt;
        for (size_t i = shuffled.size(); i > 1; --i) {
            std::swap(shuffled[i - 1], shuffled[static_cast<size_t>(engine() % i)]);
        }
        const std::vector<float> s = curve(shuffled);
        for (size_t i = 0; i < s.size(); ++i) out.random_scores[i] += s[i] / static_cast<float>(random_orders);
    }
    out.auc = out.sign * TrapezoidArea(out.fractions, out.scores);
    out.random_auc = out.sign * TrapezoidArea(out.fractions, out.random_scores);
    return out;
}

CommitmentPatchResult CommitmentPatchCheck(const DenoiserFn& model, const MaskedDiffusionResult& trajectory, const MaskedDiffusionConfig& config,
                                           const std::vector<CommitmentTarget>& targets, const CommitmentRelevance& graph) {
    const char* where = "CommitmentPatchCheck";
    const int64_t L = Length(trajectory);
    if (graph.mode != CommitmentMode::Graph) Bad(where, "the relevance must be a CommitmentMode::Graph result");
    if (static_cast<int64_t>(graph.relayed.size()) != L) Bad(where, "the relevance is for another trajectory");
    if (trajectory.steps.empty()) Bad(where, "the trajectory has no steps");
    const std::vector<ResolvedTarget> resolved = Resolve(where, trajectory, targets);
    int64_t last_target_step = 0;
    std::vector<bool> is_target(static_cast<size_t>(L), false);
    for (const ResolvedTarget& t : resolved) {
        last_target_step = std::max(last_target_step, t.step);
        is_target[static_cast<size_t>(t.position)] = true;
    }
    const std::vector<int64_t>& input = trajectory.steps.front().input;
    const double base = ReplayCommitmentValue(model, trajectory, input, config, targets);
    CommitmentPatchResult out;
    for (int64_t p = 0; p < L; ++p) {
        const int64_t step = trajectory.last_write_step[static_cast<size_t>(p)];
        if (step < 0 || step >= last_target_step || is_target[static_cast<size_t>(p)]) continue;
        out.positions.push_back(p);
        out.relayed.push_back(graph.relayed[static_cast<size_t>(p)]);
        out.effect.push_back(static_cast<float>(base - ReplayCommitmentValue(model, trajectory, input, config, targets, {p})));
    }
    if (out.positions.size() >= 3) out.spearman = SpearmanRankCorrelation(out.relayed, out.effect);
    return out;
}

}  // namespace pulsatrix
