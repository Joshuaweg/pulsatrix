#include "pulsatrix/masked_diffusion.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "pulsatrix/determinism.hpp"

namespace pulsatrix {
namespace {

[[noreturn]] void Bad(const std::string& what) { throw std::invalid_argument("SampleMaskedDiffusion: " + what); }

/** @brief A uniform draw in [0, 1) from the engine's raw output, the same on every platform. */
double Uniform(std::mt19937_64& engine) { return static_cast<double>(engine() >> 11) * 0x1.0p-53; }

/** @brief The token a masked position takes, and the model's probability of it. */
struct Choice {
    int64_t token = 0;
    float probability = 0.0f;
};

class Chooser {
public:
    Chooser(const MaskedDiffusionConfig& config, int64_t vocab) : config_(config), vocab_(vocab), allowed_(static_cast<size_t>(vocab), true) {
        allowed_[static_cast<size_t>(config.mask_token_id)] = false;
        for (int64_t t : config.suppressed_token_ids) {
            if (t < 0 || t >= vocab) Bad("a suppressed token is outside the vocabulary");
            allowed_[static_cast<size_t>(t)] = false;
        }
    }

    /** @brief @p sample false always takes the most likely allowed token. */
    Choice operator()(const std::vector<float>& logits, int64_t position, bool sample, std::mt19937_64& engine) const {
        const float* row = logits.data() + position * vocab_;
        double top = -std::numeric_limits<double>::infinity();
        for (int64_t v = 0; v < vocab_; ++v) top = std::max(top, static_cast<double>(row[v]));
        double total = 0;
        for (int64_t v = 0; v < vocab_; ++v) total += std::exp(row[v] - top);
        Choice c;
        c.token = -1;
        if (sample) {
            // Draw from softmax(logits / temperature) over the allowed tokens.
            double allowed_top = -std::numeric_limits<double>::infinity();
            for (int64_t v = 0; v < vocab_; ++v) {
                if (allowed_[static_cast<size_t>(v)]) allowed_top = std::max(allowed_top, static_cast<double>(row[v]) / config_.temperature);
            }
            double mass = 0;
            for (int64_t v = 0; v < vocab_; ++v) {
                if (allowed_[static_cast<size_t>(v)]) mass += std::exp(row[v] / config_.temperature - allowed_top);
            }
            double u = Uniform(engine) * mass;
            for (int64_t v = 0; v < vocab_; ++v) {
                if (!allowed_[static_cast<size_t>(v)]) continue;
                c.token = v;  // the last allowed token absorbs rounding
                u -= std::exp(row[v] / config_.temperature - allowed_top);
                if (u < 0) break;
            }
        } else {
            for (int64_t v = 0; v < vocab_; ++v) {
                if (allowed_[static_cast<size_t>(v)] && (c.token < 0 || row[v] > row[c.token])) c.token = v;
            }
        }
        if (c.token < 0) Bad("every token is suppressed");
        c.probability = static_cast<float>(std::exp(row[c.token] - top) / total);
        return c;
    }

    /** @brief @p token's log-probability at @p position over the allowed tokens only (DPLM sets
     *         the others' logits to -inf before its log-softmax). */
    [[nodiscard]] double allowed_log_probability(const std::vector<float>& logits, int64_t position, int64_t token) const {
        const float* row = logits.data() + position * vocab_;
        double top = -std::numeric_limits<double>::infinity();
        for (int64_t v = 0; v < vocab_; ++v) {
            if (allowed_[static_cast<size_t>(v)]) top = std::max(top, static_cast<double>(row[v]));
        }
        double total = 0;
        for (int64_t v = 0; v < vocab_; ++v) {
            if (allowed_[static_cast<size_t>(v)]) total += std::exp(row[v] - top);
        }
        return row[token] - top - std::log(total);
    }

private:
    const MaskedDiffusionConfig& config_;
    int64_t vocab_;
    std::vector<bool> allowed_;
};

}  // namespace

MaskedDiffusionResult SampleMaskedDiffusion(const DenoiserFn& model, const std::vector<int64_t>& tokens, const MaskedDiffusionConfig& config) {
    if (config.steps <= 0) Bad("steps must be positive");
    if (config.mask_token_id < 0) Bad("mask_token_id must be set");
    if (!(config.temperature > 0.0f)) Bad("temperature must be positive");
    const int64_t L = static_cast<int64_t>(tokens.size());
    const int64_t M = std::count(tokens.begin(), tokens.end(), config.mask_token_id);
    if (M == 0) Bad("the input has no masked position");
    std::mt19937_64 engine(config.seed ? *config.seed : next_seed());

    MaskedDiffusionResult result;
    result.tokens = tokens;
    result.last_write_step.assign(static_cast<size_t>(L), -1);
    std::vector<float> logits;
    std::vector<int64_t> logits_input;
    int64_t vocab = 0;
    std::optional<Chooser> choose;
    // The model's output for the current sequence; runs it only when the sequence has changed.
    const auto logits_now = [&]() -> const std::vector<float>& {
        if (logits_input == result.tokens && !logits.empty()) return logits;
        logits = model(result.tokens);
        if (logits.empty() || logits.size() % static_cast<size_t>(L) != 0) Bad("the model must return L * vocab logits");
        for (float v : logits) {
            if (!std::isfinite(v)) Bad("the model returned a non-finite logit");
        }
        logits_input = result.tokens;
        if (vocab == 0) {
            vocab = static_cast<int64_t>(logits.size()) / L;
            if (config.mask_token_id >= vocab) Bad("mask_token_id is outside the vocabulary");
            for (int64_t t : tokens) {
                if (t < 0 || t >= vocab) Bad("an input token is outside the vocabulary");
            }
            choose.emplace(config, vocab);
        } else if (static_cast<int64_t>(logits.size()) != L * vocab) {
            Bad("the model changed its vocabulary size between steps");
        }
        return logits;
    };

    if (config.schedule == UnmaskingSchedule::Remasking) {
        std::vector<int64_t> free;
        for (int64_t p = 0; p < L; ++p) {
            if (tokens[static_cast<size_t>(p)] == config.mask_token_id) free.push_back(p);
        }
        for (int64_t step = 1; step <= config.steps; ++step) {
            const std::vector<float>& out = logits_now();
            std::vector<MaskedDiffusionWrite> chosen;
            std::vector<float> scores;
            for (int64_t p : free) {
                const Choice c = (*choose)(out, p, config.do_sample, engine);
                chosen.push_back({p, c.token, c.probability});
                // DPLM's scores are float32.
                scores.push_back(static_cast<float>(choose->allowed_log_probability(out, p, c.token)));
            }
            // DPLM: cutoff_len = (n * rate).long() in float32, cutoff = sorted(scores)[cutoff_len],
            // and positions scoring strictly below the cutoff are masked.
            const float rate = static_cast<float>(1.0 - static_cast<double>(step) / static_cast<double>(config.steps));
            const auto k = static_cast<size_t>(static_cast<float>(M) * rate);
            std::vector<float> sorted = scores;
            std::sort(sorted.begin(), sorted.end());
            const float cutoff = sorted[std::min(k, sorted.size() - 1)];
            // _reparam_decoding starts from the previous tokens: a filled position that isn't
            // masked keeps its token, whatever the model would choose there now.
            MaskedDiffusionStep record{result.tokens, {}, {}};
            for (size_t i = 0; i < chosen.size(); ++i) {
                const auto p = static_cast<size_t>(chosen[i].position);
                const bool filled = result.tokens[p] != config.mask_token_id;
                if (scores[i] < cutoff) {
                    if (filled) record.remasked.push_back(chosen[i].position);
                    result.tokens[p] = config.mask_token_id;
                } else if (!filled) {
                    record.writes.push_back(chosen[i]);
                    result.tokens[p] = chosen[i].token;
                    result.last_write_step[p] = static_cast<int64_t>(result.steps.size());
                }
            }
            result.steps.push_back(std::move(record));
        }
        return result;
    }

    const int64_t base = M / config.steps, remainder = M % config.steps;
    const double t_end = 1e-5;
    for (int64_t step = 0; step < config.steps; ++step) {
        std::vector<int64_t> masked;
        for (int64_t p = 0; p < L; ++p) {
            if (result.tokens[static_cast<size_t>(p)] == config.mask_token_id) masked.push_back(p);
        }
        if (masked.empty()) break;
        std::vector<MaskedDiffusionWrite> writes;
        if (config.schedule == UnmaskingSchedule::Confidence) {
            const int64_t count = std::min<int64_t>(base + (step < remainder ? 1 : 0), static_cast<int64_t>(masked.size()));
            if (count == 0) continue;
            const std::vector<float>& out = logits_now();
            for (int64_t p : masked) {
                const Choice c = (*choose)(out, p, config.do_sample, engine);
                writes.push_back({p, c.token, c.probability});
            }
            std::stable_sort(writes.begin(), writes.end(), [](const auto& a, const auto& b) { return a.probability > b.probability; });
            writes.resize(static_cast<size_t>(count));
        } else {
            const bool last = step + 1 == config.steps;
            std::vector<int64_t> chosen;
            if (last) {
                chosen = masked;
            } else {
                const double t = 1.0 - static_cast<double>(step) * (1.0 - t_end) / static_cast<double>(config.steps);
                const double s = 1.0 - static_cast<double>(step + 1) * (1.0 - t_end) / static_cast<double>(config.steps);
                for (int64_t p : masked) {
                    if (Uniform(engine) < (t - s) / t) chosen.push_back(p);
                }
            }
            if (chosen.empty()) continue;
            const std::vector<float>& out = logits_now();
            for (int64_t p : chosen) {
                // MDLM's final noise removal takes the most likely token even when sampling.
                const Choice c = (*choose)(out, p, config.do_sample && !last, engine);
                writes.push_back({p, c.token, c.probability});
            }
        }
        std::sort(writes.begin(), writes.end(), [](const auto& a, const auto& b) { return a.position < b.position; });
        MaskedDiffusionStep record{result.tokens, writes, {}};
        for (const MaskedDiffusionWrite& w : writes) {
            result.tokens[static_cast<size_t>(w.position)] = w.token;
            result.last_write_step[static_cast<size_t>(w.position)] = static_cast<int64_t>(result.steps.size());
        }
        result.steps.push_back(std::move(record));
    }
    return result;
}

DenoiserFn MakeDenoiser(CausalLM& model, DeviceBackend* backend) {
    if (model.config().causal) throw std::invalid_argument("MakeDenoiser: the CausalLM is causal; a denoiser needs bidirectional attention");
    return [&model, backend](const std::vector<int64_t>& tokens) {
        const int64_t L = static_cast<int64_t>(tokens.size());
        return model.forward(Tensor(Shape({1, L}), backend, std::vector<float>(tokens.begin(), tokens.end()), backend->device())).to_host_vector();
    };
}

DenoiserFn MakeDenoiser(EncoderLM& model, DeviceBackend* backend) {
    return [&model, backend](const std::vector<int64_t>& tokens) {
        const int64_t L = static_cast<int64_t>(tokens.size());
        return model.forward(Tensor(Shape({1, L}), backend, std::vector<float>(tokens.begin(), tokens.end()), backend->device())).to_host_vector();
    };
}

}  // namespace pulsatrix
