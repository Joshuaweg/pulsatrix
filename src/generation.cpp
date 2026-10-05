#include "pulsatrix/generation.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#include "pulsatrix/determinism.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace {

constexpr float kRemoved = -std::numeric_limits<float>::infinity();

void CheckConfig(const GenerationConfig& c) {
    if (c.max_new_tokens < 0 || c.min_new_tokens < 0) {
        throw std::invalid_argument("GenerationConfig: max_new_tokens and min_new_tokens must not be negative");
    }
    if (!(c.temperature > 0.0f) || std::isinf(c.temperature)) {
        throw std::invalid_argument("GenerationConfig: temperature must be positive and finite");
    }
    if (c.top_k < 0) {
        throw std::invalid_argument("GenerationConfig: top_k must not be negative");
    }
    if (!(c.top_p > 0.0f && c.top_p <= 1.0f)) {
        throw std::invalid_argument("GenerationConfig: top_p must be in (0, 1]");
    }
    if (!(c.min_p >= 0.0f && c.min_p <= 1.0f)) {
        throw std::invalid_argument("GenerationConfig: min_p must be in [0, 1]");
    }
    if (!(c.repetition_penalty > 0.0f) || std::isinf(c.repetition_penalty)) {
        throw std::invalid_argument("GenerationConfig: repetition_penalty must be positive and finite");
    }
}

// Softmax probabilities, in double; -infinity logits get probability 0.
std::vector<double> Softmax(const std::vector<float>& logits) {
    const float max = *std::max_element(logits.begin(), logits.end());
    std::vector<double> p(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = logits[i] == kRemoved ? 0.0 : std::exp(static_cast<double>(logits[i]) - max);
        sum += p[i];
    }
    for (double& v : p) v /= sum;
    return p;
}

}  // namespace

void ProcessLogits(std::vector<float>& logits, const std::vector<int64_t>& tokens, int64_t num_new_tokens,
                   const GenerationConfig& config) {
    CheckConfig(config);
    const auto vocab = static_cast<int64_t>(logits.size());
    if (config.repetition_penalty != 1.0f) {
        std::vector<bool> seen(logits.size(), false);
        for (int64_t t : tokens) {
            if (t >= 0 && t < vocab) seen[static_cast<size_t>(t)] = true;
        }
        for (size_t i = 0; i < logits.size(); ++i) {
            if (!seen[i]) continue;
            logits[i] = logits[i] < 0.0f ? logits[i] * config.repetition_penalty : logits[i] / config.repetition_penalty;
        }
    }
    if (num_new_tokens < config.min_new_tokens) {
        for (int64_t e : config.eos_token_ids) {
            if (e >= 0 && e < vocab) logits[static_cast<size_t>(e)] = kRemoved;
        }
    }
    if (!config.do_sample) {
        return;
    }
    if (config.temperature != 1.0f) {
        for (float& l : logits) l /= config.temperature;
    }
    if (config.top_k > 0 && config.top_k < vocab) {
        // Hugging Face keeps everything at least as large as the k-th largest, ties included.
        std::vector<float> sorted = logits;
        std::nth_element(sorted.begin(), sorted.begin() + (config.top_k - 1), sorted.end(), std::greater<float>());
        const float kth = sorted[static_cast<size_t>(config.top_k - 1)];
        for (float& l : logits) {
            if (l < kth) l = kRemoved;
        }
    }
    if (config.top_p < 1.0f) {
        // Ascending by probability; drop the low end whose cumulative probability stays within
        // 1 - top_p, but always keep the most likely token.
        const std::vector<double> p = Softmax(logits);
        std::vector<size_t> order(logits.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return logits[a] < logits[b]; });
        double cumulative = 0.0;
        for (size_t r = 0; r + 1 < order.size(); ++r) {
            cumulative += p[order[r]];
            if (cumulative <= 1.0 - static_cast<double>(config.top_p)) {
                logits[order[r]] = kRemoved;
            }
        }
    }
    if (config.min_p > 0.0f) {
        const std::vector<double> p = Softmax(logits);
        const double top = *std::max_element(p.begin(), p.end());
        const size_t best = static_cast<size_t>(std::max_element(p.begin(), p.end()) - p.begin());
        for (size_t i = 0; i < logits.size(); ++i) {
            if (i != best && p[i] < config.min_p * top) logits[i] = kRemoved;
        }
    }
}

GenerationResult Generate(const NextTokenLogitsFn& model, const std::vector<int64_t>& prompt,
                          const GenerationConfig& config) {
    CheckConfig(config);
    if (prompt.empty()) {
        throw std::invalid_argument("Generate: the prompt is empty");
    }
    std::mt19937_64 engine(config.seed ? *config.seed : next_seed());
    GenerationResult result;
    result.tokens = prompt;
    for (int64_t step = 0; step < config.max_new_tokens; ++step) {
        std::vector<float> logits = model(result.tokens);
        if (logits.empty()) {
            throw std::invalid_argument("Generate: the model returned no logits");
        }
        for (float l : logits) {
            if (!std::isfinite(l)) {
                throw std::invalid_argument("Generate: the model returned a non-finite logit");
            }
        }
        const std::vector<double> raw = Softmax(logits);
        ProcessLogits(logits, result.tokens, step, config);
        const auto best = static_cast<size_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (logits[best] == kRemoved) {
            throw std::invalid_argument("Generate: every token was removed");
        }
        size_t chosen = best;
        if (config.do_sample) {
            const std::vector<double> p = Softmax(logits);
            // Inverse CDF on a uniform draw from the engine's raw bits (platform-independent).
            const double u = static_cast<double>(engine() >> 11) * 0x1.0p-53;
            double cumulative = 0.0;
            chosen = best;
            for (size_t i = 0; i < p.size(); ++i) {
                if (p[i] == 0.0) continue;
                cumulative += p[i];
                chosen = i;
                if (u < cumulative) break;
            }
        }
        const auto token = static_cast<int64_t>(chosen);
        result.tokens.push_back(token);
        result.new_tokens.push_back(token);
        result.log_probs.push_back(static_cast<float>(std::log(raw[chosen])));
        if (std::find(config.eos_token_ids.begin(), config.eos_token_ids.end(), token) != config.eos_token_ids.end()) {
            result.finish_reason = FinishReason::Eos;
            return result;
        }
    }
    result.finish_reason = FinishReason::Length;
    return result;
}

NextTokenLogitsFn MakeNextTokenLogits(std::vector<Module*> layers, DeviceBackend* backend) {
    if (layers.empty()) {
        throw std::invalid_argument("MakeNextTokenLogits: no layers");
    }
    for (Module* m : layers) {
        if (m == nullptr) throw std::invalid_argument("MakeNextTokenLogits: a layer is null");
    }
    return [layers = std::move(layers), backend](const std::vector<int64_t>& tokens) {
        std::vector<float> ids(tokens.begin(), tokens.end());
        const auto L = static_cast<int64_t>(tokens.size());
        Tensor x(Shape({1, L}), backend, ids);
        for (Module* m : layers) {
            x = m->forward(x);
        }
        if (x.rank() != 3 || x.shape().dim(0) != 1 || x.shape().dim(1) != L) {
            throw std::invalid_argument("MakeNextTokenLogits: the model must map (1, L) ids to (1, L, vocab) logits");
        }
        const int64_t vocab = x.shape().dim(2);
        const std::vector<float> all = x.to_host_vector();
        return std::vector<float>(all.end() - vocab, all.end());
    };
}

NextTokenLogitsFn MakeCachedNextTokenLogits(EmbeddingModule& embedding, std::vector<TransformerBlock*> blocks,
                                            std::vector<Module*> head, DeviceBackend* backend, int64_t max_length) {
    if (blocks.empty()) {
        throw std::invalid_argument("MakeCachedNextTokenLogits: no blocks");
    }
    for (TransformerBlock* b : blocks) {
        if (b == nullptr) throw std::invalid_argument("MakeCachedNextTokenLogits: a block is null");
    }
    for (Module* m : head) {
        if (m == nullptr) throw std::invalid_argument("MakeCachedNextTokenLogits: a head layer is null");
    }
    struct State {
        std::vector<KVCache> caches;
        std::vector<int64_t> processed;
        std::vector<float> last_logits;
    };
    auto state = std::make_shared<State>();
    for (TransformerBlock* b : blocks) {
        state->caches.push_back(b->mha().MakeKVCache(1, max_length));
    }
    EmbeddingModule* emb = &embedding;
    return [state, emb, blocks = std::move(blocks), head = std::move(head), backend,
            max_length](const std::vector<int64_t>& tokens) {
        if (tokens.empty()) {
            throw std::invalid_argument("MakeCachedNextTokenLogits: the sequence is empty");
        }
        if (static_cast<int64_t>(tokens.size()) > max_length) {
            throw std::invalid_argument("MakeCachedNextTokenLogits: " + std::to_string(tokens.size()) +
                                        " tokens don't fit in caches of " + std::to_string(max_length));
        }
        size_t shared = 0;
        while (shared < tokens.size() && shared < state->processed.size() && tokens[shared] == state->processed[shared]) {
            ++shared;
        }
        if (shared == tokens.size() && shared == state->processed.size()) {
            return state->last_logits;  // the same sequence again
        }
        if (shared == tokens.size()) {
            --shared;  // a prefix of what was processed: rerun its last token for its logits
        }
        for (KVCache& c : state->caches) c.truncate(static_cast<int64_t>(shared));
        const auto fresh = static_cast<int64_t>(tokens.size() - shared);
        std::vector<float> ids(tokens.begin() + static_cast<std::ptrdiff_t>(shared), tokens.end());
        Tensor x = emb->forward(Tensor(Shape({1, fresh}), backend, ids));
        for (size_t i = 0; i < blocks.size(); ++i) {
            x = blocks[i]->forward_cached(x, state->caches[i]);
        }
        const int64_t d = x.shape().dim(2);
        const std::vector<float> all = x.to_host_vector();
        Tensor last(Shape({1, d}), backend, std::vector<float>(all.end() - d, all.end()));
        for (Module* m : head) {
            last = m->forward(last);
        }
        state->processed = tokens;
        state->last_logits = last.to_host_vector();
        return state->last_logits;
    };
}

}  // namespace pulsatrix
