#include "pulsatrix/lm_scoring.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {

namespace {

void CheckExample(const PromptCompletion& e, const char* who) {
    if (e.prompt.empty()) throw std::invalid_argument(std::string(who) + ": an example has an empty prompt");
    if (e.completion.empty()) throw std::invalid_argument(std::string(who) + ": an example has an empty completion");
}

// Example indices sorted by length (stable), so each batch pads little.
std::vector<size_t> ByLength(const std::vector<size_t>& lengths) {
    std::vector<size_t> order(lengths.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lengths[a] < lengths[b]; });
    return order;
}

// Gathers rows of (rows_total, vocab) logits on the device and copies them to the host.
std::vector<float> GatherRows(const Tensor& logits, const std::vector<float>& rows, int64_t vocab, DeviceBackend* backend) {
    const auto count = static_cast<int64_t>(rows.size());
    const Tensor index(Shape({count, 1}), backend, rows, backend->device());
    Tensor out(Shape({count, vocab}), backend);
    backend->gather_rows(logits.data(), index.data(), out.data(), static_cast<size_t>(count), static_cast<size_t>(vocab));
    return out.to_host_vector();
}

// log Σ exp over one row, in double.
double LogSumExp(const float* row, int64_t n) {
    const float top = *std::max_element(row, row + n);
    double sum = 0.0;
    for (int64_t k = 0; k < n; ++k) sum += std::exp(static_cast<double>(row[k]) - top);
    return top + std::log(sum);
}

}  // namespace

std::vector<JsonValue> ReadJsonl(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("ReadJsonl: can't read " + path);
    std::vector<JsonValue> rows;
    std::string line;
    int64_t number = 0;
    while (std::getline(in, line)) {
        ++number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        try {
            rows.push_back(ParseJson(line));
        } catch (const std::exception& e) {
            throw std::runtime_error("ReadJsonl: " + path + " line " + std::to_string(number) + ": " + e.what());
        }
    }
    return rows;
}

std::vector<PromptCompletion> TokenizePromptCompletions(const std::vector<JsonValue>& rows, const TextTokenizer& tokenizer,
                                                        const std::string& prompt_key, const std::string& completion_key,
                                                        bool add_special_tokens) {
    std::vector<PromptCompletion> out;
    out.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        const JsonValue* prompt = rows[i].type() == JsonValue::Type::Object ? rows[i].find(prompt_key) : nullptr;
        const JsonValue* completion = rows[i].type() == JsonValue::Type::Object ? rows[i].find(completion_key) : nullptr;
        if (prompt == nullptr || completion == nullptr || prompt->type() != JsonValue::Type::String ||
            completion->type() != JsonValue::Type::String) {
            throw std::invalid_argument("TokenizePromptCompletions: row " + std::to_string(i) + " needs string fields \"" +
                                        prompt_key + "\" and \"" + completion_key + "\"");
        }
        PromptCompletion e;
        e.prompt = tokenizer.encode(prompt->as_string(), add_special_tokens).ids;
        e.completion = tokenizer.encode(completion->as_string(), false).ids;
        if (e.prompt.empty() || e.completion.empty()) {
            throw std::invalid_argument("TokenizePromptCompletions: row " + std::to_string(i) + " tokenizes to an empty prompt or completion");
        }
        out.push_back(std::move(e));
    }
    return out;
}

CausalLmBatch CollatePromptCompletions(const std::vector<PromptCompletion>& examples, DeviceBackend* backend, int64_t pad_id) {
    if (examples.empty()) throw std::invalid_argument("CollatePromptCompletions: no examples");
    std::vector<int64_t> lengths;
    int64_t length = 0;
    for (const PromptCompletion& e : examples) {
        CheckExample(e, "CollatePromptCompletions");
        lengths.push_back(static_cast<int64_t>(e.prompt.size() + e.completion.size()) - 1);
        length = std::max(length, lengths.back());
    }
    int64_t target_tokens = 0;
    const auto n = static_cast<int64_t>(examples.size());
    std::vector<float> ids(static_cast<size_t>(n * length), static_cast<float>(pad_id));
    std::vector<float> targets(static_cast<size_t>(n * length), static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex));
    for (int64_t r = 0; r < n; ++r) {
        const PromptCompletion& e = examples[static_cast<size_t>(r)];
        const auto p = static_cast<int64_t>(e.prompt.size());
        const auto c = static_cast<int64_t>(e.completion.size());
        float* row_ids = &ids[static_cast<size_t>(r * length)];
        float* row_targets = &targets[static_cast<size_t>(r * length)];
        for (int64_t i = 0; i < p; ++i) row_ids[i] = static_cast<float>(e.prompt[static_cast<size_t>(i)]);
        for (int64_t j = 0; j + 1 < c; ++j) row_ids[p + j] = static_cast<float>(e.completion[static_cast<size_t>(j)]);
        // Position p - 1 + j predicts completion token j.
        for (int64_t j = 0; j < c; ++j) row_targets[p - 1 + j] = static_cast<float>(e.completion[static_cast<size_t>(j)]);
        target_tokens += c;
    }
    return {Tensor(Shape({n, length}), backend, ids, backend->device()),
            Tensor(Shape({n, length}), backend, targets, backend->device()), std::move(lengths), target_tokens};
}

std::vector<double> CompletionLogProbs(CausalLM& model, const std::vector<PromptCompletion>& examples, DeviceBackend* backend,
                                       int64_t batch_size) {
    if (batch_size < 1) throw std::invalid_argument("CompletionLogProbs: batch_size must be >= 1");
    std::vector<size_t> lengths;
    for (const PromptCompletion& e : examples) {
        CheckExample(e, "CompletionLogProbs");
        lengths.push_back(e.prompt.size() + e.completion.size());
    }
    const int64_t vocab = model.config().vocab_size;
    const std::vector<size_t> order = ByLength(lengths);
    std::vector<double> out(examples.size(), 0.0);
    for (size_t start = 0; start < order.size(); start += static_cast<size_t>(batch_size)) {
        const size_t end = std::min(order.size(), start + static_cast<size_t>(batch_size));
        std::vector<PromptCompletion> chunk;
        for (size_t k = start; k < end; ++k) chunk.push_back(examples[order[k]]);
        const CausalLmBatch batch = CollatePromptCompletions(chunk, backend);
        const int64_t length = batch.ids.shape().dim(1);
        const Tensor logits = model.forward(batch.ids);
        std::vector<float> rows;
        std::vector<int64_t> tokens;
        for (size_t r = 0; r < chunk.size(); ++r) {
            const auto p = static_cast<int64_t>(chunk[r].prompt.size());
            for (size_t j = 0; j < chunk[r].completion.size(); ++j) {
                rows.push_back(static_cast<float>(static_cast<int64_t>(r) * length + p - 1 + static_cast<int64_t>(j)));
                tokens.push_back(chunk[r].completion[j]);
            }
        }
        const std::vector<float> host = GatherRows(logits, rows, vocab, backend);
        size_t row = 0;
        for (size_t r = 0; r < chunk.size(); ++r) {
            double sum = 0.0;
            for (size_t j = 0; j < chunk[r].completion.size(); ++j, ++row) {
                const float* v = &host[row * static_cast<size_t>(vocab)];
                sum += static_cast<double>(v[tokens[row]]) - LogSumExp(v, vocab);
            }
            out[order[start + r]] = sum;
        }
    }
    return out;
}

ChoiceScores ScoreChoices(CausalLM& model, const std::vector<std::vector<int64_t>>& prompts,
                          const std::vector<int64_t>& choice_tokens, DeviceBackend* backend, int64_t batch_size) {
    if (prompts.empty()) throw std::invalid_argument("ScoreChoices: no prompts");
    if (choice_tokens.empty()) throw std::invalid_argument("ScoreChoices: no choices");
    if (batch_size < 1) throw std::invalid_argument("ScoreChoices: batch_size must be >= 1");
    const int64_t vocab = model.config().vocab_size;
    for (const int64_t c : choice_tokens) {
        if (c < 0 || c >= vocab) throw std::invalid_argument("ScoreChoices: choice token " + std::to_string(c) + " is outside the vocabulary");
    }
    std::vector<size_t> lengths;
    for (const auto& p : prompts) {
        if (p.empty()) throw std::invalid_argument("ScoreChoices: a prompt is empty");
        lengths.push_back(p.size());
    }
    const std::vector<size_t> order = ByLength(lengths);
    ChoiceScores out;
    out.log_probs.assign(prompts.size(), {});
    out.choice_log_probs.assign(prompts.size(), {});
    out.choice_mass.assign(prompts.size(), 0.0);
    for (size_t start = 0; start < order.size(); start += static_cast<size_t>(batch_size)) {
        const size_t end = std::min(order.size(), start + static_cast<size_t>(batch_size));
        const auto n = static_cast<int64_t>(end - start);
        const auto length = static_cast<int64_t>(lengths[order[end - 1]]);
        std::vector<float> ids(static_cast<size_t>(n * length), 0.0f);
        std::vector<float> rows;
        for (int64_t r = 0; r < n; ++r) {
            const auto& p = prompts[order[start + static_cast<size_t>(r)]];
            for (size_t i = 0; i < p.size(); ++i) ids[static_cast<size_t>(r * length) + i] = static_cast<float>(p[i]);
            rows.push_back(static_cast<float>(r * length + static_cast<int64_t>(p.size()) - 1));
        }
        const Tensor logits = model.forward(Tensor(Shape({n, length}), backend, ids, backend->device()));
        const std::vector<float> host = GatherRows(logits, rows, vocab, backend);
        for (int64_t r = 0; r < n; ++r) {
            const float* v = &host[static_cast<size_t>(r * vocab)];
            const double lse = LogSumExp(v, vocab);
            std::vector<double> full;
            for (const int64_t c : choice_tokens) full.push_back(static_cast<double>(v[c]) - lse);
            const double top = *std::max_element(full.begin(), full.end());
            double mass = 0.0;
            for (const double f : full) mass += std::exp(f - top);
            const double log_mass = top + std::log(mass);
            std::vector<double> renormalized;
            for (const double f : full) renormalized.push_back(f - log_mass);
            const size_t i = order[start + static_cast<size_t>(r)];
            out.log_probs[i] = std::move(full);
            out.choice_log_probs[i] = std::move(renormalized);
            out.choice_mass[i] = std::exp(log_mass);
        }
    }
    return out;
}

int64_t SingleTokenId(const TextTokenizer& tokenizer, const std::string& text) {
    const std::vector<int64_t> ids = tokenizer.encode(text, false).ids;
    if (ids.size() != 1) {
        throw std::invalid_argument("SingleTokenId: \"" + text + "\" is " + std::to_string(ids.size()) + " tokens, not one");
    }
    return ids.front();
}

}  // namespace pulsatrix
