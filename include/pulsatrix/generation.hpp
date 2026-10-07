/** @file generation.hpp
 *  @brief Text generation from a language model: greedy decoding and seeded sampling with
 *         temperature, top-k, top-p and min-p, repetition penalty, and EOS handling (LLM-4).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {

/**
 * @brief The model as generation sees it: the token sequence so far in, the logits for the next
 *        token out (one per vocabulary entry). How they're computed is up to the caller: a full
 *        forward pass over the sequence (MakeNextTokenLogits), or a cached step (LLM-5).
 */
using NextTokenLogitsFn = std::function<std::vector<float>(const std::vector<int64_t>& tokens)>;

/** @brief Decoding settings. The defaults decode greedily. Field names follow Hugging Face's
 *         GenerationConfig. */
struct GenerationConfig {
    int64_t max_new_tokens = 32;
    /** @brief EOS cannot be generated before this many new tokens. */
    int64_t min_new_tokens = 0;
    /** @brief Generation stops after any of these is generated (it is kept in the output). */
    std::vector<int64_t> eos_token_ids;
    /** @brief false: always the most likely token (greedy). true: sample. */
    bool do_sample = false;
    /** @brief Divides the logits; below 1 sharpens, above 1 flattens. Must be positive. */
    float temperature = 1.0f;
    /** @brief Keep only the top_k most likely tokens; 0 keeps all. */
    int64_t top_k = 0;
    /** @brief Keep the smallest set of most likely tokens whose probability reaches top_p; 1 keeps
     *         all. */
    float top_p = 1.0f;
    /** @brief Drop tokens less likely than min_p times the most likely one; 0 keeps all. */
    float min_p = 0.0f;
    /** @brief CTRL's penalty on every token already in the sequence, prompt included: positive
     *         logits are divided by it, negative ones multiplied; 1 is off. */
    float repetition_penalty = 1.0f;
    /** @brief The sampling seed; unset draws one from the global stream (next_seed(), FND-7), so
     *         generation is reproducible after set_seed() either way. */
    std::optional<uint64_t> seed;
};

/** @brief Why generation stopped. */
enum class FinishReason {
    /** An EOS token was generated. */
    Eos,
    /** max_new_tokens were generated. */
    Length,
};

/** @brief A generated continuation. */
struct GenerationResult {
    /** @brief The prompt followed by the new tokens. */
    std::vector<int64_t> tokens;
    /** @brief Only the new tokens, EOS included if generated. */
    std::vector<int64_t> new_tokens;
    /** @brief Each new token's log-probability under the model itself (log-softmax of the raw
     *         logits, before any processing): how likely the model found it, whatever the decoding
     *         settings were. */
    std::vector<float> log_probs;
    FinishReason finish_reason = FinishReason::Length;
};

/**
 * @brief Applies the configured logit processing, in Hugging Face's order: repetition penalty,
 *        the EOS ban before min_new_tokens, then (when sampling) temperature, top-k, top-p and
 *        min-p. Removed tokens get -infinity. With do_sample false only the first two apply.
 * @param tokens The sequence so far (for the repetition penalty).
 * @param num_new_tokens How many tokens have been generated so far (for min_new_tokens).
 * @throws std::invalid_argument for an invalid config.
 */
void ProcessLogits(std::vector<float>& logits, const std::vector<int64_t>& tokens, int64_t num_new_tokens,
                   const GenerationConfig& config);

/**
 * @brief Generates up to max_new_tokens after @p prompt, one at a time: process the next-token
 *        logits, take the most likely token (greedy) or draw one from their softmax (sampling),
 *        append it, and stop after an EOS token.
 * @note Sampling draws from std::mt19937_64's raw output, so a seed gives the same text on every
 *       platform as long as the model's logits are the same.
 * @throws std::invalid_argument if the prompt is empty, the config is invalid, or the model
 *         returns an empty or non-finite set of logits, or every token gets removed.
 */
[[nodiscard]] GenerationResult Generate(const NextTokenLogitsFn& model, const std::vector<int64_t>& prompt,
                                        const GenerationConfig& config = {});

/**
 * @brief A NextTokenLogitsFn for a stack of pulsatrix modules that maps token ids `(1, L)` to
 *        logits `(1, L, vocab)` (for example EmbeddingModule, causal TransformerBlocks, an RMSNorm
 *        and a TiedLMHeadModule): it runs the whole sequence through the modules and returns the
 *        last position's logits.
 * @param layers Run in order; not owned, must outlive the function.
 * @param backend Where the ids tensor is built.
 * @note Recomputes the whole prefix at every step; MakeCachedNextTokenLogits doesn't.
 * @throws std::invalid_argument (when called) if the output isn't rank 3 with one row per token.
 */
[[nodiscard]] NextTokenLogitsFn MakeNextTokenLogits(std::vector<Module*> layers, DeviceBackend* backend);

/**
 * @brief A NextTokenLogitsFn for embedding -> transformer blocks -> head layers that keeps a KV cache
 *        per block (LLM-5): each call runs only the tokens after the longest prefix it has already
 *        processed, so generating n tokens costs one pass over the prompt and then one position per
 *        token, instead of n passes over the whole sequence. A call with a different history (a new
 *        prompt, or a shorter or edited one) truncates the caches to the shared prefix and continues
 *        from there.
 * @param embedding Maps `(1, L)` ids to `(1, L, d_model)`.
 * @param blocks Run with forward_cached, in order; each gets its own cache of max_length positions.
 * @param head Applied to the last position's hidden state `(1, d_model)`: for example a final
 *        RMSNormModule and a TiedLMHeadModule; must end in `(1, vocab)`.
 * @note The returned function owns the caches (allocated once) and is stateful: use one per
 *       sequence being generated. The modules are not owned and must outlive it.
 * @throws std::invalid_argument (when built) if blocks is empty or a pointer is null; (when called)
 *         if the sequence is empty or longer than max_length.
 */
[[nodiscard]] NextTokenLogitsFn MakeCachedNextTokenLogits(EmbeddingModule& embedding,
                                                          std::vector<TransformerBlock*> blocks,
                                                          std::vector<Module*> head, DeviceBackend* backend,
                                                          int64_t max_length, float embed_scale = 1.0f);

}  // namespace pulsatrix
