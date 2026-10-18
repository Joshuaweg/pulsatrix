/** @file lm_scoring.hpp
 *  @brief Prompt/completion data for a causal language model (LLM-10): read it from JSONL, batch
 *         it with the prompt masked out of the loss, and score it, as log P(completion | prompt)
 *         or as the probabilities of a few single-token choices ("P(True)").
 *  @ingroup dl_modules
 *
 * Fine-tuning a model to answer a question trains only the answer: CollatePromptCompletions()
 * sets every target outside the completion to TokenCrossEntropyLoss::kIgnoreIndex, so the loss is
 * the completion's log loss and nothing else. Scoring reads the same positions back.
 *
 * Batches are right-padded. Causal attention never lets a real token attend to the padding after
 * it, so padding changes no real position's logits (tests/lm_scoring_test.cpp checks a padded
 * batch against each example alone).
 *
 * The scorers copy only the logit rows they need to the host, gathered on the device: a batch's
 * full logits are (N, L, vocab), about 400 MB for SmolLM2 at N = 8, L = 256.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/tensor.hpp"
#include "pulsatrix/text_tokenizer.hpp"

namespace pulsatrix {

/** @brief One example: the prompt's token ids, then the completion's. */
struct PromptCompletion {
    std::vector<int64_t> prompt;
    std::vector<int64_t> completion;
};

/**
 * @brief Reads a JSONL file: one JSON value a line. Blank lines are skipped.
 * @throws std::runtime_error if the file can't be read, naming the line that doesn't parse.
 */
[[nodiscard]] std::vector<JsonValue> ReadJsonl(const std::string& path);

/**
 * @brief Tokenizes each row's prompt and completion strings (objects with those keys). The two
 *        are tokenized separately, so a completion like " Yes" keeps its own tokens whatever the
 *        prompt ends with.
 * @param add_special_tokens Passed to TextTokenizer::encode() for the prompt (a BOS token, for
 *        tokenizers that add one); never for the completion.
 * @throws std::invalid_argument if a row isn't an object with both strings, or a prompt or
 *         completion tokenizes to nothing.
 */
[[nodiscard]] std::vector<PromptCompletion> TokenizePromptCompletions(const std::vector<JsonValue>& rows,
                                                                      const TextTokenizer& tokenizer,
                                                                      const std::string& prompt_key = "prompt",
                                                                      const std::string& completion_key = "completion",
                                                                      bool add_special_tokens = false);

/** @brief A right-padded training batch. */
struct CausalLmBatch {
    Tensor ids;                    ///< (N, L) token ids as floats: prompt, then completion but its last token
    Tensor targets;                ///< (N, L): the next token where it is a completion token, else kIgnoreIndex
    std::vector<int64_t> lengths;  ///< each row's real length, |prompt| + |completion| - 1
    int64_t target_tokens = 0;     ///< completion tokens in the batch, the loss's token count
};

/**
 * @brief Builds a batch for TokenCrossEntropyLoss: position i predicts token i + 1, and only the
 *        completion's tokens count. The completion's last token is never an input (nothing
 *        follows it), so each row is one token shorter than prompt + completion.
 * @param pad_id The id written into padding; its value doesn't matter (see the file comment).
 * @throws std::invalid_argument if examples is empty or an example has an empty prompt or completion.
 */
[[nodiscard]] CausalLmBatch CollatePromptCompletions(const std::vector<PromptCompletion>& examples, DeviceBackend* backend,
                                                     int64_t pad_id = 0);

/**
 * @brief log P(completion | prompt) under the model, summed over the completion's tokens, in
 *        double. Runs forward passes only, batch_size examples at a time, grouped by length.
 * @throws std::invalid_argument as CollatePromptCompletions(), or if batch_size < 1.
 */
[[nodiscard]] std::vector<double> CompletionLogProbs(CausalLM& model, const std::vector<PromptCompletion>& examples,
                                                     DeviceBackend* backend, int64_t batch_size = 8);

/** @brief Scores of single-token choices after each prompt. Indexed [example][choice]. */
struct ChoiceScores {
    std::vector<std::vector<double>> log_probs;         ///< log P(choice | prompt) over the whole vocabulary
    std::vector<std::vector<double>> choice_log_probs;  ///< the same, renormalized over the choices
    std::vector<double> choice_mass;                    ///< the probability the choices take together
};

/**
 * @brief Scores the next token after each prompt against a fixed set of single-token choices,
 *        such as " Yes" and " No". For two choices, choice_log_probs[n][0] - choice_log_probs[n][1]
 *        is the log-odds of the first.
 * @throws std::invalid_argument if prompts or choice_tokens is empty, a prompt is empty, a choice
 *         is outside the vocabulary, or batch_size < 1.
 */
[[nodiscard]] ChoiceScores ScoreChoices(CausalLM& model, const std::vector<std::vector<int64_t>>& prompts,
                                        const std::vector<int64_t>& choice_tokens, DeviceBackend* backend,
                                        int64_t batch_size = 8);

/**
 * @brief The id of a string that must tokenize to exactly one token, such as " Yes".
 * @throws std::invalid_argument if it tokenizes to anything else.
 */
[[nodiscard]] int64_t SingleTokenId(const TextTokenizer& tokenizer, const std::string& text);

}  // namespace pulsatrix
