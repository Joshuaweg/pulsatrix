/** @file golden_logits.hpp
 *  @brief The golden-logit harness: a loaded model's logits against Hugging Face's on real text
 *         (LLM-6).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {

/** @brief One reference sequence's comparison. */
struct GoldenSequenceResult {
    int64_t num_tokens = 0;
    /** @brief The largest |pulsatrix - reference| over every logit of every position. */
    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    /** @brief The largest |pulsatrix - reference| / max(1, max |reference logit| at that
     *         position): what the threshold is applied to. */
    float max_scaled_diff = 0.0f;
    /** @brief Positions whose most likely next token differs from the reference's. */
    int64_t argmax_disagreements = 0;
};

/** @brief A model's comparison against a golden file. */
struct GoldenReport {
    std::vector<GoldenSequenceResult> sequences;
    /** @brief The largest max_abs_diff over all sequences. */
    float max_abs_diff = 0.0f;
    /** @brief The largest max_scaled_diff over all sequences. */
    float max_scaled_diff = 0.0f;
    float threshold = 0.0f;
    /** @brief max_scaled_diff < threshold. */
    bool passed = false;
    /** @brief The golden file's metadata: model, revision, versions. */
    std::map<std::string, std::string> metadata;
};

/**
 * @brief Compares one sequence's logits, both (L, vocab) row-major.
 * @note Each position's differences are divided by max(1, max |reference logit| there), so the
 *       threshold is absolute for logits up to 1 and relative above. fp32 sums drift with the
 *       size of what they add up, and Qwen2.5's logits reach about 24 (1.1e-3 absolute on CPU,
 *       5e-5 relative), while a layout bug moves logits by about their own size.
 * @throws std::invalid_argument if the sizes don't match L * vocab.
 */
[[nodiscard]] GoldenSequenceResult CompareLogits(const std::vector<float>& got, const std::vector<float>& expected,
                                                 int64_t num_tokens, int64_t vocab);

/**
 * @brief Runs @p model on every `ids.<k>` sequence of a golden file (as written by
 *        tools/golden/make_golden.py) and compares all its logits with `logits.<k>` (CompareLogits).
 * @param threshold The roadmap's criterion: fp32, every logit within 1e-3 of the reference,
 *        relative to the position's largest logit when that exceeds 1.
 * @throws std::invalid_argument if the file holds no sequences, a sequence's ids aren't I64 or
 *         don't fit the vocabulary, or its logits' shape doesn't match.
 */
[[nodiscard]] GoldenReport CompareToGolden(CausalLM& model, const SafetensorsFile& golden, float threshold = 1e-3f);

}  // namespace pulsatrix
