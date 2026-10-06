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
    /** @brief Positions whose most likely next token differs from the reference's. */
    int64_t argmax_disagreements = 0;
};

/** @brief A model's comparison against a golden file. */
struct GoldenReport {
    std::vector<GoldenSequenceResult> sequences;
    /** @brief The largest max_abs_diff over all sequences. */
    float max_abs_diff = 0.0f;
    float threshold = 0.0f;
    /** @brief max_abs_diff < threshold. */
    bool passed = false;
    /** @brief The golden file's metadata: model, revision, versions. */
    std::map<std::string, std::string> metadata;
};

/**
 * @brief Runs @p model on every `ids.<k>` sequence of a golden file (as written by
 *        tools/golden/make_golden.py) and compares all its logits with `logits.<k>`.
 * @param threshold The roadmap's criterion: fp32, every logit within 1e-3 of the reference.
 * @throws std::invalid_argument if the file holds no sequences, a sequence's ids aren't I64 or
 *         don't fit the vocabulary, or its logits' shape doesn't match.
 */
[[nodiscard]] GoldenReport CompareToGolden(CausalLM& model, const SafetensorsFile& golden, float threshold = 1e-3f);

}  // namespace pulsatrix
