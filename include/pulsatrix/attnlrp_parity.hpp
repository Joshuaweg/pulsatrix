/** @file attnlrp_parity.hpp
 *  @brief AttnLRP parity with LXT on a real language model (LLM-7): pulsatrix's relevance against
 *         LXT's own AttnLRP on the same Hugging Face model and tokens.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {

/** @brief One reference sequence's comparison. */
struct AttnLrpSequenceResult {
    int64_t num_tokens = 0;
    /** @brief The explained token at the last position (the reference's argmax). */
    int64_t target = 0;
    /** @brief Pearson correlation of the per-token input relevance with LXT's. */
    float token_correlation = 0.0f;
    /** @brief The lowest, over layers, Pearson correlation of the per-(KV head, token) relevance
     *         at k_proj's output with LXT's; value_correlation is the same at v_proj. */
    float key_correlation = 0.0f;
    float value_correlation = 0.0f;
    /** @brief max |pulsatrix - LXT| over the tokens, divided by the largest |LXT| token
     *         relevance: catches a scale difference, which correlation can't see. */
    float max_relative_diff = 0.0f;
    /** @brief Total input relevance, pulsatrix's and LXT's (both seeded with the logit). */
    float relevance_sum = 0.0f;
    float reference_sum = 0.0f;
};

/** @brief A model's comparison against an AttnLRP reference file. */
struct AttnLrpReport {
    std::vector<AttnLrpSequenceResult> sequences;
    /** @brief The lowest token, key and value correlations over all sequences. */
    float min_token_correlation = 1.0f;
    float min_kv_correlation = 1.0f;
    /** @brief The largest max_relative_diff over all sequences (reported; not part of passed). */
    float max_relative_diff = 0.0f;
    float threshold = 0.0f;
    /** @brief Every correlation above the threshold. */
    bool passed = false;
    /** @brief The reference file's metadata: model, revision, versions. */
    std::map<std::string, std::string> metadata;
};

/** @brief Pearson correlation of two equal-length vectors; 0 if either is constant. */
[[nodiscard]] double PearsonCorrelation(const std::vector<float>& x, const std::vector<float>& y);

/** @brief The rule configuration that matches LXT's AttnLRP on a Hugging Face model: the epsilon
 *         rule with a tiny epsilon, biases kept in the denominator as LXT's gradient * input
 *         keeps them in z (Qwen2's QKV biases), and LayerNorm with only its standard deviation
 *         held constant (LRPRuleConfig::layer_norm_detach_std; ESM-2, PLM-6). */
[[nodiscard]] LRPRuleConfig LxtAttnLrpConfig();

/**
 * @brief Explains each `ids.<k>` sequence of an AttnLRP reference file (as written by
 *        tools/golden/make_attnlrp_reference.py) from the logit of `target.<k>` at the last
 *        position, seeded with that logit as LXT's examples do, and compares the input relevance
 *        and every layer's K/V-head relevance with the reference's.
 * @param threshold The roadmap's criterion: correlation above 0.99.
 * @throws std::invalid_argument if the file holds no sequences or its tensors don't fit the model.
 */
[[nodiscard]] AttnLrpReport CompareToAttnLrp(CausalLM& model, const SafetensorsFile& reference,
                                             const LRPRuleConfig& config = LxtAttnLrpConfig(),
                                             float threshold = 0.99f);

}  // namespace pulsatrix
