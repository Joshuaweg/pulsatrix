/** @file masked_diffusion.hpp
 *  @brief Sampling from a masked diffusion language model (DIFF-2): start from a sequence with
 *         masked positions and fill them over a number of steps, recording the whole trajectory
 *         so it can be explained afterwards (DIFF-3).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "pulsatrix/encoder_lm.hpp"

namespace pulsatrix {

/**
 * @brief The model as the sampler sees it: the current sequence (masked positions included) in,
 *        logits for every position out, `L * vocab` values, row-major. MakeDenoiser builds one for
 *        an EncoderLM.
 */
using DenoiserFn = std::function<std::vector<float>(const std::vector<int64_t>& tokens)>;

/** @brief How the sampler chooses which masked positions to fill at each step. */
enum class UnmaskingSchedule {
    /**
     * @brief LLaDA's low-confidence remasking: every step fills a fixed share of the positions
     *        that started masked (spread as evenly as the step count allows, earlier steps taking
     *        the remainder), choosing the masked positions whose chosen token the model is most
     *        sure of. Ties go to the earlier position. Fully deterministic when greedy.
     */
    Confidence,
    /**
     * @brief MDLM's ancestral sampler (`ddpm_cache`): time runs linearly from 1 to 1e-5, and at
     *        each step every masked position is filled independently with probability
     *        `(t - s) / t`, where s is the next step's time. Positions still masked after the last
     *        step are filled with their most likely token, as MDLM's final noise-removal step does.
     *        Steps that fill nothing don't run the model (MDLM reuses its output the same way).
     */
    Ancestral,
};

/** @brief Sampling settings. The defaults fill greedily by confidence. */
struct MaskedDiffusionConfig {
    /** @brief Denoising steps. Must be positive. */
    int64_t steps = 64;
    /** @brief The token that marks a masked position. Must be set. */
    int64_t mask_token_id = -1;
    UnmaskingSchedule schedule = UnmaskingSchedule::Confidence;
    /** @brief false: each filled position takes its most likely token. true: sample it. */
    bool do_sample = false;
    /** @brief Divides the logits before sampling. Must be positive. */
    float temperature = 1.0f;
    /** @brief Tokens never generated. The mask token never is (MDLM's SUBS parameterization). */
    std::vector<int64_t> suppressed_token_ids;
    /** @brief The seed for sampling and for the ancestral schedule's draws; unset draws one from
     *         the global stream (next_seed(), FND-7). */
    std::optional<uint64_t> seed;
};

/** @brief One position filled at one step. */
struct MaskedDiffusionWrite {
    int64_t position = 0;
    int64_t token = 0;
    /** @brief The model's probability of the token (softmax of the raw logits, before any
     *         processing), the confidence the Confidence schedule ranks by. */
    float probability = 0.0f;
};

/** @brief One step that filled at least one position. */
struct MaskedDiffusionStep {
    /** @brief The sequence the model saw at this step. */
    std::vector<int64_t> input;
    /** @brief The positions filled from the model's output, in position order. */
    std::vector<MaskedDiffusionWrite> writes;
};

/** @brief A sampled sequence and how it was produced. */
struct MaskedDiffusionResult {
    /** @brief The final sequence: no masked position remains. */
    std::vector<int64_t> tokens;
    /** @brief Every step that filled something, in order. Each step's input is the previous
     *         step's input with the previous writes applied. */
    std::vector<MaskedDiffusionStep> steps;
    /** @brief For each position, the index into `steps` of the step that last wrote it, or -1 for
     *         a position that wasn't masked in the input (a prompt token). Neither schedule
     *         re-masks, so a position's last write is also its only one. */
    std::vector<int64_t> last_write_step;
};

/**
 * @brief Fills every masked position of @p tokens over the configured steps.
 * @note Sampling draws from std::mt19937_64's raw output, so a seed gives the same trajectory on
 *       every platform as long as the model's logits are the same.
 * @throws std::invalid_argument for an invalid config, an input with no masked position or a
 *         token outside the vocabulary the model's output implies, logits of the wrong size or not
 *         finite, or a position where every token is suppressed.
 */
[[nodiscard]] MaskedDiffusionResult SampleMaskedDiffusion(const DenoiserFn& model, const std::vector<int64_t>& tokens,
                                                          const MaskedDiffusionConfig& config);

/**
 * @brief A DenoiserFn for an EncoderLM: one forward pass over the sequence, `(1, L)` to logits
 *        `(1, L, vocab)`. The model isn't owned and must outlive the function.
 */
[[nodiscard]] DenoiserFn MakeDenoiser(EncoderLM& model, DeviceBackend* backend);

}  // namespace pulsatrix
