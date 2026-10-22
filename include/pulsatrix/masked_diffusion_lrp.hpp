/** @file masked_diffusion_lrp.hpp
 *  @brief Explaining a masked diffusion LM's generation (DIFF-3): commitment-graph LRP, which
 *         explains tokens written at any step back to the prompt by chaining AttnLRP through the
 *         tokens committed in between, and the checks that test it by replaying the trajectory.
 *  @ingroup interpretability
 *
 *  In a carry-over sampler (MDLM's, LLaDA's) no hidden state passes from one step to the next:
 *  each step is one forward pass over the current sequence, and the committed tokens are the only
 *  channel between steps. So relevance that one step's pass puts on a token committed earlier is
 *  re-seeded at the step that wrote it, onto the logit that produced it, and the steps are walked
 *  in reverse. With the forward activations fixed every rule in use is linear in the incoming
 *  relevance, so each step needs one relevance pass however many tokens it seeds.
 *  See docs/interpretability/diffusion-lrp.md (D2) for the derivation and its limits.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "pulsatrix/attnlrp_parity.hpp"  // LxtAttnLrpConfig
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/masked_diffusion.hpp"

namespace pulsatrix {

/**
 * @brief One denoising step's model as the explainer sees it: a forward pass that keeps what the
 *        relevance pass needs, and the relevance pass. EncoderLMStepModel wraps an EncoderLM.
 */
class StepRelevanceModel {
public:
    virtual ~StepRelevanceModel() = default;
    /** @brief Logits for every position, `L * vocab` values, row-major. */
    [[nodiscard]] virtual std::vector<float> forward(const std::vector<int64_t>& tokens) = 0;
    /** @brief Relevance per input position, `L` values, from relevance on the last forward()'s
     *         logits (`L * vocab`). */
    [[nodiscard]] virtual std::vector<float> relevance(const std::vector<float>& relevance_logits) = 0;
};

/** @brief An EncoderLM as a StepRelevanceModel: one sequence per pass, AttnLRP with @p config
 *         (LXT's rules by default, as EncoderExplainer uses). */
class EncoderLMStepModel final : public StepRelevanceModel {
public:
    /** @param backend Where the model runs. The model and backend must outlive this. */
    EncoderLMStepModel(EncoderLM& model, DeviceBackend* backend, LRPRuleConfig config = LxtAttnLrpConfig());
    /** @brief Turns on EncoderLM::set_keep_activations, which relevance() needs. */
    [[nodiscard]] std::vector<float> forward(const std::vector<int64_t>& tokens) override;
    [[nodiscard]] std::vector<float> relevance(const std::vector<float>& relevance_logits) override;

private:
    EncoderLM& model_;
    DeviceBackend* backend_;
    LRPRuleConfig config_;
    int64_t length_ = 0;
};

/** @brief A CausalLM with bidirectional attention (an A2D diffusion model, DIFF-4) as a
 *         StepRelevanceModel, with the same rules as EncoderLMStepModel. */
class CausalLMStepModel final : public StepRelevanceModel {
public:
    /**
     * @param backend Where the model runs. The model and backend must outlive this.
     * @throws std::invalid_argument if the model is causal.
     */
    CausalLMStepModel(CausalLM& model, DeviceBackend* backend, LRPRuleConfig config = LxtAttnLrpConfig());
    [[nodiscard]] std::vector<float> forward(const std::vector<int64_t>& tokens) override;
    [[nodiscard]] std::vector<float> relevance(const std::vector<float>& relevance_logits) override;

private:
    CausalLM& model_;
    DeviceBackend* backend_;
    LRPRuleConfig config_;
    int64_t length_ = 0;
};

/** @brief How far relevance travels. */
enum class CommitmentMode {
    /** @brief Relevance on a committed token is re-seeded at the step that wrote it, until it
     *         reaches the prompt or a mask token. */
    Graph,
    /** @brief Each target is explained at its own step only: relevance on committed tokens stops
     *         there. What the graph is compared with. */
    StepLocal,
};

/** @brief A generated token to explain. */
struct CommitmentTarget {
    /** @brief A position the trajectory wrote. */
    int64_t position = 0;
    /**
     * @brief The token whose logit is explained, at the step that last wrote the position. Unset:
     *        the token written there. Set: a fixed reference token, the check DLIG (arXiv
     *        2610.01177) found more informative than the model's own choice. Relevance relayed to
     *        the position from later steps always goes to the written token.
     */
    std::optional<int64_t> token;
};

/** @brief One step's relevance pass: what went in and where it landed. */
struct CommitmentHop {
    /** @brief The index into MaskedDiffusionResult::steps. */
    int64_t step = 0;
    /** @brief Relevance seeded on the step's logits: its targets' logits and what was relayed. */
    double seeded = 0;
    /** @brief Relevance on the step's input, by kind of token. */
    double to_prompt = 0;
    double to_committed = 0;
    double to_mask = 0;
    /** @brief What the pass didn't conserve: biases' shares, norms, the stabilizer. */
    [[nodiscard]] double leak() const { return seeded - to_prompt - to_committed - to_mask; }
};

/**
 * @brief Where the relevance of the explained tokens went. Every vector has one value per
 *        position. The ledger balances exactly: value = prompt + mask + leak (+ relayed in
 *        CommitmentMode::StepLocal, where it is an endpoint).
 */
struct CommitmentRelevance {
    CommitmentMode mode = CommitmentMode::Graph;
    /** @brief The explained value: the targets' logits, each at the step that wrote it, summed.
     *         The relevance seeded, as LXT seeds a logit. */
    double value = 0;
    /** @brief Relevance on the prompt (positions not masked in the input), from every step. Zero
     *         elsewhere. */
    std::vector<float> prompt;
    /** @brief Relevance that reached each generated position from later steps. Graph: relayed
     *         onward from the step that wrote it. StepLocal: stopped there. */
    std::vector<float> relayed;
    /** @brief Relevance each position absorbed while it was masked, summed over steps: the mask
     *         sink. */
    std::vector<float> mask;
    /** @brief One entry per step explained, in the order explained (the last step first). Steps
     *         nothing was seeded at aren't run. */
    std::vector<CommitmentHop> hops;
    /** @brief For each of the trajectory's steps, the relevance on its input (`L` values), or
     *         empty for a step that wasn't run. */
    std::vector<std::vector<float>> step_relevance;

    [[nodiscard]] double prompt_total() const;
    [[nodiscard]] double relayed_total() const;
    [[nodiscard]] double mask_total() const;
    [[nodiscard]] double leak_total() const;
};

/**
 * @brief Explains @p targets on @p trajectory, which SampleMaskedDiffusion produced with @p model's
 *        logits. Each step's input is rerun, so the model must give the logits it gave then.
 *
 * Inputs are classified by what the step saw: a mask token is a mask, a position no step had yet
 * written is the prompt, and anything else was committed by the last step that wrote it before
 * this one. That definition holds under a remasking sampler too: a revoked position is a mask
 * until it is written again.
 * @note Cost: one forward and one relevance pass per step that has something seeded.
 * @throws std::invalid_argument for no targets, a position given twice or not written by the
 *         trajectory, a token outside the vocabulary, or logits or relevance of the wrong size.
 */
[[nodiscard]] CommitmentRelevance ExplainCommitments(StepRelevanceModel& model, const MaskedDiffusionResult& trajectory,
                                                     int64_t mask_token_id, const std::vector<CommitmentTarget>& targets,
                                                     CommitmentMode mode = CommitmentMode::Graph);

// ---- checks: replaying the trajectory -------------------------------------------------------

/**
 * @brief Reruns @p trajectory from @p input with its write order frozen: each step writes the
 *        same positions as before, each taking the model's most likely allowed token for the new
 *        sequence (greedy, as explanation runs are). Positions in @p skipped are never written, and
 *        a position @p input masks that the trajectory never wrote (a deleted prompt token) stays
 *        masked, so the result may keep mask tokens.
 * @param config Only `mask_token_id` and `suppressed_token_ids` are used.
 * @throws std::invalid_argument if @p input's length differs from the trajectory's, or the model's
 *         output doesn't fit.
 */
[[nodiscard]] MaskedDiffusionResult ReplayMaskedDiffusion(const DenoiserFn& model, const MaskedDiffusionResult& trajectory,
                                                          const std::vector<int64_t>& input, const MaskedDiffusionConfig& config,
                                                          const std::vector<int64_t>& skipped = {});

/**
 * @brief The explained value on a replay (ReplayMaskedDiffusion): each target's logit at the step
 *        that wrote it in @p trajectory, for the token the trajectory wrote there (or the target's
 *        fixed token), summed. A target in @p skipped still counts at that step.
 * @throws std::invalid_argument as ReplayMaskedDiffusion, or for a target the trajectory didn't
 *         write.
 */
[[nodiscard]] double ReplayCommitmentValue(const DenoiserFn& model, const MaskedDiffusionResult& trajectory,
                                           const std::vector<int64_t>& input, const MaskedDiffusionConfig& config,
                                           const std::vector<CommitmentTarget>& targets, const std::vector<int64_t>& skipped = {});

/** @brief A prompt-deletion curve against its random baseline. */
struct CommitmentDeletionCurve {
    /** @brief Fractions of the prompt deleted: 0, 1/steps, ..., 1. */
    std::vector<float> fractions;
    /** @brief The explained value after deleting that share, most relevant first. */
    std::vector<float> scores;
    /** @brief The same in random orders (their mean). */
    std::vector<float> random_scores;
    /** @brief The sign of the undeleted value. */
    float sign = 1.0f;
    /** @brief Trapezoid areas under each curve times `sign`; a faithful explanation removes the
     *         value faster: auc < random_auc. */
    double auc = 0;
    double random_auc = 0;
};

/**
 * @brief Deletes prompt tokens (replaces them with the mask token) in order of how much they
 *        support the value (@p prompt_relevance of the value's sign, largest first), replays the
 *        trajectory each time (ReplayCommitmentValue), and records the value, beside
 *        @p random_orders random orders.
 * @note Cost: `(steps + 1) * (1 + random_orders)` replays.
 * @throws std::invalid_argument if @p prompt_relevance doesn't have one value per position, the
 *         input has no prompt, steps < 1 or random_orders < 1.
 */
[[nodiscard]] CommitmentDeletionCurve CommitmentDeletionCheck(const DenoiserFn& model, const MaskedDiffusionResult& trajectory,
                                                              const MaskedDiffusionConfig& config,
                                                              const std::vector<CommitmentTarget>& targets,
                                                              const std::vector<float>& prompt_relevance, int64_t steps = 10,
                                                              int64_t random_orders = 3, uint64_t seed = 0);

/** @brief Step-resolved patching against the relevance relayed through each committed token. */
struct CommitmentPatchResult {
    /** @brief The generated positions patched: every non-target position written before the last
     *         target's step. */
    std::vector<int64_t> positions;
    /** @brief The relevance relayed through each (CommitmentRelevance::relayed). */
    std::vector<float> relayed;
    /** @brief The value minus its value with the position left masked from its write step on. */
    std::vector<float> effect;
    /** @brief Spearman correlation of relayed relevance with the effect; 0 for fewer than 3
     *         positions. A token the graph routes much relevance through should matter more. */
    double spearman = 0;
};

/**
 * @brief Leaves each committed intermediate token masked in turn, replays the trajectory, and
 *        compares the change in the value with the relevance @p graph relayed through that token.
 * @param graph ExplainCommitments' result for the same targets, in CommitmentMode::Graph.
 * @note Cost: one replay per patched position, plus one.
 * @throws std::invalid_argument if @p graph isn't a Graph result for this trajectory, or as
 *         ReplayCommitmentValue.
 */
[[nodiscard]] CommitmentPatchResult CommitmentPatchCheck(const DenoiserFn& model, const MaskedDiffusionResult& trajectory,
                                                         const MaskedDiffusionConfig& config, const std::vector<CommitmentTarget>& targets,
                                                         const CommitmentRelevance& graph);

}  // namespace pulsatrix
