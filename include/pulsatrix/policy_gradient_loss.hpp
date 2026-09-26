/** @file policy_gradient_loss.hpp
 *  @brief REINFORCE's return-weighted negative log-likelihood loss over a batched rollout.
 *  @ingroup rl
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief loss = mean_b( -log pi(a_b | s_b) * G_b ), where `a_b` is the action actually taken on
 *        step `b` of a rollout and `G_b` its return -- the REINFORCE policy-gradient surrogate
 *        of Williams 1992.
 *
 * This is a return-weighted cross-entropy against the actions the policy itself took. It is a
 * *surrogate*: its value has no interpretation as an error, and minimizing it is only a device
 * for producing the policy gradient in backward(). A negative return makes a step's
 * contribution negative, which is correct and expected -- the sign of the loss carries no
 * information about how well the policy is doing.
 *
 * @note The gradient in backward() is **dense across every action column**, not sparse to the
 *       taken action the way DQNLoss's masked gradient is. This is the single most important
 *       structural difference from DQNLoss and the detail most likely to be gotten wrong by
 *       analogy to it. In DQN the non-taken actions are *unobserved* on a transition, so they
 *       receive exactly zero gradient. Here the non-taken actions are not unobserved at all:
 *       softmax is a normalized distribution, so raising the probability of the taken action
 *       necessarily lowers every other action's probability, and each of those columns gets a
 *       real, non-zero `returns[b] * p[b,k] / N`. A zero there would mean the policy could
 *       increase one action's probability without taking mass from anywhere -- not a
 *       distribution at all.
 * @note The policy network outputs raw logits, and the numerically stable softmax (subtract the
 *       row max before exponentiating) is computed here, fused into the loss -- exactly why
 *       CrossEntropyLoss fuses softmax with NLL rather than composing a SoftmaxModule with a
 *       separate log step. Unlike CrossEntropyLoss, which is single-example and rank-1, this
 *       loss is batched (N, action_dim), matching DQNLoss: its caller computes it once per
 *       whole collected rollout (RolloutBuffer::compute_returns()'s batched output), not once
 *       per step.
 * @note Returns are consumed exactly as given. Variance-reduction tricks -- return
 *       standardization, a learned baseline -- are training-loop-level decisions, and baking
 *       one in here would silently change the estimator every caller gets, the same
 *       "algorithm-agnostic minimum" scope cut RolloutBuffer already made.
 * @note Not a Module subclass, for MSELoss/BCEWithLogitsLoss/DQNLoss's reason: losses are the
 *       seed point relevance/gradient propagation starts *from*, not something a
 *       `propagate_relevance` rule is defined for.
 * @note backward() takes no incoming gradient: like every loss here, this is a graph root.
 */
class PolicyGradientLoss {
public:
    /**
     * @brief Constructs a policy-gradient loss.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this loss.
     */
    explicit PolicyGradientLoss(DeviceBackend* backend);

    /**
     * @brief Computes the return-weighted negative log-likelihood and caches what backward()
     *        needs.
     * @param logits The policy network's raw logits over a whole rollout, shape
     *        (N, action_dim), N >= 1, action_dim >= 1. Raw logits, *not* probabilities.
     * @param actions Float-encoded action indices, shape (N, 1) -- the same discrete-action
     *        encoding Environment::step() accepts and CategoricalPolicyAgent::act() produces,
     *        so a rollout straight out of a RolloutBuffer needs no conversion.
     * @param returns Per-step returns, shape (N, 1) -- typically
     *        RolloutBuffer::compute_returns()'s output.
     * @return The scalar mean return-weighted negative log-probability.
     * @throws std::invalid_argument if any tensor has the wrong rank, if the three batch
     *         dimensions disagree, if an encoded action is not within 1e-4 of a whole number,
     *         or if a decoded index falls outside [0, action_dim) -- all external boundaries.
     *         The integer tolerance is byte-for-byte DQNLoss::forward()'s (itself
     *         CartPoleEnv::step()'s): loose enough to tolerate a policy's float round-trip,
     *         tight enough to catch a genuinely fractional "action".
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in raw host loops
     *       (a row-wise softmax and a per-row gather at a data-dependent column have no
     *       DeviceBackend primitive). PULSATRIX_ASSERT(device() == DeviceType::Cpu) on all three
     *       inputs guards against silent UB on a CUDA-backed Tensor; see
     *       campaign_exai_dl_library_phase1_5_cuda_backend.md's scope decision. Do not remove
     *       these guards without actually retrofitting the method to route through
     *       DeviceBackend.
     */
    [[nodiscard]] float forward(const Tensor& logits, const Tensor& actions, const Tensor& returns);

    /**
     * @brief Gradient w.r.t. `logits`: `returns[b,0] * (p[b,k] - 1{k == a_b}) / N`, the
     *        softmax/cross-entropy gradient identity scaled per row by that row's return.
     * @return Gradient tensor, shape (N, action_dim) -- the shape of the logits passed to
     *         forward(), so it feeds straight into the policy network's own backward().
     * @throws std::logic_error if forward() has never been called -- uses the cached state.
     * @note **Every column is written**, including the non-taken actions, whose gradient is
     *       `returns[b] * p[b,k] / N` and is zero only where the return is zero or the
     *       probability has underflowed. Contrast DQNLoss::backward(), whose non-selected
     *       columns are exactly 0.0f by construction; see this class's note on why the two
     *       differ.
     * @note The `1/N` factor is the batch-mean scaling; N is the number of rollout steps, not
     *       the element count, because the mean is taken over steps (one log-probability each),
     *       not over all N*action_dim logits -- identical reasoning to DQNLoss's `2/N`.
     * @note Also dereferences Tensor::data() directly, but deliberately not independently
     *       device-guarded: it only reads state forward() already validated before caching, and
     *       forward()'s own guard is the only way a non-Cpu tensor could reach that cache -- a
     *       second guard here would be untestable dead code. Identical reasoning to
     *       DQNLoss::backward(); see mission_host_loop_guards.md.
     */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    /** @brief Row-wise softmax probabilities of the last forward()'s logits, shape
     *         (N, action_dim) -- cached rather than the logits themselves, since backward()
     *         needs only `p` and re-deriving it would repeat the whole stabilized softmax. */
    Tensor last_probs_;
    Tensor last_returns_;
    /** @brief Decoded, already-validated action index per rollout step -- decoded once in
     *         forward() rather than re-derived (and re-validated) in backward(). */
    std::vector<int64_t> last_action_indices_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
