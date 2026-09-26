/** @file dqn_target.hpp
 *  @brief DQN Bellman target computation (vanilla + Double DQN) and target-network hard sync.
 *  @ingroup rl
 */
#pragma once

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Vanilla DQN Bellman target (Mnih et al. 2015):
 *        `targets[b,0] = rewards[b,0] + gamma * (1 - dones[b,0]) * max_a next_q_target[b,a]`.
 *
 * @param next_q_target The *target* network's Q-values for the next state, shape
 *        (N, action_dim). Both the action selection (the max) and its evaluation come from
 *        this one tensor -- which is precisely the coupling Double DQN breaks.
 * @param rewards Immediate rewards, shape (N, 1).
 * @param dones Episode-termination flags as 0.0f/1.0f floats, shape (N, 1) -- ReplayBatch's
 *        own encoding, so a sampled batch feeds straight in.
 * @param gamma Discount factor, in [0, 1].
 * @param backend Backend to allocate the result through. Not owned; must outlive the result.
 *        Passed explicitly because Tensor exposes no accessor for the backend it was built
 *        against, and this codebase's convention is an injected, non-owned DeviceBackend*
 *        rather than a global default.
 * @return The Bellman targets, shape (N, 1) -- ready to hand straight to DQNLoss::forward().
 * @throws std::invalid_argument if next_q_target is not rank-2 with N >= 1 and
 *         action_dim >= 1, if rewards/dones are not (N, 1) for that same N, or if gamma is
 *         outside [0, 1] -- all external boundaries.
 * @note A `dones[b,0] == 1` row zeroes the bootstrapped term *exactly*: a terminal state has
 *       no successor whose value could be backed up, so its target is the reward alone. This
 *       is a hard cut, not a heavy discount, and is tested as an exact equality.
 * @note `dones` is used as a plain multiplier, not thresholded -- 0.0f/1.0f is the documented
 *       encoding, and silently reinterpreting anything else would hide a caller's bug.
 * @note Raw host loop over Tensor::data() (a row-wise max at a data-dependent column has no
 *       DeviceBackend primitive) -- PULSATRIX_ASSERT(device() == DeviceType::Cpu) guards against
 *       silent UB on a CUDA-backed Tensor; see mission_host_loop_guards.md.
 */
[[nodiscard]] Tensor ComputeDQNTarget(const Tensor& next_q_target, const Tensor& rewards, const Tensor& dones,
                                      float gamma, DeviceBackend* backend);

/**
 * @brief Double DQN Bellman target (van Hasselt et al. 2016, arXiv:1509.06461):
 *        `a* = argmax_a next_q_online[b,a]`, then
 *        `targets[b,0] = rewards[b,0] + gamma * (1 - dones[b,0]) * next_q_target[b, a*]`.
 *
 * Vanilla DQN both *selects* and *evaluates* the next action with the same network, so any
 * positive noise in a Q-estimate is systematically picked up by the max and propagated into
 * the target -- the well-documented overestimation bias. Double DQN decouples the two: the
 * online network says which action looks best, the target network says what it is worth. The
 * two only agree when the networks agree, and the divergence when they disagree is exactly
 * the property this function exists for (and is directly tested against ComputeDQNTarget).
 *
 * @param next_q_online The *online* network's Q-values for the next state, shape
 *        (N, action_dim) -- used for action selection only.
 * @param next_q_target The *target* network's Q-values for the next state, same shape --
 *        used for evaluation only.
 * @param rewards Immediate rewards, shape (N, 1).
 * @param dones Episode-termination flags as 0.0f/1.0f floats, shape (N, 1).
 * @param gamma Discount factor, in [0, 1].
 * @param backend Backend to allocate the result through. Not owned; must outlive the result.
 * @return The Bellman targets, shape (N, 1).
 * @throws std::invalid_argument on the same conditions as ComputeDQNTarget, plus if
 *         next_q_online and next_q_target have different shapes -- external boundaries.
 * @note Ties in `next_q_online` resolve to the lowest index, matching ComputeDQNTarget's own
 *       max scan and DQNAgent's argmax, so the three are consistent by construction.
 * @note A free function rather than a class or an enum-flagged variant of ComputeDQNTarget:
 *       it is a pure computation with no state to carry, and the extra `next_q_online`
 *       argument -- not a mode flag -- is the whole difference. Mission 1's training loop
 *       picks which of the two to call; nothing downstream needs to branch.
 */
[[nodiscard]] Tensor ComputeDoubleDQNTarget(const Tensor& next_q_online, const Tensor& next_q_target,
                                            const Tensor& rewards, const Tensor& dones, float gamma,
                                            DeviceBackend* backend);

/**
 * @brief Hard target-network update: copies every parameter value of `source` into
 *        `destination`, element-wise and in place (Mnih et al. 2015's periodic full copy).
 *
 * @param source Network to copy parameter values from -- typically the online Q-network.
 * @param destination Network to copy into -- typically the frozen target network.
 * @throws std::invalid_argument if the two networks expose a different number of parameters,
 *         or if any parameter pair's shapes differ, naming the offending index and shapes.
 *         External boundary: two independently-constructed networks genuinely can have
 *         mismatched architectures, and a silent partial copy would leave the target network
 *         quietly wrong for the rest of training.
 * @note A hard copy, not Polyak/soft averaging -- classic DQN. A soft variant would be a
 *       different function with an extra tau, not a flag on this one.
 * @note The copy is element-wise into `destination`'s existing parameter buffers, *not* a
 *       replacement of its Tensor objects. `destination`'s parameters must remain the same
 *       objects its own parameters() (and any optimizer already holding ParamRefs into it)
 *       point at, so subsequent forward passes see the synced values through the same
 *       storage. Replacing the Tensors would dangle every outstanding ParamRef.
 * @note Gradients are untouched. This is a pure value copy and has nothing to do with
 *       zero_grad(); a target network is never backpropagated through in DQN anyway.
 * @note No device guard of its own: it takes no caller-supplied Tensor, only Modules, whose
 *       parameters() buffers were validated when those Modules were constructed. A guard
 *       here would be untestable dead code -- the same reasoning MSELoss::backward()
 *       documents (mission_host_loop_guards.md).
 */
void SyncTargetNetwork(Module& source, Module& destination);

}  // namespace pulsatrix
