/** @file polyak_update.hpp
 *  @brief Soft (Polyak / exponential-moving-average) target-network update, as used by SAC.
 *  @ingroup rl
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Soft target-network update (Lillicrap et al. 2016 / Haarnoja et al. 2018):
 *        `destination_param[i] = tau * source_param[i] + (1 - tau) * destination_param[i]`,
 *        element-wise and in place.
 *
 * A genuinely different mechanism from SyncTargetNetwork's hard periodic copy, not a rename of
 * it. The hard copy leaves the target network frozen for `k` steps and then moves it a long way
 * at once; this moves it a little on *every* step, which is what the off-policy actor-critic
 * algorithms (DDPG/TD3/SAC) depend on for a slowly-drifting bootstrapping target. Both are
 * shipped, neither supersedes the other, and `tau` is not a flag bolted onto the existing
 * function -- the formula, not a mode, is the difference.
 *
 * @param source Network to blend parameter values *from* -- typically the online network.
 * @param destination Network to blend *into* -- typically the target network.
 * @param tau Blending coefficient, in (0, 1]. `tau` near 0 (SAC's usual 0.005) means the
 *        target barely moves per step; `tau == 1` degenerates to exactly SyncTargetNetwork's
 *        hard copy, which is legal (and directly cross-checked in the tests) though an unusual
 *        choice for a soft update.
 * @throws std::invalid_argument if `tau` is not in (0, 1] -- external boundary. `tau == 0` is
 *         rejected rather than accepted as a no-op: a target network that provably never moves
 *         is a real caller error (typically an uninitialized hyperparameter), and silently
 *         doing nothing forever is the worst possible way to report it. NaN is rejected by the
 *         same check.
 * @throws std::invalid_argument if the two networks expose a different number of parameters,
 *         or if any parameter pair's shapes differ, naming the offending index and shapes --
 *         the identical validation SyncTargetNetwork performs, deliberately not weakened.
 *         Two independently-constructed networks genuinely can have mismatched architectures,
 *         and a silent partial blend would leave the target network quietly wrong for the rest
 *         of training. A rejected update leaves the offending parameter untouched.
 * @note The blend writes element-wise into `destination`'s *existing* parameter buffers, never
 *       replacing its Tensor objects -- the same discipline SyncTargetNetwork established.
 *       `destination`'s parameters must remain the same objects its own parameters() (and any
 *       optimizer already holding ParamRefs into it) point at. Replacing the Tensors would
 *       dangle every outstanding ParamRef. Doubly load-bearing here: a soft update reads the
 *       destination's current value as an input, so it is the one place where the destination's
 *       buffer identity across *successive* calls is what makes the exponential moving average
 *       an average at all.
 * @note Gradients are untouched. This is a pure value blend and has nothing to do with
 *       zero_grad(); a target network is never backpropagated through.
 * @note Device-generic (GPU-native-kernels Mission 7): one DeviceBackend::rl_rows(PolyakBlend)
 *       pass per parameter through the destination parameter's own backend, evaluating
 *       `tau * source + (1 - tau) * destination` per element exactly as the original host loop
 *       did. A source parameter on a different device is staged onto the destination's first.
 */
void PolyakUpdate(Module& source, Module& destination, float tau);

}  // namespace pulsatrix
