/** @file sac_continuous_cartpole_training.hpp
 *  @brief The Soft Actor-Critic on-ContinuousCartPoleEnv training loop itself, shared verbatim by
 *         examples/sac_continuous_cartpole_demo.cpp and
 *         tests/sac_continuous_cartpole_integration_test.cpp.
 *
 *  @note Header-only, and deliberately *not* under include/exai/, for exactly the reason
 *        examples/dqn_cartpole_training.hpp, examples/reinforce_cartpole_training.hpp,
 *        examples/a2c_cartpole_training.hpp and examples/ppo_cartpole_training.hpp are not:
 *        mission_sac_continuous_cartpole_training.md's deliverable is a test/demo integration
 *        assembled out of pieces Phase 1 and Phase 4 Missions 0 and 1 already shipped
 *        (ContinuousCartPoleEnv, ReplayBuffer, TanhGaussianPolicy, PolyakUpdate, MSELoss,
 *        SyncTargetNetwork, AdamOptimizer), and nothing here belongs in the library's public
 *        surface. This mission made **no** production change at all -- not even an additive
 *        accessor, unlike the PPO mission's two RolloutBuffer accessors.
 *  @note It lives in one file purely because the mission requires the demo and the test to run
 *        *the same* loop with *the same* hyperparameters and seeds: two hand-copied loops would
 *        drift, and the thresholds the test asserts would then no longer be the thresholds the
 *        demo prints. The test includes it by relative path.
 *  @note Every source of randomness -- all four networks' weight init, the actor's
 *        reparameterization noise, replay sampling, environment reset -- is one of this
 *        codebase's deterministic LCGs with a fixed seed, never `<random>`. The whole run is
 *        therefore bit-reproducible, which is what makes a performance-threshold assertion a
 *        test rather than a coin flip.
 *
 *  ## What is structurally new here, relative to every prior mission in this campaign
 *
 *  This is the campaign's most involved training loop: **four trainable networks** (the actor's
 *  two independent MLPs `mean_network` / `log_std_network`, plus twin critics `Q1` / `Q2`), **two
 *  target networks** (`Q1_target` / `Q2_target`, Polyak-averaged every step rather than
 *  hard-synced periodically), and a **differentiable sampling step** between the actor and its
 *  own loss. Three things follow from that, and all three are load-bearing:
 *
 *  1. **The critic's input is a concatenation.** `Q_i(s, a)` takes one `(N, obs_dim + action_dim)`
 *     row, not two tensors -- a continuous action cannot index a column of a discrete Q-head the
 *     way DQN's does. `concat_state_action()` below is a training-loop-level host-loop copy, the
 *     same disposition as A2C's advantage computation and PPO's advantage standardization.
 *  2. **The actor's gradient arrives through the critics' `backward()`.** `d(min_q)/d(a)` is only
 *     obtainable by running a critic's backward pass and reading the *input* gradient's
 *     action-columns slice. That is the "borrow a network's backward for its input-gradient, not
 *     its parameter-gradient" pattern the GAN mission already found and documented -- and it
 *     carries the same contamination hazard, resolved the same way. See
 *     min_selected_action_gradient()'s own note, and the deliberately *separate* zero_grad calls
 *     at the call site (the fix is the caller's, so that a regression test can omit it).
 *  3. **Per-row min-selection.** `min(Q1, Q2)` is selected independently per row, so each
 *     critic's borrowed backward receives nonzero gradient only in the rows it won. Summing both
 *     resulting action-gradient slices therefore gives the correct per-row total with no
 *     double-counting -- the same per-row masking precedent DQN's masked gradient and PPO's clip
 *     mask established.
 *
 *  ## What SAC deliberately does *not* need
 *
 *  `ReplayBuffer` is reused completely unchanged, and in particular stores no log-probability.
 *  SAC recomputes everything -- both critic targets and the actor's own objective -- from the
 *  *current* actor and critics on each sampled `(s, a, r, s', done)` tuple. It never needs the
 *  log-probability the *behavior* policy assigned to the stored action, which is exactly what
 *  distinguishes it from PPO's `RolloutBuffer`.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

#include "exai/adam_optimizer.hpp"
#include "exai/continuous_cartpole_env.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/dqn_target.hpp"
#include "exai/linear_module.hpp"
#include "exai/module.hpp"
#include "exai/mse_loss.hpp"
#include "exai/polyak_update.hpp"
#include "exai/relu_module.hpp"
#include "exai/replay_buffer.hpp"
#include "exai/sequential_module.hpp"
#include "exai/shape.hpp"
#include "exai/tanh_gaussian_policy.hpp"
#include "exai/tensor.hpp"

namespace exai {
namespace sac_continuous_cartpole {

/**
 * @brief Episodes averaged at each end of training to form the performance comparison.
 * @note DQN's fixed window of 20, not REINFORCE/A2C/PPO's 10%-of-the-run fraction -- and the
 *       mission states which and why: SAC updates every *environment step*, like DQN, rather
 *       than once per rollout, so the episode (not the rollout) is already the natural unit of
 *       progress and a fixed window is the right comparison.
 */
inline constexpr int64_t kWindow = 20;

/**
 * @brief Every hyperparameter and seed of the training run, in one place.
 * @note The defaults are the tuned configuration that actually clears both of the mission's
 *       fixed bars (last-20 average >= 3x first-20 average, and >= 30% of max_steps). Demo and
 *       test both take them as-is; neither overrides anything. See the TUNING NOTE at the bottom
 *       of this file for what was measured to arrive at them.
 */
struct TrainingConfig {
    /** @brief Episode length limit. The absolute bar is expressed relative to this. */
    int64_t max_steps = 200;
    /**
     * @brief Width of the single hidden layer of *every* network here -- both actor MLPs, both
     *        critics and both targets. One knob rather than PPO's separate actor/critic widths:
     *        with four trainable networks, per-network widths would be four knobs whose
     *        interaction this demo has no way to interpret, and the targets' widths are not free
     *        at all (PolyakUpdate rejects a parameter-shape mismatch, correctly).
     */
    int64_t hidden_size = 64;
    /** @brief Total training episodes. */
    int64_t num_episodes = 150;
    /** @brief Minibatch size drawn from the replay buffer per update. */
    int64_t batch_size = 64;
    /**
     * @brief Replay transitions that must be stored before the first gradient update runs.
     *        Distinct from batch_size (DQN's warm-up *was* its batch size): SAC's critic
     *        bootstraps off a target built from a freshly-sampled actor action, so updating from
     *        a 64-transition buffer means fitting twin critics to a target dominated by the first
     *        second of a single episode. A separate, larger warm-up was measured to matter; see
     *        the TUNING NOTE.
     */
    int64_t warmup_size = 1000;
    /** @brief Replay capacity -- large enough that nothing is evicted on a run this short. */
    int64_t replay_capacity = 100000;
    /** @brief Adam step size for both actor MLPs (one optimizer, both networks -- see RunTraining). */
    float actor_learning_rate = 3e-4f;
    /**
     * @brief Adam step size for both critics. A separate knob from the actor's, and tuned
     *        separately, for the reason A2C/PPO already recorded: a critic solves a supervised
     *        regression against a moving target and tolerates a larger step than a policy whose
     *        every parameter change shifts the data distribution.
     */
    float critic_learning_rate = 1e-3f;
    /** @brief Discount factor, used only for the Bellman target `y`. */
    float gamma = 0.99f;
    /**
     * @brief Polyak blending coefficient, applied to *both* targets on *every* update step.
     *        SAC's standard 0.005. Not DQN's periodic hard sync -- see PolyakUpdate's own header
     *        for why the two are genuinely different mechanisms rather than one with a flag.
     */
    float tau = 0.005f;
    /**
     * @brief Fixed entropy temperature. Not auto-tuned -- the campaign doc's Phase 4 Recon scope
     *        cut, so this is a constructor-level hyperparameter exactly like gamma.
     * @note Must be read against the *bounded* scale of this policy's log-probability: the
     *       tanh squash-correction term is capped at `-log(kLogProbStabilizer) ~= 13.8`, so the
     *       largest entropy bonus alpha can inject into a Bellman target is ~13.8*alpha against
     *       a per-step reward of exactly 1.0. That is why the tuned value is well below the
     *       literature's usual 0.2 on unbounded-reward benchmarks; see the TUNING NOTE.
     */
    float alpha = 0.05f;

    uint32_t critic_weight_seed = 12345u;
    uint32_t actor_weight_seed = 54321u;
    uint32_t env_seed = 7u;
    uint32_t buffer_seed = 2024u;
    uint32_t noise_seed = 99u;
};

/** @brief Everything a caller (demo or test) needs to report on or assert against. */
struct TrainingResult {
    /** @brief Length of every completed episode, in order. */
    std::vector<int64_t> episode_lengths;
    /** @brief Episodes averaged at each end -- min(kWindow, episode count). */
    int64_t window = 0;
    /** @brief Mean length of the first `window` episodes. */
    double first_window_average = 0.0;
    /** @brief Mean length of the last `window` episodes. */
    double last_window_average = 0.0;

    /** @brief Environment steps taken over the whole run. */
    int64_t total_steps = 0;
    /** @brief Gradient updates performed (one per environment step past the warm-up). */
    int64_t update_count = 0;
    /** @brief PolyakUpdate() calls per target network -- one per update by construction. */
    int64_t polyak_count = 0;

    /** @brief False if any single update's critic1, critic2 *or* actor loss was non-finite. */
    bool all_losses_finite = true;
    /** @brief Index of the first update at which any of the three losses was non-finite, or -1. */
    int64_t first_non_finite_update = -1;

    /** @brief Each of the three losses at every update, in order. */
    std::vector<float> critic1_losses;
    std::vector<float> critic2_losses;
    std::vector<float> actor_losses;

    float first_critic1_loss = 0.0f;
    float last_critic1_loss = 0.0f;
    float first_critic2_loss = 0.0f;
    float last_critic2_loss = 0.0f;
    float first_actor_loss = 0.0f;
    float last_actor_loss = 0.0f;

    /** @brief Mean log-probability of the actor's sampled batch action, per update -- the
     *         quantity the entropy term prices. Rises toward 0 as the policy sharpens. */
    std::vector<float> mean_log_probs;
    /** @brief Mean `exp(log_std)` over the batch, per update -- the policy's own exploration
     *         scale, which nothing external schedules (contrast DQN's epsilon schedule). */
    std::vector<float> mean_stds;
    /** @brief Fraction of rows in which Q1 (not Q2) was the min-selected critic, per update.
     *         Near 0 or 1 everywhere would mean one critic is dead weight. */
    std::vector<double> q1_selected_fractions;

    /** @brief Both critics' and both targets' first-layer weight buffers, before/after training. */
    std::vector<float> initial_q1_weight;
    std::vector<float> final_q1_weight;
    std::vector<float> initial_q1_target_weight;
    std::vector<float> final_q1_target_weight;
    std::vector<float> initial_q2_target_weight;
    std::vector<float> final_q2_target_weight;
    /** @brief Both actor MLPs' first-layer weight buffers, before/after training. */
    std::vector<float> initial_mean_weight;
    std::vector<float> final_mean_weight;
    std::vector<float> initial_log_std_weight;
    std::vector<float> final_log_std_weight;

    double max_q1_weight_delta = 0.0;
    /** @brief max |final - initial| over Q1_target's first layer -- nonzero only if
     *         PolyakUpdate genuinely moved it. */
    double max_q1_target_weight_delta = 0.0;
    double max_q2_target_weight_delta = 0.0;
    double max_mean_weight_delta = 0.0;
    double max_log_std_weight_delta = 0.0;

    /**
     * @brief max |Q1_target param - Q1 param| at the end of training, over every parameter.
     *        A soft update *lags*: the target must have moved (above) and must still not have
     *        caught up (this), or tau is doing nothing recognizable as an exponential average.
     */
    double max_target_lag = 0.0;
};

/**
 * @brief Deterministic small-random values in [-0.1, 0.1] from this codebase's standard
 *        Numerical-Recipes LCG -- the same initialization pattern every prior training mission in
 *        this campaign, ToyGAN and ToyVAE use.
 * @note Required for all four networks, not merely conventional: LinearModule zero-initializes,
 *       and a zero-initialized ReLU network has a dead hidden layer whose first-layer weights
 *       receive exactly zero gradient forever. For `log_std_network` specifically, note what an
 *       all-zero network means here -- `log_std == 0`, i.e. `std == 1` -- which is a *usable*
 *       starting exploration scale rather than a degenerate one; it is still initialized randomly
 *       so that its first layer can learn at all.
 */
inline std::vector<float> lcg_weights(size_t count, uint32_t seed) {
    std::vector<float> values;
    values.reserve(count);
    uint32_t state = seed;
    for (size_t i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        const float unit = static_cast<float>((state >> 8) & 0xFFFFu) / 65535.0f;  // [0, 1]
        values.push_back((unit - 0.5f) * 0.2f);                                    // [-0.1, 0.1]
    }
    return values;
}

/** @brief Flattened copy of a Tensor's values. */
inline std::vector<float> tensor_values(const Tensor& t) {
    return std::vector<float>(t.data(), t.data() + t.numel());
}

/** @brief Flattened copy of every parameter *value* a module exposes, in parameters() order. */
inline std::vector<float> parameter_values(Module& module) {
    std::vector<float> out;
    for (const ParamRef& p : module.parameters()) {
        out.insert(out.end(), p.value->data(), p.value->data() + p.value->numel());
    }
    return out;
}

/** @brief Flattened copy of every parameter *gradient* a module exposes, in parameters() order.
 *         The regression test reads contamination out of exactly this. */
inline std::vector<float> parameter_grads(Module& module) {
    std::vector<float> out;
    for (const ParamRef& p : module.parameters()) {
        out.insert(out.end(), p.grad->data(), p.grad->data() + p.grad->numel());
    }
    return out;
}

/**
 * @brief Deterministic standard-normal noise source for the reparameterization step.
 *
 * TanhGaussianPolicy takes `epsilon` as a caller-supplied input on purpose (this codebase's
 * "randomness is a deterministic, testable input" convention, shared with Reparameterize and
 * NoiseSchedule), so *something* at the training-loop level has to produce it. Nothing in the
 * library does: every prior use site either hand-wrote fixed epsilon values or, in ToyVAE's case,
 * used the plain *uniform* LCG draw, which is not a standard normal and would silently mis-scale
 * SAC's log-density.
 *
 * @note Box-Muller on top of the same Numerical-Recipes LCG (1664525 / 1013904223) every other
 *       deterministic sampler here uses -- never `<random>`, whose engines and distributions are
 *       implementation-defined and would make the threshold assertions unreproducible across
 *       toolchains.
 * @note One normal per *two* uniforms, using only the cosine branch and discarding the sine one.
 *       Box-Muller's second output is free, but keeping it would require carrying a one-value
 *       cache across calls, which makes the generator's state -- and therefore the whole run's
 *       reproducibility -- depend on how many draws each call happened to ask for. Discarding it
 *       makes the state a pure function of the number of values drawn so far.
 * @note The uniforms are mapped to the *open* interval (0, 1) as `(raw + 1) / 65537`, not the
 *       closed [0, 1] the weight initializer uses: `log(0)` is -inf and would produce an infinite
 *       epsilon, poisoning every downstream tensor in one step.
 */
struct NormalNoise {
    uint32_t state;

    explicit NormalNoise(uint32_t seed) : state(seed) {}

    /** @brief Next uniform draw, strictly inside (0, 1). */
    [[nodiscard]] double next_open_unit() {
        state = state * 1664525u + 1013904223u;
        return (static_cast<double>((state >> 8) & 0xFFFFu) + 1.0) / 65537.0;
    }

    /** @brief One standard-normal draw. */
    [[nodiscard]] float next_normal() {
        const double u1 = next_open_unit();
        const double u2 = next_open_unit();
        constexpr double kTwoPi = 6.28318530717958647692;
        return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2));
    }

    /** @brief A tensor of the given shape filled with independent standard-normal draws. */
    [[nodiscard]] Tensor sample(const Shape& shape, DeviceBackend* backend) {
        Tensor out(shape, backend);
        for (int64_t i = 0; i < out.numel(); ++i) {
            out.data()[i] = next_normal();
        }
        return out;
    }
};

/**
 * @brief `[observations | actions]` as one `(N, obs_dim + action_dim)` block -- the single input a
 *        continuous-action `Q(s, a)` network takes.
 * @param observations `(N, obs_dim)`.
 * @param actions `(N, action_dim)`, with the same N.
 * @param backend Backend to allocate the result through.
 * @note A plain host-loop row copy, deliberately *not* a new production class. DQN never needed
 *       this because a discrete Q-network emits one column per action and selects by index; a
 *       continuous action has to be *fed in*, and this is the whole of what that costs. Same
 *       disposition as A2C's advantage computation and PPO's advantage standardization.
 * @note Observation columns come first and action columns last, and that ordering is a contract
 *       between this function and action_columns() below -- the two are the forward and backward
 *       halves of the same layout decision.
 */
inline Tensor concat_state_action(const Tensor& observations, const Tensor& actions, DeviceBackend* backend) {
    const int64_t n = observations.shape().dim(0);
    const int64_t obs_dim = observations.shape().dim(1);
    const int64_t action_dim = actions.shape().dim(1);
    Tensor out(Shape({n, obs_dim + action_dim}), backend);
    for (int64_t row = 0; row < n; ++row) {
        float* dst = out.data() + row * (obs_dim + action_dim);
        const float* obs_src = observations.data() + row * obs_dim;
        const float* act_src = actions.data() + row * action_dim;
        for (int64_t c = 0; c < obs_dim; ++c) {
            dst[c] = obs_src[c];
        }
        for (int64_t c = 0; c < action_dim; ++c) {
            dst[obs_dim + c] = act_src[c];
        }
    }
    return out;
}

/**
 * @brief The action-columns slice of a critic's input-gradient -- the inverse of
 *        concat_state_action()'s layout.
 * @param grad_input `(N, obs_dim + action_dim)`, as returned by `Q_i.backward(...)`.
 * @param obs_dim How many leading columns to skip.
 * @param backend Backend to allocate the result through.
 * @return `(N, action_dim)`, the gradient of whatever the critic's output was differentiated
 *         against, w.r.t. the action fed into it.
 */
inline Tensor action_columns(const Tensor& grad_input, int64_t obs_dim, DeviceBackend* backend) {
    const int64_t n = grad_input.shape().dim(0);
    const int64_t width = grad_input.shape().dim(1);
    const int64_t action_dim = width - obs_dim;
    Tensor out(Shape({n, action_dim}), backend);
    for (int64_t row = 0; row < n; ++row) {
        const float* src = grad_input.data() + row * width + obs_dim;
        float* dst = out.data() + row * action_dim;
        for (int64_t c = 0; c < action_dim; ++c) {
            dst[c] = src[c];
        }
    }
    return out;
}

/**
 * @brief `mean_n( alpha*log_prob[n] - min(q1[n], q2[n]) )` -- SAC's actor objective, exactly as
 *        the mission states it.
 * @param q1 `(N, 1)` Q1 evaluated at the *freshly sampled* action.
 * @param q2 `(N, 1)` Q2 evaluated at the same action.
 * @param log_prob `(N, 1)` that action's log-density, from TanhGaussianPolicy::forward().
 * @param alpha Entropy temperature.
 * @note Factored out of the loop precisely so the entropy term can be checked *as a term*: the
 *       mission requires a test confirming `actor_loss(alpha = 0.5) - actor_loss(alpha = 0.0)`
 *       equals the hand-computable `0.5 * mean(log_prob)` exactly. That is a statement about this
 *       function, and only a separately-callable function can carry it.
 * @note Accumulated in double, then narrowed once -- the same disposition
 *       TanhGaussianPolicy::forward() applies to its own per-row sum. The two terms differ in
 *       magnitude by ~two orders of magnitude at the tuned alpha (Q values reach ~1/(1-gamma) on
 *       a reward-1-per-step task), so a float accumulator would lose the entropy term's low bits
 *       and make the exactness the test above asserts a matter of luck.
 */
[[nodiscard]] inline float actor_loss_value(const Tensor& q1, const Tensor& q2, const Tensor& log_prob, float alpha) {
    const int64_t n = log_prob.numel();
    if (n == 0) {
        return 0.0f;
    }
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double min_q = static_cast<double>(q1.data()[i] <= q2.data()[i] ? q1.data()[i] : q2.data()[i]);
        sum += static_cast<double>(alpha) * static_cast<double>(log_prob.data()[i]) - min_q;
    }
    return static_cast<float>(sum / static_cast<double>(n));
}

/** @brief The per-row `min`-selection mask and the action-gradient it produces. */
struct MinSelectedActionGradient {
    /** @brief `(N, action_dim)` -- `d(actor_loss)/d(action)`, summed over both critics. */
    Tensor grad_action;
    /** @brief Fraction of rows in which Q1 was the selected (smaller) critic. */
    double q1_selected_fraction = 0.0;
};

/**
 * @brief Runs both critics' backward passes to extract `d(actor_loss)/d(action)`, masked per row
 *        by which critic the `min` selected.
 *
 * @param q1 Online critic 1. Its most recent forward() must be the one at the sampled action.
 * @param q2 Online critic 2, same requirement.
 * @param q1_values `(N, 1)` q1's output at the sampled action -- used only to decide the mask.
 * @param q2_values `(N, 1)` q2's output at the same action.
 * @param obs_dim Leading columns of the critics' input that are observation, not action.
 * @param backend Backend to allocate through.
 * @return The summed action-gradient and the mask's Q1 share.
 *
 * @warning **THIS FUNCTION CONTAMINATES `q1` AND `q2`'s PARAMETER GRADIENTS, BY CONSTRUCTION.**
 *          `Module::backward()` has exactly one way to hand back an input-gradient, and it
 *          *always* accumulates into the module's own parameter gradients as a side effect. Here
 *          only the input-gradient is wanted; the parameter-gradient it deposits belongs to the
 *          *actor's* objective and must never reach a critic parameter update. The caller must
 *          therefore call `critic_optimizer.zero_grad(q1)` / `zero_grad(q2)` immediately after
 *          this returns, before either critic's next real training step.
 *
 *          This is the identical hazard the GAN mission (Phase 6 Phase 5 Mission 4) found and
 *          documented for its generator step routing gradient through the discriminator, and it
 *          is resolved the identical way. The fix is deliberately left to the *caller* rather
 *          than performed here, for one reason: a regression test has to be able to omit it and
 *          measure what the omission costs. `tests/sac_continuous_cartpole_integration_test.cpp`
 *          does exactly that, and it is why the parameter is not merely a documented convention.
 *
 * @note The `-1/N` in the mask is `d(actor_loss)/d(min_q[n])`: the loss subtracts `min_q` and
 *       averages over N rows. The *un*selected critic receives a structural zero in that row, so
 *       summing the two resulting action-gradient slices reconstructs the correct per-row total
 *       with no double-counting -- which is only true because `min` is selecting, not blending.
 * @note Ties (`q1[n] == q2[n]`) go to Q1, arbitrarily but deterministically. `min` is not
 *       differentiable at a tie; picking a side is the standard subgradient convention, and the
 *       same `<=` tie-break actor_loss_value() uses, so the loss and its gradient always agree
 *       about which critic a tied row belongs to.
 */
[[nodiscard]] inline MinSelectedActionGradient min_selected_action_gradient(Module& q1, Module& q2,
                                                                           const Tensor& q1_values,
                                                                           const Tensor& q2_values, int64_t obs_dim,
                                                                           DeviceBackend* backend) {
    const int64_t n = q1_values.shape().dim(0);
    const float scale = -1.0f / static_cast<float>(n);

    Tensor grad_q1(q1_values.shape(), backend);
    Tensor grad_q2(q2_values.shape(), backend);
    int64_t q1_selected = 0;
    for (int64_t i = 0; i < n; ++i) {
        const bool use_q1 = q1_values.data()[i] <= q2_values.data()[i];
        grad_q1.data()[i] = use_q1 ? scale : 0.0f;
        grad_q2.data()[i] = use_q1 ? 0.0f : scale;
        q1_selected += use_q1 ? 1 : 0;
    }

    const Tensor q1_input_grad = q1.backward(grad_q1);
    const Tensor q2_input_grad = q2.backward(grad_q2);
    const Tensor from_q1 = action_columns(q1_input_grad, obs_dim, backend);
    const Tensor from_q2 = action_columns(q2_input_grad, obs_dim, backend);

    Tensor grad_action(from_q1.shape(), backend);
    for (int64_t i = 0; i < grad_action.numel(); ++i) {
        grad_action.data()[i] = from_q1.data()[i] + from_q2.data()[i];
    }

    return MinSelectedActionGradient{std::move(grad_action),
                                     static_cast<double>(q1_selected) / static_cast<double>(n)};
}

/**
 * @brief Builds the `(N, action_dim)` broadcast `grad_log_prob` TanhGaussianPolicy::backward()
 *        asks for: the constant `alpha / N` repeated across every column of every row.
 * @note The shape is deliberately `(N, action_dim)`, not log_prob's own `(N, 1)` -- backward()'s
 *       documented convention, because log_prob is a *sum* over action_dim and every element of a
 *       row therefore receives the identical incoming scalar. Passing `(N, 1)` throws.
 * @note `alpha / N` and not `alpha`: the `1/N` is the mean in the actor loss, the same `1/N` the
 *       min-selection mask carries on its own branch.
 */
[[nodiscard]] inline Tensor entropy_gradient(int64_t n, int64_t action_dim, float alpha, DeviceBackend* backend) {
    Tensor out(Shape({n, action_dim}), backend);
    const float value = alpha / static_cast<float>(n);
    for (int64_t i = 0; i < out.numel(); ++i) {
        out.data()[i] = value;
    }
    return out;
}

/**
 * @brief Runs the full SAC training loop on ContinuousCartPoleEnv and returns its statistics.
 * @param config Hyperparameters and seeds.
 * @param on_episode Optional per-episode callback (episode index, episode length) -- the demo's
 *        progress printing, nothing the test needs.
 *
 * @note Algorithm is exactly mission_sac_continuous_cartpole_training.md's stated loop. Per
 *       environment step: sample an action from the current actor with fresh noise, step the
 *       environment, store the transition; then, once `warmup_size` transitions exist, sample a
 *       batch and run (1) both critics' Bellman regression, (2) the actor's entropy-regularized
 *       objective, (3) a Polyak blend into both targets.
 *
 * @note **Ordering is load-bearing in three separate places**, and every one of them is a
 *       consequence of `Module::backward()` consuming the most recent `forward()`'s cache:
 *       - The Bellman target `y` is built *before* either critic is stepped, from the *target*
 *         critics and a fresh actor sample of `s'`. Building it after would regress each critic
 *         against a target contaminated by the other's update.
 *       - Each critic's `forward -> MSELoss -> backward -> step -> zero_grad` is one complete,
 *         uninterrupted cycle. Q1 and Q2 are separate Module objects with separate caches, so the
 *         two cycles cannot interfere -- but the *same* critic's target-building forward and its
 *         prediction forward would, which is why the targets are built through `Q1_target` /
 *         `Q2_target` and never through the online critics.
 *       - The actor update's `mean_network.forward()` / `log_std_network.forward()` precede the
 *         critics' forward *and* backward passes, and nothing between them touches either actor
 *         MLP -- so their caches survive intact until their own `backward()` calls at the end.
 *
 * @note **The contamination fix.** `min_selected_action_gradient()` borrows both critics'
 *       backward passes for their input-gradients and unavoidably deposits the actor objective's
 *       gradient into the critics' parameter buffers. The two `critic_optimizer.zero_grad()`
 *       calls immediately after it are the fix, and they are the *entire* reason the next
 *       iteration's critic step sees a clean gradient. See min_selected_action_gradient()'s
 *       warning, and the regression test that removes them.
 *
 * @note **Three separate TanhGaussianPolicy instances**, not one. The class caches its forward
 *       for its backward, and three distinct sampling steps are live in a single iteration: the
 *       environment-interaction sample, the target-building sample of `s'`, and the actor
 *       update's sample of `s`. Only the third's cache is ever consumed by a `backward()`, but
 *       sharing one instance would mean the other two silently overwrite it -- a bug that
 *       produces plausible-looking numbers rather than an error.
 *
 * @note **One Adam optimizer per role, not per network.** `actor_optimizer` steps both actor
 *       MLPs and `critic_optimizer` steps both critics. That is safe and not a shortcut: Adam's
 *       per-parameter moment state is keyed by the parameter Tensor's *pointer identity*
 *       (AdamOptimizer's own documented design), and four independently-constructed networks own
 *       four disjoint sets of Tensors. Two networks sharing an optimizer object therefore share
 *       no state whatsoever -- only the learning rate, which is exactly the intent.
 */
inline TrainingResult RunTraining(const TrainingConfig& config,
                                  const std::function<void(int64_t, int64_t)>& on_episode = {}) {
    CPUBackend backend;
    ContinuousCartPoleEnv env(&backend, config.max_steps, config.env_seed);

    const int64_t obs_dim = env.observation_dim();
    const int64_t action_dim = env.action_dim();
    const int64_t critic_in_dim = obs_dim + action_dim;

    // ---- actor: two independent single-hidden-layer MLPs, NOT one two-headed network ----
    // Mission 1's own scope note: a shared trunk would need gradient-merging plumbing
    // Module::backward() does not provide for free (two heads' gradients would both have to flow
    // into one trunk backward call), whereas two independent networks need none at all.
    LinearModule mean_in(obs_dim, config.hidden_size, &backend);
    ReluModule mean_relu(&backend);
    LinearModule mean_out(config.hidden_size, action_dim, &backend);
    SequentialModule mean_network({&mean_in, &mean_relu, &mean_out});

    LinearModule log_std_in(obs_dim, config.hidden_size, &backend);
    ReluModule log_std_relu(&backend);
    LinearModule log_std_out(config.hidden_size, action_dim, &backend);
    SequentialModule log_std_network({&log_std_in, &log_std_relu, &log_std_out});

    // ---- critics: twin Q(s, a) networks over the concatenated (obs | action) row ----
    LinearModule q1_in(critic_in_dim, config.hidden_size, &backend);
    ReluModule q1_relu(&backend);
    LinearModule q1_out(config.hidden_size, 1, &backend);
    SequentialModule q1({&q1_in, &q1_relu, &q1_out});

    LinearModule q2_in(critic_in_dim, config.hidden_size, &backend);
    ReluModule q2_relu(&backend);
    LinearModule q2_out(config.hidden_size, 1, &backend);
    SequentialModule q2({&q2_in, &q2_relu, &q2_out});

    LinearModule q1_target_in(critic_in_dim, config.hidden_size, &backend);
    ReluModule q1_target_relu(&backend);
    LinearModule q1_target_out(config.hidden_size, 1, &backend);
    SequentialModule q1_target({&q1_target_in, &q1_target_relu, &q1_target_out});

    LinearModule q2_target_in(critic_in_dim, config.hidden_size, &backend);
    ReluModule q2_target_relu(&backend);
    LinearModule q2_target_out(config.hidden_size, 1, &backend);
    SequentialModule q2_target({&q2_target_in, &q2_target_relu, &q2_target_out});

    // Independent initializations for the two critics: twin critics whose *entire point* is
    // decorrelated overestimation error would be a single critic computed twice if they started
    // identical and saw identical batches, which they do.
    q1_in.set_weight(lcg_weights(static_cast<size_t>(critic_in_dim * config.hidden_size), config.critic_weight_seed));
    q1_out.set_weight(lcg_weights(static_cast<size_t>(config.hidden_size), config.critic_weight_seed + 1u));
    q2_in.set_weight(lcg_weights(static_cast<size_t>(critic_in_dim * config.hidden_size), config.critic_weight_seed + 2u));
    q2_out.set_weight(lcg_weights(static_cast<size_t>(config.hidden_size), config.critic_weight_seed + 3u));

    mean_in.set_weight(lcg_weights(static_cast<size_t>(obs_dim * config.hidden_size), config.actor_weight_seed));
    mean_out.set_weight(lcg_weights(static_cast<size_t>(config.hidden_size * action_dim), config.actor_weight_seed + 1u));
    log_std_in.set_weight(lcg_weights(static_cast<size_t>(obs_dim * config.hidden_size), config.actor_weight_seed + 2u));
    log_std_out.set_weight(
        lcg_weights(static_cast<size_t>(config.hidden_size * action_dim), config.actor_weight_seed + 3u));

    // Each target starts as an exact hard copy of the critic it shadows -- a snapshot that never
    // matched what it snapshots would be a modeling error, exactly as in the DQN mission. This is
    // SyncTargetNetwork's job, not PolyakUpdate's: `tau = 1` would be equivalent (and is
    // cross-checked in polyak_update_test.cpp) but says something different about intent.
    SyncTargetNetwork(q1, q1_target);
    SyncTargetNetwork(q2, q2_target);

    ReplayBuffer replay(config.replay_capacity, obs_dim, action_dim, &backend, config.buffer_seed);
    // Three policies, three distinct live caches -- see RunTraining's note.
    TanhGaussianPolicy act_policy(&backend);
    TanhGaussianPolicy target_policy(&backend);
    TanhGaussianPolicy actor_policy(&backend);
    MSELoss critic1_loss_fn(&backend);
    MSELoss critic2_loss_fn(&backend);
    AdamOptimizer actor_optimizer(config.actor_learning_rate, &backend);
    AdamOptimizer critic_optimizer(config.critic_learning_rate, &backend);
    NormalNoise noise(config.noise_seed);

    TrainingResult result;
    result.initial_q1_weight = tensor_values(q1_in.weight());
    result.initial_q1_target_weight = tensor_values(q1_target_in.weight());
    result.initial_q2_target_weight = tensor_values(q2_target_in.weight());
    result.initial_mean_weight = tensor_values(mean_in.weight());
    result.initial_log_std_weight = tensor_values(log_std_in.weight());

    for (int64_t episode = 0; episode < config.num_episodes; ++episode) {
        Tensor observation = env.reset();
        int64_t episode_length = 0;

        for (int64_t t = 0; t < config.max_steps; ++t) {
            // ---- (0) act: a fresh reparameterized sample from the current policy ----
            // No external exploration schedule at all (contrast DQN's epsilon-greedy): the
            // policy's own learned log_std *is* the exploration, which is what the entropy term
            // in the actor objective is there to keep from collapsing.
            const Tensor act_mean = mean_network.forward(observation);
            const Tensor act_log_std = log_std_network.forward(observation);
            const Tensor act_epsilon = noise.sample(act_mean.shape(), &backend);
            const TanhGaussianSample act_sample = act_policy.forward(act_mean, act_log_std, act_epsilon);

            const StepResult step_result = env.step(act_sample.action);
            replay.add(observation, act_sample.action, step_result.reward, step_result.observation, step_result.done);
            observation = step_result.observation;
            ++episode_length;
            ++result.total_steps;

            if (replay.size() >= config.warmup_size) {
                const ReplayBatch batch = replay.sample(config.batch_size);
                const int64_t n = config.batch_size;

                // ---- (1) critic update ----
                // The target's action comes from a *fresh* sample of s' under the current actor,
                // not from the replay's stored action. That is what makes this SAC's soft Bellman
                // backup rather than a plain off-policy TD(0) backup.
                const Tensor next_mean = mean_network.forward(batch.next_observations);
                const Tensor next_log_std = log_std_network.forward(batch.next_observations);
                const Tensor next_epsilon = noise.sample(next_mean.shape(), &backend);
                const TanhGaussianSample next_sample =
                    target_policy.forward(next_mean, next_log_std, next_epsilon);

                const Tensor next_input = concat_state_action(batch.next_observations, next_sample.action, &backend);
                const Tensor q1_target_next = q1_target.forward(next_input);
                const Tensor q2_target_next = q2_target.forward(next_input);

                Tensor y(Shape({n, 1}), &backend);
                for (int64_t i = 0; i < n; ++i) {
                    const float min_q = q1_target_next.data()[i] <= q2_target_next.data()[i]
                                            ? q1_target_next.data()[i]
                                            : q2_target_next.data()[i];
                    // The entropy bonus rides *inside* the bootstrap, so it is discounted and
                    // propagated exactly like reward -- SAC's maximum-entropy objective, not a
                    // reward-shaping bolt-on.
                    const float soft_min_q = min_q - config.alpha * next_sample.log_prob.data()[i];
                    // (1 - done) zeroes the bootstrap exactly, the same convention every prior
                    // Bellman-style target in this campaign uses.
                    y.data()[i] = batch.rewards.data()[i] +
                                  config.gamma * (1.0f - batch.dones.data()[i]) * soft_min_q;
                }

                // The *stored*, behavior-policy actions -- this is what makes the critic update
                // off-policy-correct.
                const Tensor pred_input = concat_state_action(batch.observations, batch.actions, &backend);

                const Tensor q1_pred = q1.forward(pred_input);
                const float critic1_loss = critic1_loss_fn.forward(q1_pred, y);
                (void)q1.backward(critic1_loss_fn.backward());
                critic_optimizer.step(q1);
                critic_optimizer.zero_grad(q1);

                const Tensor q2_pred = q2.forward(pred_input);
                const float critic2_loss = critic2_loss_fn.forward(q2_pred, y);
                (void)q2.backward(critic2_loss_fn.backward());
                critic_optimizer.step(q2);
                critic_optimizer.zero_grad(q2);

                // ---- (2) actor update ----
                // A fresh sample of s, *not* the replay's stored action: the actor's objective is
                // about what it would do now, and the reparameterization is what makes that
                // sample differentiable w.r.t. its own parameters.
                const Tensor mean = mean_network.forward(batch.observations);
                const Tensor log_std = log_std_network.forward(batch.observations);
                const Tensor epsilon = noise.sample(mean.shape(), &backend);
                const TanhGaussianSample sample = actor_policy.forward(mean, log_std, epsilon);

                const Tensor new_input = concat_state_action(batch.observations, sample.action, &backend);
                const Tensor q1_new = q1.forward(new_input);
                const Tensor q2_new = q2.forward(new_input);
                const float actor_loss = actor_loss_value(q1_new, q2_new, sample.log_prob, config.alpha);

                const MinSelectedActionGradient selected =
                    min_selected_action_gradient(q1, q2, q1_new, q2_new, obs_dim, &backend);

                // ***THE FIX***, and nothing else in this loop performs it. The two backward()
                // calls inside min_selected_action_gradient() deposited the *actor's* objective
                // into the critics' parameter-gradient buffers as an unavoidable side effect of
                // asking for an input-gradient. Zeroing here, before either critic's next real
                // step, is what keeps that contamination out of a critic parameter update.
                // tests/sac_continuous_cartpole_integration_test.cpp removes these two lines and
                // measures exactly what they are worth.
                critic_optimizer.zero_grad(q1);
                critic_optimizer.zero_grad(q2);

                const Tensor grad_log_prob = entropy_gradient(n, action_dim, config.alpha, &backend);
                const TanhGaussianGrad actor_grad = actor_policy.backward(selected.grad_action, grad_log_prob);
                (void)mean_network.backward(actor_grad.grad_mean);
                (void)log_std_network.backward(actor_grad.grad_log_std);
                actor_optimizer.step(mean_network);
                actor_optimizer.step(log_std_network);
                actor_optimizer.zero_grad(mean_network);
                actor_optimizer.zero_grad(log_std_network);

                // ---- (3) soft target update, every step (not periodic) ----
                PolyakUpdate(q1, q1_target, config.tau);
                PolyakUpdate(q2, q2_target, config.tau);
                ++result.polyak_count;

                // ---- bookkeeping ----
                if (!std::isfinite(critic1_loss) || !std::isfinite(critic2_loss) || !std::isfinite(actor_loss)) {
                    result.all_losses_finite = false;
                    if (result.first_non_finite_update < 0) {
                        result.first_non_finite_update = result.update_count;
                    }
                }
                if (result.update_count == 0) {
                    result.first_critic1_loss = critic1_loss;
                    result.first_critic2_loss = critic2_loss;
                    result.first_actor_loss = actor_loss;
                }
                result.last_critic1_loss = critic1_loss;
                result.last_critic2_loss = critic2_loss;
                result.last_actor_loss = actor_loss;
                result.critic1_losses.push_back(critic1_loss);
                result.critic2_losses.push_back(critic2_loss);
                result.actor_losses.push_back(actor_loss);

                double log_prob_sum = 0.0;
                for (int64_t i = 0; i < n; ++i) {
                    log_prob_sum += static_cast<double>(sample.log_prob.data()[i]);
                }
                result.mean_log_probs.push_back(static_cast<float>(log_prob_sum / static_cast<double>(n)));

                double std_sum = 0.0;
                for (int64_t i = 0; i < log_std.numel(); ++i) {
                    std_sum += std::exp(static_cast<double>(log_std.data()[i]));
                }
                result.mean_stds.push_back(
                    static_cast<float>(std_sum / static_cast<double>(log_std.numel())));
                result.q1_selected_fractions.push_back(selected.q1_selected_fraction);
                ++result.update_count;
            }

            if (step_result.done) {
                break;
            }
        }

        result.episode_lengths.push_back(episode_length);
        if (on_episode) {
            on_episode(episode, episode_length);
        }
    }

    result.final_q1_weight = tensor_values(q1_in.weight());
    result.final_q1_target_weight = tensor_values(q1_target_in.weight());
    result.final_q2_target_weight = tensor_values(q2_target_in.weight());
    result.final_mean_weight = tensor_values(mean_in.weight());
    result.final_log_std_weight = tensor_values(log_std_in.weight());

    const auto max_delta = [](const std::vector<float>& before, const std::vector<float>& after) {
        double worst = 0.0;
        for (size_t i = 0; i < after.size() && i < before.size(); ++i) {
            const double delta =
                std::fabs(static_cast<double>(after[i]) - static_cast<double>(before[i]));
            if (delta > worst) {
                worst = delta;
            }
        }
        return worst;
    };
    result.max_q1_weight_delta = max_delta(result.initial_q1_weight, result.final_q1_weight);
    result.max_q1_target_weight_delta = max_delta(result.initial_q1_target_weight, result.final_q1_target_weight);
    result.max_q2_target_weight_delta = max_delta(result.initial_q2_target_weight, result.final_q2_target_weight);
    result.max_mean_weight_delta = max_delta(result.initial_mean_weight, result.final_mean_weight);
    result.max_log_std_weight_delta = max_delta(result.initial_log_std_weight, result.final_log_std_weight);

    // A soft update *lags* by construction, so the online/target gap over *every* parameter is
    // the other half of "PolyakUpdate is genuinely doing an exponential average": a zero gap
    // would mean tau behaved like a hard copy.
    {
        const std::vector<float> online = parameter_values(q1);
        const std::vector<float> target = parameter_values(q1_target);
        result.max_target_lag = max_delta(online, target);
    }

    const int64_t total = static_cast<int64_t>(result.episode_lengths.size());
    result.window = total < kWindow ? total : kWindow;
    double head = 0.0;
    double tail = 0.0;
    for (int64_t i = 0; i < result.window; ++i) {
        head += static_cast<double>(result.episode_lengths[static_cast<size_t>(i)]);
        tail += static_cast<double>(result.episode_lengths[static_cast<size_t>(total - 1 - i)]);
    }
    result.first_window_average = result.window > 0 ? head / static_cast<double>(result.window) : 0.0;
    result.last_window_average = result.window > 0 ? tail / static_cast<double>(result.window) : 0.0;

    return result;
}

// ---------------------------------------------------------------------------------------
// TUNING NOTE
// ---------------------------------------------------------------------------------------
// What the tuning that produced TrainingConfig's defaults actually measured.
//
// Recorded here rather than in a commit message because several of the design notes above cite
// it, and a claim about an experiment is only worth as much as the numbers behind it -- the same
// standard the REINFORCE, A2C and PPO missions' own TUNING NOTEs set. All figures are Release
// builds (Debug produces bit-identical numbers -- verified -- just slower: ~7s vs ~50s for the
// tuned run). "ratio" is last-20/first-20 mean episode length (bar: >= 3.00x); "final" is the
// last-20 mean (bar: >= 60.0, i.e. 30% of max_steps=200); "std" is the policy's own mean
// exp(log_std) at the last update; "logp" is the mean log-probability of the sampled batch action
// at the last update. Every cell in every table below kept all three losses finite at every
// update; no configuration tried in this mission ever produced a NaN.
//
// The first window is 30.15 in almost every cell, and that is not a coincidence: nothing the
// actor does before the warm-up is cleared can affect the first ~33 episodes, so any
// configuration sharing hidden_size and the actor seed shares its first window exactly. It moves
// only when hidden_size or a seed changes (table 3 and 5 below).
//
// (1) alpha x actor lr, AT THE FINAL CONFIGURATION (150 episodes, critic lr 1e-3, hidden 64,
//     batch 64, tau 0.005, warm-up 1000). This is the table that chose alpha, and it is the one
//     that matters most, because alpha is the knob this mission's exit gate is actually about.
//       alpha=0.000: alr 1e-4 -> 4.88x (147.15) | 3e-4 -> 4.85x (146.30) | 1e-3 -> 4.33x (130.65)
//       alpha=0.005: alr 1e-4 -> 6.25x (188.50) | 3e-4 -> 6.20x (187.05) | 1e-3 -> 4.40x (132.70)
//       alpha=0.020: alr 1e-4 -> 4.68x (141.00) | 3e-4 -> 5.01x (150.95) | 1e-3 -> 5.51x (166.00)
//       alpha=0.050: alr 1e-4 -> 5.38x (162.25) | 3e-4 -> 6.48x (195.30) | 1e-3 -> 6.57x (198.10)
//       alpha=0.200: alr 1e-4 -> 6.03x (181.95) | 3e-4 -> 4.56x (137.60) | 1e-3 -> 5.02x (151.30)
//     All 15 cells clear both bars, so the bars alone cannot choose alpha. What *does* separate
//     the alpha=0 row from every other row is the thing the entropy term exists to control, and
//     it is unambiguous in the diagnostics rather than in the bars:
//       - At alpha = 0 the policy's exploration scale **collapses**: final mean exp(log_std) is
//         0.036 / 0.016 / 0.019 across the three learning rates, and the mean log-probability
//         runs away upward to +3.50 / +5.74 / +11.04 (a near-deterministic tanh-squashed policy
//         has an unboundedly large density). That is the maximum-entropy objective with its
//         entropy term switched off, observed rather than asserted.
//       - At every alpha > 0 the scale stabilizes in 0.31 .. 0.67 and the log-probability stays
//         within ~2 of zero. The entropy term is holding the policy stochastic, which is exactly
//         its job, and it is doing so without any external exploration schedule at all (contrast
//         DQN's epsilon decay, which this loop does not have and does not need).
//     alpha = 0.05 chosen: interior on the swept axis, best-in-table at the chosen actor lr, and
//     in the region where the entropy bonus is a real but not dominant share of the Bellman
//     target (alpha * 13.8 = 0.69 at the tanh stabilizer's bound, against a per-step reward of
//     exactly 1.0). alpha = 0.2 would put that bound at 2.76, i.e. the entropy bonus could nearly
//     triple the effective reward, and its row is visibly the noisiest of the five.
//
// (2) critic lr x actor lr, at alpha = 0.05 (measured over a 220-episode budget, before the
//     budget itself was cut to 150 -- see (6)):
//       clr=3e-4: alr 1e-4 -> 3.50x (105.50) | 3e-4 -> 6.62x (199.70) | 1e-3 -> 2.34x (70.70) FAIL
//       clr=1e-3: alr 1e-4 -> 5.61x (169.25) | 3e-4 -> 6.33x (190.75) | 1e-3 -> 6.47x (195.00)
//       clr=3e-3: alr 1e-4 -> 6.16x (185.65) | 3e-4 -> 6.46x (194.80) | 1e-3 -> 6.05x (182.45)
//     The single failing cell in this whole mission's tuning is clr=3e-4 / alr=1e-3, at 2.34x --
//     and note it still clears the *absolute* bar (70.70 >= 60.0) while missing the relative one.
//     The reading is the obvious one and it is why clr=3e-4 was dropped: a critic that learns too
//     slowly relative to its actor gives the actor a stale action-gradient to climb, and the
//     effect gets worse as the actor's step size grows (3.50x -> 6.62x -> 2.34x is not monotone,
//     but the two extremes of the actor lr axis are the two weak cells). clr = 1e-3 chosen:
//     interior, and the only row whose worst cell (5.61x) is comfortably clear.
//
// (3) hidden size x batch size, at the chosen learning rates (220-episode budget):
//       h=32:  batch 32 -> 6.40x (167.95) | 64 -> 7.27x (190.85) | 128 -> 7.35x (193.00)
//       h=64:  batch 32 -> 5.07x (153.00) | 64 -> 6.33x (190.75) | 128 -> 6.00x (181.05)
//       h=128: batch 32 -> 4.83x (157.95) | 64 -> 5.43x (177.50) | 128 -> 5.45x (178.35)
//     Every cell clears both bars, and the systematic trends are mild: batch 32 is the weakest
//     column at every width (a 32-row minibatch is a noisy estimate of a target that already
//     contains a fresh stochastic actor sample), and the first window rises with width
//     (26.25 / 30.15 / 32.70) because a wider randomly-initialized actor happens to push harder,
//     which mechanically *depresses* the ratio without making the run worse. h=64 / batch=64
//     kept: interior on both axes, and this mission has four networks at this width, so the
//     suite is not paying for width it cannot show a benefit from.
//
// (4) tau x warm-up size, at the chosen configuration (220-episode budget):
//       tau=0.001: warm 200 -> 5.49x (143.40) | 1000 -> 6.58x (198.25) | 3000 -> 3.55x (107.05)
//       tau=0.005: warm 200 -> 5.53x (165.45) | 1000 -> 6.33x (190.75) | 3000 -> 6.38x (192.45)
//       tau=0.020: warm 200 -> 6.69x (196.40) | 1000 -> 6.31x (190.10) | 3000 -> 6.63x (200.00)
//     All nine clear both bars. Two readings kept:
//       - warm-up 1000 is the best cell of every tau row but one, and the reason a separate
//         warm-up knob exists at all (rather than DQN's "warm-up == batch_size") is visible in
//         the tau=0.001 / warm=3000 cell: a slow target plus a late start leaves only 4743
//         updates in the budget, and the run does not finish converging. The warm-up is not free
//         -- it trades updates for target quality -- and 1000 is where that trade was best.
//       - tau's effect shows up in the *magnitudes* rather than the bars: the largest critic1
//         loss anywhere in the run is 10.1 at tau=0.001, 170 at tau=0.005 and 625 at tau=0.020.
//         A faster-moving target is a less stable regression problem, exactly as expected, and
//         0.005 (SAC's own literature value) sits between a target too sluggish to finish in
//         budget and one that makes the critics chase themselves.
//
// (5) Seed robustness -- the check that matters most for a thresholded test, run over five
//     independent seed-sets (both critic init, both actor init and the environment varied
//     together), at the final configuration (150 episodes):
//       6.48x (30.15 -> 195.30) | 6.42x (26.35 -> 169.10) | 5.08x (24.55 -> 124.70)
//       6.62x (30.10 -> 199.30) | 6.97x (27.10 -> 189.00)
//     Worst seed-set 5.08x / 124.70, against bars of 3.00x / 60.0 -- a margin of 1.7x on the
//     relative bar and 2.1x on the absolute one. The same five seed-sets over the 220-episode
//     budget give 6.33x / 7.37x / 5.42x / 6.03x / 7.32x (worst 5.42x / 132.95), so the budget cut
//     costs a little worst-case margin and nothing else.
//
// (6) Episode budget, at the chosen configuration:
//       150 -> 6.48x (195.30), 15556 updates | 220 -> 6.33x (190.75), 29238 updates
//       300 -> 6.63x (200.00), 45205 updates | 400 -> 6.63x (200.00), 65205 updates
//     150 chosen, and this is a cost decision made on evidence rather than a guess: it is the
//     *best* of the four on both bars while costing half the updates of the next budget up and a
//     quarter of the largest. The run reaches a 190+ final window well before episode 150 (the
//     demo's rolling mean crosses 190 at episode 100), so the extra budgets are buying saturation
//     the bars cannot see. In wall-clock the tuned run costs ~7s Release / ~50s Debug, which is
//     what the full suite pays for this mission's headline test.
//
// (7) What is NOT tuned, and why. gamma is pinned at 0.99, this campaign's value in every prior
//     mission, and was not swept: changing it changes the scale of every Q value and therefore
//     the meaning of every critic-loss number reported above, which would make the tables
//     incomparable for no expected gain on a task whose horizon is 200 steps. The log_std network
//     is deliberately **not** clamped to the [-20, 2] range reference SAC implementations use.
//     That clamp is a saturating nonlinearity whose backward pass needs its own mask, and it was
//     not needed: across every configuration in every table above, the policy's exploration scale
//     stayed in 0.003 .. 0.85, nowhere near a bound a clamp would have engaged at. Adding
//     untested masking plumbing to prevent a failure mode that never occurred would be
//     speculative -- but if a future task does drive log_std to an extreme, this is the first
//     place to look.

}  // namespace sac_continuous_cartpole
}  // namespace exai
