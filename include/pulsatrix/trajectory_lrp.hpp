/** @file trajectory_lrp.hpp
 *  @brief Explaining a continuous diffusion sampler end to end (DIFF-5): trajectory LRP (design
 *         D1), which seeds a scalar of the final sample x_0 once and carries its relevance back
 *         through every reverse step to the starting noise x_T, the conditioning, and named sinks.
 *  @ingroup interpretability
 *
 *  A DDPM or DDIM step is `x_prev = skip * x_t - network * eps_theta(x_t, t, c) + noise * z`, a
 *  residual block with a third, injected-noise term. The relevance on each element of x_prev is
 *  split across the three terms, by default keeping each term's sign (TrajectorySplit::SignedRatio). The skip share goes straight to x_t; the network share runs
 *  through eps_theta with its own LRP rules onto x_t, the conditioning c and the timestep
 *  embedding (a sink); the noise share is a sink too, exactly zero under DDIM. Only the
 *  trajectory is stored, and each step's forward pass is rerun just before its relevance pass.
 *  See docs/interpretability/diffusion-lrp.md (D1) for the derivation and its limits.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/noise_schedule.hpp"

namespace pulsatrix {

/** @brief Where the network's share of one step landed. */
struct DenoiserRelevance {
    /** @brief On x_t, one value per element. */
    std::vector<float> x;
    /** @brief On the conditioning, one value per conditioning input. */
    std::vector<float> condition;
    /** @brief On the timestep embedding: a sink, since the embedding is a constant. */
    double time = 0;
};

/**
 * @brief The noise predictor eps_theta as the explainer sees it, with its conditioning fixed: a
 *        forward pass that keeps what the relevance pass needs, and the relevance pass.
 */
class ContinuousDenoiser {
public:
    virtual ~ContinuousDenoiser() = default;
    /** @brief The predicted noise at timestep @p t, the same size as @p x_t. */
    [[nodiscard]] virtual std::vector<float> predict(const std::vector<float>& x_t, int64_t t) = 0;
    /** @brief Relevance on the last predict()'s inputs, from @p relevance_eps on its output. */
    [[nodiscard]] virtual DenoiserRelevance relevance(const std::vector<float>& relevance_eps) = 0;
};

/**
 * @brief The network rules ConcatDenoiser uses unless told otherwise: the epsilon rule with the bias
 *        in the denominator (Zennit's), so the biases absorb their share. The codebase's default
 *        pre-bias rule divides by `x W` alone, which a hidden layer's bias can leave near zero while
 *        the unit is active; chained over the steps of a sampler its relevance blew up (DIFF-5's
 *        planted DDPM: conditioning shares up to 570 times the explained value).
 */
[[nodiscard]] LRPRuleConfig TrajectoryNetworkRules();

/**
 * @brief A Module that reads `[x_t, embedding(t), c]` as one row, conditioned by concatenation as
 *        in tests/diffusion_integration_test.cpp: x_t, then a sinusoidal timestep embedding
 *        (SinusoidalTimestepEmbedding), then a fixed conditioning vector such as a one-hot class.
 *        Its input relevance is split back into those three parts.
 */
class ConcatDenoiser final : public ContinuousDenoiser {
public:
    /**
     * @param network Maps `(1, dim + time_dim + condition.size())` to `(1, dim)`. It and the backend
     *        must outlive this.
     * @param config The rules the network's relevance pass uses.
     * @throws std::invalid_argument if dim <= 0 or time_dim < 0.
     */
    ConcatDenoiser(Module& network, int64_t dim, int64_t time_dim, std::vector<float> condition, DeviceBackend* backend,
                   LRPRuleConfig config = TrajectoryNetworkRules());
    /** @brief Replaces the conditioning, for instance with another class. */
    void set_condition(std::vector<float> condition);
    [[nodiscard]] const std::vector<float>& condition() const { return condition_; }
    /** @brief The row the network reads at timestep @p t. */
    [[nodiscard]] std::vector<float> input_row(const std::vector<float>& x_t, int64_t t) const;
    [[nodiscard]] std::vector<float> predict(const std::vector<float>& x_t, int64_t t) override;
    [[nodiscard]] DenoiserRelevance relevance(const std::vector<float>& relevance_eps) override;

private:
    Module& network_;
    int64_t dim_;
    int64_t time_dim_;
    std::vector<float> condition_;
    DeviceBackend* backend_;
    LRPRuleConfig config_;
};

/** @brief Which reverse process to run. */
enum class ContinuousSampler {
    /** @brief Ancestral DDPM (Ho et al. 2020): every timestep T..1, noise sqrt(beta_t), none at t = 1. */
    Ddpm,
    /** @brief DDIM (Song et al. 2021) on evenly spaced timesteps; deterministic when eta = 0. */
    Ddim,
};

struct ContinuousSamplerConfig {
    ContinuousSampler sampler = ContinuousSampler::Ddim;
    /** @brief DDIM only: how many steps. The timesteps are `round(T (k / steps))` for k = steps..1,
     *         deduplicated, ending at the clean sample. DDPM always takes every timestep. */
    int64_t steps = 50;
    /** @brief DDIM only: 0 is deterministic, 1 has DDPM's posterior variance. */
    float eta = 0.0f;
    /** @brief Seeds the injected noise (an own Box-Muller over mt19937_64, the same on every platform). */
    uint64_t seed = 0;
};

/** @brief A reverse process, recorded so it can be explained: every state and every noise draw. */
struct ContinuousTrajectory {
    /** @brief `x[0]` is x_T, `x[k + 1]` the state after step k, `x.back()` the sample x_0. */
    std::vector<std::vector<float>> x;
    /** @brief The timestep each step reads (descending). */
    std::vector<int64_t> timesteps;
    /** @brief Each step's coefficients. */
    std::vector<ReverseStepCoefficients> coefficients;
    /** @brief Each step's noise z (zero when the step injects none). */
    std::vector<std::vector<float>> noise;

    [[nodiscard]] const std::vector<float>& sample() const { return x.back(); }
    [[nodiscard]] int64_t steps() const { return static_cast<int64_t>(timesteps.size()); }
};

/**
 * @brief Runs the reverse process from @p x_T and records it.
 * @throws std::invalid_argument for an empty x_T, steps < 1, a negative eta, or a prediction of the
 *         wrong size.
 */
[[nodiscard]] ContinuousTrajectory SampleContinuousDiffusion(ContinuousDenoiser& model, const NoiseSchedule& schedule,
                                                             const std::vector<float>& x_T, const ContinuousSamplerConfig& config);

/** @brief Draws @p n standard normal values from @p seed, as SampleContinuousDiffusion does. */
[[nodiscard]] std::vector<float> StandardNormal(size_t n, uint64_t seed);

// ---- seeds: a scalar of x_0 -------------------------------------------------------------------

/** @brief The sum of @p sample over @p region: relevance x0_i on the region's elements, 0 elsewhere.
 *  @throws std::invalid_argument for an index outside the sample. */
[[nodiscard]] std::vector<float> RegionSeed(const std::vector<float>& sample, const std::vector<int64_t>& region);

/** @brief The reconstruction error `||x_0 - reference||^2`, as LRP for autoencoders seeds it
 *         (Kobayashi et al. 2023): relevance `(x0_i - ref_i)^2` on each element.
 *  @throws std::invalid_argument if the sizes differ. */
[[nodiscard]] std::vector<float> ReconstructionSeed(const std::vector<float>& sample, const std::vector<float>& reference);

// ---- the explainer --------------------------------------------------------------------------

/** @brief How the relevance on x_prev is split across skip, network and noise (h_1, h_2, h_3). */
enum class TrajectorySplit {
    /**
     * @brief `R_k = R |h_k| / (sum_j |h_j| + epsilon)`, the absolute-value ratio of LRP for ResNets
     *        (Otsuki et al. 2024), D1's first design. Bounded however much the terms cancel, and
     *        conserving up to the epsilon, which only handles 0/0. But it loses sign: a term that
     *        opposes x_prev still gets a share of R's sign. The network term opposes the skip term
     *        whenever eps_theta removes noise, so everything that acts through eps_theta comes back
     *        with the wrong sign. On DIFF-5's planted DDPM the class got the wrong sign in 9 of 12
     *        samples.
     */
    AbsoluteRatio,
    /**
     * @brief The signed epsilon rule, `R_k = R h_k / (s + e sign(s))` with `s = sum_j h_j` and the
     *        stabilizer e = epsilon * sum_j |h_j|. Shares grow without bound when the terms cancel;
     *        what the stabilizer absorbs is reported.
     */
    SignedEpsilon,
    /**
     * @brief `R_k = R h_k sign(s) / (sum_j |h_j| + epsilon)`: each term keeps the sign of its
     *        contribution to x_prev and is bounded by |R|. What the terms cancel,
     *        `R (1 - |s| / sum_j |h_j|)`, is absorbed and reported. It is the signed epsilon rule
     *        with the stabilizer `sum_j |h_j| - |s|`, just large enough to bound every share.
     */
    SignedRatio,
};

struct TrajectoryLrpConfig {
    /** @brief SignedRatio: on DIFF-5's planted DDPM its per-step conditioning relevance tracked the
     *         effect of removing the class at that step (median Spearman 0.98); AbsoluteRatio's didn't. */
    TrajectorySplit split = TrajectorySplit::SignedRatio;
    /** @brief AbsoluteRatio and SignedRatio: added to the denominator. SignedEpsilon: the
     *         stabilizer's scale. */
    double epsilon = 1e-9;
    /** @brief Relative tolerance for the rerun check: recomputing each step from the stored x_t, z
     *         and coefficients must give the stored x_prev, or the explanation would be of another
     *         trajectory. Negative turns the check off. */
    double recompute_tolerance = 1e-4;
};

/** @brief One reverse step's split, aggregated over elements. */
struct TrajectoryHop {
    /** @brief The index into ContinuousTrajectory::timesteps, and that timestep. */
    int64_t step = 0;
    int64_t timestep = 0;
    /** @brief Relevance arriving on x_prev. */
    double seeded = 0;
    /** @brief Its split: skip (to x_t), network (into eps_theta), noise (a sink), and what the
     *         split's epsilon absorbed (a sink). seeded = skip + network + noise + absorbed. */
    double skip = 0;
    double network = 0;
    double noise = 0;
    double absorbed = 0;
    /** @brief The signed fractions, logged next to the split: `sum_i R_i h_k,i / s_i` for each
     *         term, where `s_i = sum_k h_k,i` (elements with s_i = 0 are left out). They sum to
     *         seeded over those elements and grow without bound when the terms cancel. */
    double signed_skip = 0;
    double signed_network = 0;
    double signed_noise = 0;
    /** @brief Where the network share landed: x_t, the conditioning, the timestep embedding. */
    double network_to_x = 0;
    double network_to_condition = 0;
    double network_to_time = 0;
    /** @brief What the network's relevance pass didn't conserve: biases, norms, its stabilizers. */
    [[nodiscard]] double network_leak() const { return network - network_to_x - network_to_condition - network_to_time; }
};

/**
 * @brief Where the seed's relevance went. The budget balances exactly:
 *        value = x_T + condition + time + noise + absorbed + leak.
 */
struct TrajectoryRelevance {
    /** @brief The seed's total, the explained value. */
    double value = 0;
    /** @brief Relevance on the starting noise x_T, one value per element. */
    std::vector<float> x_T;
    /** @brief Relevance on the conditioning, summed over steps. */
    std::vector<float> condition;
    /** @brief Relevance on the conditioning at each step (`steps x condition` values), so its
     *         concentration over time can be read. */
    std::vector<std::vector<float>> condition_by_step;
    /** @brief One entry per step, in sampling order (step 0 reads x_T). */
    std::vector<TrajectoryHop> hops;

    [[nodiscard]] double x_T_total() const;
    [[nodiscard]] double condition_total() const;
    [[nodiscard]] double time_total() const;
    [[nodiscard]] double noise_total() const;
    [[nodiscard]] double absorbed_total() const;
    [[nodiscard]] double leak_total() const;
};

/**
 * @brief Explains @p seed, relevance on the trajectory's sample x_0 (RegionSeed, ReconstructionSeed,
 *        or a downstream classifier's input relevance), back through @p trajectory, which
 *        SampleContinuousDiffusion produced with @p model.
 * @note Cost: one forward and one relevance pass per step. Memory: one step's activations.
 * @throws std::invalid_argument if the seed's size differs from the sample's, or a model output has
 *         the wrong size.
 * @throws std::runtime_error if a step, rerun from the stored state, doesn't reproduce the stored
 *         next state within config.recompute_tolerance (the model isn't the one that sampled, or
 *         isn't deterministic).
 */
[[nodiscard]] TrajectoryRelevance ExplainTrajectory(ContinuousDenoiser& model, const ContinuousTrajectory& trajectory,
                                                    const std::vector<float>& seed, const TrajectoryLrpConfig& config = {});

}  // namespace pulsatrix
