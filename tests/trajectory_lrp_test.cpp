/** @file trajectory_lrp_test.cpp
 *  @brief Trajectory LRP (DIFF-5, design D1): the split by hand, the budget, the noise sink under
 *         DDPM and DDIM, the rerun check, cancellation, and a planted class-conditional DDPM.
 */
#include "pulsatrix/trajectory_lrp.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/sinusoidal_timestep_embedding.hpp"

namespace pulsatrix {
namespace {

double Total(const std::vector<float>& v) {
    double total = 0;
    for (float x : v) total += x;
    return total;
}

void ExpectBudgetBalances(const TrajectoryRelevance& r) {
    const double sum = r.x_T_total() + r.condition_total() + r.time_total() + r.noise_total() + r.absorbed_total() + r.leak_total();
    EXPECT_NEAR(sum, r.value, 1e-5 * std::max(1.0, std::abs(r.value)));
    for (const TrajectoryHop& h : r.hops) {
        EXPECT_NEAR(h.skip + h.network + h.noise + h.absorbed, h.seeded, 1e-9 * std::max(1.0, std::abs(h.seeded)));
    }
}

// ---- a linear denoiser, by hand ---------------------------------------------------------------

/** @brief eps = [x, c] W with no bias and no time input: its epsilon-rule LRP is exact. */
struct LinearDenoiser {
    CPUBackend cpu;
    LinearModule net{3, 2, &cpu, false};
    ConcatDenoiser model{net, 2, 0, {1.0f}, &cpu};
    LinearDenoiser() { net.set_weight({0.5f, -0.2f, 0.3f, 0.8f, 0.4f, -0.6f}); }
};

TEST(TrajectoryLrp, OneDdimStepSplitsAndFollowsTheNetworkByHand) {
    // Linear schedule 0.1..0.5 over 5 steps, one DDIM step from t = 5 to the clean sample. The
    // numbers were computed in Python (double) from the step and the epsilon rule.
    LinearDenoiser d;
    const NoiseSchedule schedule(5, 0.1f, 0.5f);
    ContinuousSamplerConfig config;
    config.steps = 1;
    const ContinuousTrajectory trajectory = SampleContinuousDiffusion(d.model, schedule, {1.0f, -0.5f}, config);
    ASSERT_EQ(trajectory.timesteps, std::vector<int64_t>{5});
    EXPECT_NEAR(trajectory.sample()[0], 0.7947199402146967, 1e-5);
    EXPECT_NEAR(trajectory.sample()[1], 1.5573428449615037, 1e-5);

    TrajectoryLrpConfig absolute;
    absolute.split = TrajectorySplit::AbsoluteRatio;
    const TrajectoryRelevance r = ExplainTrajectory(d.model, trajectory, RegionSeed(trajectory.sample(), {0, 1}), absolute);
    EXPECT_NEAR(r.value, 2.3520627851762006, 1e-5);
    EXPECT_NEAR(r.hops[0].skip, 0.4699766307234543 + 0.484983077237755, 1e-5);
    EXPECT_NEAR(r.hops[0].network, 0.3247433094912425 + 1.0723597677237486, 1e-5);
    EXPECT_NEAR(r.x_T[0], 0.8651987983382408, 1e-5);
    EXPECT_NEAR(r.x_T[1], 0.7774876712474228, 1e-5);
    EXPECT_NEAR(r.condition[0], 0.709376315590537, 1e-5);
    EXPECT_EQ(r.noise_total(), 0.0);
    EXPECT_NEAR(r.leak_total(), 0.0, 1e-5);
    ExpectBudgetBalances(r);

    // The default, SignedRatio: element 1's skip term (-1.29) opposes its sum (+1.56), so its share
    // is negative, and the cancelled part is absorbed.
    const TrajectoryRelevance s = ExplainTrajectory(d.model, trajectory, RegionSeed(trajectory.sample(), {0, 1}));
    EXPECT_NEAR(s.hops[0].skip, -0.01500644651430072, 1e-5);
    EXPECT_NEAR(s.hops[0].network, 0.7476164582325061, 1e-5);
    EXPECT_NEAR(s.absorbed_total(), 1.6194527734579949, 1e-5);
    EXPECT_NEAR(s.x_T[0], 0.43220771901658406, 1e-5);
    EXPECT_NEAR(s.x_T[1], -0.06258115943159026, 1e-5);
    EXPECT_NEAR(s.condition[0], 0.3629834521332116, 1e-5);
    ExpectBudgetBalances(s);
}

TEST(TrajectoryLrp, DdimHasNoNoiseSinkAndDdpmDoes) {
    LinearDenoiser d;
    const NoiseSchedule schedule = NoiseSchedule::Cosine(20);
    ContinuousSamplerConfig ddim;
    ddim.steps = 10;
    const ContinuousTrajectory a = SampleContinuousDiffusion(d.model, schedule, {0.3f, -1.1f}, ddim);
    EXPECT_EQ(a.steps(), 10);
    const TrajectoryRelevance ra = ExplainTrajectory(d.model, a, ReconstructionSeed(a.sample(), {0.0f, 0.0f}));
    EXPECT_EQ(ra.noise_total(), 0.0);  // exactly: every z is zero
    ExpectBudgetBalances(ra);

    ContinuousSamplerConfig ddpm;
    ddpm.sampler = ContinuousSampler::Ddpm;
    ddpm.seed = 3;
    const ContinuousTrajectory b = SampleContinuousDiffusion(d.model, schedule, {0.3f, -1.1f}, ddpm);
    EXPECT_EQ(b.steps(), 20);
    EXPECT_EQ(b.coefficients.back().noise, 0.0f);  // the last DDPM step injects nothing
    const TrajectoryRelevance rb = ExplainTrajectory(d.model, b, RegionSeed(b.sample(), {0, 1}));
    EXPECT_NE(rb.noise_total(), 0.0);
    EXPECT_EQ(rb.hops.back().noise, 0.0);
    ExpectBudgetBalances(rb);
    // The same seed gives the same trajectory.
    EXPECT_EQ(SampleContinuousDiffusion(d.model, schedule, {0.3f, -1.1f}, ddpm).x, b.x);
}

TEST(TrajectoryLrp, DdimTimestepsAreEvenlySpacedAndEndAtTheCleanSample) {
    LinearDenoiser d;
    const NoiseSchedule schedule(10);
    ContinuousSamplerConfig config;
    config.steps = 4;
    const ContinuousTrajectory t = SampleContinuousDiffusion(d.model, schedule, {0.1f, 0.2f}, config);
    EXPECT_EQ(t.timesteps, (std::vector<int64_t>{10, 8, 5, 3}));  // round(10 k / 4), k = 4..1
    config.steps = 30;  // more steps than timesteps: every timestep once
    EXPECT_EQ(SampleContinuousDiffusion(d.model, schedule, {0.1f, 0.2f}, config).steps(), 10);
}

TEST(TrajectoryLrp, TheRerunCheckRejectsAModelThatDidntSample) {
    LinearDenoiser d;
    const NoiseSchedule schedule(5, 0.1f, 0.5f);
    const ContinuousTrajectory trajectory = SampleContinuousDiffusion(d.model, schedule, {1.0f, -0.5f}, {});
    d.net.set_weight({0.5f, -0.2f, 0.3f, 0.8f, 0.4f, -0.5f});
    EXPECT_THROW((void)ExplainTrajectory(d.model, trajectory, RegionSeed(trajectory.sample(), {0})), std::runtime_error);
    TrajectoryLrpConfig unchecked;
    unchecked.recompute_tolerance = -1;
    EXPECT_NO_THROW((void)ExplainTrajectory(d.model, trajectory, RegionSeed(trajectory.sample(), {0}), unchecked));
}

TEST(TrajectoryLrp, WhenSkipAndNetworkCancelEachSplitBehavesAsDocumented) {
    // eps = k x with k chosen so the network term cancels the skip term to 0.1%.
    CPUBackend cpu;
    const NoiseSchedule schedule(5, 0.1f, 0.5f);
    const ReverseStepCoefficients c = schedule.ddim_coefficients(5, 0);
    const float k = 0.999f * c.skip / c.network;
    LinearModule net(2, 2, &cpu, false);
    net.set_weight({k, 0.0f, 0.0f, k});
    ConcatDenoiser model(net, 2, 0, {}, &cpu);
    ContinuousSamplerConfig config;
    config.steps = 1;
    const ContinuousTrajectory trajectory = SampleContinuousDiffusion(model, schedule, {1.0f, -2.0f}, config);
    const std::vector<float> seed = {1.0f, 1.0f};

    TrajectoryLrpConfig absolute;
    absolute.split = TrajectorySplit::AbsoluteRatio;
    const TrajectoryRelevance a = ExplainTrajectory(model, trajectory, seed, absolute);
    EXPECT_NEAR(a.hops[0].skip, 1.0, 1e-3);  // about half each, whatever the cancellation
    EXPECT_NEAR(a.hops[0].network, 1.0, 1e-3);
    EXPECT_GT(a.hops[0].signed_skip, 500.0);  // the logged signed fractions show it
    EXPECT_LT(a.hops[0].signed_network, -500.0);
    ExpectBudgetBalances(a);

    TrajectoryLrpConfig signed_config;
    signed_config.split = TrajectorySplit::SignedEpsilon;
    signed_config.epsilon = 1e-6;
    const TrajectoryRelevance s = ExplainTrajectory(model, trajectory, seed, signed_config);
    EXPECT_GT(s.hops[0].skip, 500.0);
    EXPECT_LT(s.hops[0].network, -500.0);
    ExpectBudgetBalances(s);

    // SignedRatio: half each with its sign, bounded, and the cancelled 99.95% absorbed.
    const TrajectoryRelevance r = ExplainTrajectory(model, trajectory, seed);
    EXPECT_NEAR(r.hops[0].skip, 1.0, 1e-3);
    EXPECT_NEAR(r.hops[0].network, -1.0, 1e-3);
    EXPECT_NEAR(r.absorbed_total(), 2.0, 2e-3);
    ExpectBudgetBalances(r);
}

TEST(TrajectoryLrp, SeedsAndInputsAreChecked) {
    EXPECT_EQ(RegionSeed({1.0f, -2.0f, 3.0f}, {0, 2}), (std::vector<float>{1.0f, 0.0f, 3.0f}));
    EXPECT_THROW((void)RegionSeed({1.0f}, {1}), std::invalid_argument);
    EXPECT_EQ(ReconstructionSeed({1.0f, -2.0f}, {0.5f, -1.0f}), (std::vector<float>{0.25f, 1.0f}));
    EXPECT_THROW((void)ReconstructionSeed({1.0f}, {1.0f, 2.0f}), std::invalid_argument);

    LinearDenoiser d;
    const NoiseSchedule schedule(5);
    EXPECT_THROW((void)SampleContinuousDiffusion(d.model, schedule, {}, {}), std::invalid_argument);
    ContinuousSamplerConfig zero;
    zero.steps = 0;
    EXPECT_THROW((void)SampleContinuousDiffusion(d.model, schedule, {1.0f, 2.0f}, zero), std::invalid_argument);
    EXPECT_THROW((void)SampleContinuousDiffusion(d.model, schedule, {1.0f, 2.0f, 3.0f}, {}), std::invalid_argument);
    const ContinuousTrajectory t = SampleContinuousDiffusion(d.model, schedule, {1.0f, 2.0f}, {});
    EXPECT_THROW((void)ExplainTrajectory(d.model, t, {1.0f}), std::invalid_argument);
}

TEST(TrajectoryLrp, StandardNormalIsDeterministicAndStandard) {
    const std::vector<float> a = StandardNormal(20001, 9);
    EXPECT_EQ(a, StandardNormal(20001, 9));
    double mean = 0, square = 0;
    for (float v : a) {
        mean += v;
        square += static_cast<double>(v) * v;
    }
    mean /= static_cast<double>(a.size());
    EXPECT_NEAR(mean, 0.0, 0.03);
    EXPECT_NEAR(square / static_cast<double>(a.size()) - mean * mean, 1.0, 0.04);
}

// ---- a planted class-conditional DDPM ---------------------------------------------------------

constexpr int64_t kDim = 8, kTime = 8, kClasses = 2, kHidden = 64, kT = 100;

/** @brief Uniform values in [-scale, scale] from a small LCG, the same on every platform. */
std::vector<float> Uniform(size_t n, float scale, uint32_t seed) {
    std::vector<float> out(n);
    for (float& v : out) {
        seed = seed * 1664525u + 1013904223u;
        v = scale * (static_cast<float>(seed >> 8) / 8388608.0f - 1.0f);
    }
    return out;
}

/** @brief An MLP denoiser, `[x_t, embedding(t), one-hot class]` -> 64 -> ReLU -> 64 -> ReLU -> 8. */
struct PlantedDdpm {
    CPUBackend cpu;
    LinearModule l1{kDim + kTime + kClasses, kHidden, &cpu};
    ReluModule r1{&cpu};
    LinearModule l2{kHidden, kHidden, &cpu};
    ReluModule r2{&cpu};
    LinearModule l3{kHidden, kDim, &cpu};
    SequentialModule net{{&l1, &r1, &l2, &r2, &l3}};
    // Linear to 0.05: alpha_bar_T is 0.08, so the first DDIM step amplifies the noise prediction's
    // error about 3.5 times. The cosine schedule's alpha_bar_T of 1e-4 amplifies it about 100 times,
    // which samplers hide by clipping x0_hat, and clipping isn't part of the step D1 splits.
    NoiseSchedule schedule{kT, 1e-4f, 0.05f};

    explicit PlantedDdpm(uint32_t seed) { Initialize(seed); }

    void Initialize(uint32_t seed) {
        l1.set_weight(Uniform(static_cast<size_t>((kDim + kTime + kClasses) * kHidden), 1.0f / std::sqrt(18.0f), seed));
        l2.set_weight(Uniform(static_cast<size_t>(kHidden * kHidden), 1.0f / 8.0f, seed + 1));
        l3.set_weight(Uniform(static_cast<size_t>(kHidden * kDim), 1.0f / 8.0f, seed + 2));
        l1.set_bias(std::vector<float>(kHidden, 0.0f));
        l2.set_bias(std::vector<float>(kHidden, 0.0f));
        l3.set_bias(std::vector<float>(kDim, 0.0f));
    }

    static std::vector<float> OneHot(int64_t c) {
        std::vector<float> v(kClasses, 0.0f);
        v[static_cast<size_t>(c)] = 1.0f;
        return v;
    }

    /**
     * @brief The planted data: the class sets the sign of elements 0-3, `(+-)(0.8, 0.6, 0.8, 0.6)`
     *        plus a little noise; elements 4-7 are +-0.6 at random, whatever the class.
     */
    static std::vector<float> Sample(int64_t c, std::mt19937_64& rng) {
        // StandardNormal, not std::normal_distribution, so every platform trains the same model.
        const std::vector<float> small = StandardNormal(kDim, rng());
        std::vector<float> x(kDim);
        const float sign = c == 0 ? 1.0f : -1.0f;
        for (int64_t i = 0; i < 4; ++i) x[i] = sign * (i % 2 == 0 ? 0.8f : 0.6f) + 0.05f * small[i];
        for (int64_t i = 4; i < kDim; ++i) x[i] = (rng() % 2 ? 0.6f : -0.6f) + 0.05f * small[i];
        return x;
    }

    /** @brief L_simple with Adam; returns the mean loss over the last 100 steps. */
    double Train(int steps, int64_t batch) {
        AdamOptimizer adam(2e-3f, &cpu);
        MSELoss loss(&cpu);
        std::mt19937_64 rng(11);
        const int64_t width = kDim + kTime + kClasses;
        double recent = 0;
        for (int step = 0; step < steps; ++step) {
            std::vector<float> rows, targets;
            for (int64_t n = 0; n < batch; ++n) {
                const auto c = static_cast<int64_t>(rng() % kClasses);
                const auto t = static_cast<int64_t>(1 + rng() % kT);
                const std::vector<float> x0 = Sample(c, rng);
                const float a = schedule.alpha_bar(t);
                const std::vector<float> eps = StandardNormal(kDim, rng());
                for (int64_t i = 0; i < kDim; ++i) rows.push_back(std::sqrt(a) * x0[i] + std::sqrt(1.0f - a) * eps[i]);
                const std::vector<float> embedding = SinusoidalTimestepEmbedding(t, kTime, &cpu).to_host_vector();
                rows.insert(rows.end(), embedding.begin(), embedding.end());
                const std::vector<float> onehot = OneHot(c);
                rows.insert(rows.end(), onehot.begin(), onehot.end());
                targets.insert(targets.end(), eps.begin(), eps.end());
            }
            adam.zero_grad(net);
            const Tensor out = net.forward(Tensor(Shape({batch, width}), &cpu, rows));
            const float value = loss.forward(out, Tensor(Shape({batch, kDim}), &cpu, targets));
            (void)net.backward(loss.backward());
            adam.step(net);
            if (step >= steps - 100) recent += value / 100.0;
        }
        return recent;
    }
};

/** @brief Spearman rank correlation (no ties expected). */
double Spearman(const std::vector<double>& a, const std::vector<double>& b) {
    const auto ranks = [](const std::vector<double>& v) {
        std::vector<size_t> order(v.size());
        for (size_t i = 0; i < v.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return v[x] < v[y]; });
        std::vector<double> r(v.size());
        for (size_t i = 0; i < v.size(); ++i) r[order[i]] = static_cast<double>(i);
        return r;
    };
    const std::vector<double> ra = ranks(a), rb = ranks(b);
    const double n = static_cast<double>(a.size());
    double d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += (ra[i] - rb[i]) * (ra[i] - rb[i]);
    return 1.0 - 6.0 * d / (n * (n * n - 1.0));
}

/** @brief Reruns a deterministic trajectory with the conditioning replaced at step @p k only, or
 *         at every step for k = -1. */
std::vector<float> RerunWithConditionAt(ConcatDenoiser& model, const ContinuousTrajectory& t, int64_t k, const std::vector<float>& other) {
    const std::vector<float> own = model.condition();
    std::vector<float> x = t.x[0];
    for (int64_t j = 0; j < t.steps(); ++j) {
        model.set_condition(j == k || k < 0 ? other : own);
        const std::vector<float> eps = model.predict(x, t.timesteps[static_cast<size_t>(j)]);
        const ReverseStepCoefficients& c = t.coefficients[static_cast<size_t>(j)];
        for (size_t i = 0; i < x.size(); ++i) x[i] = c.skip * x[i] - c.network * eps[i];
    }
    model.set_condition(own);
    return x;
}

double Median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

/**
 * The class sets region A (elements 0-3) and not region B (4-7). Ground truth per step: remove the
 * class at that step only (a zero one-hot), rerun the rest of the trajectory, and measure how far
 * region A's sum moves. In this model that effect grows from 0.3% of the region at the first of 20
 * DDIM steps to 28% at the last. The default split's conditioning relevance follows it step by step
 * and has the region's sign; AbsoluteRatio's has the wrong sign in most samples.
 */
TEST(TrajectoryLrp, OnAPlantedClassConditionalDdpmTheClassExplainsItsRegionStepByStep) {
    PlantedDdpm ddpm(17);
    const double loss = ddpm.Train(3000, 128);
    ContinuousSamplerConfig config;
    config.steps = 20;
    TrajectoryLrpConfig absolute;
    absolute.split = TrajectorySplit::AbsoluteRatio;
    int follows = 0, right_sign = 0, absolute_wrong_sign = 0;
    std::vector<double> rho, relevance_a, relevance_b, moved_a, moved_b;
    std::vector<double> effect_by_step(20, 0.0), relevance_by_step(20, 0.0);
    for (int64_t c = 0; c < kClasses; ++c) {
        ConcatDenoiser model(ddpm.net, kDim, kTime, PlantedDdpm::OneHot(c), &ddpm.cpu);
        for (uint64_t seed = 1; seed <= 6; ++seed) {
            const ContinuousTrajectory t = SampleContinuousDiffusion(model, ddpm.schedule, StandardNormal(kDim, 100 + seed), config);
            const std::vector<float>& x0 = t.sample();
            const double region = x0[0] + x0[1] + x0[2] + x0[3];
            follows += (c == 0) == (region > 0);
            const TrajectoryRelevance a = ExplainTrajectory(model, t, RegionSeed(x0, {0, 1, 2, 3}));
            const TrajectoryRelevance b = ExplainTrajectory(model, t, RegionSeed(x0, {4, 5, 6, 7}));
            ExpectBudgetBalances(a);
            ExpectBudgetBalances(b);
            right_sign += (a.condition_total() > 0) == (region > 0);
            relevance_a.push_back(std::abs(a.condition_total()));
            relevance_b.push_back(std::abs(b.condition_total()));
            const std::vector<float> none = RerunWithConditionAt(model, t, -1, {0.0f, 0.0f});
            moved_a.push_back(std::abs(region - (none[0] + none[1] + none[2] + none[3])));
            moved_b.push_back(std::abs(x0[4] + x0[5] + x0[6] + x0[7] - (none[4] + none[5] + none[6] + none[7])));
            const TrajectoryRelevance abs_a = ExplainTrajectory(model, t, RegionSeed(x0, {0, 1, 2, 3}), absolute);
            absolute_wrong_sign += (abs_a.condition_total() > 0) != (region > 0);
            std::vector<double> effect, relevance;
            for (int64_t k = 0; k < t.steps(); ++k) {
                const std::vector<float> y = RerunWithConditionAt(model, t, k, {0.0f, 0.0f});
                effect.push_back(region - (y[0] + y[1] + y[2] + y[3]));
                relevance.push_back(Total(a.condition_by_step[static_cast<size_t>(k)]));
                effect_by_step[static_cast<size_t>(k)] += effect.back() / region / 12.0;
                relevance_by_step[static_cast<size_t>(k)] += relevance.back() / a.value / 12.0;
            }
            rho.push_back(Spearman(relevance, effect));
        }
    }
    std::printf("[planted DDPM] L_simple %.4f; median |condition relevance|: region A %.3f, region B %.3f; median move without "
                "the class: A %.3f, B %.3f; Spearman with the per-step effect: median %.2f, min %.2f; AbsoluteRatio's sign wrong "
                "in %d of 12\n",
                loss, Median(relevance_a), Median(relevance_b), Median(moved_a), Median(moved_b), Median(rho),
                *std::min_element(rho.begin(), rho.end()), absolute_wrong_sign);
    for (size_t k = 0; k < 20; k += 3) {
        std::printf("[planted DDPM] step %2zu: effect / region %.4f, condition relevance / value %.4f\n", k, effect_by_step[k],
                    relevance_by_step[k]);
    }
    EXPECT_LT(loss, 0.3);
    EXPECT_EQ(follows, 12);  // the model learned the task
    EXPECT_EQ(right_sign, 12);
    EXPECT_GT(Median(relevance_a), 3.0 * Median(relevance_b));
    EXPECT_GT(Median(rho), 0.9);
    EXPECT_GE(absolute_wrong_sign, 6);
}

}  // namespace
}  // namespace pulsatrix
