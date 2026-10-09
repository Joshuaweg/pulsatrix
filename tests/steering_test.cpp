// FEAT-7: steering directions, the steering hook, and the reliability report on a behavior whose
// answers are known exactly.
#include "pulsatrix/steering.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hf_model.hpp"
#include "pulsatrix/topk_sparse_autoencoder.hpp"

namespace pulsatrix {
namespace {

TEST(Steering, DirectionsAreTheDifferenceOfMeansOrADecoderRow) {
    const std::vector<float> pos = {1, 2, 3, 3, 4, 5}, neg = {0, 0, 1};
    EXPECT_EQ(DifferenceOfMeans(pos, neg, 3), (std::vector<float>{2, 3, 3}));
    EXPECT_THROW((void)DifferenceOfMeans({}, neg, 3), std::invalid_argument);
    EXPECT_THROW((void)DifferenceOfMeans(pos, {1, 2}, 3), std::invalid_argument);
    CPUBackend cpu;
    TopKSaeOptions o;
    o.k = 2;
    TopKSparseAutoencoder sae(4, 8, &cpu, o);
    EXPECT_EQ(FeaturizerDirection(sae, 3), sae.decoder_direction(3));
}

TEST(Steering, TheHookAddsTheDirectionAtItsPosition) {
    CPUBackend cpu;
    const Tensor h(Shape({1, 2, 3}), &cpu, std::vector<float>{1, 1, 1, 2, 2, 2});
    const HiddenStateHook hook = SteeringHook(2, {1, 0, -1}, 0.5f);
    EXPECT_EQ(hook(2, h).to_host_vector(), (std::vector<float>{1.5f, 1, 0.5f, 2.5f, 2, 1.5f}));
    EXPECT_EQ(hook(1, h).to_host_vector(), h.to_host_vector());  // other positions untouched
    const HiddenStateHook second_row = SteeringHook(2, {1, 0, -1}, 2.0f, {false, true});
    EXPECT_EQ(second_row(2, h).to_host_vector(), (std::vector<float>{1, 1, 1, 4, 2, 0}));
    EXPECT_THROW((void)SteeringHook(2, {1, 0}, 1.0f)(2, h), std::invalid_argument);
    EXPECT_THROW((void)SteeringHook(2, {1, 0, 0}, 1.0f, {true})(2, h), std::invalid_argument);

    // In a model: coefficient 0 changes nothing, anything else changes the logits.
    const HfModelConfig config = ReadHfConfig(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/qwen2/config.json");
    CausalLM model(config, &cpu);
    uint64_t s = 7;
    for (const NamedParamRef& p : model.named_parameters()) {
        std::vector<float> v(static_cast<size_t>(p.ref.value->numel()));
        for (float& w : v) {
            s = s * 6364136223846793005ULL + 1442695040888963407ULL;
            w = 0.2f * (static_cast<float>(static_cast<uint32_t>(s >> 32)) / 4294967296.0f - 0.5f);
        }
        *p.ref.value = Tensor(p.ref.value->shape(), &cpu, v);
    }
    const Tensor ids(Shape({1, 4}), &cpu, std::vector<float>{1, 4, 2, 6});
    const std::vector<float> clean = model.forward(ids).to_host_vector();
    const std::vector<float> direction(static_cast<size_t>(config.hidden_size), 0.3f);
    model.set_hidden_state_hook(SteeringHook(1, direction, 0.0f));
    EXPECT_EQ(model.forward(ids).to_host_vector(), clean);
    model.set_hidden_state_hook(SteeringHook(1, direction, 1.0f));
    EXPECT_NE(model.forward(ids).to_host_vector(), clean);
}

TEST(Steering, TheReportMeasuresPerInputSteerabilityAgainstRandomDirections) {
    CPUBackend cpu;
    // Input i's behavior reads its steered hidden state through its own readout w_i, so its
    // steerability is exactly w_i · v. Three of ten readouts point against v.
    const std::vector<float> v = {1, 0, 0, 0};
    std::vector<std::vector<float>> w;
    for (int i = 0; i < 10; ++i) w.push_back({i < 3 ? -0.5f : 1.0f + 0.1f * static_cast<float>(i), 0.2f, -0.1f, 0.05f});
    int calls = 0;
    auto behavior = [&](int64_t i, const HiddenStateHook& hook) {
        ++calls;
        const Tensor h(Shape({1, 1, 4}), &cpu, std::vector<float>{0.3f, -0.2f, 0.4f, 0.1f});
        const std::vector<float> steered = hook(5, h).to_host_vector();
        double b = 0;
        for (size_t j = 0; j < 4; ++j) b += static_cast<double>(w[static_cast<size_t>(i)][j]) * steered[j];
        return b;
    };
    SteeringOptions o;
    o.random_directions = 4;
    const SteeringReport r = MeasureSteering(behavior, 10, 5, v, o);
    EXPECT_EQ(calls, 10 * 5 * (1 + 4));
    double mean = 0;
    for (int i = 0; i < 10; ++i) {
        EXPECT_NEAR(r.steerability[static_cast<size_t>(i)], w[static_cast<size_t>(i)][0], 1e-6) << i;
        mean += w[static_cast<size_t>(i)][0] / 10.0;
    }
    EXPECT_NEAR(r.mean_steerability, mean, 1e-6);
    EXPECT_DOUBLE_EQ(r.anti_steerable_fraction, 0.3);
    double var = 0;
    for (int i = 0; i < 10; ++i) var += (w[static_cast<size_t>(i)][0] - mean) * (w[static_cast<size_t>(i)][0] - mean) / 10.0;
    EXPECT_NEAR(r.steerability_sd, std::sqrt(var), 1e-6);
    // The mean behavior is linear in the coefficient with slope mean_steerability.
    EXPECT_EQ(r.coefficients, (std::vector<double>{-0.5, -0.25, 0.0, 0.25, 0.5}));  // the default
    EXPECT_NEAR(r.mean_behavior[4] - r.mean_behavior[2], (r.coefficients[4] - r.coefficients[2]) * mean, 1e-5);
    EXPECT_GT(r.random_max_abs_steerability, 0.0);
    EXPECT_NEAR(r.over_random, r.mean_steerability / r.random_max_abs_steerability, 1e-12);
    EXPECT_GT(r.over_random, 1.0);  // v lines up with the readouts; random directions mostly don't

    EXPECT_THROW((void)MeasureSteering(behavior, 0, 5, v), std::invalid_argument);
    EXPECT_THROW((void)MeasureSteering(behavior, 10, 5, {0, 0, 0, 0}), std::invalid_argument);
    SteeringOptions one;
    one.coefficients = {1.0, 1.0};
    EXPECT_THROW((void)MeasureSteering(behavior, 10, 5, v, one), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
