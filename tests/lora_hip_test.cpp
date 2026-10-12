// TRN-7 on the GPU: a PEFT adapter on the tiny Qwen2 gives PEFT's logits through hipBLAS, and an
// adapter's gradients, merge and split relevance (TRN-8) agree with the CPU's.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hip_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lora.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }

std::vector<float> Random(std::size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> v(n);
    for (float& x : v) x = normal(rng);
    return v;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    float worst = 0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

TEST(LoraHip, PeftAdapterGivesPeftsLogits) {
    HIPBackend hip;
    CPUBackend cpu;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Fixture("qwen2"), &hip);
    LoadLoraAdapter(*model, Fixture("qwen2_lora"), HuggingFaceMapping(model->config()));
    const std::vector<float> expected =
        SafetensorsFile::Read(Fixture("qwen2_lora") + "/reference.safetensors").tensor("logits", &cpu).to_host_vector();
    const Tensor tokens(Shape({1, 6}), &hip, std::vector<float>{3, 17, 42, 5, 60, 8}, DeviceType::Hip);
    EXPECT_LT(MaxAbsDiff(model->forward(tokens).to_host_vector(), expected), 1e-4f);
    MergeLora(*model);
    EXPECT_LT(MaxAbsDiff(model->forward(tokens).to_host_vector(), expected), 1e-4f);
}

TEST(LoraHip, GradientsMergeAndRelevanceMatchTheCpu) {
    HIPBackend hip;
    CPUBackend cpu;
    const int64_t n = 64, in = 96, out = 80;
    const std::vector<float> w = Random(static_cast<std::size_t>(in * out), 1), x = Random(static_cast<std::size_t>(n * in), 2),
                             b = Random(static_cast<std::size_t>(8 * out), 3), g = Random(static_cast<std::size_t>(n * out), 4);
    LoRAConfig config;
    config.rank = 8;
    config.alpha = 16;
    std::vector<std::vector<float>> results[2];
    int k = 0;
    for (DeviceBackend* backend : {static_cast<DeviceBackend*>(&cpu), static_cast<DeviceBackend*>(&hip)}) {
        const DeviceType device = backend->device();
        LinearModule l(in, out, backend, device);
        l.set_weight(w);
        l.attach_lora(config);
        l.lora()->set_b(b);
        const Tensor input(Shape({n, in}), backend, x, device);
        auto& r = results[k++];
        r.push_back(l.forward(input).to_host_vector());
        r.push_back(l.backward(Tensor(Shape({n, out}), backend, g, device)).to_host_vector());
        r.push_back(l.lora()->a_grad().to_host_vector());
        r.push_back(l.lora()->b_grad().to_host_vector());
        LRPRuleConfig gamma;
        gamma.rule = LRPRule::Gamma;
        // epsilon 0.1: with the default 1e-6, a z near 0 amplifies the backends' different rounding.
        for (const LRPRuleConfig& rule : {LRPRuleConfig{0.1f}, gamma}) {
            const Tensor relevance(Shape({n, out}), backend, g, device);
            (void)l.forward(input);
            r.push_back(l.propagate_relevance(relevance, rule).to_host_vector());
            r.push_back(l.lora()->last_adapter_relevance().to_host_vector());
        }
        l.merge_lora();
        r.push_back(l.forward(input).to_host_vector());
    }
    for (std::size_t i = 0; i < results[0].size(); ++i) {
        float largest = 0;
        for (float v : results[0][i]) largest = std::max(largest, std::fabs(v));
        EXPECT_LT(MaxAbsDiff(results[0][i], results[1][i]), 1e-5f * largest + 1e-5f) << "result " << i;
    }
}

}  // namespace
}  // namespace pulsatrix
