// KS-9: torchvision ResNet and VGG in pulsatrix, checked against torchvision's logits and Zennit
// 1.0.0's heatmaps (tools/golden/make_vision_golden.py). Shared by vision_models_test.cpp (CPU)
// and the GPU suites, which run the same cases on their own backend.
#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/vision_models.hpp"

namespace pulsatrix::vision_cases {

inline std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/vision/" + name; }

inline std::vector<float> Host(const Tensor& t) {
    CPUBackend cpu;
    if (t.device() == DeviceType::Cpu) return t.to_host_vector();
    Tensor host = t;
    host.to(DeviceType::Cpu, &cpu);
    return host.to_host_vector();
}

/** @brief Every element within @p rel of the reference's largest magnitude. */
inline void ExpectClose(const Tensor& actual, const std::vector<float>& expected, float rel, const std::string& what) {
    const std::vector<float> a = Host(actual);
    ASSERT_EQ(a.size(), expected.size()) << what;
    float scale = 0.0f;
    for (float e : expected) scale = std::max(scale, std::fabs(e));
    const float tol = rel * std::max(scale, 1e-12f);
    size_t bad = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!(std::fabs(a[i] - expected[i]) <= tol) && bad++ < 5) {
            ADD_FAILURE() << what << " element " << i << ": " << a[i] << " vs " << expected[i] << " (tol " << tol << ")";
        }
    }
    EXPECT_EQ(bad, 0u) << what;
}

struct CompositeCase {
    const char* name;
    LRP lrp;
};

inline std::vector<CompositeCase> ZennitPresets() {
    return {{"EpsilonPlus", LRP::epsilon_plus()},
            {"EpsilonAlpha2Beta1", LRP::epsilon_alpha2_beta1()},
            {"EpsilonGammaBox", LRP::epsilon_gamma_box(-3.0f, 3.0f)}};
}

/**
 * @brief Loads @p weights into @p model, checks the logits and the three Zennit presets'
 *        heatmaps against @p golden. A ResNet's BatchNorms are folded while it is explained.
 */
template <typename Model>
void CheckAgainstGolden(Model& model, const std::string& weights, const std::string& golden, const std::vector<int64_t>& targets,
                        DeviceBackend* backend, float logit_rel = 1e-4f, float relevance_rel = 1e-3f) {
    LoadTorchvisionWeights(model, weights);
    model.set_training(false);
    const SafetensorsFile ref = SafetensorsFile::Map(golden);
    CPUBackend cpu;
    // On the caller's backend: the model caches its input, so it can't hold this function's.
    const Tensor x_host = ref.tensor("input", &cpu);
    const Tensor x(x_host.shape(), backend, x_host.to_host_vector(), backend->device());

    ExpectClose(model.forward(x), ref.tensor("logits", &cpu).to_host_vector(), logit_rel, "logits");

    std::vector<std::unique_ptr<BatchNormFold>> folds;
    if constexpr (std::is_same_v<Model, TorchvisionResNet>) folds = model.fold_batch_norms();
    ExplainerContext ctx(model.layers());
    for (CompositeCase& c : ZennitPresets()) {
        SCOPED_TRACE(c.name);
        const Attribution a = c.lrp.explain(ctx, x, LRPTarget{targets, {}, LRPSeed::OneHot}, backend);
        ExpectClose(a.values, ref.tensor(std::string("relevance.") + c.name, &cpu).to_host_vector(), relevance_rel, c.name);
    }
}

inline ResNetConfig TinyResNetConfig() {
    ResNetConfig c;
    c.blocks = {1, 2};
    c.width = 4;
    c.num_classes = 5;
    return c;
}

inline VGGConfig TinyVGGConfig() {
    VGGConfig c;
    c.features = {4, 0, 8, 8, 0};
    c.pool_size = 2;
    c.hidden = 6;
    c.num_classes = 5;
    return c;
}

inline void TinyResNetMatchesTorchvisionAndZennit(DeviceBackend* backend) {
    TorchvisionResNet model(TinyResNetConfig(), backend);
    CheckAgainstGolden(model, Fixture("tiny_resnet.safetensors"), Fixture("tiny_resnet_golden.safetensors"), {1, 3}, backend);
}

inline void TinyVGGMatchesTorchvisionAndZennit(DeviceBackend* backend) {
    TorchvisionVGG model(TinyVGGConfig(), backend);
    CheckAgainstGolden(model, Fixture("tiny_vgg.safetensors"), Fixture("tiny_vgg_golden.safetensors"), {0, 4}, backend);
}

/** @brief The published ImageNet models, when PULSATRIX_TORCHVISION_DIR holds the converted
 *         weights and the generator's goldens; skipped otherwise. */
inline void PublishedModelMatches(const std::string& name, DeviceBackend* backend) {
    const char* dir = std::getenv("PULSATRIX_TORCHVISION_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set PULSATRIX_TORCHVISION_DIR to run the published " << name;
    const std::string weights = std::string(dir) + "/" + name + ".safetensors", golden = std::string(dir) + "/" + name + "_golden.safetensors";
    CPUBackend cpu;
    const SafetensorsFile ref = SafetensorsFile::Map(golden);
    auto [ptr, size] = ref.bytes("targets");
    std::vector<int64_t> targets(size / sizeof(int64_t));
    std::memcpy(targets.data(), ptr, size);
    if (name == "resnet18") {
        TorchvisionResNet model(TorchvisionResNet::ResNet18(), backend);
        CheckAgainstGolden(model, weights, golden, targets, backend, 1e-4f, 2e-3f);
    } else {
        TorchvisionVGG model(TorchvisionVGG::VGG16(), backend);
        CheckAgainstGolden(model, weights, golden, targets, backend, 1e-4f, 2e-3f);
    }
}

}  // namespace pulsatrix::vision_cases
