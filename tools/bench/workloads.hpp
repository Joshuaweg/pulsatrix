// Fixed models and inputs shared by pulsatrix_bench (KS-2) and hip_profile_workloads (HIP-1), so
// both measure exactly the same work on any backend.
//
//   mlp      Linear(256->512) -> ReLU -> Linear(512->512) -> ReLU -> Linear(512->10), batch 64
//   cnn      Conv2D(3->16) -> BatchNorm -> ReLU -> Conv2D(16->32, stride 2) -> BatchNorm -> ReLU
//            -> Flatten -> Linear, on 32 images of 3x32x32: im2col, col2im, per-channel norms
//   convnet  cnn without the BatchNorms, for explanations (LRP has no BatchNorm rule; FND-5
//            folds BatchNorm away first)
#pragma once

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix::bench {

// Small deterministic values: matrices +-1/sqrt(fan_in), 1-D parameters 1 (norm scales) or 0
// (biases, so LRP's conservation is exact up to the epsilon stabilizer).
inline void InitParameters(Module& m) {
    uint64_t s = 12345;
    auto uniform = [&s] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(s >> 40) / static_cast<float>(1ULL << 24);
    };
    for (const NamedParamRef& p : m.named_parameters()) {
        Tensor& t = *p.ref.value;
        std::vector<float> v(static_cast<size_t>(t.numel()));
        if (t.rank() >= 2) {
            const float a = 1.0f / std::sqrt(static_cast<float>(t.numel() / t.shape().dim(t.rank() - 1)));
            for (float& x : v) x = (2.0f * uniform() - 1.0f) * a;
        } else {
            const float fill = p.name.find("weight") != std::string::npos ? 1.0f : 0.0f;
            for (float& x : v) x = fill;
        }
        t = Tensor(t.shape(), t.backend(), v, t.device());
    }
}

inline std::vector<float> Pattern(size_t n, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * static_cast<float>(static_cast<int>((i * 7919) % 17) - 8);
    return v;
}

// A classifier, its layers (for an ExplainerContext), a batch and its labels.
struct Classifier {
    std::vector<std::unique_ptr<Module>> owned;
    std::vector<Module*> layers;
    std::unique_ptr<SequentialModule> model;
    Tensor x;
    Tensor y;
    // One input of x, batch dimension kept, for explanations.
    Tensor single;
};

inline Tensor Labels(int64_t n, DeviceBackend* backend) {
    std::vector<float> labels(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) labels[static_cast<size_t>(i)] = static_cast<float>(i % 10);
    return Tensor(Shape({n}), backend, labels);
}

template <typename T, typename... Args>
T* Add(Classifier& c, Args&&... args) {
    c.owned.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    T* m = static_cast<T*>(c.owned.back().get());
    c.layers.push_back(m);
    return m;
}

inline Classifier MakeMlp(DeviceBackend* backend) {
    constexpr int64_t n = 64;
    Classifier c{{}, {}, nullptr, Tensor(Shape({n, 256}), backend, Pattern(n * 256, 0.05f)), Labels(n, backend),
                 Tensor(Shape({1, 256}), backend, Pattern(256, 0.05f))};
    Add<LinearModule>(c, 256, 512, backend);
    Add<ReluModule>(c, backend);
    Add<LinearModule>(c, 512, 512, backend);
    Add<ReluModule>(c, backend);
    Add<LinearModule>(c, 512, 10, backend);
    c.model = std::make_unique<SequentialModule>(c.layers);
    InitParameters(*c.model);
    return c;
}

inline Classifier MakeConvNet(DeviceBackend* backend, bool batch_norm) {
    constexpr int64_t n = 32;
    Classifier c{{}, {}, nullptr, Tensor(Shape({n, 3, 32, 32}), backend, Pattern(n * 3 * 32 * 32, 0.1f)),
                 Labels(n, backend), Tensor(Shape({1, 3, 32, 32}), backend, Pattern(3 * 32 * 32, 0.1f))};
    Add<Conv2DModule>(c, 3, 16, 3, 3, backend, 1, 1);
    if (batch_norm) Add<BatchNormModule>(c, 16, backend);
    Add<ReluModule>(c, backend);
    Add<Conv2DModule>(c, 16, 32, 3, 3, backend, 2, 1);
    if (batch_norm) Add<BatchNormModule>(c, 32, backend);
    Add<ReluModule>(c, backend);
    Add<FlattenModule>(c, backend);
    Add<LinearModule>(c, 32 * 16 * 16, 10, backend);
    c.model = std::make_unique<SequentialModule>(c.layers);
    InitParameters(*c.model);
    return c;
}

inline Classifier MakeCnn(DeviceBackend* backend) { return MakeConvNet(backend, true); }

}  // namespace pulsatrix::bench
