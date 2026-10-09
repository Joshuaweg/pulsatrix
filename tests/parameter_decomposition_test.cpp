// FEAT-9: parameter decomposition (SPD and VPD), against a PyTorch rendering
// (tools/golden/make_spd_golden.py), and on a toy model of superposition.
#include "pulsatrix/parameter_decomposition.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Floats(const SafetensorsFile& f, const std::string& name) {
    CPUBackend cpu;
    return f.tensor(name, &cpu).to_host_vector();
}

double MaxDiff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    double d = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) d = std::max(d, std::abs(static_cast<double>(a[i]) - b[i]));
    return d;
}

const char* kParamNames[] = {"V", "U", "gin", "gib", "gout", "gob"};

/** @brief The golden's two-layer model, decomposed, with its noise. */
struct GoldenModel {
    CPUBackend* cpu;
    LinearModule l0, l1;
    ReluModule relu;
    ComponentLinear c0, c1;
    SequentialModule model;

    GoldenModel(const SafetensorsFile& g, CPUBackend* backend)
        : cpu(backend),
          l0(3, 4, backend),
          l1(4, 2, backend),
          relu(backend),
          c0((Init(l0, g, 0), l0), 5, backend, 3),
          c1((Init(l1, g, 1), l1), 5, backend, 3),
          model({&c0, &relu, &c1}) {
        Load(c0, g, 0);
        Load(c1, g, 1);
    }
    static void Init(LinearModule& l, const SafetensorsFile& g, int i) {
        l.set_weight(Floats(g, "w" + std::to_string(i)));
        l.set_bias(Floats(g, "b" + std::to_string(i)));
    }
    static void Load(ComponentLinear& c, const SafetensorsFile& g, int l) {
        std::vector<NamedParamRef> p = c.named_parameters();
        for (size_t j = 0; j < 6; ++j) {
            *p[j].ref.value = Tensor(p[j].ref.value->shape(), p[j].ref.value->backend(), Floats(g, "init." + std::to_string(l) + "." + kParamNames[j]));
        }
    }
};

ParameterDecomposition::NoiseFn GoldenNoise(const SafetensorsFile& g, const std::string& method) {
    return [&g, method](const char* pass, int64_t layer, int64_t count) {
        const std::string p = pass;
        std::vector<float> v = p == "route" ? Floats(g, "route") : Floats(g, method + "." + p + "." + std::to_string(layer));
        EXPECT_EQ(static_cast<int64_t>(v.size()), count) << pass;
        return v;
    };
}

DecompositionOptions GoldenOptions(const std::string& method) {
    DecompositionOptions o;
    if (method == "spd") {
        o.importance_coefficient = 3e-3f;
        o.p = 1.0f;
        return o;
    }
    o.method = DecompositionMethod::VPD;
    o.importance_coefficient = 2e-2f;
    o.p = 0.9f;
    o.adversarial_coefficient = 0.5f;
    o.pgd_steps = 0;  // the sources are the golden's
    return o;
}

void SetSources(ParameterDecomposition& d, const SafetensorsFile& g) {
    d.adversarial_sources() = {Floats(g, "vpd.sources.0"), Floats(g, "vpd.sources.1")};
}

class DecompositionGolden : public ::testing::TestWithParam<std::string> {};

TEST_P(DecompositionGolden, MatchesPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = SafetensorsFile::Map(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/spd/spd_golden.safetensors");
    const std::string method = GetParam();
    const Tensor x(Shape({4, 3}), &cpu, Floats(g, "x"));
    GoldenModel m(g, &cpu);
    ParameterDecomposition d(m.model, {&m.c0, &m.c1}, &cpu, GoldenOptions(method));
    d.set_noise(GoldenNoise(g, method));
    if (method == "vpd") SetSources(d, g);
    AdamOptimizer opt(1e-3f, &cpu);
    opt.zero_grad(d.parameters_module());
    const DecompositionLoss loss = d.loss_and_backward(x);
    EXPECT_NEAR(loss.total, Floats(g, method + ".total")[0], 2e-5);
    EXPECT_NEAR(loss.faithfulness, Floats(g, method + ".faithfulness")[0], 1e-5);
    EXPECT_NEAR(loss.stochastic, Floats(g, method + ".stochastic")[0], 1e-5);
    EXPECT_NEAR(loss.importance, Floats(g, method + ".importance")[0], 2e-5);
    if (method == "spd") EXPECT_NEAR(loss.layerwise, Floats(g, "spd.layerwise")[0], 1e-5);
    else EXPECT_NEAR(loss.adversarial, Floats(g, "vpd.adversarial")[0], 1e-5);
    const std::vector<NamedParamRef> p = d.parameters_module().named_parameters();
    ASSERT_EQ(p.size(), 12u);
    for (size_t i = 0; i < 12; ++i) {
        const std::string want = method + ".grad." + std::to_string(i / 6) + "." + kParamNames[i % 6];
        EXPECT_LT(MaxDiff(p[i].ref.grad->to_host_vector(), Floats(g, want)), 2e-5) << p[i].name;
    }
    // One Adam step, from fresh state.
    GoldenModel m2(g, &cpu);
    ParameterDecomposition d2(m2.model, {&m2.c0, &m2.c1}, &cpu, GoldenOptions(method));
    d2.set_noise(GoldenNoise(g, method));
    if (method == "vpd") SetSources(d2, g);
    AdamOptimizer opt2(1e-3f, &cpu);
    (void)TrainDecomposition(d2, x, opt2);
    const std::vector<NamedParamRef> q = d2.parameters_module().named_parameters();
    for (size_t i = 0; i < 12; ++i) {
        const std::string want = method + ".after." + std::to_string(i / 6) + "." + kParamNames[i % 6];
        EXPECT_LT(MaxDiff(q[i].ref.value->to_host_vector(), Floats(g, want)), 1e-5) << q[i].name;
    }
}

INSTANTIATE_TEST_SUITE_P(Methods, DecompositionGolden, ::testing::Values("spd", "vpd"));

std::vector<float> Wave(int64_t n, float a, float b) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = std::sin(a * static_cast<float>(i) + b);
    return v;
}

TEST(ComponentLinear, MasksOfOneWithDeltaGiveTheTargetAndBackwardMatchesFiniteDifferences) {
    CPUBackend cpu;
    LinearModule target(4, 3, &cpu);
    target.set_weight(Wave(12, 0.7f, 0.3f));
    target.set_bias({0.1f, -0.2f, 0.05f});
    ComponentLinear c(target, 6, &cpu, 4, 3);
    const Tensor x(Shape({5, 4}), &cpu, Wave(20, 0.41f, 1.0f));
    const std::vector<float> want = target.forward(x).to_host_vector();
    c.use_target();
    EXPECT_LT(MaxDiff(c.forward(x).to_host_vector(), want), 1e-6);
    c.use_components(std::vector<float>(30, 1.0f), std::vector<float>(5, 1.0f));
    EXPECT_LT(MaxDiff(c.forward(x).to_host_vector(), want), 1e-5);
    // Without Δ, masks of 1 give V U.
    c.use_components(std::vector<float>(30, 1.0f));
    const std::vector<float> vu = c.component_weight(), y = c.forward(x).to_host_vector(), xv = x.to_host_vector();
    for (int64_t r = 0; r < 5; ++r) {
        for (int64_t k = 0; k < 3; ++k) {
            double s = target.bias().to_host_vector()[static_cast<size_t>(k)];
            for (int64_t j = 0; j < 4; ++j) s += static_cast<double>(xv[static_cast<size_t>(r * 4 + j)]) * vu[static_cast<size_t>(j * 3 + k)];
            EXPECT_NEAR(y[static_cast<size_t>(r * 3 + k)], s, 1e-5);
        }
    }
    // Backward: the input's gradient and the masks', against finite differences of 0.5 |y|².
    std::vector<float> masks = Wave(30, 0.9f, 0.2f), delta = Wave(5, 1.3f, 0.5f);
    for (float& e : masks) e = 0.5f + 0.4f * e;
    for (float& e : delta) e = 0.5f + 0.4f * e;
    auto f = [&](const std::vector<float>& in, const std::vector<float>& m, const std::vector<float>& md) {
        c.use_components(m, md);
        double s = 0;
        for (float v : c.forward(Tensor(Shape({5, 4}), &cpu, in)).to_host_vector()) s += 0.5 * v * v;
        return s;
    };
    c.use_components(masks, delta);
    const std::vector<float> out = c.forward(x).to_host_vector();
    const std::vector<float> gx = c.backward(Tensor(Shape({5, 3}), &cpu, out)).to_host_vector();
    const std::vector<float> gm = c.mask_grad(), gd = c.delta_mask_grad();
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> up = xv, down = xv;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(gx[i], (f(up, masks, delta) - f(down, masks, delta)) / 2e-3, 2e-3) << "x " << i;
    }
    for (size_t i = 0; i < masks.size(); ++i) {
        std::vector<float> up = masks, down = masks;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(gm[i], (f(xv, up, delta) - f(xv, down, delta)) / 2e-3, 2e-3) << "m " << i;
    }
    for (size_t i = 0; i < delta.size(); ++i) {
        std::vector<float> up = delta, down = delta;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(gd[i], (f(xv, masks, up) - f(xv, masks, down)) / 2e-3, 2e-3) << "Δ " << i;
    }
    EXPECT_THROW(ComponentLinear(target, 0, &cpu), std::invalid_argument);
    c.use_components(std::vector<float>(7, 1.0f));
    EXPECT_THROW((void)c.forward(x), std::invalid_argument);
}

TEST(ParameterDecomposition, TheKlLossOnLogitsHasTheRightGradient) {
    CPUBackend cpu;
    LinearModule l0(3, 5, &cpu), l1(5, 4, &cpu);
    l0.set_weight(Wave(15, 0.8f, 0.1f));
    l1.set_weight(Wave(20, 0.33f, 0.7f));
    ReluModule relu(&cpu);
    ComponentLinear c0(l0, 4, &cpu, 3, 1), c1(l1, 4, &cpu, 3, 2);
    SequentialModule model({&c0, &relu, &c1});
    DecompositionOptions o;
    o.divergence = OutputDivergence::KlOnLogits;
    o.importance_coefficient = 0.01f;
    o.p = 2.0f;
    ParameterDecomposition d(model, {&c0, &c1}, &cpu, o);
    const std::vector<float> noise = Wave(64, 0.77f, 0.3f);
    d.set_noise([&](const char*, int64_t, int64_t count) {
        std::vector<float> v(noise.begin(), noise.begin() + count);
        for (float& e : v) e = 0.5f + 0.45f * e;
        return v;
    });
    const Tensor x(Shape({3, 3}), &cpu, Wave(9, 0.6f, 0.9f));
    AdamOptimizer opt(1e-3f, &cpu);
    opt.zero_grad(d.parameters_module());
    (void)d.loss_and_backward(x);
    // Finite differences of the total loss in a few V, U and gate parameters (the gradients
    // are read first: every loss_and_backward() adds to them).
    std::vector<NamedParamRef> p = d.parameters_module().named_parameters();
    std::vector<std::vector<float>> grads;
    for (const NamedParamRef& r : p) grads.push_back(r.ref.grad->to_host_vector());
    for (size_t which : {0u, 1u, 2u, 5u, 6u, 7u}) {
        Tensor* value = p[which].ref.value;
        const std::vector<float>& grad = grads[which];
        const std::vector<float> base = value->to_host_vector();
        for (size_t i = 0; i < std::min<size_t>(base.size(), 3); ++i) {
            auto total = [&](float delta) {
                std::vector<float> v = base;
                v[i] += delta;
                *value = Tensor(value->shape(), value->backend(), v);
                const float t = d.loss_and_backward(x).total;
                *value = Tensor(value->shape(), value->backend(), base);
                return static_cast<double>(t);
            };
            EXPECT_NEAR(grad[i], (total(1e-3f) - total(-1e-3f)) / 2e-3, 2e-3) << p[which].name << " " << i;
        }
    }
}


/** @brief Toy model of superposition (Elhage et al.), tied as they train it: 5 sparse features
 *         through 2 dimensions, x̂ = ReLU(x W Wᵀ + b), each feature U[0, 1] with probability
 *         0.05. Returns the layers W and Wᵀ (with b) to decompose separately. */
struct Tms {
    CPUBackend* cpu;
    LinearModule w1, w2;
    ReluModule relu;
    uint64_t s;
    Tms(CPUBackend* backend, uint64_t seed) : cpu(backend), w1(5, 2, backend, false), w2(2, 5, backend), relu(backend), s(seed) {}
    double u() {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(static_cast<uint32_t>(s >> 32)) / 4294967296.0;
    }
    Tensor Batch(int64_t n) {
        std::vector<float> x(static_cast<size_t>(n * 5), 0.0f);
        for (float& e : x) {
            if (u() < 0.05) e = static_cast<float>(u());
        }
        return Tensor(Shape({n, 5}), cpu, x);
    }
    void Train(int steps) {
        const int64_t F = 5, H = 2, n = 1024;
        std::vector<double> W(10), b(5, 0.0), mW(10, 0), vW(10, 0), mb(5, 0), vb(5, 0);
        for (double& e : W) e = (u() - 0.5) * 0.6;
        for (int step = 0; step < steps; ++step) {
            const std::vector<float> x = Batch(n).to_host_vector();
            std::vector<double> gW(10, 0.0), gb(5, 0.0);
            for (int64_t r = 0; r < n; ++r) {
                const float* xr = x.data() + r * F;
                double h[2] = {0, 0}, dh[2] = {0, 0};
                for (int64_t k = 0; k < H; ++k) {
                    for (int64_t j = 0; j < F; ++j) h[k] += xr[j] * W[static_cast<size_t>(j * H + k)];
                }
                for (int64_t j = 0; j < F; ++j) {
                    double pre = b[static_cast<size_t>(j)];
                    for (int64_t k = 0; k < H; ++k) pre += h[k] * W[static_cast<size_t>(j * H + k)];
                    if (pre <= 0) continue;
                    const double dy = 2.0 * (pre - xr[j]) / static_cast<double>(n * F);
                    gb[static_cast<size_t>(j)] += dy;
                    for (int64_t k = 0; k < H; ++k) {
                        gW[static_cast<size_t>(j * H + k)] += dy * h[k];
                        dh[k] += dy * W[static_cast<size_t>(j * H + k)];
                    }
                }
                for (int64_t j = 0; j < F; ++j) {
                    for (int64_t k = 0; k < H; ++k) gW[static_cast<size_t>(j * H + k)] += xr[j] * dh[k];
                }
            }
            const double t = step + 1;
            auto adam = [&](std::vector<double>& p, const std::vector<double>& g, std::vector<double>& m, std::vector<double>& v) {
                for (size_t i = 0; i < p.size(); ++i) {
                    m[i] = 0.9 * m[i] + 0.1 * g[i];
                    v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i];
                    p[i] -= 5e-3 * (m[i] / (1 - std::pow(0.9, t))) / (std::sqrt(v[i] / (1 - std::pow(0.999, t))) + 1e-8);
                }
            };
            adam(W, gW, mW, vW);
            adam(b, gb, mb, vb);
        }
        std::vector<float> w(W.begin(), W.end()), wt(10);
        for (int64_t j = 0; j < F; ++j) {
            for (int64_t k = 0; k < H; ++k) wt[static_cast<size_t>(k * F + j)] = w[static_cast<size_t>(j * H + k)];
        }
        w1.set_weight(w);
        w2.set_weight(wt);
        w2.set_bias(std::vector<float>(b.begin(), b.end()));
    }
};

TEST(ParameterDecomposition, GivesEachFeatureOfAToyModelOfSuperpositionItsOwnSubcomponent) {
#ifndef NDEBUG
    GTEST_SKIP() << "3,000 SPD steps take minutes at -O0; runs in Release builds (CI)";
#endif
    CPUBackend cpu;
    Tms tms(&cpu, 3);
    tms.Train(3000);
    ComponentLinear c1(tms.w1, 20, &cpu, 16, 1), c2(tms.w2, 20, &cpu, 16, 2);
    SequentialModule model({&c1, &c2, &tms.relu});
    DecompositionOptions o;
    o.importance_coefficient = 1e-2f;
    ParameterDecomposition d(model, {&c1, &c2}, &cpu, o);
    // SPD v1's quarter-cosine decay; a higher rate and penalty than its 40,000 steps of 4,096 use.
    AdamOptimizer opt(1e-2f, &cpu);
    const int steps = 3000;
    for (int step = 0; step < steps; ++step) {
        opt.set_learning_rate(1e-2f * std::cos(1.5707964f * static_cast<float>(step) / static_cast<float>(steps - 1)));
        (void)TrainDecomposition(d, tms.Batch(1024), opt);
    }
    const ComponentAlignment a = AlignComponentsToRows(c1);
    const std::vector<ImportanceStats> st = MeasureImportance(d, tms.Batch(8192));
    // One feature at a time: which subcomponents of each layer it needs.
    std::vector<float> onehot(25, 0.0f);
    for (int j = 0; j < 5; ++j) onehot[static_cast<size_t>(j * 5 + j)] = 0.75f;
    const std::vector<std::vector<float>> ci = d.causal_importances(Tensor(Shape({5, 5}), &cpu, onehot));
    const std::vector<float> v = c1.V().weight().to_host_vector(), uw = c1.U().weight().to_host_vector(), &w = c1.target_weight();
    std::vector<int64_t> owner;
    int single = 0;
    double share = 0;
    for (int64_t j = 0; j < 5; ++j) {
        int n[2] = {0, 0};
        int64_t top = 0;
        for (int layer = 0; layer < 2; ++layer) {
            for (int64_t c = 0; c < 20; ++c) {
                const float g = ci[static_cast<size_t>(layer)][static_cast<size_t>(j * 20 + c)];
                n[layer] += g > 0.1f ? 1 : 0;
                if (layer == 0 && g > ci[0][static_cast<size_t>(j * 20 + top)]) top = c;
            }
        }
        single += n[0] == 1 && n[1] == 1 ? 1 : 0;
        owner.push_back(top);
        double row = 0, comp = 0;
        for (int64_t k = 0; k < 2; ++k) {
            row += static_cast<double>(w[static_cast<size_t>(j * 2 + k)]) * w[static_cast<size_t>(j * 2 + k)];
            const double cw = static_cast<double>(v[static_cast<size_t>(j * 20 + top)]) * uw[static_cast<size_t>(top * 2 + k)];
            comp += cw * cw;
        }
        share += std::sqrt(comp / row) / 5;
    }
    std::sort(owner.begin(), owner.end());
    const auto distinct = std::unique(owner.begin(), owner.end()) - owner.begin();
    std::printf("[SPD] TMS 5-2: MMCS %.3f; alive %ld + %ld of 20; one important subcomponent per layer for %d of 5 features, %ld distinct; "
                "their share of the feature's row %.3f\n",
                a.mean_max_cosine, static_cast<long>(st[0].alive), static_cast<long>(st[1].alive), single, static_cast<long>(distinct), share);
    EXPECT_GT(a.mean_max_cosine, 0.97);
    EXPECT_EQ(distinct, 5);       // every feature has its own subcomponent
    EXPECT_GT(share, 0.9);        // which carries its weight row: 0.93 to 1.00
    EXPECT_LE(st[0].alive, 8);    // of 20
    EXPECT_LE(st[1].alive, 8);
}

}  // namespace
}  // namespace pulsatrix
