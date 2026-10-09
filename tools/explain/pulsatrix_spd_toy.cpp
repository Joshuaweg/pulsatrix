// pulsatrix_spd_toy: parameter decomposition of a toy model of superposition (roadmap FEAT-9),
// the first check of SPD (Bushnaq, Braun and Sharkey, arXiv 2506.20790).
//
//   pulsatrix_spd_toy [--features 5] [--hidden 2] [--method spd|vpd] [--steps 40000] [--batch 4096]
//                     [--components 20] [--importance 3e-3] [--p 1] [--lr 1e-3] [--seed 0]
//
// The target is the tied x̂ = ReLU(x W Wᵀ + b) (Elhage et al.), each feature U[0, 1] with
// probability 0.05, trained with Adam at 5e-3 for --target-steps (10000) batches of 1024. W and Wᵀ
// are decomposed separately (untied, as the VPD toy configs do), into
// --components subcomponents each; training uses SPD v1's quarter-cosine learning-rate decay
// (VPD: p annealed from 2 to --p). The reference runs 40000 steps of 4096 (5-2: C 20, importance
// 3e-3, p 1; 40-10: C 200, importance 1e-4, p 2).
//
// Reports:
//   - MMCS and ML2R of W1's rows (one per feature) against the subcomponents;
//   - one feature at a time (x = 0.75 e_j): how many subcomponents of each layer are causally
//     important (above 0.1), whether each feature has its own, and that subcomponent's share of
//     the feature's weight row in W1 (its norm over the row's);
//   - alive subcomponents (above 0.1 on some input) and the mean causal importance L0 on data.
// PULSATRIX_SPD_DUMP=1 also prints every subcomponent's part of each feature's W1 row, with its
// causal importance on that feature alone.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/parameter_decomposition.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_spd_toy: " << problem << "\n"
              << "usage: pulsatrix_spd_toy [--features N] [--hidden N] [--method spd|vpd] [--steps N] [--batch N] [--components C]\n"
              << "       [--importance X] [--p X] [--lr X] [--target-steps N] [--seed S]\n";
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace pulsatrix;
    int64_t F = 5, H = 2, steps = 40000, batch = 4096, C = 20, target_steps = 10000;
    std::string method = "spd";
    float importance = 3e-3f, p = 1.0f, lr = 1e-3f;
    uint64_t seed = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--features") F = std::atoll(value().c_str());
        else if (a == "--hidden") H = std::atoll(value().c_str());
        else if (a == "--method") method = value();
        else if (a == "--steps") steps = std::atoll(value().c_str());
        else if (a == "--batch") batch = std::atoll(value().c_str());
        else if (a == "--components") C = std::atoll(value().c_str());
        else if (a == "--importance") importance = std::strtof(value().c_str(), nullptr);
        else if (a == "--p") p = std::strtof(value().c_str(), nullptr);
        else if (a == "--lr") lr = std::strtof(value().c_str(), nullptr);
        else if (a == "--target-steps") target_steps = std::atoll(value().c_str());
        else if (a == "--seed") seed = std::strtoull(value().c_str(), nullptr, 10);
        else Usage("unknown option " + a);
    }
    if (method != "spd" && method != "vpd") Usage("--method must be spd or vpd");
    const bool vpd = method == "vpd";
    CPUBackend cpu;
    uint64_t s = seed * 2654435761ULL + 99;
    auto u = [&] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(static_cast<uint32_t>(s >> 32)) / 4294967296.0;
    };
    auto data = [&](int64_t n) {
        std::vector<float> x(static_cast<size_t>(n * F), 0.0f);
        for (float& e : x) {
            if (u() < 0.05) e = static_cast<float>(u());
        }
        return Tensor(Shape({n, F}), &cpu, x);
    };

    // The target, tied as Elhage et al. train it: x̂ = ReLU(x W Wᵀ + b), with Adam on W and b.
    LinearModule w1(F, H, &cpu, false), w2(H, F, &cpu);
    ReluModule relu(&cpu);
    {
        std::vector<double> W(static_cast<size_t>(F * H)), b(static_cast<size_t>(F), 0.0), mW(W.size(), 0), vW(W.size(), 0), mb(b.size(), 0),
            vb(b.size(), 0);
        for (double& e : W) e = (u() - 0.5) * 0.6;
        const int64_t n = 1024;
        double last = 0;
        for (int64_t step = 0; step < target_steps; ++step) {
            const std::vector<float> x = data(n).to_host_vector();
            std::vector<double> gW(W.size(), 0.0), gb(b.size(), 0.0);
            last = 0;
            for (int64_t r = 0; r < n; ++r) {
                const float* xr = x.data() + r * F;
                std::vector<double> h(static_cast<size_t>(H), 0.0), dy(static_cast<size_t>(F)), dh(static_cast<size_t>(H), 0.0);
                for (int64_t k = 0; k < H; ++k) {
                    for (int64_t j = 0; j < F; ++j) h[static_cast<size_t>(k)] += xr[j] * W[static_cast<size_t>(j * H + k)];
                }
                for (int64_t j = 0; j < F; ++j) {
                    double pre = b[static_cast<size_t>(j)];
                    for (int64_t k = 0; k < H; ++k) pre += h[static_cast<size_t>(k)] * W[static_cast<size_t>(j * H + k)];
                    const double y = std::max(0.0, pre), e = y - xr[j];
                    last += e * e / static_cast<double>(n * F);
                    dy[static_cast<size_t>(j)] = pre > 0 ? 2.0 * e / static_cast<double>(n * F) : 0.0;
                    gb[static_cast<size_t>(j)] += dy[static_cast<size_t>(j)];
                    for (int64_t k = 0; k < H; ++k) {
                        gW[static_cast<size_t>(j * H + k)] += dy[static_cast<size_t>(j)] * h[static_cast<size_t>(k)];
                        dh[static_cast<size_t>(k)] += dy[static_cast<size_t>(j)] * W[static_cast<size_t>(j * H + k)];
                    }
                }
                for (int64_t j = 0; j < F; ++j) {
                    for (int64_t k = 0; k < H; ++k) gW[static_cast<size_t>(j * H + k)] += xr[j] * dh[static_cast<size_t>(k)];
                }
            }
            const double t = static_cast<double>(step + 1);
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
        std::vector<float> w(W.begin(), W.end()), wt(static_cast<size_t>(H * F));
        for (int64_t j = 0; j < F; ++j) {
            for (int64_t k = 0; k < H; ++k) wt[static_cast<size_t>(k * F + j)] = w[static_cast<size_t>(j * H + k)];
        }
        w1.set_weight(w);
        w2.set_weight(wt);
        w2.set_bias(std::vector<float>(b.begin(), b.end()));
        int64_t represented = 0;
        for (int64_t j = 0; j < F; ++j) {
            double sq = 0;
            for (int64_t k = 0; k < H; ++k) sq += W[static_cast<size_t>(j * H + k)] * W[static_cast<size_t>(j * H + k)];
            represented += sq > 0.25 ? 1 : 0;
        }
        std::printf("target: TMS %ld-%ld, tied, reconstruction MSE %.2e; %ld features with |W row| > 0.5\n", static_cast<long>(F),
                    static_cast<long>(H), last, static_cast<long>(represented));
    }

    // The decomposition.
    ComponentLinear c1(w1, C, &cpu, 16, seed + 1), c2(w2, C, &cpu, 16, seed + 2);
    SequentialModule model({&c1, &c2, &relu});
    DecompositionOptions o;
    o.method = vpd ? DecompositionMethod::VPD : DecompositionMethod::SPD;
    o.importance_coefficient = importance;
    o.p = p;
    o.stochastic_coefficient = vpd ? 0.5f : 1.0f;
    o.seed = seed;
    ParameterDecomposition d(model, {&c1, &c2}, &cpu, o);
    AdamOptimizer opt(lr, &cpu);
    DecompositionLoss l;
    for (int64_t step = 0; step < steps; ++step) {
        const float t = steps > 1 ? static_cast<float>(step) / static_cast<float>(steps - 1) : 0.0f;
        opt.set_learning_rate(lr * std::cos(1.5707964f * t));
        if (vpd) d.set_p(2.0f + (p - 2.0f) * t);
        l = TrainDecomposition(d, data(batch), opt);
        if ((step + 1) % std::max<int64_t>(1, steps / 10) == 0) {
            std::printf("step %ld: faithfulness %.2e, reconstruction %.2e, %s %.2e, importance %.3f\n", static_cast<long>(step + 1), l.faithfulness,
                        l.stochastic, vpd ? "adversarial" : "layerwise", vpd ? l.adversarial : l.layerwise, l.importance);
            std::fflush(stdout);
        }
    }

    const ComponentAlignment a = AlignComponentsToRows(c1);
    const std::vector<ImportanceStats> st = MeasureImportance(d, data(8192));
    std::printf("\n%s, %ld steps of %ld: MMCS %.3f, ML2R %.3f; alive %ld + %ld of %ld; causal importance L0 on data %.2f + %.2f\n",
                vpd ? "VPD" : "SPD", static_cast<long>(steps), static_cast<long>(batch), a.mean_max_cosine, a.mean_norm_ratio,
                static_cast<long>(st[0].alive), static_cast<long>(st[1].alive), static_cast<long>(C), st[0].l0, st[1].l0);

    // One feature at a time.
    std::vector<float> onehot(static_cast<size_t>(F * F), 0.0f);
    for (int64_t j = 0; j < F; ++j) onehot[static_cast<size_t>(j * F + j)] = 0.75f;
    const std::vector<std::vector<float>> ci = d.causal_importances(Tensor(Shape({F, F}), &cpu, onehot));
    const std::vector<float> v = c1.V().weight().to_host_vector(), uw = c1.U().weight().to_host_vector();
    const std::vector<float>& w = c1.target_weight();
    std::set<int64_t> owners;
    double important[2] = {0, 0}, share = 0;
    int64_t single = 0;
    for (int64_t j = 0; j < F; ++j) {
        int64_t n[2] = {0, 0}, top = 0;
        float best = -1;
        for (int layer = 0; layer < 2; ++layer) {
            for (int64_t c = 0; c < C; ++c) {
                const float g = ci[static_cast<size_t>(layer)][static_cast<size_t>(j * C + c)];
                n[layer] += g > 0.1f ? 1 : 0;
                if (layer == 0 && g > best) {
                    best = g;
                    top = c;
                }
            }
            important[layer] += static_cast<double>(n[layer]) / static_cast<double>(F);
        }
        single += n[0] == 1 && n[1] == 1 ? 1 : 0;
        owners.insert(top);
        double row = 0, comp = 0;
        for (int64_t k = 0; k < H; ++k) {
            row += static_cast<double>(w[static_cast<size_t>(j * H + k)]) * w[static_cast<size_t>(j * H + k)];
            const double cw = static_cast<double>(v[static_cast<size_t>(j * C + top)]) * uw[static_cast<size_t>(top * H + k)];
            comp += cw * cw;
        }
        share += (row > 0 ? std::sqrt(comp / row) : 0) / static_cast<double>(F);
    }
    std::printf("one feature at a time: important subcomponents per feature %.2f in W1, %.2f in W2; features with exactly one in each "
                "%ld of %ld; distinct W1 subcomponents %zu; the top one's share of the feature's W1 row %.3f\n",
                important[0], important[1], static_cast<long>(single), static_cast<long>(F), owners.size(), share);
    if (std::getenv("PULSATRIX_SPD_DUMP") != nullptr) {
        for (int64_t j = 0; j < F; ++j) {
            std::printf("feature %ld: W1 row (", static_cast<long>(j));
            for (int64_t k = 0; k < H; ++k) std::printf("%s%.3f", k ? ", " : "", w[static_cast<size_t>(j * H + k)]);
            std::printf(")\n");
            for (int64_t c = 0; c < C; ++c) {
                double n = 0;
                for (int64_t k = 0; k < H; ++k) {
                    const double cw = static_cast<double>(v[static_cast<size_t>(j * C + c)]) * uw[static_cast<size_t>(c * H + k)];
                    n += cw * cw;
                }
                if (std::sqrt(n) < 0.05) continue;
                std::printf("   c %2ld: row (", static_cast<long>(c));
                for (int64_t k = 0; k < H; ++k) std::printf("%s%.3f", k ? ", " : "", v[static_cast<size_t>(j * C + c)] * uw[static_cast<size_t>(c * H + k)]);
                std::printf("), V %.3f, |U| %.3f, CI on the feature %.3f\n", v[static_cast<size_t>(j * C + c)], std::sqrt(n) / std::abs(v[static_cast<size_t>(j * C + c)]),
                            ci[0][static_cast<size_t>(j * C + c)]);
            }
        }
    }
    return 0;
}
