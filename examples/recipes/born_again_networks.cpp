// Recipe: Is the student born better? (KD-4)
//
// Furlanello et al. (ICML 2018) trained a network on the labels, then a second network of the same
// size on the first one's outputs, then a third on the second's, and found the students often beat
// their teachers. This recipe repeats that on MNIST, with a small fully connected network, over
// several seeds, with three controls:
//
//   - a second network trained on the labels from another start, so the gain can be set against
//     the spread between two runs that differ only in their seed;
//   - Furlanello et al.'s permuted teacher (DKPP): the student of generation 1 again, from the same
//     start and with the same batches, but taught by generation 0's outputs with every class but the
//     top one shuffled. It keeps the teacher's answer and confidence and scrambles the rest;
//   - an ensemble of three networks trained on the labels, the fair comparison for the ensemble of
//     three born-again generations.
//
// And one question distillation's accuracy can't answer: does the student use its teacher's
// evidence? The gradient of the true class with respect to the pixels (saliency), on held-out
// digits, compared between generation 0 and generation 1, and between generation 0 and the other
// network trained on the labels (KD-2). Not gradient x input: it is zero on every blank pixel, the
// same pixels for every network, and with it every comparison in a first run came out at a
// Spearman correlation of 0.995, the students' and the independent network's alike.
//
// The predictions, fixed before the run:
//   1. Generation 1 is more accurate than generation 0 on average, by less than one percentage
//      point, and not on every seed.
//   2. The permuted teacher's student is less accurate than generation 1 on average: the gain, if
//      any, comes from what the teacher says about the wrong classes.
//   3. The ensemble of three label-trained networks is at least as accurate as the ensemble of three
//      born-again generations, which learned from one another and so make more of the same errors.
//   4. Generation 1's explanations agree with generation 0's more than the other label-trained
//      network's do, on every seed: the student learns the teacher's evidence, not only its answers.
//
// Usage: born_again_networks_recipe MNIST_DIR OUT_DIR [--device cpu|hip] [--seeds N]
//            [--train N] [--epochs N] [--generations N] [--temperature T] [--smoke]
//   MNIST_DIR holds train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte and
//   t10k-labels-idx1-ubyte (tools/fetch_mnist.py downloads them). --smoke runs on made-up data in
//   a few seconds, to check the plumbing; its numbers mean nothing, and MNIST_DIR is ignored.
//
// Docs: docs/recipes/deep-learning/born_again_networks.md
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/born_again.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/distillation_loss.hpp"
#include "pulsatrix/explanation_agreement.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mnist_loader.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"
#ifdef PULSATRIX_DEMO_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

struct Options {
    std::string mnist_dir;
    std::string out;
    bool hip = false;
    int seeds = 5;
    int64_t train = 10000;
    int epochs = 10;
    int generations = 3;
    float temperature = 1.0f;
    bool smoke = false;
};

struct SplitMix {
    uint64_t state;
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }
};

constexpr int64_t kClasses = 10;

// Images as rows of a matrix, and their labels.
struct Data {
    std::vector<float> x;
    std::vector<int64_t> y;
    int64_t d = 0;
    [[nodiscard]] int64_t size() const { return static_cast<int64_t>(y.size()); }
};

Data FromMnist(const MnistDataset& m) {
    Data out;
    out.d = 28 * 28;
    for (size_t i = 0; i < m.images.size(); ++i) {
        const std::vector<float> v = m.images[i].to_host_vector();
        out.x.insert(out.x.end(), v.begin(), v.end());
        out.y.push_back(m.labels[i]);
    }
    return out;
}

// --smoke's data: ten Gaussian clusters in 20 dimensions.
Data Clusters(int64_t n, uint64_t seed) {
    Data out;
    out.d = 20;
    SplitMix centers{7};
    std::vector<double> c(static_cast<size_t>(kClasses * out.d));
    for (double& v : c) v = 2 * centers.uniform() - 1;
    SplitMix rng{seed};
    for (int64_t i = 0; i < n; ++i) {
        const int64_t k = static_cast<int64_t>(rng.next() % kClasses);
        for (int64_t j = 0; j < out.d; ++j) {
            out.x.push_back(static_cast<float>(c[static_cast<size_t>(k * out.d + j)] + 0.6 * (rng.uniform() + rng.uniform() - 1)));
        }
        out.y.push_back(k);
    }
    return out;
}

// He-uniform weights, U(-sqrt(6 / in), sqrt(6 / in)), and zero biases: a LinearModule starts at
// zero, which a network can't learn from.
void Initialize(LinearModule& layer, int64_t in, int64_t out, SplitMix& rng) {
    const double bound = std::sqrt(6.0 / static_cast<double>(in));
    std::vector<float> w(static_cast<size_t>(in * out));
    for (float& v : w) v = static_cast<float>((2 * rng.uniform() - 1) * bound);
    layer.set_weight(w);
}

// LeNet-300-100: d -> 300 -> 100 -> 10, with ReLUs.
struct Net {
    LinearModule l1, l2, l3;
    ReluModule r1, r2;
    SequentialModule seq;
    Net(int64_t d, uint64_t seed, DeviceBackend* backend)
        : l1(d, 300, backend), l2(300, 100, backend), l3(100, kClasses, backend), r1(backend), r2(backend),
          seq({&l1, &r1, &l2, &r2, &l3}) {
        SplitMix rng{seed};
        Initialize(l1, d, 300, rng);
        Initialize(l2, 300, 100, rng);
        Initialize(l3, 100, kClasses, rng);
    }
};

Tensor Rows(const std::vector<float>& x, int64_t d, const std::vector<size_t>& idx, DeviceBackend* backend) {
    std::vector<float> out;
    out.reserve(idx.size() * static_cast<size_t>(d));
    for (size_t i : idx) out.insert(out.end(), x.begin() + static_cast<std::ptrdiff_t>(i * d), x.begin() + static_cast<std::ptrdiff_t>((i + 1) * d));
    return Tensor(Shape({static_cast<int64_t>(idx.size()), d}), backend, out, backend->device());
}

// The network's logits for every row, (n, 10), on the host.
std::vector<float> Logits(Net& net, const Data& data, DeviceBackend* backend) {
    std::vector<float> out;
    for (int64_t start = 0; start < data.size(); start += 1000) {
        std::vector<size_t> idx;
        for (int64_t i = start; i < std::min(data.size(), start + 1000); ++i) idx.push_back(static_cast<size_t>(i));
        const std::vector<float> l = net.seq.forward(Rows(data.x, data.d, idx, backend)).to_host_vector();
        out.insert(out.end(), l.begin(), l.end());
    }
    return out;
}

double Accuracy(const std::vector<float>& logits, const std::vector<int64_t>& y) {
    int64_t right = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        const auto row = logits.begin() + static_cast<std::ptrdiff_t>(i * kClasses);
        right += std::max_element(row, row + kClasses) - row == y[i] ? 1 : 0;
    }
    return static_cast<double>(right) / static_cast<double>(y.size());
}

// What a network learns from: the labels, or a teacher's logits for every training row (permuted
// afresh each epoch when `permute` is set).
struct Teacher {
    const std::vector<float>* logits = nullptr;
    bool permute = false;
};

// Adam, batches of 128 in an order fixed by `order_seed`, so two networks given the same seed see
// the same batches.
void Train(Net& net, const Data& data, const Teacher& teacher, const Options& o, uint64_t order_seed, DeviceBackend* backend) {
    AdamOptimizer opt(1e-3f, backend);
    TokenCrossEntropyLoss hard(backend);
    DistillationOptions dopt;
    dopt.temperature = o.temperature;
    DistillationLoss soft(backend, dopt);
    CPUBackend cpu;
    SplitMix rng{order_seed};
    std::vector<size_t> order(static_cast<size_t>(data.size()));
    std::iota(order.begin(), order.end(), size_t{0});
    for (int epoch = 0; epoch < o.epochs; ++epoch) {
        for (size_t i = order.size(); i > 1; --i) std::swap(order[i - 1], order[static_cast<size_t>(rng.next() % i)]);
        std::vector<float> targets;
        if (teacher.logits && teacher.permute) {
            const Tensor all(Shape({data.size(), kClasses}), &cpu, *teacher.logits);
            targets = PermuteDarkKnowledge(all, order_seed * 1000 + static_cast<uint64_t>(epoch), &cpu).to_host_vector();
        }
        const std::vector<float>& source = teacher.permute ? targets : (teacher.logits ? *teacher.logits : targets);
        for (size_t start = 0; start < order.size(); start += 128) {
            const std::vector<size_t> idx(order.begin() + static_cast<std::ptrdiff_t>(start),
                                          order.begin() + static_cast<std::ptrdiff_t>(std::min(order.size(), start + 128)));
            opt.zero_grad(net.seq);
            const Tensor logits = net.seq.forward(Rows(data.x, data.d, idx, backend));
            if (teacher.logits) {
                (void)soft.forward(logits, Rows(source, kClasses, idx, backend));
                (void)net.seq.backward(soft.backward());
            } else {
                std::vector<float> y;
                for (size_t i : idx) y.push_back(static_cast<float>(data.y[i]));
                (void)hard.forward(logits, Tensor(Shape({static_cast<int64_t>(idx.size())}), backend, y, backend->device()));
                (void)net.seq.backward(hard.backward());
            }
            opt.step(net.seq);
        }
    }
}

// The gradient of each row's true class with respect to its pixels, on the host.
std::vector<Attribution> Explain(Net& net, const Data& data, int64_t count, DeviceBackend* backend, DeviceBackend* host) {
    std::vector<size_t> idx(static_cast<size_t>(count));
    std::iota(idx.begin(), idx.end(), size_t{0});
    (void)net.seq.forward(Rows(data.x, data.d, idx, backend));
    std::vector<float> seed(static_cast<size_t>(count * kClasses), 0.0f);
    for (int64_t i = 0; i < count; ++i) seed[static_cast<size_t>(i * kClasses + data.y[static_cast<size_t>(i)])] = 1.0f;
    const std::vector<float> g = net.seq.backward(Tensor(Shape({count, kClasses}), backend, seed, backend->device())).to_host_vector();
    std::vector<Attribution> out;
    for (int64_t i = 0; i < count; ++i) {
        const std::vector<float> v(g.begin() + static_cast<std::ptrdiff_t>(i * data.d), g.begin() + static_cast<std::ptrdiff_t>((i + 1) * data.d));
        out.push_back(Attribution{"saliency", Tensor(Shape({1, data.d}), host, v), {}});
    }
    return out;
}

std::string Fixed(double v, int digits) {
    std::ostringstream s;
    s.setf(std::ios::fixed);
    s.precision(digits);
    s << v;
    return s.str();
}

double Mean(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()); }

double Sd(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;
    const double m = Mean(v);
    double s = 0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size() - 1));
}

std::string Pp(double fraction) { return Fixed(100 * fraction, 2) + "%"; }

// Per seed: every accuracy and agreement the predictions need.
struct SeedResult {
    std::vector<double> generation;  // test accuracy of generations 0..G
    double teacher_confidence = 0;   // generation 0's mean top probability on its training images
    double retrain = 0, permuted = 0, born_again_ensemble = 0, label_ensemble = 0;
    AgreementReport student, other;  // generation 1 vs 0, and the other label network vs 0
};

int Run(DeviceBackend* backend, const Options& o) {
    std::filesystem::create_directories(o.out);
    Data train, test;
    if (o.smoke) {
        train = Clusters(400, 1);
        test = Clusters(200, 2);
    } else {
        CPUBackend cpu;
        train = FromMnist(MnistIdxLoader::Load(o.mnist_dir + "/train-images-idx3-ubyte", o.mnist_dir + "/train-labels-idx1-ubyte", &cpu, o.train));
        test = FromMnist(MnistIdxLoader::Load(o.mnist_dir + "/t10k-images-idx3-ubyte", o.mnist_dir + "/t10k-labels-idx1-ubyte", &cpu));
    }
    std::cout << "Data: " << train.size() << " training and " << test.size() << " test images" << (o.smoke ? " (made up: --smoke)" : " (MNIST)")
              << "\nNetwork: " << train.d << " -> 300 -> 100 -> 10, Adam lr 1e-3, " << o.epochs << " epochs, batches of 128; students at T = "
              << Fixed(o.temperature, 1) << "\n\n";
    const int64_t explained = std::min<int64_t>(200, test.size());
    AgreementOptions agree;
    agree.seed = 1;
    std::vector<SeedResult> results;
    const auto start = std::chrono::steady_clock::now();
    for (int s = 0; s < o.seeds; ++s) {
        const uint64_t base = 1000 * static_cast<uint64_t>(s + 1);
        SeedResult r;
        std::vector<std::unique_ptr<Net>> gens;
        std::vector<std::vector<float>> train_logits, test_logits;
        for (int g = 0; g <= o.generations; ++g) {
            gens.push_back(std::make_unique<Net>(train.d, base + static_cast<uint64_t>(g), backend));
            Teacher t;
            if (g > 0) t.logits = &train_logits.back();
            Train(*gens.back(), train, t, o, base + 500 + static_cast<uint64_t>(g), backend);
            train_logits.push_back(Logits(*gens.back(), train, backend));
            test_logits.push_back(Logits(*gens.back(), test, backend));
            r.generation.push_back(Accuracy(test_logits.back(), test.y));
        }
        // How much the first teacher says beyond the labels: its mean top probability on the images
        // it trained on (1 would leave nothing for the student but the labels).
        for (int64_t i = 0; i < train.size(); ++i) {
            const auto row = train_logits[0].begin() + i * kClasses;
            const float top = *std::max_element(row, row + kClasses);
            double sum = 0;
            for (int64_t k = 0; k < kClasses; ++k) sum += std::exp(static_cast<double>(row[k] - top) / o.temperature);
            r.teacher_confidence += 1.0 / sum / static_cast<double>(train.size());
        }
        // The permuted teacher's student: generation 1's start and batches.
        Net permuted(train.d, base + 1, backend);
        Train(permuted, train, Teacher{&train_logits[0], true}, o, base + 501, backend);
        r.permuted = Accuracy(Logits(permuted, test, backend), test.y);
        // Label-trained networks from other starts: one for the spread and the explanations, and
        // with generation 0, as many as the born-again ensemble has members.
        std::vector<std::unique_ptr<Net>> others;
        std::vector<std::vector<float>> label_logits{test_logits[0]};
        for (int k = 1; k < o.generations; ++k) {
            others.push_back(std::make_unique<Net>(train.d, base + 100 + static_cast<uint64_t>(k), backend));
            Train(*others.back(), train, Teacher{}, o, base + 600 + static_cast<uint64_t>(k), backend);
            label_logits.push_back(Logits(*others.back(), test, backend));
        }
        r.retrain = Accuracy(label_logits.at(1), test.y);
        CPUBackend cpu;  // the ensembles and the explanations are compared on the host
        auto ensemble = [&](const std::vector<std::vector<float>>& members) {
            std::vector<Tensor> t;
            for (const auto& m : members) t.emplace_back(Shape({test.size(), kClasses}), &cpu, m);
            return Accuracy(EnsembleLogProbabilities(t, &cpu).to_host_vector(), test.y);
        };
        r.born_again_ensemble = ensemble(std::vector<std::vector<float>>(test_logits.begin() + 1, test_logits.end()));
        r.label_ensemble = ensemble(label_logits);
        const std::vector<Attribution> teacher_maps = Explain(*gens[0], test, explained, backend, &cpu);
        r.student = CompareExplanations(teacher_maps, Explain(*gens[1], test, explained, backend, &cpu), agree);
        r.other = CompareExplanations(teacher_maps, Explain(*others.at(0), test, explained, backend, &cpu), agree);
        std::cout << "seed " << s << ": generations";
        for (double a : r.generation) std::cout << " " << Pp(a);
        std::cout << "; other label network " << Pp(r.retrain) << ", permuted teacher " << Pp(r.permuted) << "; ensembles: born-again "
                  << Pp(r.born_again_ensemble) << ", label " << Pp(r.label_ensemble) << "\n";
        results.push_back(std::move(r));
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "(" << Fixed(seconds, 0) << " s)\n\n";

    auto column = [&](auto get) {
        std::vector<double> v;
        for (const SeedResult& r : results) v.push_back(get(r));
        return v;
    };
    auto summary = [&](const std::string& name, const std::vector<double>& v) {
        std::cout << "  " << name << std::string(name.size() < 34 ? 34 - name.size() : 1, ' ') << Pp(Mean(v)) << " ± " << Fixed(100 * Sd(v), 2) << "\n";
    };
    std::cout << "Test accuracy, mean ± sd over " << o.seeds << " seeds:\n";
    for (int g = 0; g <= o.generations; ++g) {
        summary(g == 0 ? "generation 0 (labels)" : "generation " + std::to_string(g), column([&](const SeedResult& r) { return r.generation[static_cast<size_t>(g)]; }));
    }
    summary("another network on the labels", column([](const SeedResult& r) { return r.retrain; }));
    summary("student of the permuted teacher", column([](const SeedResult& r) { return r.permuted; }));
    summary("ensemble of generations 1-" + std::to_string(o.generations), column([](const SeedResult& r) { return r.born_again_ensemble; }));
    summary("ensemble of " + std::to_string(o.generations) + " label networks", column([](const SeedResult& r) { return r.label_ensemble; }));

    std::cout << "\nGeneration 0's mean top probability on its training images, at T = " << Fixed(o.temperature, 1) << ": "
              << Pp(Mean(column([](const SeedResult& r) { return r.teacher_confidence; }))) << "\n";
    const std::vector<double> gain = column([](const SeedResult& r) { return r.generation[1] - r.generation[0]; });
    const std::vector<double> spread = column([](const SeedResult& r) { return r.retrain - r.generation[0]; });
    const std::vector<double> dark = column([](const SeedResult& r) { return r.generation[1] - r.permuted; });
    const std::vector<double> ens = column([](const SeedResult& r) { return r.label_ensemble - r.born_again_ensemble; });
    auto wins = [](const std::vector<double>& v) { return static_cast<size_t>(std::count_if(v.begin(), v.end(), [](double x) { return x > 0; })); };
    std::cout << "\nThe predictions:\n";
    const bool p1 = Mean(gain) > 0 && Mean(gain) < 0.01 && wins(gain) < gain.size();
    std::cout << "  1. generation 1 minus generation 0: " << (Mean(gain) >= 0 ? "+" : "") << Fixed(100 * Mean(gain), 2) << " points, higher on "
              << wins(gain) << " of " << gain.size() << " seeds (two label networks differ by " << Fixed(100 * Mean(spread), 2) << " ± "
              << Fixed(100 * Sd(spread), 2) << "): " << (p1 ? "as predicted" : "not as predicted") << "\n";
    const bool p2 = Mean(dark) > 0;
    std::cout << "  2. generation 1 minus the permuted teacher's student: " << (Mean(dark) >= 0 ? "+" : "") << Fixed(100 * Mean(dark), 2)
              << " points, higher on " << wins(dark) << " of " << dark.size() << " seeds: " << (p2 ? "as predicted" : "not as predicted") << "\n";
    const bool p3 = Mean(ens) >= 0;
    std::cout << "  3. label ensemble minus born-again ensemble: " << (Mean(ens) >= 0 ? "+" : "") << Fixed(100 * Mean(ens), 2) << " points, higher on "
              << wins(ens) << " of " << ens.size() << " seeds: " << (p3 ? "as predicted" : "not as predicted") << "\n";
    int closer = 0;
    std::cout << "  4. explanations of " << explained << " test digits (the gradient of the true class), Spearman with generation 0's:\n";
    for (size_t s = 0; s < results.size(); ++s) {
        const SeedResult& r = results[s];
        closer += r.student.spearman > r.other.spearman ? 1 : 0;
        std::cout << "       seed " << s << ": generation 1 " << Fixed(r.student.spearman, 3) << " (p " << Fixed(r.student.p_value, 3) << ", CKA "
                  << Fixed(r.student.cka, 3) << "), the other label network " << Fixed(r.other.spearman, 3) << " (p " << Fixed(r.other.p_value, 3)
                  << ", CKA " << Fixed(r.other.cka, 3) << ")\n";
    }
    const bool p4 = closer == static_cast<int>(results.size());
    std::cout << "     generation 1 agrees more on " << closer << " of " << results.size() << " seeds: " << (p4 ? "as predicted" : "not as predicted") << "\n";

    JsonValue::Array seeds;
    for (const SeedResult& r : results) {
        JsonValue::Array g;
        for (double a : r.generation) g.push_back(JsonValue(a));
        seeds.push_back(JsonValue(JsonValue::Object{
            {"generations", JsonValue(std::move(g))},
            {"other_label_network", JsonValue(r.retrain)},
            {"permuted_teacher", JsonValue(r.permuted)},
            {"teacher_confidence", JsonValue(r.teacher_confidence)},
            {"born_again_ensemble", JsonValue(r.born_again_ensemble)},
            {"label_ensemble", JsonValue(r.label_ensemble)},
            {"agreement_generation_1", ToJson(r.student)},
            {"agreement_other_label_network", ToJson(r.other)},
        }));
    }
    const std::string path = o.out + "/born_again.json";
    std::ofstream(path, std::ios::binary) << WriteJson(JsonValue(JsonValue::Object{{"seeds", JsonValue(std::move(seeds))}})) << "\n";
    std::cout << "\nWrote " << path << "\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: born_again_networks_recipe MNIST_DIR OUT_DIR [--device cpu|hip] [--seeds N] [--train N] [--epochs N] "
                     "[--generations N] [--temperature T] [--smoke]\n");
        return 2;
    }
    Options o;
    o.mnist_dir = argv[1];
    o.out = argv[2];
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) {
            o.hip = std::string(argv[++i]) == "hip";
        } else if (a == "--seeds" && i + 1 < argc) {
            o.seeds = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--train" && i + 1 < argc) {
            o.train = std::max(100, std::atoi(argv[++i]));
        } else if (a == "--epochs" && i + 1 < argc) {
            o.epochs = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--generations" && i + 1 < argc) {
            o.generations = std::max(2, std::atoi(argv[++i]));
        } else if (a == "--temperature" && i + 1 < argc) {
            o.temperature = static_cast<float>(std::atof(argv[++i]));
        } else if (a == "--smoke") {
            o.smoke = true;
        } else {
            std::fprintf(stderr, "born_again_networks_recipe: unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (o.smoke) {
        o.seeds = std::min(o.seeds, 2);
        o.epochs = std::min(o.epochs, 2);
        o.generations = 2;
    }
    try {
        if (o.hip) {
#ifdef PULSATRIX_DEMO_WITH_HIP
            HIPBackend gpu;
            return Run(&gpu, o);
#else
            std::fprintf(stderr, "this build has no HIP backend\n");
            return 2;
#endif
        }
        CPUBackend cpu;
        return Run(&cpu, o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "born_again_networks_recipe: %s\n", e.what());
        return 1;
    }
}
