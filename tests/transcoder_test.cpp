// FEAT-5: transcoders and skip transcoders, against a PyTorch rendering
// (tools/golden/make_transcoder_golden.py), on a synthetic MLP, and spliced into real blocks
// through the MLP hook.
#include "pulsatrix/transcoder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/featurizer_metrics.hpp"
#include "pulsatrix/hf_model.hpp"
#include "pulsatrix/safetensors.hpp"

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

Transcoder GoldenTranscoder(const SafetensorsFile& g, CPUBackend* cpu) {
    TranscoderOptions o;
    o.k = 3;
    o.k_aux = 4;
    o.skip = true;
    o.dead_after = 100;
    Transcoder t(6, 5, 20, cpu, o);
    t.encoder().set_weight(Floats(g, "w_enc"));
    t.encoder().set_bias(Floats(g, "b_enc"));
    t.decoder().set_weight(Floats(g, "w_dec"));
    t.decoder().set_bias(Floats(g, "b_dec"));
    t.skip()->set_weight(Floats(g, "w_skip"));
    auto [ptr, size] = g.bytes("dead");
    std::vector<int64_t> dead(size / sizeof(int64_t));
    std::memcpy(dead.data(), ptr, size);
    std::vector<int64_t> since(20, 0);
    for (int64_t d : dead) since[static_cast<size_t>(d)] = 100;
    t.set_inputs_since_fired(since);
    return t;
}

TEST(Transcoder, SkipTranscoderMatchesPyTorch) {
    CPUBackend cpu;
    const SafetensorsFile g = SafetensorsFile::Map(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/sae/transcoder_golden.safetensors");
    Transcoder t = GoldenTranscoder(g, &cpu);
    const Tensor x(Shape({9, 6}), &cpu, Floats(g, "x")), y(Shape({9, 5}), &cpu, Floats(g, "y"));
    const FeaturizerLoss loss = t.loss_and_backward_with_target(x, y);
    EXPECT_NEAR(loss.total, Floats(g, "loss")[0], 1e-5);
    EXPECT_NEAR(loss.reconstruction, Floats(g, "reconstruction")[0], 1e-5);
    EXPECT_NEAR(loss.sparsity, Floats(g, "aux")[0], 1e-6);
    const std::vector<NamedParamRef> p = t.named_parameters();
    ASSERT_EQ(p.size(), 5u);
    EXPECT_EQ(p[4].name, "skip.weight");
    const char* names[] = {"grad_w_enc", "grad_b_enc", "grad_w_dec", "grad_b_dec", "grad_w_skip"};
    for (size_t i = 0; i < 5; ++i) EXPECT_LT(MaxDiff(p[i].ref.grad->to_host_vector(), Floats(g, names[i])), 1e-5) << p[i].name;

    // Two Adam steps, the dead set held as the reference holds it.
    Transcoder trained = GoldenTranscoder(g, &cpu);
    AdamOptimizer opt(1e-2f, &cpu);
    for (int step = 0; step < 2; ++step) {
        std::vector<int64_t> since(20, 0);
        for (int64_t d : {0, 3, 8, 12, 19}) since[static_cast<size_t>(d)] = 100;
        trained.set_inputs_since_fired(since);
        (void)TrainFeaturizer(trained, x, y, opt);
    }
    const char* after[] = {"after_w_enc", "after_b_enc", "after_w_dec", "after_b_dec", "after_w_skip"};
    const std::vector<NamedParamRef> q = trained.named_parameters();
    for (size_t i = 0; i < 5; ++i) EXPECT_LT(MaxDiff(q[i].ref.value->to_host_vector(), Floats(g, after[i])), 1e-5) << q[i].name;
}

TEST(Transcoder, IsAModuleFromInputToPrediction) {
    CPUBackend cpu;
    TranscoderOptions o;
    o.k = 4;
    o.skip = true;
    Transcoder t(5, 3, 12, &cpu, o);
    std::vector<float> skip(15);
    for (size_t i = 0; i < skip.size(); ++i) skip[i] = 0.1f * static_cast<float>(i % 4) - 0.15f;
    t.skip()->set_weight(skip);
    std::vector<float> xv(4 * 5);
    for (size_t i = 0; i < xv.size(); ++i) xv[i] = std::sin(1.1f * static_cast<float>(i)) + 0.4f;
    const Tensor x(Shape({4, 5}), &cpu, xv);
    const std::vector<float> y = t.forward(x).to_host_vector(), p = t.predict(x).to_host_vector();
    for (size_t i = 0; i < y.size(); ++i) EXPECT_FLOAT_EQ(y[i], p[i]);
    EXPECT_EQ(t.output_dim(), 3);
    EXPECT_EQ(t.decoder_direction(0).size(), 3u);
    // decode() leaves the skip connection out.
    const std::vector<float> d = t.decode(t.encode(x)).to_host_vector();
    EXPECT_GT(MaxDiff(d, p), 1e-3);
    // Backward against finite differences of 0.5 |ŷ|².
    auto f = [&](const std::vector<float>& in) {
        double s = 0;
        for (float v : t.forward(Tensor(Shape({4, 5}), &cpu, in)).to_host_vector()) s += 0.5 * v * v;
        return s;
    };
    (void)t.forward(x);
    const std::vector<float> gx = t.backward(Tensor(Shape({4, 3}), &cpu, y)).to_host_vector();
    for (size_t i = 0; i < xv.size(); ++i) {
        std::vector<float> up = xv, down = xv;
        up[i] += 1e-3f;
        down[i] -= 1e-3f;
        EXPECT_NEAR(gx[i], (f(up) - f(down)) / 2e-3, 3e-3) << i;
    }
    (void)t.forward(x);
    EXPECT_EQ(t.propagate_relevance(Tensor(Shape({4, 3}), &cpu, y), LRPRuleConfig{}).shape(), Shape({4, 5}));
    // A target is required, and absorption (an autoencoder metric) refuses it.
    EXPECT_THROW((void)t.loss_and_backward(x), std::invalid_argument);
    EXPECT_THROW((void)FeatureAbsorption(t, x, {0, 1, 0, 1}), std::invalid_argument);
    EXPECT_THROW((void)t.loss_and_backward_with_target(x, Tensor(Shape({4, 5}), &cpu, xv)), std::invalid_argument);
    // Checkpoints carry the skip connection.
    const std::string path = (std::filesystem::temp_directory_path() / "pulsatrix_transcoder.safetensors").string();
    SaveCheckpoint(path, t);
    Transcoder other(5, 3, 12, &cpu, o);
    LoadCheckpoint(path, other);
    EXPECT_EQ(other.predict(x).to_host_vector(), p);
    std::filesystem::remove(path);
}

/** @brief A synthetic MLP: a large linear part plus 8 sparse ReLU units. */
struct MlpData {
    std::vector<float> x, y;
};

MlpData MakeMlp(int64_t n, int64_t d, uint64_t seed) {
    uint64_t s = seed;
    auto u = [&] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(static_cast<uint32_t>(s >> 32)) / 4294967296.0;
    };
    std::vector<double> a(static_cast<size_t>(d * d)), w1(static_cast<size_t>(8 * d)), w2(static_cast<size_t>(8 * d));
    for (double& v : a) v = (u() * 2 - 1) * 0.6;
    for (double& v : w1) v = u() * 2 - 1;
    for (double& v : w2) v = u() * 2 - 1;
    MlpData m;
    for (int64_t r = 0; r < n; ++r) {
        std::vector<double> x(static_cast<size_t>(d));
        for (double& v : x) v = u() * 2 - 1;
        for (int64_t j = 0; j < d; ++j) {
            double out = 0;
            for (int64_t i = 0; i < d; ++i) out += a[static_cast<size_t>(i * d + j)] * x[static_cast<size_t>(i)];
            m.y.push_back(static_cast<float>(out));
        }
        for (int h = 0; h < 8; ++h) {
            double pre = -1.2;
            for (int64_t i = 0; i < d; ++i) pre += w1[static_cast<size_t>(h * d + i)] * x[static_cast<size_t>(i)];
            if (pre <= 0) continue;
            for (int64_t j = 0; j < d; ++j) m.y[static_cast<size_t>(r * d + j)] += static_cast<float>(pre * w2[static_cast<size_t>(h * d + j)]);
        }
        m.x.insert(m.x.end(), x.begin(), x.end());
    }
    return m;
}

TEST(Transcoder, TheSkipConnectionTakesTheLinearPartOffTheLatents) {
    CPUBackend cpu;
    const int64_t d = 8, n = 2048;
    const MlpData m = MakeMlp(n, d, 4);
    const Tensor x(Shape({n, d}), &cpu, m.x), y(Shape({n, d}), &cpu, m.y);
    auto train = [&](bool skip) {
        TranscoderOptions o;
        o.k = 3;
        o.skip = skip;
        o.dead_after = 4096;
        auto t = std::make_unique<Transcoder>(d, d, 32, &cpu, o);
        t->initialize_bias(y);
        AdamOptimizer opt(5e-3f, &cpu);
        for (int step = 0; step < 600; ++step) (void)TrainFeaturizer(*t, x, y, opt);
        return EvaluatePrediction(*t, x, y);
    };
    const ReconstructionMetrics plain = train(false), skip = train(true);
    std::printf("[Transcoder] k = 3 on a linear + sparse MLP: explained variance %.3f plain, %.3f with skip\n", plain.explained_variance,
                skip.explained_variance);
    EXPECT_GT(skip.explained_variance, 0.95);
    // Unexplained variance: 10.9% plain, 1.5% with the skip connection.
    EXPECT_LT(1.0 - skip.explained_variance, (1.0 - plain.explained_variance) / 3);
}

TEST(Transcoder, SplicesIntoAnEncoderBlocksMlp) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm", &cpu);
    const int64_t h = model->config().hidden_size, layer = 0;
    // Training pairs from the MLP hook, which reads without changing anything.
    std::vector<float> ins, outs;
    std::vector<std::vector<float>> id_rows;
    for (int s = 0; s < 24; ++s) {
        std::vector<float> ids(12);
        for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<float>(4 + (s * 7 + static_cast<int>(i) * 5) % 20);
        ids.front() = 0;
        ids.back() = 2;
        id_rows.push_back(ids);
    }
    model->layer(layer).set_mlp_hook([&](const Tensor& in, const Tensor& out) {
        const std::vector<float> a = in.to_host_vector(), b = out.to_host_vector();
        ins.insert(ins.end(), a.begin(), a.end());
        outs.insert(outs.end(), b.begin(), b.end());
        return out;
    });
    const std::vector<float> clean0 = model->forward(Tensor(Shape({1, 12}), &cpu, id_rows[0])).to_host_vector();
    for (size_t s = 1; s < id_rows.size(); ++s) (void)model->forward(Tensor(Shape({1, 12}), &cpu, id_rows[s]));
    model->layer(layer).set_mlp_hook({});
    EXPECT_EQ(model->forward(Tensor(Shape({1, 12}), &cpu, id_rows[0])).to_host_vector(), clean0);  // reading changed nothing
    const auto n = static_cast<int64_t>(ins.size()) / h;
    const Tensor x(Shape({n, h}), &cpu, ins), y(Shape({n, h}), &cpu, outs);
    TranscoderOptions o;
    o.k = 8;
    o.skip = true;
    Transcoder t(h, h, 4 * h, &cpu, o);
    t.initialize_bias(y);
    AdamOptimizer opt(3e-3f, &cpu);
    for (int step = 0; step < 400; ++step) (void)TrainFeaturizer(t, x, y, opt);
    const ReconstructionMetrics fit = EvaluatePrediction(t, x, y);

    const Tensor ids(Shape({1, 12}), &cpu, id_rows[3]);
    const std::vector<float> clean = model->forward(ids).to_host_vector();
    auto loss = [&](const MlpHook& hook) {
        model->layer(layer).set_mlp_hook(hook);
        const std::vector<float> v = model->forward(ids).to_host_vector();
        model->layer(layer).set_mlp_hook({});
        double s = 0;
        for (size_t i = 0; i < v.size(); ++i) s += (static_cast<double>(v[i]) - clean[i]) * (static_cast<double>(v[i]) - clean[i]);
        return s;
    };
    const LossRecovered r = MeasureMlpLossRecovered(t, loss);
    std::printf("[Transcoder] tiny ESM layer 0 MLP: explained variance %.3f, loss recovered %.3f\n", fit.explained_variance, r.recovered);
    EXPECT_EQ(r.clean, 0.0);
    EXPECT_GT(r.ablated, 0.0);
    EXPECT_GT(fit.explained_variance, 0.8);
    EXPECT_GT(r.recovered, 0.8);
    EXPECT_LT(r.recovered, 1.0);
    // Only the rows asked for are replaced.
    EXPECT_EQ(MeasureMlpLossRecovered(t, loss, std::vector<bool>(12, false)).spliced, 0.0);
    model->layer(layer).set_mlp_hook([](const Tensor&, const Tensor& out) { return Tensor(Shape({1}), out.backend(), std::vector<float>{0}); });
    EXPECT_THROW((void)model->forward(ids), std::invalid_argument);
}

TEST(Transcoder, TheMlpHookWorksInACausalModel) {
    CPUBackend cpu;
    const HfModelConfig config = ReadHfConfig(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/qwen2/config.json");
    CausalLM model(config, &cpu);
    uint64_t seed = 5;
    for (const NamedParamRef& p : model.named_parameters()) {
        std::vector<float> v(static_cast<size_t>(p.ref.value->numel()));
        for (float& w : v) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            w = 0.2f * (static_cast<float>(static_cast<uint32_t>(seed >> 32)) / 4294967296.0f - 0.5f);
        }
        *p.ref.value = Tensor(p.ref.value->shape(), &cpu, v);
    }
    const Tensor ids(Shape({2, 3}), &cpu, std::vector<float>{1, 5, 2, 7, 7, 3});
    const std::vector<float> clean = model.forward(ids).to_host_vector();
    int calls = 0;
    model.layer(0).set_mlp_hook([&](const Tensor& in, const Tensor& out) {
        ++calls;
        EXPECT_EQ(in.shape(), Shape({2, 3, config.hidden_size}));
        EXPECT_EQ(out.shape(), in.shape());
        return out;
    });
    EXPECT_EQ(model.forward(ids).to_host_vector(), clean);
    EXPECT_EQ(calls, 1);
    model.layer(0).set_mlp_hook(MlpAblationHook());
    EXPECT_NE(model.forward(ids).to_host_vector(), clean);
    model.layer(0).set_mlp_hook({});
    EXPECT_EQ(model.forward(ids).to_host_vector(), clean);
}

}  // namespace
}  // namespace pulsatrix
