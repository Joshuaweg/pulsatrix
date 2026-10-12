// TRN-7: LoRA adapters on LinearModule, ApplyLora() on whole models, merging, and PEFT adapter files,
// checked against PEFT's own logits and gradients (tools/golden/make_lora_golden.py).
#include "pulsatrix/lora.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }
const std::vector<float> kTokens = {3, 17, 42, 5, 60, 8};

std::vector<float> Random(std::size_t n, uint32_t seed, float scale = 1.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> v(n);
    for (float& x : v) x = scale * normal(rng);
    return v;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    float worst = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

// x (n, in) @ m (in, out) in double.
std::vector<double> MatMul(const std::vector<float>& x, const std::vector<float>& m, int64_t n, int64_t in, int64_t out) {
    std::vector<double> y(static_cast<std::size_t>(n * out));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t k = 0; k < in; ++k)
            for (int64_t j = 0; j < out; ++j)
                y[static_cast<std::size_t>(i * out + j)] += static_cast<double>(x[static_cast<std::size_t>(i * in + k)]) * m[static_cast<std::size_t>(k * out + j)];
    return y;
}

class LoraLayerTest : public ::testing::Test {
protected:
    CPUBackend backend;
    const int64_t in = 6, out = 5, n = 4;

    LinearModule Layer(uint32_t seed) {
        LinearModule l(in, out, &backend);
        l.set_weight(Random(static_cast<std::size_t>(in * out), seed, 0.5f));
        l.set_bias(Random(static_cast<std::size_t>(out), seed + 1, 0.5f));
        return l;
    }
    static LoRAConfig Config(int64_t rank, float alpha) {
        LoRAConfig c;
        c.rank = rank;
        c.alpha = alpha;
        c.seed = 7;
        return c;
    }
    Tensor Input(uint32_t seed) { return Tensor(Shape({n, in}), &backend, Random(static_cast<std::size_t>(n * in), seed)); }
};

TEST_F(LoraLayerTest, StartsAsTheBaseLayerWithPeftsInitialization) {
    LinearModule l = Layer(1);
    const Tensor x = Input(2);
    const std::vector<float> before = l.forward(x).to_host_vector();
    l.attach_lora(Config(3, 6.0f));
    ASSERT_NE(l.lora(), nullptr);
    EXPECT_EQ(l.forward(x).to_host_vector(), before);  // B starts at zero
    EXPECT_FLOAT_EQ(l.lora()->scale(), 2.0f);
    const std::vector<float> a = l.lora()->a().to_host_vector();
    ASSERT_EQ(l.lora()->a().shape(), Shape({in, 3}));
    const float bound = 1.0f / std::sqrt(static_cast<float>(in));
    EXPECT_LE(*std::max_element(a.begin(), a.end()), bound);
    EXPECT_GE(*std::min_element(a.begin(), a.end()), -bound);
    EXPECT_GT(*std::max_element(a.begin(), a.end()) - *std::min_element(a.begin(), a.end()), bound);  // spread out
    for (float b : l.lora()->b().to_host_vector()) EXPECT_EQ(b, 0.0f);
    std::vector<std::string> names;
    for (const NamedParamRef& p : l.named_parameters()) names.push_back(p.name);
    EXPECT_EQ(names, (std::vector<std::string>{"weight", "bias", "lora_A.weight", "lora_B.weight"}));
}

TEST_F(LoraLayerTest, ForwardAddsTheScaledLowRankUpdate) {
    LinearModule l = Layer(3);
    l.attach_lora(Config(2, 8.0f));
    const std::vector<float> a = Random(static_cast<std::size_t>(in * 2), 4), b = Random(static_cast<std::size_t>(2 * out), 5);
    l.lora()->set_a(a);
    l.lora()->set_b(b);
    const Tensor x = Input(6);
    const std::vector<float> got = l.forward(x).to_host_vector();
    const std::vector<float> xv = x.to_host_vector(), w = l.weight().to_host_vector(), bias = l.bias().to_host_vector();
    const std::vector<double> base = MatMul(xv, w, n, in, out);
    std::vector<float> xa(static_cast<std::size_t>(n * 2));
    const std::vector<double> xa_d = MatMul(xv, a, n, in, 2);
    for (std::size_t i = 0; i < xa.size(); ++i) xa[i] = static_cast<float>(xa_d[i]);
    const std::vector<double> update = MatMul(xa, b, n, 2, out);
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < out; ++j) {
            const std::size_t k = static_cast<std::size_t>(i * out + j);
            EXPECT_NEAR(got[k], base[k] + bias[static_cast<std::size_t>(j)] + 4.0 * update[k], 1e-4);
        }
    }
}

// The loss sum(y * u) is linear in A, in B and in x, so central differences are exact up to rounding.
TEST_F(LoraLayerTest, GradientsMatchFiniteDifferences) {
    LinearModule l = Layer(7);
    l.attach_lora(Config(3, 3.0f));
    l.lora()->set_b(Random(static_cast<std::size_t>(3 * out), 8, 0.5f));
    const Tensor x = Input(9);
    const std::vector<float> u = Random(static_cast<std::size_t>(n * out), 10);
    const Tensor upstream(Shape({n, out}), &backend, u);
    auto loss = [&](LinearModule& m, const Tensor& input) {
        const std::vector<float> y = m.forward(input).to_host_vector();
        double s = 0;
        for (std::size_t i = 0; i < y.size(); ++i) s += static_cast<double>(y[i]) * u[i];
        return s;
    };
    (void)l.forward(x);
    const std::vector<float> grad_x = l.backward(upstream).to_host_vector();
    const std::vector<float> grad_a = l.lora()->a_grad().to_host_vector(), grad_b = l.lora()->b_grad().to_host_vector();
    const double h = 1e-2;
    for (std::size_t i = 0; i < grad_a.size(); ++i) {
        std::vector<float> a = l.lora()->a().to_host_vector();
        const float saved = a[i];
        a[i] = saved + static_cast<float>(h);
        l.lora()->set_a(a);
        const double up = loss(l, x);
        a[i] = saved - static_cast<float>(h);
        l.lora()->set_a(a);
        const double down = loss(l, x);
        a[i] = saved;
        l.lora()->set_a(a);
        EXPECT_NEAR(grad_a[i], (up - down) / (2 * h), 2e-3) << "A " << i;
    }
    for (std::size_t i = 0; i < grad_b.size(); ++i) {
        std::vector<float> b = l.lora()->b().to_host_vector();
        const float saved = b[i];
        b[i] = saved + static_cast<float>(h);
        l.lora()->set_b(b);
        const double up = loss(l, x);
        b[i] = saved - static_cast<float>(h);
        l.lora()->set_b(b);
        const double down = loss(l, x);
        b[i] = saved;
        l.lora()->set_b(b);
        EXPECT_NEAR(grad_b[i], (up - down) / (2 * h), 2e-3) << "B " << i;
    }
    std::vector<float> xv = x.to_host_vector();
    for (std::size_t i = 0; i < xv.size(); ++i) {
        const float saved = xv[i];
        xv[i] = saved + static_cast<float>(h);
        const double up = loss(l, Tensor(x.shape(), &backend, xv));
        xv[i] = saved - static_cast<float>(h);
        const double down = loss(l, Tensor(x.shape(), &backend, xv));
        xv[i] = saved;
        EXPECT_NEAR(grad_x[i], (up - down) / (2 * h), 2e-3) << "x " << i;
    }
}

TEST_F(LoraLayerTest, MergingGivesTheSameOutputAndUnmergingRestoresTheWeight) {
    LinearModule l = Layer(11);
    l.attach_lora(Config(4, 16.0f));
    l.lora()->set_b(Random(static_cast<std::size_t>(4 * out), 12, 0.5f));
    const Tensor x = Input(13);
    const std::vector<float> w = l.weight().to_host_vector();
    const std::vector<float> adapted = l.forward(x).to_host_vector();
    l.merge_lora();
    EXPECT_TRUE(l.lora()->merged());
    EXPECT_LT(MaxAbsDiff(l.forward(x).to_host_vector(), adapted), 1e-5f);
    EXPECT_GT(MaxAbsDiff(l.weight().to_host_vector(), w), 0.1f);
    // While merged, the adapter gets no gradient: its update is already in W.
    (void)l.backward(Tensor(Shape({n, out}), &backend, Random(static_cast<std::size_t>(n * out), 14)));
    for (float g : l.lora()->b_grad().to_host_vector()) EXPECT_EQ(g, 0.0f);
    EXPECT_THROW(l.merge_lora(), std::logic_error);
    EXPECT_THROW(l.set_lora_enabled(false), std::logic_error);
    l.unmerge_lora();
    EXPECT_LT(MaxAbsDiff(l.weight().to_host_vector(), w), 1e-6f);
    EXPECT_THROW(l.unmerge_lora(), std::logic_error);
}

TEST_F(LoraLayerTest, DisablingRunsTheBaseLayer) {
    LinearModule l = Layer(15);
    const Tensor x = Input(16);
    const std::vector<float> base = l.forward(x).to_host_vector();
    l.attach_lora(Config(2, 2.0f));
    l.lora()->set_b(Random(static_cast<std::size_t>(2 * out), 17));
    EXPECT_GT(MaxAbsDiff(l.forward(x).to_host_vector(), base), 0.1f);
    l.set_lora_enabled(false);
    EXPECT_EQ(l.forward(x).to_host_vector(), base);
    EXPECT_THROW(l.merge_lora(), std::logic_error);
    l.set_lora_enabled(true);
    l.detach_lora();
    EXPECT_EQ(l.lora(), nullptr);
    EXPECT_EQ(l.forward(x).to_host_vector(), base);
}

TEST_F(LoraLayerTest, RelevanceIsThatOfTheAdaptedWeight) {
    LinearModule l = Layer(18);
    l.attach_lora(Config(2, 4.0f));
    l.lora()->set_b(Random(static_cast<std::size_t>(2 * out), 19, 0.5f));
    LinearModule plain = Layer(18);
    plain.set_weight(l.lora()->effective_weight(l.weight()).to_host_vector());
    const Tensor x = Input(20);
    const Tensor r(Shape({n, out}), &backend, Random(static_cast<std::size_t>(n * out), 21));
    // LoRARelevance::Merged: the layer as its adapted weight. (The default, Split, is TRN-8's.)
    LRPRuleConfig epsilon;
    epsilon.lora_relevance = LoRARelevance::Merged;
    LRPRuleConfig gamma = epsilon;
    gamma.rule = LRPRule::Gamma;
    gamma.gamma = 0.25f;
    for (const LRPRuleConfig& config : {epsilon, gamma}) {
        (void)l.forward(x);
        (void)plain.forward(x);
        const std::vector<float> want = plain.propagate_relevance(r, config).to_host_vector();
        float largest = 0;
        for (float v : want) largest = std::max(largest, std::fabs(v));
        // z = x W + s (x A) B rounds differently from x (W + s A B), and the rules divide by z.
        EXPECT_LT(MaxAbsDiff(l.propagate_relevance(r, config).to_host_vector(), want), 1e-5f * largest);
    }
}

TEST_F(LoraLayerTest, BadSettingsThrow) {
    LinearModule l = Layer(22);
    EXPECT_THROW(l.attach_lora(Config(0, 1.0f)), std::invalid_argument);
    LoRAConfig c = Config(2, 1.0f);
    c.dropout = 1.0f;
    EXPECT_THROW(l.attach_lora(c), std::invalid_argument);
    EXPECT_THROW(l.merge_lora(), std::logic_error);
    l.attach_lora(Config(2, 1.0f));
    EXPECT_THROW(l.attach_lora(Config(2, 1.0f)), std::logic_error);
    EXPECT_THROW(l.lora()->set_a({1.0f}), std::invalid_argument);
}

// Only the adapters train, and they fit a target the base layer can't reach.
TEST(LoraTraining, OnlyTheAdaptersTrainAndTheyFit) {
    CPUBackend backend;
    LinearModule l1(8, 16, &backend);
    ReluModule relu(&backend);
    LinearModule l2(16, 4, &backend);
    l1.set_weight(Random(8 * 16, 1, 0.3f));
    l2.set_weight(Random(16 * 4, 2, 0.3f));
    SequentialModule model({&l1, &relu, &l2});
    const std::vector<float> w1 = l1.weight().to_host_vector();
    LoRAConfig c;
    c.rank = 4;
    c.alpha = 4;
    EXPECT_EQ(ApplyLora(model, c), (std::vector<std::string>{"0", "2"}));
    int trainable = 0;
    for (const NamedParamRef& p : model.named_parameters()) trainable += p.ref.value->requires_grad() ? 1 : 0;
    EXPECT_EQ(trainable, 4);

    const Tensor x(Shape({32, 8}), &backend, Random(32 * 8, 3));
    const Tensor target(Shape({32, 4}), &backend, Random(32 * 4, 4, 0.5f));
    AdamOptimizer opt(1e-2f, &backend);
    MSELoss mse(&backend);
    double first = 0, last = 0;
    for (int step = 0; step < 300; ++step) {
        opt.zero_grad(model);
        const Tensor y = model.forward(x);
        const double loss = mse.forward(y, target);
        (void)model.backward(mse.backward());
        opt.step(model);
        if (step == 0) first = loss;
        last = loss;
    }
    EXPECT_LT(last, 0.5 * first) << first << " -> " << last;
    EXPECT_EQ(l1.weight().to_host_vector(), w1);  // frozen
}

TEST(LoraModels, NamedModulesFollowTheParameterNames) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> lm = LoadCausalLM(Fixture("qwen2"), &backend);
    std::unique_ptr<EncoderLM> esm = LoadEncoderLM(Fixture("esm"), &backend);
    for (Module* model : {static_cast<Module*>(lm.get()), static_cast<Module*>(esm.get())}) {
        std::map<std::string, const Tensor*> params;
        for (const NamedParamRef& p : model->named_parameters()) params[p.name] = p.ref.value;
        int linear = 0;
        for (const NamedModuleRef& m : NamedModules(*model)) {
            if (auto* l = dynamic_cast<LinearModule*>(m.module)) {
                ++linear;
                ASSERT_TRUE(params.count(m.name + ".weight")) << m.name;
                EXPECT_EQ(params.at(m.name + ".weight"), &l->weight()) << m.name;
            }
        }
        EXPECT_GE(linear, 12);
    }
}

TEST(LoraModels, ApplyLoraSelectsLayersAsPeftDoes) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> lm = LoadCausalLM(Fixture("qwen2"), &backend);
    const std::vector<std::string> all = ApplyLora(*lm);
    EXPECT_EQ(all.size(), 14u);  // 7 projections in each of 2 layers; never lm_head
    EXPECT_EQ(std::count(all.begin(), all.end(), "lm_head"), 0);
    EXPECT_EQ(all.front(), "layers.0.mha.q_proj");
    EXPECT_THROW((void)ApplyLora(*lm), std::invalid_argument);  // already adapted
    RemoveLora(*lm);
    EXPECT_TRUE(LoraLayers(*lm).empty());

    LoRAConfig qv;
    qv.target_modules = {"q_proj", "v_proj"};
    EXPECT_EQ(ApplyLora(*lm, qv), (std::vector<std::string>{"layers.0.mha.q_proj", "layers.0.mha.v_proj",
                                                            "layers.1.mha.q_proj", "layers.1.mha.v_proj"}));
    RemoveLora(*lm);
    LoRAConfig bad;
    bad.target_modules = {"qq_proj"};
    EXPECT_THROW((void)ApplyLora(*lm, bad), std::invalid_argument);
    bad.target_modules = {"norm1"};  // an RMSNorm
    EXPECT_THROW((void)ApplyLora(*lm, bad), std::invalid_argument);
    EXPECT_TRUE(LoraLayers(*lm).empty());
}

class PeftParityTest : public ::testing::Test {
protected:
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Fixture("qwen2"), &backend);
    SafetensorsFile reference = SafetensorsFile::Read(Fixture("qwen2_lora") + "/reference.safetensors");

    std::vector<float> Logits() { return model->forward(Tensor(Shape({1, 6}), &backend, kTokens)).to_host_vector(); }
};

TEST_F(PeftParityTest, LogitsAndGradientsMatchPeft) {
    LoadLoraAdapter(*model, Fixture("qwen2_lora"), HuggingFaceMapping(model->config()));
    ASSERT_EQ(LoraLayers(*model).size(), 14u);
    EXPECT_FLOAT_EQ(LoraLayers(*model)[0].second->lora()->scale(), 2.0f);
    const std::vector<float> expected = reference.tensor("logits", &backend).to_host_vector();
    std::vector<float> base_model;
    {
        SetLoraEnabled(*model, false);
        base_model = Logits();
        SetLoraEnabled(*model, true);
    }
    EXPECT_GT(MaxAbsDiff(base_model, expected), 0.05f);  // the adapter matters
    EXPECT_LT(MaxAbsDiff(Logits(), expected), 1e-4f);

    (void)model->backward(reference.tensor("upstream", &backend));
    const std::map<std::string, std::string> names = [&] {
        std::map<std::string, std::string> m;
        for (const WeightMapping& w : HuggingFaceMapping(model->config())) m[w.target] = w.source;
        return m;
    }();
    int checked = 0;
    for (const auto& [name, layer] : LoraLayers(*model)) {
        const std::string module = "base_model.model." + names.at(name + ".weight").substr(0, names.at(name + ".weight").size() - 7);
        const LoRAAdapter& a = *layer->lora();
        const int64_t in = layer->in_features(), out = layer->out_features(), r = a.rank();
        const std::vector<float> ga = reference.tensor(module + ".lora_A.weight", &backend).to_host_vector();  // (r, in)
        const std::vector<float> gb = reference.tensor(module + ".lora_B.weight", &backend).to_host_vector();  // (out, r)
        const std::vector<float> ours_a = a.a_grad().to_host_vector(), ours_b = a.b_grad().to_host_vector();
        float scale = 0;
        for (float g : gb) scale = std::max(scale, std::fabs(g));
        for (int64_t i = 0; i < in; ++i)
            for (int64_t k = 0; k < r; ++k)
                EXPECT_NEAR(ours_a[static_cast<std::size_t>(i * r + k)], ga[static_cast<std::size_t>(k * in + i)], 1e-3) << name;
        for (int64_t k = 0; k < r; ++k)
            for (int64_t j = 0; j < out; ++j)
                EXPECT_NEAR(ours_b[static_cast<std::size_t>(k * out + j)], gb[static_cast<std::size_t>(j * r + k)], 1e-3) << name;
        ++checked;
    }
    EXPECT_EQ(checked, 14);

    MergeLora(*model);
    EXPECT_LT(MaxAbsDiff(Logits(), expected), 1e-4f);
    UnmergeLora(*model);
    EXPECT_LT(MaxAbsDiff(Logits(), expected), 1e-4f);
}

TEST_F(PeftParityTest, SavingWritesPeftsFile) {
    LoadLoraAdapter(*model, Fixture("qwen2_lora"), HuggingFaceMapping(model->config()));
    const std::string dir = (std::filesystem::temp_directory_path() / "pulsatrix_lora_save").string();
    std::filesystem::remove_all(dir);
    SaveLoraAdapter(*model, dir, HuggingFaceMapping(model->config()));
    const SafetensorsFile ours = SafetensorsFile::Read(dir + "/adapter_model.safetensors");
    const SafetensorsFile peft = SafetensorsFile::Read(Fixture("qwen2_lora") + "/adapter_model.safetensors");
    EXPECT_EQ(std::set<std::string>(ours.names().begin(), ours.names().end()),
              std::set<std::string>(peft.names().begin(), peft.names().end()));
    for (const std::string& key : peft.names()) {
        EXPECT_EQ(ours.info(key).shape, peft.info(key).shape) << key;
        EXPECT_EQ(ours.tensor(key, &backend).to_host_vector(), peft.tensor(key, &backend).to_host_vector()) << key;
    }
    std::ifstream in(dir + "/adapter_config.json");
    std::stringstream text;
    text << in.rdbuf();
    const JsonValue config = ParseJson(text.str());
    EXPECT_EQ(config.find("r")->as_int64(), 4);
    EXPECT_EQ(config.find("lora_alpha")->as_float(), 8.0f);
    EXPECT_EQ(config.find("peft_type")->as_string(), "LORA");
    EXPECT_EQ(config.find("target_modules")->as_array().size(), 14u);

    // And it loads back onto a fresh model.
    std::unique_ptr<CausalLM> again = LoadCausalLM(Fixture("qwen2"), &backend);
    LoadLoraAdapter(*again, dir, HuggingFaceMapping(again->config()));
    EXPECT_EQ(again->forward(Tensor(Shape({1, 6}), &backend, kTokens)).to_host_vector(), Logits());
    std::filesystem::remove_all(dir);
}

TEST_F(PeftParityTest, ApplyLorasDefaultIsPeftsAllLinear) {
    (void)ApplyLora(*model);
    const std::string dir = (std::filesystem::temp_directory_path() / "pulsatrix_lora_all_linear").string();
    std::filesystem::remove_all(dir);
    SaveLoraAdapter(*model, dir, HuggingFaceMapping(model->config()));
    const SafetensorsFile ours = SafetensorsFile::Read(dir + "/adapter_model.safetensors");
    const SafetensorsFile peft = SafetensorsFile::Read(Fixture("qwen2_lora") + "/adapter_model.safetensors");
    EXPECT_EQ(std::set<std::string>(ours.names().begin(), ours.names().end()),
              std::set<std::string>(peft.names().begin(), peft.names().end()));
    std::filesystem::remove_all(dir);
}

TEST_F(PeftParityTest, LoadingRefusesWhatItCantRunAndChangesNothing) {
    const std::string dir = (std::filesystem::temp_directory_path() / "pulsatrix_lora_bad").string();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::filesystem::copy_file(Fixture("qwen2_lora") + "/adapter_model.safetensors", dir + "/adapter_model.safetensors");
    std::ifstream in(Fixture("qwen2_lora") + "/adapter_config.json");
    std::stringstream text;
    text << in.rdbuf();
    const std::string good = text.str();
    auto write = [&](const std::string& from, const std::string& to) {
        std::string s = good;
        const std::size_t at = s.find(from);
        ASSERT_NE(at, std::string::npos) << from;
        s.replace(at, from.size(), to);
        std::ofstream(dir + "/adapter_config.json") << s;
    };
    const auto mapping = HuggingFaceMapping(model->config());
    write("\"use_dora\": false", "\"use_dora\": true");
    EXPECT_THROW(LoadLoraAdapter(*model, dir, mapping), std::invalid_argument);
    write("\"use_rslora\": false", "\"use_rslora\": true");
    EXPECT_THROW(LoadLoraAdapter(*model, dir, mapping), std::invalid_argument);
    write("\"modules_to_save\": null", "\"modules_to_save\": [\"lm_head\"]");
    EXPECT_THROW(LoadLoraAdapter(*model, dir, mapping), std::invalid_argument);
    write("\"peft_type\": \"LORA\"", "\"peft_type\": \"IA3\"");
    EXPECT_THROW(LoadLoraAdapter(*model, dir, mapping), std::invalid_argument);
    // A key for a layer the model doesn't have.
    std::unique_ptr<CausalLM> tied = LoadCausalLM(Fixture("llama"), &backend);  // 4 heads / 2 K/V: k_proj is narrower
    write("\"use_dora\": false", "\"use_dora\": false");
    EXPECT_THROW(LoadLoraAdapter(*tied, dir, HuggingFaceMapping(tied->config())), std::invalid_argument);
    EXPECT_THROW(LoadLoraAdapter(*model, dir), std::invalid_argument);  // no mapping: pulsatrix names
    EXPECT_TRUE(LoraLayers(*model).empty());
    EXPECT_TRUE(LoraLayers(*tied).empty());
    std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace pulsatrix
