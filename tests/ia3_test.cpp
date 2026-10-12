// TRN-10: IA3 rescaling vectors on LinearModule, ApplyIA3() on whole models, merging, and PEFT adapter
// files, checked against PEFT's own logits and gradients (tools/golden/make_ia3_golden.py).
#include "pulsatrix/ia3.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lora.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }
const std::vector<float> kTokens = {3, 17, 42, 5, 60, 8};

std::vector<float> Random(std::size_t n, uint32_t seed, float scale = 1.0f, float offset = 0.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> v(n);
    for (float& x : v) x = offset + scale * normal(rng);
    return v;
}

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    float m = 0;
    for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

float MaxAbs(const std::vector<float>& v) {
    float m = 0;
    for (float x : v) m = std::max(m, std::fabs(x));
    return m;
}

class IA3LayerTest : public ::testing::TestWithParam<bool> {
protected:
    CPUBackend backend;
    const int64_t in = 6, out = 5, n = 4;

    LinearModule Layer(uint32_t seed) {
        LinearModule l(in, out, &backend);
        l.set_weight(Random(static_cast<std::size_t>(in * out), seed, 0.5f));
        l.set_bias(Random(static_cast<std::size_t>(out), seed + 1, 0.5f));
        return l;
    }
    Tensor Input(uint32_t seed) { return Tensor(Shape({n, in}), &backend, Random(static_cast<std::size_t>(n * in), seed)); }
    // The layer IA3 makes, as a plain layer: W diag(l) and b ⊙ l, or diag(l) W.
    LinearModule Merged(const LinearModule& l) {
        const bool ff = l.ia3()->feedforward();
        std::vector<float> w = l.weight().to_host_vector(), b = l.bias().to_host_vector();
        const std::vector<float> s = l.ia3()->scale().to_host_vector();
        for (int64_t i = 0; i < in; ++i)
            for (int64_t j = 0; j < out; ++j) w[static_cast<std::size_t>(i * out + j)] *= s[static_cast<std::size_t>(ff ? i : j)];
        if (!ff)
            for (int64_t j = 0; j < out; ++j) b[static_cast<std::size_t>(j)] *= s[static_cast<std::size_t>(j)];
        LinearModule m(in, out, &backend);
        m.set_weight(w);
        m.set_bias(b);
        return m;
    }
};

TEST_P(IA3LayerTest, StartsAsTheBaseLayerAndRescalesAsPeftDoes) {
    const bool ff = GetParam();
    LinearModule l = Layer(1);
    const Tensor x = Input(2);
    const std::vector<float> base = l.forward(x).to_host_vector();
    l.attach_ia3(ff);
    EXPECT_EQ(l.forward(x).to_host_vector(), base);  // l starts at 1
    EXPECT_EQ(l.ia3()->scale().numel(), ff ? in : out);
    EXPECT_EQ(l.named_parameters().back().name, "ia3_l");
    l.ia3()->set_scale(Random(static_cast<std::size_t>(ff ? in : out), 3, 0.5f, 1.0f));
    EXPECT_LT(MaxAbsDiff(l.forward(x).to_host_vector(), Merged(l).forward(x).to_host_vector()), 1e-5f);
    EXPECT_THROW(l.attach_ia3(ff), std::logic_error);
    LoRAConfig lora;
    EXPECT_THROW(l.attach_lora(lora), std::logic_error);
}

TEST_P(IA3LayerTest, GradientsMatchFiniteDifferences) {
    const bool ff = GetParam();
    LinearModule l = Layer(4);
    l.attach_ia3(ff);
    l.ia3()->set_scale(Random(static_cast<std::size_t>(ff ? in : out), 5, 0.5f, 1.0f));
    const Tensor x = Input(6);
    const std::vector<float> u = Random(static_cast<std::size_t>(n * out), 7);
    auto loss = [&](const Tensor& input) {
        const std::vector<float> y = l.forward(input).to_host_vector();
        double s = 0;
        for (std::size_t i = 0; i < y.size(); ++i) s += static_cast<double>(y[i]) * u[i];
        return s;
    };
    (void)l.forward(x);
    const std::vector<float> grad_x = l.backward(Tensor(Shape({n, out}), &backend, u)).to_host_vector();
    const std::vector<float> grad_l = l.ia3()->scale_grad().to_host_vector();
    const std::vector<float> w_grad = l.weight_grad().to_host_vector();
    const double h = 1e-2;
    std::vector<float> s = l.ia3()->scale().to_host_vector();
    for (std::size_t i = 0; i < s.size(); ++i) {
        const float saved = s[i];
        s[i] = saved + static_cast<float>(h);
        l.ia3()->set_scale(s);
        const double up = loss(x);
        s[i] = saved - static_cast<float>(h);
        l.ia3()->set_scale(s);
        const double down = loss(x);
        s[i] = saved;
        l.ia3()->set_scale(s);
        EXPECT_NEAR(grad_l[i], (up - down) / (2 * h), 2e-3) << "l " << i;
    }
    std::vector<float> xv = x.to_host_vector();
    for (std::size_t i = 0; i < xv.size(); ++i) {
        const float saved = xv[i];
        xv[i] = saved + static_cast<float>(h);
        const double up = loss(Tensor(x.shape(), &backend, xv));
        xv[i] = saved - static_cast<float>(h);
        const double down = loss(Tensor(x.shape(), &backend, xv));
        xv[i] = saved;
        EXPECT_NEAR(grad_x[i], (up - down) / (2 * h), 2e-3) << "x " << i;
    }
    // And the base weight's gradient, had it been trainable, is that of the rescaled layer's input or output.
    LinearModule m = Layer(4);
    if (ff) {
        std::vector<float> scaled = x.to_host_vector();
        for (int64_t r = 0; r < n; ++r)
            for (int64_t i = 0; i < in; ++i) scaled[static_cast<std::size_t>(r * in + i)] *= s[static_cast<std::size_t>(i)];
        (void)m.forward(Tensor(x.shape(), &backend, scaled));
        (void)m.backward(Tensor(Shape({n, out}), &backend, u));
    } else {
        std::vector<float> g = u;
        for (int64_t r = 0; r < n; ++r)
            for (int64_t j = 0; j < out; ++j) g[static_cast<std::size_t>(r * out + j)] *= s[static_cast<std::size_t>(j)];
        (void)m.forward(x);
        (void)m.backward(Tensor(Shape({n, out}), &backend, g));
    }
    EXPECT_LT(MaxAbsDiff(w_grad, m.weight_grad().to_host_vector()), 1e-4f);
}

TEST_P(IA3LayerTest, MergeUnmergeAndDisable) {
    const bool ff = GetParam();
    LinearModule l = Layer(8);
    const Tensor x = Input(9);
    const std::vector<float> base = l.forward(x).to_host_vector(), w = l.weight().to_host_vector(), b = l.bias().to_host_vector();
    l.attach_ia3(ff);
    l.ia3()->set_scale(Random(static_cast<std::size_t>(ff ? in : out), 10, 0.5f, 1.0f));
    const std::vector<float> adapted = l.forward(x).to_host_vector();
    l.merge_ia3();
    EXPECT_LT(MaxAbsDiff(l.forward(x).to_host_vector(), adapted), 1e-5f);
    EXPECT_EQ(MaxAbsDiff(l.bias().to_host_vector(), b) == 0.0f, ff);  // only output scaling touches the bias
    EXPECT_THROW(l.set_ia3_enabled(false), std::logic_error);
    l.unmerge_ia3();
    EXPECT_LT(MaxAbsDiff(l.weight().to_host_vector(), w), 1e-5f);
    EXPECT_LT(MaxAbsDiff(l.bias().to_host_vector(), b), 1e-5f);
    // Unmerging divides by l + 1e-8, so W is back only to float rounding.
    l.set_ia3_enabled(false);
    EXPECT_LT(MaxAbsDiff(l.forward(x).to_host_vector(), base), 1e-5f);
    l.set_ia3_enabled(true);
    l.detach_ia3();
    EXPECT_LT(MaxAbsDiff(l.forward(x).to_host_vector(), base), 1e-5f);
}

TEST_P(IA3LayerTest, RelevanceIsThatOfTheRescaledLayer) {
    const bool ff = GetParam();
    LinearModule l = Layer(11);
    l.attach_ia3(ff);
    l.ia3()->set_scale(Random(static_cast<std::size_t>(ff ? in : out), 12, 0.5f, 1.0f));
    LinearModule m = Merged(l);
    const Tensor x = Input(13);
    const Tensor r(Shape({n, out}), &backend, Random(static_cast<std::size_t>(n * out), 14));
    LRPRuleConfig gamma;
    gamma.rule = LRPRule::Gamma;
    for (const LRPRuleConfig& config : {LRPRuleConfig{}, gamma}) {
        (void)l.forward(x);
        (void)m.forward(x);
        const std::vector<float> want = m.propagate_relevance(r, config).to_host_vector();
        EXPECT_LT(MaxAbsDiff(l.propagate_relevance(r, config).to_host_vector(), want), 1e-5f * MaxAbs(want));
    }
}

INSTANTIATE_TEST_SUITE_P(OutputAndFeedforward, IA3LayerTest, ::testing::Values(false, true),
                         [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "Feedforward" : "Output"; });

TEST(IA3Models, TheDefaultIsKeysValuesAndTheMlpsInnerActivations) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> lm = LoadCausalLM(Fixture("qwen2"), &backend);
    EXPECT_EQ(ApplyIA3(*lm), (std::vector<std::string>{"layers.0.mha.k_proj", "layers.0.mha.v_proj", "layers.0.swiglu.down_proj",
                                                        "layers.1.mha.k_proj", "layers.1.mha.v_proj", "layers.1.swiglu.down_proj"}));
    EXPECT_TRUE(lm->layer(0).swiglu().down_proj().ia3()->feedforward());
    EXPECT_FALSE(lm->layer(0).mha().k_proj().ia3()->feedforward());
    int trainable = 0;
    for (const NamedParamRef& p : lm->named_parameters()) trainable += p.ref.value->requires_grad() ? 1 : 0;
    EXPECT_EQ(trainable, 6);
    EXPECT_THROW((void)ApplyIA3(*lm), std::invalid_argument);
    RemoveIA3(*lm);

    std::unique_ptr<EncoderLM> esm = LoadEncoderLM(Fixture("esm"), &backend);
    const std::vector<std::string> esm_layers = ApplyIA3(*esm);
    EXPECT_EQ(esm_layers.size(), 3 * static_cast<std::size_t>(esm->num_layers()));
    EXPECT_EQ(esm_layers[2], "layers.0.mlp.fc2");

    IA3Config bad;
    bad.target_modules = {"q_proj"};
    bad.feedforward_modules = {"down_proj"};  // not a target
    EXPECT_THROW((void)ApplyIA3(*lm, bad), std::invalid_argument);
    EXPECT_TRUE(IA3Layers(*lm).empty());
}

class PeftIA3Test : public ::testing::Test {
protected:
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Fixture("qwen2"), &backend);
    SafetensorsFile reference = SafetensorsFile::Read(Fixture("qwen2_ia3") + "/reference.safetensors");
    std::vector<float> Logits() { return model->forward(Tensor(Shape({1, 6}), &backend, kTokens)).to_host_vector(); }
};

TEST_F(PeftIA3Test, LogitsAndGradientsMatchPeft) {
    LoadIA3Adapter(*model, Fixture("qwen2_ia3"), HuggingFaceMapping(model->config()));
    ASSERT_EQ(IA3Layers(*model).size(), 6u);
    const std::vector<float> expected = reference.tensor("logits", &backend).to_host_vector();
    SetIA3Enabled(*model, false);
    EXPECT_GT(MaxAbsDiff(Logits(), expected), 0.05f);  // the vectors matter
    SetIA3Enabled(*model, true);
    EXPECT_LT(MaxAbsDiff(Logits(), expected), 1e-4f);

    (void)model->backward(reference.tensor("upstream", &backend));
    std::map<std::string, std::string> names;
    for (const WeightMapping& w : HuggingFaceMapping(model->config())) names[w.target] = w.source;
    for (const auto& [name, layer] : IA3Layers(*model)) {
        const std::string source = names.at(name + ".weight");
        const std::string key = "base_model.model." + source.substr(0, source.size() - 7) + ".ia3_l";
        EXPECT_LT(MaxAbsDiff(layer->ia3()->scale_grad().to_host_vector(), reference.tensor(key, &backend).to_host_vector()), 1e-3f)
            << name;
    }
    MergeIA3(*model);
    EXPECT_LT(MaxAbsDiff(Logits(), expected), 1e-4f);
    UnmergeIA3(*model);
    EXPECT_LT(MaxAbsDiff(Logits(), expected), 1e-4f);
}

TEST_F(PeftIA3Test, SavingWritesPeftsFileAndLoadsBack) {
    LoadIA3Adapter(*model, Fixture("qwen2_ia3"), HuggingFaceMapping(model->config()));
    const std::string dir = (std::filesystem::temp_directory_path() / "pulsatrix_ia3_save").string();
    std::filesystem::remove_all(dir);
    SaveIA3Adapter(*model, dir, HuggingFaceMapping(model->config()));
    {
        const SafetensorsFile ours = SafetensorsFile::Read(dir + "/adapter_model.safetensors");
        const SafetensorsFile peft = SafetensorsFile::Read(Fixture("qwen2_ia3") + "/adapter_model.safetensors");
        EXPECT_EQ(std::set<std::string>(ours.names().begin(), ours.names().end()),
                  std::set<std::string>(peft.names().begin(), peft.names().end()));
        for (const std::string& key : peft.names()) {
            EXPECT_EQ(ours.info(key).shape, peft.info(key).shape) << key;
            EXPECT_EQ(ours.tensor(key, &backend).to_host_vector(), peft.tensor(key, &backend).to_host_vector()) << key;
        }
    }
    std::stringstream text;
    {
        std::ifstream in(dir + "/adapter_config.json");  // closed before remove_all: Windows won't delete open files
        text << in.rdbuf();
    }
    const JsonValue config = ParseJson(text.str());
    EXPECT_EQ(config.find("peft_type")->as_string(), "IA3");
    EXPECT_EQ(config.find("target_modules")->as_array().size(), 6u);
    EXPECT_EQ(config.find("feedforward_modules")->as_array().size(), 2u);

    std::unique_ptr<CausalLM> again = LoadCausalLM(Fixture("qwen2"), &backend);
    LoadIA3Adapter(*again, dir, HuggingFaceMapping(again->config()));
    EXPECT_EQ(again->forward(Tensor(Shape({1, 6}), &backend, kTokens)).to_host_vector(), Logits());
    std::filesystem::remove_all(dir);
}

TEST_F(PeftIA3Test, LoadingRefusesWhatItCantRun) {
    const std::string dir = (std::filesystem::temp_directory_path() / "pulsatrix_ia3_bad").string();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    std::filesystem::copy_file(Fixture("qwen2_ia3") + "/adapter_model.safetensors", dir + "/adapter_model.safetensors");
    std::string good;
    {
        std::ifstream in(Fixture("qwen2_ia3") + "/adapter_config.json");
        std::stringstream text;
        text << in.rdbuf();
        good = text.str();
    }
    auto write = [&](const std::string& from, const std::string& to) {
        std::string s = good;
        const std::size_t at = s.find(from);
        ASSERT_NE(at, std::string::npos) << from;
        s.replace(at, from.size(), to);
        std::ofstream(dir + "/adapter_config.json") << s;
    };
    const auto mapping = HuggingFaceMapping(model->config());
    write("\"fan_in_fan_out\": false", "\"fan_in_fan_out\": true");
    EXPECT_THROW(LoadIA3Adapter(*model, dir, mapping), std::invalid_argument);
    write("\"modules_to_save\": null", "\"modules_to_save\": [\"lm_head\"]");
    EXPECT_THROW(LoadIA3Adapter(*model, dir, mapping), std::invalid_argument);
    write("\"peft_type\": \"IA3\"", "\"peft_type\": \"LORA\"");
    EXPECT_THROW(LoadIA3Adapter(*model, dir, mapping), std::invalid_argument);
    write("\"down_proj\"\n  ]", "\"q_proj\"\n  ]");  // feedforward_modules no longer names down_proj
    EXPECT_THROW(LoadIA3Adapter(*model, dir, mapping), std::invalid_argument);
    EXPECT_TRUE(IA3Layers(*model).empty());
    std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace pulsatrix
