// IO-5: Hugging Face config.json and single or sharded safetensors checkpoints. The configs in
// tests/fixtures/hf are the real ones (trimmed to the fields that matter) for the v1.2 target
// models, fetched from the Hub on 2026-10-05.

#include "pulsatrix/hf_model.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& name) {
    return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf/" + name + ".config.json";
}

bool Mentions(const std::vector<std::string>& list, const std::string& text) {
    for (const auto& s : list) {
        if (s.find(text) != std::string::npos) return true;
    }
    return false;
}

TEST(HfConfigTest, SmolLM2IsAPlainLlamaThatPulsatrixRunsExactly) {
    HfModelConfig c = ReadHfConfig(Fixture("smollm2-135m"));
    EXPECT_EQ(c.architecture, "LlamaForCausalLM");
    EXPECT_EQ(c.model_type, "llama");
    EXPECT_EQ(c.hidden_size, 576);
    EXPECT_EQ(c.intermediate_size, 1536);
    EXPECT_EQ(c.num_hidden_layers, 30);
    EXPECT_EQ(c.num_attention_heads, 9);
    EXPECT_EQ(c.num_key_value_heads, 3);
    EXPECT_EQ(c.head_dim, 64);  // 576 / 9, not given
    EXPECT_EQ(c.vocab_size, 49152);
    EXPECT_FLOAT_EQ(c.rms_norm_eps, 1e-5f);
    EXPECT_FLOAT_EQ(c.rope_theta, 100000.0f);
    EXPECT_FALSE(c.rope_scaling.has_value());
    EXPECT_TRUE(c.tie_word_embeddings);
    EXPECT_FALSE(c.qkv_bias);
    EXPECT_FALSE(c.out_bias);
    EXPECT_FALSE(c.qk_norm);
    EXPECT_EQ(c.eos_token_ids, (std::vector<int64_t>{0}));
    EXPECT_EQ(c.dtype, "bfloat16");
    EXPECT_TRUE(c.unsupported.empty());

    AttentionConfig a = ToAttentionConfig(c);
    EXPECT_EQ(a.d_model, 576);
    EXPECT_EQ(a.num_kv_heads, 3);
    EXPECT_EQ(a.rope_layout, RoPELayout::RotateHalf);
    EXPECT_TRUE(a.causal);
    CPUBackend backend;
    MultiHeadAttentionModule attention(a, &backend);  // builds: the shapes fit together
    EXPECT_EQ(attention.q_proj().weight().shape(), Shape({576, 576}));
    EXPECT_EQ(attention.k_proj().weight().shape(), Shape({576, 192}));
}

TEST(HfConfigTest, Qwen2HasQkvBiasesWithoutSayingSo) {
    HfModelConfig c = ReadHfConfig(Fixture("qwen2.5-0.5b"));
    EXPECT_EQ(c.model_type, "qwen2");
    EXPECT_TRUE(c.qkv_bias);   // no attention_bias field: the architecture has them
    EXPECT_FALSE(c.out_bias);  // and never on o_proj
    EXPECT_EQ(c.num_key_value_heads, 2);
    EXPECT_FLOAT_EQ(c.rope_theta, 1e6f);
    // sliding_window 32768 is in the file, but use_sliding_window is false.
    EXPECT_FALSE(c.sliding_window.has_value());
    EXPECT_TRUE(c.unsupported.empty());
}

TEST(HfConfigTest, Qwen3HasItsOwnHeadDimAndQkNorm) {
    HfModelConfig c = ReadHfConfig(Fixture("qwen3-0.6b"));
    EXPECT_EQ(c.head_dim, 128);  // 1024 / 16 would be 64
    EXPECT_TRUE(c.qk_norm);
    EXPECT_EQ(c.eos_token_ids, (std::vector<int64_t>{151645}));
    AttentionConfig a = ToAttentionConfig(c);
    EXPECT_TRUE(a.use_qk_norm);
    EXPECT_FLOAT_EQ(a.qk_norm_eps, 1e-6f);
    EXPECT_EQ(a.head_dim, 128);
    EXPECT_TRUE(c.unsupported.empty());
}

TEST(HfConfigTest, Llama32NeedsRopeScaling) {
    HfModelConfig c = ReadHfConfig(Fixture("llama-3.2-1b"));
    ASSERT_TRUE(c.rope_scaling.has_value());
    EXPECT_EQ(c.rope_scaling->type, "llama3");
    EXPECT_FLOAT_EQ(c.rope_scaling->factor, 32.0f);
    EXPECT_FLOAT_EQ(c.rope_scaling->high_freq_factor, 4.0f);
    EXPECT_EQ(c.rope_scaling->original_max_position_embeddings, 8192);
    EXPECT_TRUE(Mentions(c.unsupported, "llama3"));
    EXPECT_EQ(c.unsupported.size(), 1u);
}

TEST(HfConfigTest, GemmaListsEverythingPulsatrixCantRunYet) {
    HfModelConfig c = ReadHfConfig(Fixture("gemma-3-270m"));
    EXPECT_EQ(c.model_type, "gemma3_text");
    EXPECT_EQ(c.head_dim, 256);
    EXPECT_EQ(c.hidden_act, "gelu_pytorch_tanh");  // from hidden_activation
    EXPECT_EQ(c.sliding_window, 512);
    EXPECT_TRUE(c.qk_norm);
    EXPECT_EQ(c.layer_types.size(), 4u);
    EXPECT_TRUE(Mentions(c.unsupported, "sliding-window"));
    EXPECT_TRUE(Mentions(c.unsupported, "(1 + w)"));
    EXPECT_TRUE(Mentions(c.unsupported, "gelu_pytorch_tanh"));
    EXPECT_FALSE(Mentions(c.unsupported, "query_pre_attn_scalar"));  // 256 == head_dim
}

TEST(HfConfigTest, NewerAndMultimodalLayouts) {
    // transformers 5 writes rope_parameters; a multimodal config nests the text model; eos may be a list.
    HfModelConfig c = ParseHfConfig(R"({
        "architectures": ["SomeVLM"],
        "eos_token_id": [1, 106],
        "text_config": {"model_type": "llama", "hidden_size": 64, "intermediate_size": 128,
                        "num_hidden_layers": 2, "num_attention_heads": 4, "vocab_size": 100,
                        "rope_parameters": {"rope_type": "default", "rope_theta": 250000.0},
                        "dtype": "float32", "tie_word_embeddings": false}
    })");
    EXPECT_EQ(c.architecture, "SomeVLM");
    EXPECT_EQ(c.hidden_size, 64);
    EXPECT_EQ(c.num_key_value_heads, 4);
    EXPECT_FLOAT_EQ(c.rope_theta, 250000.0f);
    EXPECT_FALSE(c.rope_scaling.has_value());
    EXPECT_EQ(c.eos_token_ids, (std::vector<int64_t>{1, 106}));
    EXPECT_EQ(c.dtype, "float32");
    EXPECT_FALSE(c.tie_word_embeddings);
    // "SomeVLM" is not a causal LM class, so it is flagged.
    EXPECT_TRUE(Mentions(c.unsupported, "not a decoder-only causal language model"));

    // An encoder such as BERT or XLM-RoBERTa parses, but is flagged too.
    HfModelConfig encoder = ParseHfConfig(R"({"architectures": ["XLMRobertaModel"], "model_type": "xlm-roberta",
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
        "vocab_size": 100, "hidden_act": "gelu"})");
    EXPECT_TRUE(Mentions(encoder.unsupported, "XLMRobertaModel"));
}

TEST(HfConfigTest, RejectsBrokenConfigs) {
    const std::string base = R"("intermediate_size": 8, "num_hidden_layers": 1, "vocab_size": 10)";
    EXPECT_THROW((void)ParseHfConfig("[]"), std::invalid_argument);
    EXPECT_THROW((void)ParseHfConfig("{"), std::invalid_argument);
    EXPECT_THROW((void)ParseHfConfig("{" + base + R"(, "num_attention_heads": 2})"), std::invalid_argument);  // no hidden_size
    EXPECT_THROW((void)ParseHfConfig("{" + base + R"(, "hidden_size": 6, "num_attention_heads": 4})"),
                 std::invalid_argument);  // 6 / 4
    EXPECT_THROW((void)ParseHfConfig("{" + base + R"(, "hidden_size": 8, "num_attention_heads": 4, "num_key_value_heads": 3})"),
                 std::invalid_argument);
    EXPECT_THROW((void)ParseHfConfig("{" + base + R"(, "hidden_size": "8", "num_attention_heads": 4})"),
                 std::invalid_argument);
    EXPECT_THROW((void)ParseHfConfig("{" + base + R"(, "hidden_size": 8.5, "num_attention_heads": 4})"),
                 std::invalid_argument);
    EXPECT_THROW((void)ReadHfConfig("/nonexistent/config.json"), std::runtime_error);
}

// ---- checkpoints --------------------------------------------------------------------------

class HfCheckpointTest : public ::testing::Test {
protected:
    CPUBackend backend;
    std::filesystem::path dir;

    void SetUp() override {
        dir = std::filesystem::path(::testing::TempDir()) /
              ("pulsatrix_hf_" + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
    }
    void TearDown() override { std::filesystem::remove_all(dir); }

    void WriteShard(const std::string& file, const std::vector<std::pair<std::string, float>>& tensors) {
        std::vector<Tensor> owned;
        for (const auto& [name, v] : tensors) owned.emplace_back(Shape({2}), &backend, std::vector<float>{v, v + 1});
        std::vector<std::pair<std::string, const Tensor*>> refs;
        for (size_t i = 0; i < tensors.size(); ++i) refs.emplace_back(tensors[i].first, &owned[i]);
        WriteSafetensors((dir / file).string(), refs);
    }
    void WriteIndex(const std::string& weight_map) {
        std::ofstream((dir / "model.safetensors.index.json").string())
            << R"({"metadata": {"total_size": 32}, "weight_map": {)" << weight_map << "}}";
    }
};

TEST_F(HfCheckpointTest, ReadsShardsThroughTheIndex) {
    WriteShard("model-00001-of-00002.safetensors", {{"model.embed_tokens.weight", 1.0f}, {"model.norm.weight", 3.0f}});
    WriteShard("model-00002-of-00002.safetensors", {{"lm_head.weight", 5.0f}});
    WriteIndex(R"("model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                  "model.norm.weight": "model-00001-of-00002.safetensors",
                  "lm_head.weight": "model-00002-of-00002.safetensors")");
    HfCheckpoint ck = HfCheckpoint::Open(dir.string());
    EXPECT_EQ(ck.names(), (std::vector<std::string>{"lm_head.weight", "model.embed_tokens.weight", "model.norm.weight"}));
    EXPECT_EQ(ck.shard_of("lm_head.weight"), "model-00002-of-00002.safetensors");
    EXPECT_EQ(ck.shards().size(), 2u);
    EXPECT_EQ(ck.tensor("lm_head.weight", &backend).to_host_vector(), (std::vector<float>{5.0f, 6.0f}));
    EXPECT_EQ(ck.tensor("model.norm.weight", &backend).to_host_vector(), (std::vector<float>{3.0f, 4.0f}));
    EXPECT_EQ(ck.info("model.embed_tokens.weight").shape, (std::vector<int64_t>{2}));
    EXPECT_FALSE(ck.contains("missing"));
    EXPECT_THROW((void)ck.tensor("missing", &backend), std::invalid_argument);
}

TEST_F(HfCheckpointTest, ReadsASingleFile) {
    WriteShard("model.safetensors", {{"a", 1.0f}, {"b", 2.0f}});
    HfCheckpoint ck = HfCheckpoint::Open(dir.string());
    EXPECT_EQ(ck.names().size(), 2u);
    EXPECT_EQ(ck.tensor("b", &backend).to_host_vector(), (std::vector<float>{2.0f, 3.0f}));
}

TEST_F(HfCheckpointTest, RefusesIndexesThatDontMatchTheirShards) {
    EXPECT_THROW((void)HfCheckpoint::Open(dir.string()), std::runtime_error);  // empty directory

    WriteShard("s1.safetensors", {{"a", 1.0f}, {"b", 2.0f}});
    WriteIndex(R"("a": "s1.safetensors")");  // b is in the shard but not indexed
    EXPECT_THROW((void)HfCheckpoint::Open(dir.string()), std::invalid_argument);
    WriteIndex(R"("a": "s1.safetensors", "b": "s1.safetensors", "c": "s1.safetensors")");  // no c
    EXPECT_THROW((void)HfCheckpoint::Open(dir.string()), std::invalid_argument);
    WriteIndex(R"("a": "s1.safetensors", "b": "missing.safetensors")");
    EXPECT_THROW((void)HfCheckpoint::Open(dir.string()), std::runtime_error);
    for (const char* escape : {"../s1.safetensors", "sub/s1.safetensors", "..", "C:s1.safetensors", "a\\\\b"}) {
        WriteIndex(std::string(R"("a": ")") + escape + R"(", "b": "s1.safetensors")");
        EXPECT_THROW((void)HfCheckpoint::Open(dir.string()), std::invalid_argument) << escape;
    }
    WriteIndex(R"("a": 3)");
    EXPECT_THROW((void)HfCheckpoint::Open(dir.string()), std::invalid_argument);
    std::ofstream((dir / "model.safetensors.index.json").string()) << "{not json";
    EXPECT_THROW((void)HfCheckpoint::Open(dir.string()), std::invalid_argument);
}

// The memory-mapped reader sees exactly what the reading one does, and rejects the same files.
TEST_F(HfCheckpointTest, MappedFilesMatchReadFiles) {
    WriteShard("x.safetensors", {{"w", 7.0f}, {"v", -1.0f}});
    const std::string path = (dir / "x.safetensors").string();
    SafetensorsFile mapped = SafetensorsFile::Map(path);
    SafetensorsFile read = SafetensorsFile::Read(path);
    EXPECT_EQ(mapped.names(), read.names());
    EXPECT_EQ(mapped.tensor("w", &backend).to_host_vector(), read.tensor("w", &backend).to_host_vector());
    SafetensorsFile copy = mapped;  // copies share the mapping
    EXPECT_EQ(copy.tensor("v", &backend).to_host_vector(), (std::vector<float>{-1.0f, 0.0f}));
    std::ofstream((dir / "empty.safetensors").string());
    EXPECT_THROW((void)SafetensorsFile::Map((dir / "empty.safetensors").string()), std::invalid_argument);
    std::ofstream((dir / "junk.safetensors").string()) << "not a safetensors file at all";
    EXPECT_THROW((void)SafetensorsFile::Map((dir / "junk.safetensors").string()), std::invalid_argument);
    EXPECT_THROW((void)SafetensorsFile::Map((dir / "absent.safetensors").string()), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
