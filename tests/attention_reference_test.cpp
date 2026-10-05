// LLM-1: MultiHeadAttentionModule against Hugging Face Transformers' own attention classes
// (LlamaAttention, Qwen2Attention, Qwen3Attention), on the layouts real checkpoints use:
// grouped-query and multi-query attention, head_dim independent of d_model, Q/K/V biases,
// "rotate half" RoPE with a custom base, a position offset, QK-Norm, a causal mask, and left
// and right padding.
//
// Reference values generated offline by tools/generate_attention_reference_values.py
// (torch 2.14.1, transformers 5.18.0; see that file for the exact configurations). Weights
// and inputs are rebuilt here from the same Det() formula, so no weights are stored.
// Checked: the forward output, and the input gradient of sum(output * G) with G zero on
// padding queries.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/multihead_attention_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Det(int64_t n, int64_t mul, int64_t add, int64_t div) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = static_cast<float>((i * mul + add) % 17 - 8) / static_cast<float>(div);
    }
    return v;
}

void ExpectMatchesReference(const Tensor& actual, const std::vector<float>& expected) {
    ASSERT_EQ(static_cast<size_t>(actual.numel()), expected.size());
    const std::vector<float> got = actual.to_host_vector();
    for (size_t i = 0; i < expected.size(); ++i) {
        const float tol = 1e-4f * std::max(1.0f, std::fabs(expected[i]));
        EXPECT_NEAR(got[i], expected[i], tol) << "element " << i;
    }
}

// ---- Reference values: tools/generate_attention_reference_values.py output, verbatim ----
const std::vector<float> kLlamaOutput = {
    6.28125f, 10.6624994f, -0.468749523f, 0.512499809f, 6.38125038f, -6.23750067f, -0.156249523f, 0.718749762f,
    -8.89896965f, 5.73666191f, 2.79074645f, -0.898544312f, 1.80714047f, -2.50294518f, -1.23872972f, -3.39226723f,
    -6.22172737f, -6.82956553f, 7.92926979f, -7.77032614f, 4.61183643f, 7.18019295f, -7.6904912f, 3.41667318f,
    -1.92429543f, -1.81777155f, -2.14936566f, 5.68099833f, 5.90130901f, -4.23116684f, 5.37568188f, 2.57039237f,
    -2.0432353f, -1.3158685f, 4.30234432f, 1.14716101f, -3.04585075f, 9.82735634f, 2.10928273f, -3.35872912f,
    0.0718750358f, 3.18437505f, 6.98749971f, -3.65937495f, 8.484375f, 3.2562499f, -3.56562471f, 4.75312471f,
    2.02980852f, 1.93259633f, 2.23088145f, -2.12202024f, 6.80392647f, -1.92033434f, -2.44884896f, 3.60805106f,
    -7.45481586f, 0.59121573f, 5.76033211f, 2.13500237f, -4.37227345f, 3.08553243f, 2.48204112f, -7.75178242f,
    0.555373847f, -2.29580951f, 2.54111576f, -3.38031721f, 2.79690027f, -1.39742494f, -3.49385691f, 11.6083603f,
    -0.441665411f, 3.68215728f, 11.03722f, -3.96870565f, -1.33257914f, 8.30685997f, -3.67094183f, -4.36967134f,
};
const std::vector<float> kLlamaInputGrad = {
    -8.90266609f, 6.07271957f, -12.6433954f, 13.9003487f, 1.62727904f, -3.8032589f, 4.69130898f, -19.5449257f,
    -6.51310062f, 4.06256294f, -0.704827547f, -3.30151343f, 5.9753294f, -2.43655348f, 2.74105906f, -1.22590601f,
    -9.0213623f, 10.7862425f, -15.0089464f, 11.4201279f, -14.1220779f, -11.4001799f, 16.3749542f, -22.7480106f,
    -0.886680603f, 0.610924959f, 1.71829987f, 9.80862617f, 2.14700031f, -0.973502636f, 4.07392597f, -3.96764827f,
    -16.1124573f, -8.39334965f, 0.479942679f, 9.40786743f, 22.9192333f, -22.5429649f, 5.3543911f, 7.91329384f,
    4.56399393f, -3.27886653f, -3.76646256f, -1.97126746f, -1.94095159f, 1.92919266f, -2.69863391f, 3.37439847f,
    0.76952374f, 0.651918828f, 0.132376626f, 5.66338778f, -2.33205438f, -2.265589f, 1.31093848f, 0.972465038f,
    -1.29911423f, 3.24256063f, 1.93921208f, 0.967739344f, 0.860878587f, -1.8740449f, -0.521536708f, -0.501142085f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};
const std::vector<float> kQwen2Output = {
    -0.439682543f, -1.03405488f, 1.72496402f, 0.0242424011f, -0.570129871f, -0.688600421f, -1.39069271f,
    -0.854256868f, 2.67496371f, -0.719696999f, -0.183261037f, -1.22294378f, -1.18611217f, -0.901911318f, 2.5241015f,
    -0.587439775f, 0.0439573526f, -1.28307045f, -1.1197871f, -0.982812226f, 2.36985493f, -0.292032272f,
    0.136560678f, -1.35005903f, -1.62310433f, -0.323407948f, 2.81889105f, -0.7452299f, -0.460658312f, -1.44833827f,
    1.24567103f, -2.84415579f, 1.29617596f, 1.94696963f, -2.14285707f, -0.0386000276f, 1.38135004f, -2.56864548f,
    1.20471108f, 1.82823157f, -2.01915765f, -0.152391911f, 0.844670951f, -1.41877377f, 2.50886607f, 0.637236834f,
    -1.58416724f, -0.983457088f, 0.811860681f, -1.33747602f, 2.50149202f, 0.877296269f, -1.53137827f, -1.6305778f,
    0.883511901f, -1.08626461f, 2.30878735f, 1.02337313f, -1.4529283f, -1.79046714f,
};
const std::vector<float> kQwen2InputGrad = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -0.318049431f, 0.021097064f, -1.4410491f, -0.180958807f, 1.1208117f,
    -2.08653355f, 0.183080748f, 0.316604286f, 1.70100033f, -0.0866975784f, 0.11041081f, 0.67845124f, -0.379038274f,
    0.651093125f, 0.267553478f, -0.821062446f, 0.680048704f, -0.137649775f, 0.0593799055f, -0.899280667f,
    0.384966671f, -1.094221f, 0.325969607f, -0.741077065f, 0.480850577f, -1.98571396f, 2.12194467f, 0.679545105f,
    -1.4684031f, -0.643404722f, 0.00638299063f, -1.12948287f, -0.245398134f, 0.163399875f, -0.242194608f,
    -0.611437678f, -1.96414185f, 2.60326171f, 1.21346486f, -2.11433458f, 2.91130638f, 0.199152589f, -0.333185643f,
    0.211944565f, -1.31353402f, -0.326235563f, 0.106753871f, -0.028781414f, 0.113901988f, -0.619728565f,
    -1.2083236f, 0.00425339863f, -0.00861439109f, -0.383837879f,
};
const std::vector<float> kQwen3Output = {
    0.635416687f, -1.64236116f, -0.496527791f, 1.82986104f, -3.04513884f, 1.40625012f, 3.02430558f, -1.96875012f,
    -0.732177496f, 1.61817491f, -0.531353295f, -1.33072662f, 4.06122017f, -0.961188138f, -1.76583171f, 4.36025763f,
    1.15284204f, -1.10786045f, 1.33434939f, -1.65401447f, -2.29896903f, -0.273469329f, -3.33712697f, -2.68311834f,
    0.0237505659f, -0.0288071185f, 0.243811101f, 0.0793703422f, 0.0824589133f, 0.447946638f, 0.268225878f,
    0.32735607f, -2.13586879f, 2.75897598f, -0.964323878f, -2.04518151f, 6.19014883f, -1.32508874f, -1.76066995f,
    6.18407059f, 4.55555582f, 6.00694466f, -2.93055582f, -1.83333325f, 5.63888884f, -9.31944466f, -8.22222328f,
    4.91666698f, 4.34684277f, 5.04252386f, -2.51955748f, -1.31478453f, 4.26674747f, -8.18836021f, -7.06924391f,
    3.64122391f, -1.76559305f, 0.337813377f, -3.77354908f, 0.400061637f, -1.04707479f, 0.650969386f, 5.54475975f,
    -0.619475126f, 1.29008722f, -1.05234122f, -0.0871863961f, 1.30130172f, -2.86401248f, 0.373255551f, 1.27100241f,
    -2.81192136f, 2.80946207f, -2.88472819f, 1.69383752f, 1.35470223f, -5.06237507f, 0.303959489f, -0.570219398f,
    -5.32547379f,
};
const std::vector<float> kQwen3InputGrad = {
    -3.48898911f, 2.47345448f, -2.05679917f, -1.25269341f, -0.0849578381f, 0.71914798f, -1.70630169f, -1.18281841f,
    3.88024902f, 2.16923499f, -7.24998045f, 0.757847369f, -10.6458654f, 0.666429043f, -6.64554501f, 7.65493345f,
    -3.29593754f, 0.278926969f, -2.58820581f, -0.516058803f, -2.52454376f, 0.456593931f, -0.565853119f,
    -0.655700564f, -1.42411447f, -1.02016151f, -0.201518357f, -0.776829541f, 0.521843433f, 0.683480144f,
    0.717383623f, 0.818084717f, 0.573536634f, 1.03478205f, -1.4943924f, 4.45177841f, -1.38882184f, 2.60907793f,
    -1.14576757f, 2.41740155f, 5.75970602f, -0.478779018f, 7.54682493f, 1.36264408f, 7.48112011f, 1.29693913f,
    -2.26516438f, -0.65527761f, -2.037359f, -0.797901034f, -4.2989769f, -0.486575007f, -6.35037327f, -2.58663797f,
    -3.28184986f, -1.11732018f, -4.34242868f, 0.97675693f, -4.36486864f, -1.67910051f, -2.96274805f, 0.260426551f,
    -0.408691585f, -0.079723388f, 0.626392722f, -0.793476284f, -0.93253541f, -0.095215112f, -1.55276871f,
    -1.0004499f, 0.111669481f, 0.309286356f, 0.508275628f, -2.20835567f, -0.121923327f, -0.307006478f,
    -0.143214703f, -0.759378254f, 0.427828252f, 0.191692695f,
};

// Runs one case the way the generator does: forward, then backward of sum(output * G), G being
// Det(g_*) with the padding queries' rows zeroed.
void RunCase(MultiHeadAttentionModule& attn, const std::vector<float>& keep, int64_t d_model, int64_t x_mul,
             int64_t x_add, int64_t x_div, int64_t g_mul, int64_t g_add, int64_t g_div,
             const std::vector<float>& expected_output, const std::vector<float>& expected_grad) {
    const int64_t N = 2, L = 5;
    DeviceBackend* be = attn.q_proj().weight().backend();
    attn.set_key_padding_mask(Tensor(Shape({N, L}), be, keep));
    Tensor x(Shape({N, L, d_model}), be, Det(N * L * d_model, x_mul, x_add, x_div));
    Tensor out = attn.forward(x);
    ExpectMatchesReference(out, expected_output);

    std::vector<float> g = Det(N * L * d_model, g_mul, g_add, g_div);
    for (int64_t t = 0; t < N * L; ++t) {
        for (int64_t c = 0; c < d_model; ++c) {
            g[static_cast<size_t>(t * d_model + c)] *= keep[static_cast<size_t>(t)];
        }
    }
    Tensor grad_in = attn.backward(Tensor(Shape({N, L, d_model}), be, g));
    ExpectMatchesReference(grad_in, expected_grad);
}

TEST(AttentionReferenceTest, MatchesHuggingFaceLlamaAttention) {
    CPUBackend backend;
    AttentionConfig c;
    c.d_model = 8;
    c.num_heads = 4;
    c.num_kv_heads = 2;
    c.head_dim = 4;
    c.rope_layout = RoPELayout::RotateHalf;
    c.rope_base = 100.0f;
    c.qkv_bias = false;
    c.out_bias = false;
    c.causal = true;
    MultiHeadAttentionModule attn(c, &backend);
    attn.q_proj().set_weight(Det(8 * 16, 5, 1, 7));
    attn.k_proj().set_weight(Det(8 * 8, 3, 2, 9));
    attn.v_proj().set_weight(Det(8 * 8, 7, 3, 8));
    attn.out_proj().set_weight(Det(16 * 8, 11, 4, 10));
    RunCase(attn, {1, 1, 1, 1, 1, 1, 1, 1, 0, 0}, 8, 5, 3, 4, 7, 2, 5, kLlamaOutput, kLlamaInputGrad);
}

TEST(AttentionReferenceTest, MatchesHuggingFaceQwen2AttentionWithLeftPaddingAndOffset) {
    CPUBackend backend;
    AttentionConfig c;
    c.d_model = 6;
    c.num_heads = 3;
    c.num_kv_heads = 1;
    c.head_dim = 4;
    c.rope_layout = RoPELayout::RotateHalf;
    c.rope_base = 1000.0f;
    c.qkv_bias = true;
    c.out_bias = false;
    c.causal = true;
    MultiHeadAttentionModule attn(c, &backend);
    attn.q_proj().set_weight(Det(6 * 12, 5, 2, 6));
    attn.q_proj().set_bias(Det(12, 3, 1, 5));
    attn.k_proj().set_weight(Det(6 * 4, 3, 1, 7));
    attn.k_proj().set_bias(Det(4, 3, 4, 6));
    attn.v_proj().set_weight(Det(6 * 4, 7, 5, 9));
    attn.v_proj().set_bias(Det(4, 3, 2, 7));
    attn.out_proj().set_weight(Det(12 * 6, 13, 3, 11));
    attn.set_position_offset(3);
    RunCase(attn, {0, 1, 1, 1, 1, 1, 1, 1, 1, 1}, 6, 3, 1, 4, 5, 4, 6, kQwen2Output, kQwen2InputGrad);
}

TEST(AttentionReferenceTest, MatchesHuggingFaceQwen3AttentionWithQKNorm) {
    CPUBackend backend;
    AttentionConfig c;
    c.d_model = 8;
    c.num_heads = 2;
    c.num_kv_heads = 1;
    c.head_dim = 6;
    c.rope_layout = RoPELayout::RotateHalf;
    c.rope_base = 10000.0f;
    c.use_qk_norm = true;
    c.qkv_bias = false;
    c.out_bias = false;
    c.causal = true;
    MultiHeadAttentionModule attn(c, &backend);
    attn.q_proj().set_weight(Det(8 * 12, 5, 1, 6));
    attn.k_proj().set_weight(Det(8 * 6, 3, 4, 7));
    attn.v_proj().set_weight(Det(8 * 6, 7, 2, 8));
    attn.out_proj().set_weight(Det(12 * 8, 11, 5, 9));
    std::vector<float> q_gamma = Det(6, 5, 1, 20);
    std::vector<float> k_gamma = Det(6, 7, 3, 20);
    for (float& v : q_gamma) v += 1.0f;
    for (float& v : k_gamma) v += 1.0f;
    attn.q_norm()->set_gamma(q_gamma);
    attn.k_norm()->set_gamma(k_gamma);
    RunCase(attn, std::vector<float>(10, 1.0f), 8, 7, 1, 4, 3, 5, 5, kQwen3Output, kQwen3InputGrad);
}

}  // namespace
}  // namespace pulsatrix
