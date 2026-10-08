// PLM-7: the masked-LM collator and cluster sampler, ESM initialization, a training step and three
// AdamW steps against transformers (tools/golden/make_esm_training_golden.py) on the tiny random
// ESM-2, evaluation, and the fine-tuning heads.
#include "pulsatrix/protein_training.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string TinyEsm() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm"; }

TEST(MaskedLMCollator, MasksResiduesOnlyAtEsmsRates) {
    CPUBackend cpu;
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    MaskedLMCollator collator(tok, &cpu, {}, 11);
    const std::string seq(400, 'L');
    const std::vector<std::string> batch(50, seq);
    const MaskedLMBatch b = collator.collate({batch.begin(), batch.end()});
    const std::vector<float> ids = b.ids.to_host_vector(), targets = b.targets.to_host_vector(), keep = b.keep.to_host_vector();
    const int64_t L = b.ids.shape().dim(1), mask = *tok.token_to_id("<mask>"), leu = *tok.token_to_id("L");
    int64_t picked = 0, masked = 0, kept = 0, swapped = 0;
    for (int64_t n = 0; n < 50; ++n) {
        EXPECT_EQ(targets[static_cast<size_t>(n * L)], -100.0f);          // <cls>
        EXPECT_EQ(targets[static_cast<size_t>(n * L + L - 1)], -100.0f);  // <eos>
        for (int64_t l = 1; l + 1 < L; ++l) {
            const size_t k = static_cast<size_t>(n * L + l);
            EXPECT_EQ(keep[k], 1.0f);
            if (targets[k] < 0) {
                EXPECT_EQ(ids[k], static_cast<float>(leu));  // unpicked residues are untouched
                continue;
            }
            EXPECT_EQ(targets[k], static_cast<float>(leu));
            ++picked;
            if (ids[k] == static_cast<float>(mask)) ++masked;
            else if (ids[k] == static_cast<float>(leu)) ++kept;
            else ++swapped;
        }
    }
    EXPECT_EQ(picked, b.num_targets);
    // 20,000 residues: 15% picked, then 80/10/10 (a swap to L itself counts as kept: 1 in 20).
    EXPECT_NEAR(picked / 20000.0, 0.15, 0.01);
    EXPECT_NEAR(masked / static_cast<double>(picked), 0.80, 0.03);
    EXPECT_NEAR(swapped / static_cast<double>(picked), 0.10 * 19 / 20, 0.02);
    EXPECT_NEAR(kept / static_cast<double>(picked), 0.10 + 0.10 / 20, 0.02);

    // The same seed gives the same batch.
    MaskedLMCollator again(tok, &cpu, {}, 11);
    EXPECT_EQ(again.collate({batch.begin(), batch.end()}).ids.to_host_vector(), ids);
}

TEST(MaskedLMCollator, CropsLongSequencesAndPadsShortOnes) {
    CPUBackend cpu;
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    MaskedLMOptions o;
    o.max_tokens = 12;
    o.mask_probability = 0.0;
    MaskedLMCollator collator(tok, &cpu, o, 5);
    const std::string long_seq = "ACDEFGHIKLMNPQRSTVWY";
    const MaskedLMBatch b = collator.collate({long_seq, "MK"});
    ASSERT_EQ(b.ids.shape(), Shape({2, 12}));
    EXPECT_EQ(b.num_targets, 0);
    const std::vector<float> ids = b.ids.to_host_vector(), keep = b.keep.to_host_vector();
    // A window of 10 consecutive residues.
    std::string window;
    for (int l = 1; l <= 10; ++l) window += tok.id_to_token(static_cast<int64_t>(ids[static_cast<size_t>(l)]))->front();
    EXPECT_NE(long_seq.find(window), std::string::npos) << window;
    EXPECT_EQ(ids[11], static_cast<float>(*tok.token_to_id("<eos>")));
    // "MK" is <cls> M K <eos>, then padding the model ignores.
    EXPECT_EQ(keep[12 + 3], 1.0f);
    EXPECT_EQ(keep[12 + 4], 0.0f);
    EXPECT_EQ(ids[12 + 4], static_cast<float>(*tok.token_to_id("<pad>")));

    EXPECT_THROW((void)collator.collate({}), std::invalid_argument);
    EXPECT_THROW((void)collator.collate({""}), std::invalid_argument);
    EXPECT_THROW((void)collator.collate({"MK1"}), std::invalid_argument);
    o.mask_fraction = 0.95;
    EXPECT_THROW(MaskedLMCollator(tok, &cpu, o), std::invalid_argument);
}

TEST(ClusterSampler, DrawsClustersUniformlyNotSequences) {
    std::vector<std::string> clusters(9, "big");
    clusters.push_back("small");
    ClusterSampler sampler(clusters, 2);
    EXPECT_EQ(sampler.num_clusters(), 2);
    int small = 0;
    for (int i = 0; i < 10000; ++i) small += sampler.next() == 9 ? 1 : 0;
    EXPECT_NEAR(small / 10000.0, 0.5, 0.02);  // not 0.1
    EXPECT_THROW(ClusterSampler({}), std::invalid_argument);
}

TEST(InitializeEsm, MatchesTransformersInitialization) {
    CPUBackend cpu;
    const EncoderLMConfig config = ReadEsmConfig(TinyEsm() + "/config.json");
    EncoderLM model(config, &cpu);
    InitializeEsm(model, 4);
    for (const NamedParamRef& p : model.named_parameters()) {
        const std::vector<float> v = p.ref.value->to_host_vector();
        if (p.ref.value->rank() >= 2) {
            double sq = 0;
            for (float x : v) sq += static_cast<double>(x) * x;
            EXPECT_NEAR(std::sqrt(sq / static_cast<double>(v.size())), 0.02, 0.004) << p.name;
        } else {
            const bool gain = p.name.find("norm") != std::string::npos && p.name.find("weight") != std::string::npos;
            for (float x : v) EXPECT_EQ(x, gain ? 1.0f : 0.0f) << p.name;
        }
    }
    // The <pad> row of the embeddings is zero.
    const std::vector<float> e = model.named_parameters().front().ref.value->to_host_vector();
    ASSERT_EQ(model.named_parameters().front().name, "embed_tokens.weight");
    for (int64_t j = 0; j < config.hidden_size; ++j) EXPECT_EQ(e[static_cast<size_t>(config.pad_token_id * config.hidden_size + j)], 0.0f);
}

/** @brief Our parameter (value or gradient) in transformers' layout. */
std::vector<float> AsTransformers(const Tensor& t, WeightTransform transform) {
    std::vector<float> v = t.to_host_vector();
    if (transform != WeightTransform::Transpose) return v;
    const int64_t rows = t.shape().dim(0), cols = t.shape().dim(1);
    std::vector<float> out(v.size());
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) out[static_cast<size_t>(c * rows + r)] = v[static_cast<size_t>(r * cols + c)];
    }
    return out;
}

double RelativeError(const std::vector<float>& a, const std::vector<float>& b) {
    double diff = 0, scale = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        diff = std::max(diff, std::abs(static_cast<double>(a[i]) - b[i]));
        scale = std::max(scale, std::abs(static_cast<double>(b[i])));
    }
    return scale > 0 ? diff / scale : diff;
}

TEST(MaskedLMTraining, StepAndAdamWMatchTransformers) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const SafetensorsFile golden = SafetensorsFile::Map(TinyEsm() + "/esm_training_golden.safetensors");
    const MaskedLMBatch batch{golden.tensor("ids", &cpu), golden.tensor("keep", &cpu), golden.tensor("targets", &cpu), 0};
    TokenCrossEntropyLoss loss(&cpu);
    std::map<std::string, ParamRef> ours;
    for (const NamedParamRef& p : model->named_parameters()) ours[p.name] = p.ref;

    const float value = MaskedLMForwardBackward(*model, batch, loss);  // gradients start at zero
    EXPECT_NEAR(value, golden.tensor("loss", &cpu).to_host_vector()[0], 1e-5);
    int compared = 0;
    for (const WeightMapping& m : EsmMapping(model->config())) {
        const std::string name = "grad." + m.source;
        if (!golden.contains(name)) continue;
        ASSERT_TRUE(ours.count(m.target)) << m.target;
        EXPECT_LT(RelativeError(AsTransformers(*ours[m.target].grad, m.transform), golden.tensor(name, &cpu).to_host_vector()), 1e-4)
            << m.source;
        ++compared;
    }
    EXPECT_EQ(compared, 40);  // every trained tensor (the decoder shares the embedding table)

    // Three AdamW steps, as torch.optim.AdamW takes them.
    std::unique_ptr<EncoderLM> trained = LoadEncoderLM(TinyEsm(), &cpu);
    AdamWOptimizer opt(1e-3f, &cpu, 0.01f, 0.9f, 0.98f, 1e-8f);
    const std::vector<float> step_losses = golden.tensor("step_losses", &cpu).to_host_vector();
    for (int s = 0; s < 3; ++s) {
        opt.zero_grad(*trained);
        EXPECT_NEAR(MaskedLMForwardBackward(*trained, batch, loss), step_losses[static_cast<size_t>(s)], 2e-5) << "step " << s;
        opt.step(*trained);
    }
    std::map<std::string, ParamRef> after;
    for (const NamedParamRef& p : trained->named_parameters()) after[p.name] = p.ref;
    for (const WeightMapping& m : EsmMapping(trained->config())) {
        const std::string name = "final." + m.source;
        if (!golden.contains(name)) continue;
        EXPECT_LT(RelativeError(AsTransformers(*after[m.target].value, m.transform), golden.tensor(name, &cpu).to_host_vector()), 1e-4)
            << m.source;
    }
}

TEST(MaskedLMTraining, EvaluationIsRepeatableAndPerplexityIsExpLoss) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    const std::vector<std::string> seqs = {"MKTAYIAKQRQISFVKSHFSRQ", "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQD", "GSHMLEDP"};
    const MaskedLMEvaluation a = EvaluateMaskedLM(*model, tok, &cpu, seqs, 2, 9), b = EvaluateMaskedLM(*model, tok, &cpu, seqs, 3, 9);
    EXPECT_GT(a.tokens, 0);
    EXPECT_EQ(a.tokens, b.tokens);
    EXPECT_NEAR(a.loss, b.loss, 1e-5);  // batching changes nothing
    EXPECT_NEAR(a.perplexity, std::exp(a.loss), 1e-9);
    EXPECT_GE(a.accuracy, 0.0);
    EXPECT_LE(a.accuracy, 1.0);
    EXPECT_TRUE(model->keep_activations());
}

/** @brief A head's squared-error loss on top of the tiny model, and its gradient checked against
 *         finite differences of one encoder weight and one head weight. */
void ExpectHeadGradientsMatchFiniteDifferences(SequenceHead::Pooling pooling) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    MaskedLMOptions none;
    none.mask_probability = 0.0;
    MaskedLMCollator collator(tok, &cpu, none);
    const MaskedLMBatch b = collator.collate({"MKTAYIAKQR", "MQIFVK"});
    SequenceHead head(model->config().hidden_size, 2, pooling, &cpu);
    const Tensor residues = ResidueMask(b.ids, &b.keep, tok, &cpu);
    head.set_residue_mask(residues);
    const int64_t N = 2, L = b.ids.shape().dim(1), C = 2;
    const int64_t outputs = pooling == SequenceHead::Pooling::Mean ? N * C : N * L * C;
    std::vector<float> target(static_cast<size_t>(outputs));
    for (size_t i = 0; i < target.size(); ++i) target[i] = static_cast<float>(i % 3) - 1.0f;
    auto loss_value = [&] {
        model->set_padding_mask(b.keep);
        (void)model->forward(b.ids);
        const std::vector<float> y = head.forward(model->last_hidden_state()).to_host_vector();
        double s = 0;
        for (size_t i = 0; i < y.size(); ++i) s += 0.5 * (y[i] - target[i]) * (y[i] - target[i]);
        return s;
    };
    // Analytic gradients (they start at zero).
    model->set_padding_mask(b.keep);
    (void)model->forward(b.ids);
    const Tensor y = head.forward(model->last_hidden_state());
    std::vector<float> dy = y.to_host_vector();
    for (size_t i = 0; i < dy.size(); ++i) dy[i] -= target[i];
    (void)model->backward_hidden(head.backward(Tensor(y.shape(), &cpu, dy)));

    auto check = [&](Tensor& value, Tensor& grad, size_t index, const std::string& what) {
        const std::vector<float> g = grad.to_host_vector();
        std::vector<float> v = value.to_host_vector();
        const float original = v[index], h = 1e-2f;
        v[index] = original + h;
        value = Tensor(value.shape(), &cpu, v);
        const double up = loss_value();
        v[index] = original - h;
        value = Tensor(value.shape(), &cpu, v);
        const double down = loss_value();
        v[index] = original;
        value = Tensor(value.shape(), &cpu, v);
        EXPECT_NEAR(g[index], (up - down) / (2 * h), 2e-3 * std::max(1.0, std::abs(static_cast<double>(g[index])))) << what;
    };
    for (const NamedParamRef& p : model->named_parameters()) {
        if (p.name == "layers.0.mlp.fc1.weight") check(*p.ref.value, *p.ref.grad, 7, p.name);
        if (p.name == "norm.weight") check(*p.ref.value, *p.ref.grad, 3, p.name);
    }
    for (const NamedParamRef& p : head.named_parameters()) check(*p.ref.value, *p.ref.grad, 1, p.name);
}

TEST(SequenceHead, MeanPooledGradientsMatchFiniteDifferences) { ExpectHeadGradientsMatchFiniteDifferences(SequenceHead::Pooling::Mean); }
TEST(SequenceHead, PerResidueGradientsMatchFiniteDifferences) { ExpectHeadGradientsMatchFiniteDifferences(SequenceHead::Pooling::PerResidue); }

TEST(SequenceHead, MeanPoolingAveragesResiduesOnlyAndShapesAreChecked) {
    CPUBackend cpu;
    SequenceHead head(2, 1, SequenceHead::Pooling::Mean, &cpu);
    head.linear().set_weight({1.0f, 0.0f});
    head.linear().set_bias({0.5f});
    // (1, 4, 2): <cls>, two residues, <eos>; only the residues count.
    head.set_residue_mask(Tensor(Shape({1, 4}), &cpu, {0, 1, 1, 0}));
    const Tensor y = head.forward(Tensor(Shape({1, 4, 2}), &cpu, {100, 0, 2, 0, 4, 0, -100, 0}));
    EXPECT_FLOAT_EQ(y.to_host_vector()[0], 3.5f);
    // Relevance: each residue's share of the pooled value (bias left out).
    LRPRuleConfig config{1e-9f};
    const std::vector<float> r = head.propagate_relevance(Tensor(Shape({1, 1}), &cpu, {3.0f}), config).to_host_vector();
    EXPECT_NEAR(r[2], 1.0f, 1e-5);
    EXPECT_NEAR(r[4], 2.0f, 1e-5);
    EXPECT_EQ(r[0], 0.0f);
    EXPECT_THROW((void)head.forward(Tensor(Shape({1, 3, 2}), &cpu, std::vector<float>(6, 1.0f))), std::invalid_argument);
    EXPECT_THROW((void)head.forward(Tensor(Shape({4, 2}), &cpu, std::vector<float>(8, 1.0f))), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
