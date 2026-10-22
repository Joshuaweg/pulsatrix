// DIFF-3: commitment-graph LRP. A hand-built linear model, whose relevance is exact, checks the
// bookkeeping to the last digit; the tiny MDLM checks that the real model's pass is linear in its
// seed (what one pass per step relies on); a model trained on a planted copy task checks that the
// relevance reaches the token that was copied.
#include "pulsatrix/masked_diffusion_lrp.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/protein_training.hpp"  // InitializeEsm, MaskedLMBatch, MaskedLMForwardBackward
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {
namespace {

// ---- a linear model with exact relevance ------------------------------------------------------

constexpr int64_t kV = 4, kMask = 3, kL = 5;

/**
 * @brief logits[p][v] = sum_q A[p][q] * E[token_q][v], every term positive. Relevance is LRP-0,
 *        each position's share of each logit, so it conserves exactly. Position 2 reads position
 *        0, position 3 reads position 2 (and a little of its own mask), position 4 reads position
 *        3 and half of position 1.
 */
class ChainModel final : public StepRelevanceModel {
public:
    explicit ChainModel(int* forwards = nullptr) : forwards_(forwards) {
        a_[2][0] = 1.0;
        a_[3][2] = 1.0;
        a_[3][3] = 0.5;
        a_[4][3] = 1.0;
        a_[4][1] = 0.5;
    }
    static double E(int64_t token, int64_t v) {
        if (token == kMask) return v == kMask ? 0.0 : 0.5;
        return (token == v ? 3.0 : 0.0) + 1.0;
    }
    std::vector<float> forward(const std::vector<int64_t>& tokens) override {
        if (forwards_ != nullptr) ++*forwards_;
        tokens_ = tokens;
        std::vector<float> z(static_cast<size_t>(kL * kV), 0.0f);
        for (int64_t p = 0; p < kL; ++p) {
            for (int64_t v = 0; v < kV; ++v) z[static_cast<size_t>(p * kV + v)] = static_cast<float>(Logit(p, v));
        }
        return z;
    }
    std::vector<float> relevance(const std::vector<float>& seed) override {
        std::vector<float> r(static_cast<size_t>(kL), 0.0f);
        for (int64_t p = 0; p < kL; ++p) {
            for (int64_t v = 0; v < kV; ++v) {
                const double s = seed[static_cast<size_t>(p * kV + v)];
                if (s == 0.0) continue;
                for (int64_t q = 0; q < kL; ++q) r[static_cast<size_t>(q)] += static_cast<float>(s * a_[p][q] * E(tokens_[q], v) / Logit(p, v));
            }
        }
        return r;
    }
    DenoiserFn denoiser() {
        return [this](const std::vector<int64_t>& t) { return forward(t); };
    }

private:
    double Logit(int64_t p, int64_t v) const {
        double z = 0;
        for (int64_t q = 0; q < kL; ++q) z += a_[p][q] * E(tokens_[static_cast<size_t>(q)], v);
        return z;
    }
    double a_[kL][kL] = {};
    std::vector<int64_t> tokens_;
    int* forwards_;
};

MaskedDiffusionConfig ChainConfig() {
    MaskedDiffusionConfig c;
    c.mask_token_id = kMask;
    return c;
}

/** @brief The prompt (tokens 2 and 1), then positions 2, 3, 4 written one per step in that order. */
MaskedDiffusionResult ChainTrajectory(ChainModel& model) {
    MaskedDiffusionResult order;
    order.tokens.assign(kL, kMask);
    for (int64_t p : {2, 3, 4}) order.steps.push_back({{}, {{p, 0, 0.0f}}});
    return ReplayMaskedDiffusion(model.denoiser(), order, {2, 1, kMask, kMask, kMask}, ChainConfig());
}

TEST(CommitmentLrp, TheChainModelWritesWhatItsWeightsSay) {
    ChainModel model;
    const MaskedDiffusionResult t = ChainTrajectory(model);
    // Position 2 copies token 2; 3 copies 2; 4 sees 3's token 2 at weight 1 and 1's token 1 at 0.5.
    EXPECT_EQ(t.tokens, (std::vector<int64_t>{2, 1, 2, 2, 2}));
    EXPECT_EQ(t.last_write_step, (std::vector<int64_t>{-1, -1, 0, 1, 2}));
}

TEST(CommitmentLrp, GraphCarriesRelevanceBackThroughEachCommitToThePrompt) {
    int forwards = 0;
    ChainModel model(&forwards);
    const MaskedDiffusionResult t = ChainTrajectory(model);
    forwards = 0;
    const CommitmentRelevance r = ExplainCommitments(model, t, kMask, {{4, std::nullopt}});
    EXPECT_EQ(forwards, 3);
    // Position 4's logit for token 2 is 4 (from position 3) + 0.5 (from position 1). Position 3's
    // 4 goes back to step 1, where position 3's logit was 4 from position 2 and 0.25 from its own
    // mask; position 2's share goes back to step 0 and lands on position 0.
    EXPECT_NEAR(r.value, 4.5, 1e-6);
    EXPECT_NEAR(r.prompt[0], 4.0 * 4.0 / 4.25, 1e-5);
    EXPECT_NEAR(r.prompt[1], 0.5, 1e-6);
    EXPECT_NEAR(r.mask[3], 4.0 * 0.25 / 4.25, 1e-6);
    EXPECT_NEAR(r.relayed[3], 4.0, 1e-6);
    EXPECT_NEAR(r.relayed[2], 4.0 * 4.0 / 4.25, 1e-5);
    EXPECT_NEAR(r.leak_total(), 0.0, 1e-5);
    EXPECT_NEAR(r.prompt_total() + r.mask_total() + r.leak_total(), r.value, 1e-9);
    ASSERT_EQ(r.hops.size(), 3u);
    EXPECT_EQ(r.hops[0].step, 2);
    EXPECT_EQ(r.hops[2].step, 0);
    EXPECT_NEAR(r.hops[1].seeded, 4.0, 1e-6);
    for (const auto& s : r.step_relevance) EXPECT_EQ(s.size(), static_cast<size_t>(kL));
}

TEST(CommitmentLrp, StepLocalStopsAtTheCommittedTokens) {
    ChainModel model;
    const MaskedDiffusionResult t = ChainTrajectory(model);
    const CommitmentRelevance r = ExplainCommitments(model, t, kMask, {{4, std::nullopt}}, CommitmentMode::StepLocal);
    ASSERT_EQ(r.hops.size(), 1u);
    EXPECT_NEAR(r.prompt[0], 0.0, 1e-9);
    EXPECT_NEAR(r.prompt[1], 0.5, 1e-6);
    EXPECT_NEAR(r.relayed[3], 4.0, 1e-6);
    EXPECT_NEAR(r.prompt_total() + r.relayed_total() + r.mask_total() + r.leak_total(), r.value, 1e-9);
    EXPECT_TRUE(r.step_relevance[0].empty());
}

TEST(CommitmentLrp, RunsOnlyTheStepsUpstreamOfTheTargets) {
    int forwards = 0;
    ChainModel model(&forwards);
    const MaskedDiffusionResult t = ChainTrajectory(model);
    forwards = 0;
    const CommitmentRelevance r = ExplainCommitments(model, t, kMask, {{2, std::nullopt}});
    EXPECT_EQ(forwards, 1);
    EXPECT_NEAR(r.prompt[0], 4.0, 1e-6);
}

TEST(CommitmentLrp, SeveralTargetsShareOnePassPerStepAndAddUp) {
    int forwards = 0;
    ChainModel model(&forwards);
    const MaskedDiffusionResult t = ChainTrajectory(model);
    forwards = 0;
    const CommitmentRelevance both = ExplainCommitments(model, t, kMask, {{4, std::nullopt}, {3, std::nullopt}});
    EXPECT_EQ(forwards, 3);
    const CommitmentRelevance a = ExplainCommitments(model, t, kMask, {{4, std::nullopt}});
    const CommitmentRelevance b = ExplainCommitments(model, t, kMask, {{3, std::nullopt}});
    EXPECT_NEAR(both.value, a.value + b.value, 1e-6);
    for (int64_t p = 0; p < kL; ++p) {
        EXPECT_NEAR(both.prompt[p], a.prompt[p] + b.prompt[p], 1e-5);
        EXPECT_NEAR(both.mask[p], a.mask[p] + b.mask[p], 1e-5);
    }
}

TEST(CommitmentLrp, AFixedTokenExplainsThatTokensLogit) {
    ChainModel model;
    const MaskedDiffusionResult t = ChainTrajectory(model);
    // Position 4's logit for token 1: 1 from position 3 (token 2) and 0.5 * 4 from position 1.
    const CommitmentRelevance r = ExplainCommitments(model, t, kMask, {{4, int64_t{1}}}, CommitmentMode::StepLocal);
    EXPECT_NEAR(r.value, 3.0, 1e-6);
    EXPECT_NEAR(r.prompt[1], 2.0, 1e-6);
    EXPECT_NEAR(r.relayed[3], 1.0, 1e-6);
}

TEST(CommitmentLrp, ReplayReproducesTheTrajectoryAndItsValue) {
    ChainModel model;
    const MaskedDiffusionResult t = ChainTrajectory(model);
    const MaskedDiffusionResult again = ReplayMaskedDiffusion(model.denoiser(), t, t.steps[0].input, ChainConfig());
    EXPECT_EQ(again.tokens, t.tokens);
    EXPECT_EQ(again.last_write_step, t.last_write_step);
    EXPECT_NEAR(ReplayCommitmentValue(model.denoiser(), t, t.steps[0].input, ChainConfig(), {{4, std::nullopt}}),
                ExplainCommitments(model, t, kMask, {{4, std::nullopt}}).value, 1e-6);
    // A skipped position stays masked.
    const MaskedDiffusionResult skipped = ReplayMaskedDiffusion(model.denoiser(), t, t.steps[0].input, ChainConfig(), {3});
    EXPECT_EQ(skipped.tokens[3], kMask);
    EXPECT_EQ(skipped.last_write_step[3], -1);
    EXPECT_EQ(skipped.steps.size(), t.steps.size());
}

TEST(CommitmentLrp, PatchingLeavesEachIntermediateMaskedAndReplays) {
    ChainModel model;
    const MaskedDiffusionResult t = ChainTrajectory(model);
    const std::vector<CommitmentTarget> target = {{4, std::nullopt}};
    const CommitmentRelevance graph = ExplainCommitments(model, t, kMask, target);
    const CommitmentPatchResult r = CommitmentPatchCheck(model.denoiser(), t, ChainConfig(), target, graph);
    ASSERT_EQ(r.positions, (std::vector<int64_t>{2, 3}));
    // With position 3 masked, position 4's logit for token 2 is 0.5 + 0.5: down 3.5. With position
    // 2 masked, position 3 sees only masks, ties and takes token 0, and position 4's logit for
    // token 2 is 1 + 0.5: down 3.
    EXPECT_NEAR(r.effect[1], 3.5, 1e-6);
    EXPECT_NEAR(r.effect[0], 3.0, 1e-6);
    EXPECT_NEAR(r.relayed[1], 4.0, 1e-6);
    EXPECT_EQ(r.spearman, 0.0);  // too few positions
    EXPECT_THROW((void)CommitmentPatchCheck(model.denoiser(), t, ChainConfig(), target,
                                            ExplainCommitments(model, t, kMask, target, CommitmentMode::StepLocal)),
                 std::invalid_argument);
}

TEST(CommitmentLrp, DeletionRemovesTheMostRelevantPromptTokenFirst) {
    ChainModel model;
    const MaskedDiffusionResult t = ChainTrajectory(model);
    const std::vector<CommitmentTarget> target = {{4, std::nullopt}};
    const CommitmentRelevance graph = ExplainCommitments(model, t, kMask, target);
    const CommitmentDeletionCurve d = CommitmentDeletionCheck(model.denoiser(), t, ChainConfig(), target, graph.prompt, 2, 4, 1);
    ASSERT_EQ(d.scores.size(), 3u);
    EXPECT_NEAR(d.scores[0], 4.5, 1e-6);
    // Deleting position 0 first: position 2 sees only masks and takes token 0 (the tie's first),
    // which the chain copies, so position 4's logit for token 2 falls to 1 + 0.5. Deleting position
    // 1 first would leave 4 + 0.25.
    EXPECT_NEAR(d.scores[1], 1.5, 1e-6);
    EXPECT_LT(d.auc, d.random_auc);
}

TEST(CommitmentLrp, RejectsBadTargets) {
    ChainModel model;
    const MaskedDiffusionResult t = ChainTrajectory(model);
    EXPECT_THROW((void)ExplainCommitments(model, t, kMask, {}), std::invalid_argument);
    EXPECT_THROW((void)ExplainCommitments(model, t, kMask, {{0, std::nullopt}}), std::invalid_argument);  // prompt
    EXPECT_THROW((void)ExplainCommitments(model, t, kMask, {{7, std::nullopt}}), std::invalid_argument);
    EXPECT_THROW((void)ExplainCommitments(model, t, kMask, {{4, std::nullopt}, {4, std::nullopt}}), std::invalid_argument);
    EXPECT_THROW((void)ExplainCommitments(model, t, kMask, {{4, int64_t{9}}}), std::invalid_argument);
    EXPECT_THROW((void)ExplainCommitments(model, t, -1, {{4, std::nullopt}}), std::invalid_argument);
    EXPECT_THROW((void)ReplayMaskedDiffusion(model.denoiser(), t, {0, 1}, ChainConfig()), std::invalid_argument);
}

// ---- the tiny MDLM ------------------------------------------------------------------------------

TEST(CommitmentLrp, OnTheTinyMdlmOnePassPerStepEqualsExplainingEachTargetAlone) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/mdlm", &cpu);
    const int64_t mask = model->config().mask_token_id;
    std::vector<int64_t> input(10, mask);
    input[0] = 3;
    input[1] = 11;
    input[2] = 5;
    MaskedDiffusionConfig config;
    config.steps = 7;
    config.mask_token_id = mask;
    const MaskedDiffusionResult t = SampleMaskedDiffusion(MakeDenoiser(*model, &cpu), input, config);
    ASSERT_EQ(t.steps.size(), 7u);

    EncoderLMStepModel step(*model, &cpu);
    const std::vector<CommitmentTarget> targets = {{8, std::nullopt}, {9, std::nullopt}, {5, std::nullopt}};
    const CommitmentRelevance all = ExplainCommitments(step, t, mask, targets);
    EXPECT_NEAR(all.value, ReplayCommitmentValue(MakeDenoiser(*model, &cpu), t, input, config, targets), 1e-4);
    double scale = 0;
    for (float v : all.prompt) scale += std::abs(v);
    ASSERT_GT(scale, 0.0);
    std::vector<double> prompt(10, 0.0), mask_sink(10, 0.0);
    for (const CommitmentTarget& target : targets) {
        const CommitmentRelevance one = ExplainCommitments(step, t, mask, {target});
        for (size_t p = 0; p < 10; ++p) {
            prompt[p] += one.prompt[p];
            mask_sink[p] += one.mask[p];
        }
    }
    for (size_t p = 0; p < 10; ++p) {
        EXPECT_NEAR(all.prompt[p], prompt[p], 1e-4 * scale) << p;
        EXPECT_NEAR(all.mask[p], mask_sink[p], 1e-4 * scale) << p;
        EXPECT_TRUE(std::isfinite(all.relayed[p]));
    }
    EXPECT_NEAR(all.prompt_total() + all.mask_total() + all.leak_total(), all.value, 1e-6 * std::abs(all.value) + 1e-6);
    // With one step there is nothing to relay, and the two modes agree.
    MaskedDiffusionConfig once = config;
    once.steps = 1;
    const MaskedDiffusionResult one_step = SampleMaskedDiffusion(MakeDenoiser(*model, &cpu), input, once);
    const CommitmentRelevance g = ExplainCommitments(step, one_step, mask, {{6, std::nullopt}});
    const CommitmentRelevance l = ExplainCommitments(step, one_step, mask, {{6, std::nullopt}}, CommitmentMode::StepLocal);
    EXPECT_EQ(g.prompt, l.prompt);
    EXPECT_EQ(g.mask, l.mask);
}

// ---- a planted copy task --------------------------------------------------------------------

/**
 * @brief Trains a two-layer masked diffusion LM on `x0 x1 x2 x3 y0 y1` with `y0 = x1` and
 *        `y1 = x1 + 1 (mod 6)`, the prompt always given and each y masked as the masked-diffusion
 *        objective masks it. Returns the share of masked y's predicted correctly afterwards.
 */
double TrainCopyTask(EncoderLM& model, CPUBackend& cpu) {
    constexpr int64_t kContent = 6, kMaskId = 6, kLen = 6, kBatch = 64;
    InitializeEsm(model, 3);
    AdamOptimizer adam(3e-3f, &cpu);
    TokenCrossEntropyLoss loss(&cpu);
    std::mt19937_64 rng(5);
    const auto batch = [&]() {
        std::vector<float> ids, targets;
        for (int64_t n = 0; n < kBatch; ++n) {
            std::vector<int64_t> s(kLen);
            for (int64_t p = 0; p < 4; ++p) s[p] = static_cast<int64_t>(rng() % kContent);
            s[4] = s[1];
            s[5] = (s[1] + 1) % kContent;
            // Mask each y with probability t, at least one.
            const double t = 0.1 + 0.9 * static_cast<double>(rng() >> 11) * 0x1.0p-53;
            bool m4 = static_cast<double>(rng() >> 11) * 0x1.0p-53 < t, m5 = static_cast<double>(rng() >> 11) * 0x1.0p-53 < t;
            if (!m4 && !m5) (rng() % 2 ? m4 : m5) = true;
            for (int64_t p = 0; p < kLen; ++p) {
                const bool masked = (p == 4 && m4) || (p == 5 && m5);
                ids.push_back(static_cast<float>(masked ? kMaskId : s[p]));
                targets.push_back(masked ? static_cast<float>(s[p]) : -100.0f);
            }
        }
        const int64_t count = std::count_if(targets.begin(), targets.end(), [](float v) { return v >= 0; });
        return MaskedLMBatch{Tensor(Shape({kBatch, kLen}), &cpu, ids), Tensor(Shape({kBatch, kLen}), &cpu, std::vector<float>(ids.size(), 1.0f)),
                             Tensor(Shape({kBatch, kLen}), &cpu, targets), count};
    };
    for (int step = 0; step < 400; ++step) {
        const MaskedLMBatch b = batch();
        adam.zero_grad(model);
        (void)MaskedLMForwardBackward(model, b, loss);
        adam.step(model);
    }
    const MaskedLMBatch b = batch();
    model.set_padding_mask(b.keep);
    const std::vector<float> logits = model.forward(b.ids).to_host_vector(), targets = b.targets.to_host_vector();
    model.clear_padding_mask();
    int64_t right = 0, total = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
        if (targets[i] < 0) continue;
        const float* row = logits.data() + i * 7;
        right += std::max_element(row, row + kContent) - row == static_cast<int64_t>(targets[i]);
        ++total;
    }
    return static_cast<double>(right) / static_cast<double>(total);
}

TEST(CommitmentLrp, OnAPlantedCopyTaskRelevanceReachesTheCopiedToken) {
    CPUBackend cpu;
    EncoderLMConfig c;
    c.architecture = "MDLM";
    c.model_type = "mdlm";
    c.vocab_size = 7;
    c.hidden_size = 32;
    c.intermediate_size = 64;
    c.num_hidden_layers = 2;
    c.num_attention_heads = 4;
    c.layer_norm_eps = 1e-5f;
    c.max_position_embeddings = 16;
    c.mask_token_id = 6;
    c.head = EncoderLMHead::Linear;
    c.attention_bias = false;
    c.activation = ElementwiseOp::GeluTanh;
    EncoderLM model(c, &cpu);
    ASSERT_GT(TrainCopyTask(model, cpu), 0.98);

    MaskedDiffusionConfig config;
    config.steps = 2;
    config.mask_token_id = 6;
    EncoderLMStepModel step(model, &cpu);
    const DenoiserFn denoise = MakeDenoiser(model, &cpu);
    int checked = 0;
    for (const std::vector<int64_t>& prompt : {std::vector<int64_t>{0, 3, 5, 1}, {4, 2, 2, 0}, {1, 5, 0, 3}}) {
        std::vector<int64_t> input = prompt;
        input.push_back(6);
        input.push_back(6);
        const MaskedDiffusionResult t = SampleMaskedDiffusion(denoise, input, config);
        ASSERT_EQ(t.tokens[4], prompt[1]);
        ASSERT_EQ(t.tokens[5], (prompt[1] + 1) % 6);
        for (int64_t target : {4, 5}) {
            const CommitmentRelevance r = ExplainCommitments(step, t, 6, {{target, std::nullopt}});
            double others = 0;
            for (int64_t p : {0, 2, 3}) others = std::max(others, static_cast<double>(std::abs(r.prompt[p])));
            EXPECT_GT(r.prompt[1], 2.0 * others) << "prompt " << prompt[0] << prompt[1] << prompt[2] << prompt[3] << ", target " << target;
            // What the step-local explanation leaves on the other y, committed first, the graph
            // carries on, and x1 gains.
            const CommitmentRelevance local = ExplainCommitments(step, t, 6, {{target, std::nullopt}}, CommitmentMode::StepLocal);
            if (local.relayed_total() > 0.1) EXPECT_GT(r.prompt[1], local.prompt[1] + 0.1);
            ++checked;
        }
        // Deleting by relevance destroys the value faster than at random.
        const std::vector<CommitmentTarget> both = {{4, std::nullopt}, {5, std::nullopt}};
        const CommitmentRelevance r = ExplainCommitments(step, t, 6, both);
        const CommitmentDeletionCurve d = CommitmentDeletionCheck(denoise, t, config, both, r.prompt, 4, 4, 2);
        EXPECT_LT(d.auc, d.random_auc);
    }
    EXPECT_EQ(checked, 6);
}

}  // namespace
}  // namespace pulsatrix
