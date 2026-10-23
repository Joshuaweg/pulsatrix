// DIFF-2: the masked diffusion sampler. Most tests use a hand-built denoiser, so which position is
// filled when and with what is known exactly; the last runs the tiny MDLM end to end.
#include "pulsatrix/masked_diffusion.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"

namespace pulsatrix {
namespace {

constexpr int64_t kVocab = 8, kMask = 7;

/**
 * @brief A denoiser that ignores its input: position p prefers token p % 5 + 1, more strongly the
 *        earlier p is, and always prefers the mask token even more (which the sampler must never
 *        generate). Counts its calls.
 */
DenoiserFn FixedDenoiser(int* calls = nullptr) {
    return [calls](const std::vector<int64_t>& tokens) {
        if (calls != nullptr) ++*calls;
        std::vector<float> logits(tokens.size() * kVocab, 0.0f);
        for (size_t p = 0; p < tokens.size(); ++p) {
            logits[p * kVocab + (p % 5 + 1)] = 5.0f - 0.25f * static_cast<float>(p);
            logits[p * kVocab + kMask] = 20.0f;
        }
        return logits;
    };
}

std::vector<int64_t> Masked(int64_t length, int64_t prompt) {
    std::vector<int64_t> t(static_cast<size_t>(length), kMask);
    for (int64_t p = 0; p < prompt; ++p) t[static_cast<size_t>(p)] = 0;
    return t;
}

void ExpectConsistentTrajectory(const MaskedDiffusionResult& r, const std::vector<int64_t>& input) {
    std::vector<int64_t> seq = input;
    for (size_t s = 0; s < r.steps.size(); ++s) {
        EXPECT_EQ(r.steps[s].input, seq) << "step " << s;
        EXPECT_FALSE(r.steps[s].writes.empty());
        for (const MaskedDiffusionWrite& w : r.steps[s].writes) {
            EXPECT_EQ(seq[static_cast<size_t>(w.position)], kMask) << "a step wrote a position that wasn't masked";
            seq[static_cast<size_t>(w.position)] = w.token;
            EXPECT_EQ(r.last_write_step[static_cast<size_t>(w.position)], static_cast<int64_t>(s));
        }
    }
    EXPECT_EQ(seq, r.tokens);
    for (size_t p = 0; p < input.size(); ++p) {
        if (input[p] != kMask) EXPECT_EQ(r.last_write_step[p], -1);
        EXPECT_NE(r.tokens[p], kMask);
    }
}

TEST(MaskedDiffusion, ConfidenceFillsTheSurestPositionsFirstInEvenShares) {
    const std::vector<int64_t> input = Masked(12, 2);  // 10 masked positions
    MaskedDiffusionConfig config;
    config.steps = 4;
    config.mask_token_id = kMask;
    const MaskedDiffusionResult r = SampleMaskedDiffusion(FixedDenoiser(), input, config);
    ASSERT_EQ(r.steps.size(), 4u);
    // 10 over 4 steps: the first two take the remainder.
    const std::vector<size_t> counts = {3, 3, 2, 2};
    int64_t next = 2;  // confidence falls with position, so positions fill left to right
    for (size_t s = 0; s < 4; ++s) {
        ASSERT_EQ(r.steps[s].writes.size(), counts[s]);
        for (const MaskedDiffusionWrite& w : r.steps[s].writes) {
            EXPECT_EQ(w.position, next++);
            EXPECT_EQ(w.token, w.position % 5 + 1);
        }
    }
    ExpectConsistentTrajectory(r, input);
}

TEST(MaskedDiffusion, RecordsTheModelsOwnProbabilityOfEachToken) {
    MaskedDiffusionConfig config;
    config.steps = 1;
    config.mask_token_id = kMask;
    const MaskedDiffusionResult r = SampleMaskedDiffusion(FixedDenoiser(), Masked(3, 0), config);
    // Position 0: logit 5 for token 1, 20 for the mask, 0 for the other six.
    const double want = std::exp(5.0) / (std::exp(5.0) + std::exp(20.0) + 6.0);
    EXPECT_NEAR(r.steps[0].writes[0].probability, want, 1e-9);
}

TEST(MaskedDiffusion, MoreStepsThanMaskedPositionsSkipsEmptySteps) {
    int calls = 0;
    MaskedDiffusionConfig config;
    config.steps = 10;
    config.mask_token_id = kMask;
    const MaskedDiffusionResult r = SampleMaskedDiffusion(FixedDenoiser(&calls), Masked(5, 2), config);
    EXPECT_EQ(r.steps.size(), 3u);
    EXPECT_EQ(calls, 3);
}

TEST(MaskedDiffusion, NeverGeneratesSuppressedTokens) {
    MaskedDiffusionConfig config;
    config.steps = 2;
    config.mask_token_id = kMask;
    config.suppressed_token_ids = {1, 2};
    const MaskedDiffusionResult r = SampleMaskedDiffusion(FixedDenoiser(), Masked(6, 0), config);
    for (int64_t t : r.tokens) {
        EXPECT_NE(t, 1);
        EXPECT_NE(t, 2);
        EXPECT_NE(t, kMask);
    }
}

TEST(MaskedDiffusion, AncestralIsReproducibleRunsTheModelOnlyWhenSomethingChangesAndFinishesEverything) {
    MaskedDiffusionConfig config;
    config.steps = 16;
    config.mask_token_id = kMask;
    config.schedule = UnmaskingSchedule::Ancestral;
    config.seed = 7;
    int calls = 0;
    const std::vector<int64_t> input = Masked(20, 4);
    const MaskedDiffusionResult a = SampleMaskedDiffusion(FixedDenoiser(&calls), input, config);
    const MaskedDiffusionResult b = SampleMaskedDiffusion(FixedDenoiser(), input, config);
    EXPECT_EQ(calls, static_cast<int>(a.steps.size()));
    EXPECT_LE(a.steps.size(), 16u);
    ExpectConsistentTrajectory(a, input);
    ASSERT_EQ(a.steps.size(), b.steps.size());
    for (size_t s = 0; s < a.steps.size(); ++s) {
        ASSERT_EQ(a.steps[s].writes.size(), b.steps[s].writes.size());
        for (size_t w = 0; w < a.steps[s].writes.size(); ++w) EXPECT_EQ(a.steps[s].writes[w].position, b.steps[s].writes[w].position);
    }
    // Greedy: every position ends with its preferred token, whatever the order.
    for (int64_t p = 4; p < 20; ++p) EXPECT_EQ(a.tokens[static_cast<size_t>(p)], p % 5 + 1);
    // Another seed fills in another order.
    config.seed = 8;
    const MaskedDiffusionResult c = SampleMaskedDiffusion(FixedDenoiser(), input, config);
    EXPECT_NE(a.last_write_step, c.last_write_step);
}

TEST(MaskedDiffusion, SamplingIsSeededAndColdSamplingIsGreedy) {
    MaskedDiffusionConfig config;
    config.steps = 3;
    config.mask_token_id = kMask;
    config.do_sample = true;
    config.seed = 3;
    const std::vector<int64_t> input = Masked(9, 0);
    EXPECT_EQ(SampleMaskedDiffusion(FixedDenoiser(), input, config).tokens, SampleMaskedDiffusion(FixedDenoiser(), input, config).tokens);
    config.temperature = 1e-3f;
    const MaskedDiffusionResult cold = SampleMaskedDiffusion(FixedDenoiser(), input, config);
    for (int64_t p = 0; p < 9; ++p) EXPECT_EQ(cold.tokens[static_cast<size_t>(p)], p % 5 + 1);
}

TEST(MaskedDiffusion, RejectsBadInput) {
    MaskedDiffusionConfig config;
    config.mask_token_id = kMask;
    EXPECT_THROW((void)SampleMaskedDiffusion(FixedDenoiser(), Masked(4, 4), config), std::invalid_argument);  // nothing masked
    MaskedDiffusionConfig zero = config;
    zero.steps = 0;
    EXPECT_THROW((void)SampleMaskedDiffusion(FixedDenoiser(), Masked(4, 0), zero), std::invalid_argument);
    MaskedDiffusionConfig unset;
    EXPECT_THROW((void)SampleMaskedDiffusion(FixedDenoiser(), Masked(4, 0), unset), std::invalid_argument);
    EXPECT_THROW((void)SampleMaskedDiffusion([](const std::vector<int64_t>&) { return std::vector<float>(3, 0.0f); }, Masked(4, 0), config),
                 std::invalid_argument);
    EXPECT_THROW((void)SampleMaskedDiffusion(
                     [](const std::vector<int64_t>& t) {
                         std::vector<float> l(t.size() * kVocab, 0.0f);
                         l[1] = std::numeric_limits<float>::quiet_NaN();
                         return l;
                     },
                     Masked(4, 0), config),
                 std::invalid_argument);
    MaskedDiffusionConfig all = config;
    all.suppressed_token_ids = {0, 1, 2, 3, 4, 5, 6};
    EXPECT_THROW((void)SampleMaskedDiffusion(FixedDenoiser(), Masked(4, 0), all), std::invalid_argument);
}

/**
 * @brief Four positions that all start masked (no prompt). Position 0 prefers token 1 strongly
 *        while position 1 is masked, and token 2 weakly once it isn't; positions 1, 2 and 3 prefer
 *        tokens 3, 4 and 5 with falling confidence. The mask token's logit is highest everywhere.
 */
DenoiserFn ContextDenoiser(int* calls = nullptr) {
    return [calls](const std::vector<int64_t>& tokens) {
        if (calls != nullptr) ++*calls;
        std::vector<float> logits(tokens.size() * kVocab, 0.0f);
        const bool one_masked = tokens[1] == kMask;
        logits[0 * kVocab + (one_masked ? 1 : 2)] = one_masked ? 4.0f : 0.5f;
        logits[1 * kVocab + 3] = 3.0f;
        logits[2 * kVocab + 4] = 2.0f;
        logits[3 * kVocab + 5] = 1.0f;
        for (size_t p = 0; p < tokens.size(); ++p) logits[p * kVocab + kMask] = 20.0f;
        return logits;
    };
}

/** @brief Each step's input is the previous one with its writes and re-masks applied. */
void ExpectConsistentRemaskingTrajectory(const MaskedDiffusionResult& r, const std::vector<int64_t>& input) {
    std::vector<int64_t> seq = input;
    for (size_t s = 0; s < r.steps.size(); ++s) {
        EXPECT_EQ(r.steps[s].input, seq) << "step " << s;
        for (const MaskedDiffusionWrite& w : r.steps[s].writes) {
            EXPECT_EQ(seq[static_cast<size_t>(w.position)], kMask) << "a step wrote a position that wasn't masked";
            seq[static_cast<size_t>(w.position)] = w.token;
        }
        for (int64_t p : r.steps[s].remasked) {
            EXPECT_NE(r.steps[s].input[static_cast<size_t>(p)], kMask) << "re-masked a position that was masked already";
            seq[static_cast<size_t>(p)] = kMask;
        }
    }
    EXPECT_EQ(seq, r.tokens);
}

TEST(MaskedDiffusion, RemaskingKeepsTheSurestShareAndTakesBackAPositionThatLostConfidence) {
    int calls = 0;
    MaskedDiffusionConfig config;
    config.steps = 4;
    config.mask_token_id = kMask;
    config.schedule = UnmaskingSchedule::Remasking;
    const std::vector<int64_t> input = Masked(4, 0);
    const MaskedDiffusionResult r = SampleMaskedDiffusion(ContextDenoiser(&calls), input, config);
    // Every step runs the model, and step t of 4 masks floor(4 * (1 - t / 4)) positions: 3, 2, 1, 0.
    EXPECT_EQ(calls, 4);
    ASSERT_EQ(r.steps.size(), 4u);
    const auto positions = [](const MaskedDiffusionStep& s) {
        std::vector<int64_t> p;
        for (const MaskedDiffusionWrite& w : s.writes) p.push_back(w.position);
        return p;
    };
    // Step 1: position 0 is surest (4 while position 1 is masked). Step 2: positions 0 and 1 are
    // the surest; position 0 keeps its token and position 1 is filled.
    EXPECT_EQ(positions(r.steps[0]), (std::vector<int64_t>{0}));
    EXPECT_EQ(positions(r.steps[1]), (std::vector<int64_t>{1}));
    // Step 3: with position 1 filled, the model's confidence at position 0 drops to 0.5, the
    // lowest, and it is masked again while positions 2 and 3 are filled.
    EXPECT_EQ(positions(r.steps[2]), (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(r.steps[2].remasked, (std::vector<int64_t>{0}));
    // Step 4 masks nothing: position 0 takes the token it now prefers.
    EXPECT_EQ(positions(r.steps[3]), (std::vector<int64_t>{0}));
    EXPECT_EQ(r.steps[3].writes[0].token, 2);
    EXPECT_TRUE(r.steps[3].remasked.empty());
    EXPECT_EQ(r.tokens, (std::vector<int64_t>{2, 3, 4, 5}));
    EXPECT_EQ(r.last_write_step, (std::vector<int64_t>{3, 1, 2, 2}));
    ExpectConsistentRemaskingTrajectory(r, input);
}

TEST(MaskedDiffusion, RemaskingMasksOnlyScoresStrictlyBelowTheCutoffAsDplmDoes) {
    // Two positions with the same confidence and one step to go of two: the cutoff is the lower
    // score, and nothing is strictly below it, so both are kept.
    MaskedDiffusionConfig config;
    config.steps = 2;
    config.mask_token_id = kMask;
    config.schedule = UnmaskingSchedule::Remasking;
    const DenoiserFn tied = [](const std::vector<int64_t>& tokens) {
        std::vector<float> logits(tokens.size() * kVocab, 0.0f);
        for (size_t p = 0; p < tokens.size(); ++p) logits[p * kVocab + 2] = 3.0f;
        return logits;
    };
    const MaskedDiffusionResult r = SampleMaskedDiffusion(tied, Masked(3, 1), config);
    ASSERT_EQ(r.steps.size(), 2u);  // the second step changes nothing but is recorded
    EXPECT_EQ(r.steps[0].writes.size(), 2u);
    EXPECT_TRUE(r.steps[1].writes.empty());
    EXPECT_EQ(r.last_write_step, (std::vector<int64_t>{-1, 0, 0}));
}

TEST(MaskedDiffusion, RemaskingRanksByTheProbabilityOverTheAllowedTokens) {
    // Position 0: token 1 at logit 2, token 2 suppressed at logit 10. Position 1: token 3 at 3,
    // token 4 at 2.9. By raw probability position 1 is surer (token 2 takes most of position 0's
    // mass); over the allowed tokens position 0 is, as in DPLM, which sets suppressed logits to
    // -inf first.
    MaskedDiffusionConfig config;
    config.steps = 2;
    config.mask_token_id = kMask;
    config.schedule = UnmaskingSchedule::Remasking;
    config.suppressed_token_ids = {2};
    const DenoiserFn model = [](const std::vector<int64_t>& tokens) {
        std::vector<float> logits(tokens.size() * kVocab, 0.0f);
        logits[0 * kVocab + 1] = 2.0f;
        logits[0 * kVocab + 2] = 10.0f;
        logits[1 * kVocab + 3] = 3.0f;
        logits[1 * kVocab + 4] = 2.9f;
        return logits;
    };
    const MaskedDiffusionResult r = SampleMaskedDiffusion(model, Masked(2, 0), config);
    ASSERT_EQ(r.steps[0].writes.size(), 1u);
    EXPECT_EQ(r.steps[0].writes[0].position, 0);
    EXPECT_LT(r.steps[0].writes[0].probability, 0.01f);  // the recorded probability is still the raw one
}

TEST(MaskedDiffusion, SamplesFromTheTinyMdlmAndEachStepReproducesTheModelsOutput) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/mdlm", &cpu);
    const int64_t V = model->config().vocab_size, mask = model->config().mask_token_id;
    std::vector<int64_t> input(10, mask);
    input[0] = 3;
    input[1] = 11;
    for (UnmaskingSchedule schedule : {UnmaskingSchedule::Confidence, UnmaskingSchedule::Ancestral, UnmaskingSchedule::Remasking}) {
        MaskedDiffusionConfig config;
        config.steps = 4;
        config.mask_token_id = mask;
        config.schedule = schedule;
        config.seed = 1;
        const DenoiserFn denoise = MakeDenoiser(*model, &cpu);
        const MaskedDiffusionResult r = SampleMaskedDiffusion(denoise, input, config);
        for (int64_t t : r.tokens) EXPECT_NE(t, mask);
        EXPECT_EQ(r.last_write_step[0], -1);
        // Rerunning the model on a step's recorded input gives back the probabilities it recorded:
        // what DIFF-3 relies on to explain a step after the fact.
        for (const MaskedDiffusionStep& step : r.steps) {
            const std::vector<float> logits = denoise(step.input);
            for (const MaskedDiffusionWrite& w : step.writes) {
                const float* row = logits.data() + w.position * V;
                double top = row[0], total = 0;
                for (int64_t v = 1; v < V; ++v) top = std::max(top, static_cast<double>(row[v]));
                for (int64_t v = 0; v < V; ++v) total += std::exp(row[v] - top);
                EXPECT_NEAR(w.probability, std::exp(row[w.token] - top) / total, 1e-6);
            }
        }
    }
}

}  // namespace
}  // namespace pulsatrix
