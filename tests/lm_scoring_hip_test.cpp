// LLM-10 on the GPU: scoring on the HIP backend, which gathers the logit rows on the device,
// gives the CPU's scores, and a collated batch gives the CPU's training loss.
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hip_backend.hpp"
#include "pulsatrix/lm_scoring.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {
namespace {

std::string TinyLlama() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/llama"; }

const std::vector<PromptCompletion> kExamples = {
    {{1, 5, 9}, {12}}, {{3, 4, 5, 6, 7, 8}, {20, 21}}, {{2}, {30, 31, 32}}, {{11, 13, 17, 19}, {40}}};

TEST(LmScoringHipTest, CompletionLogProbsMatchTheCpu) {
    CPUBackend cpu;
    HIPBackend hip;
    auto on_cpu = LoadCausalLM(TinyLlama(), &cpu);
    auto on_hip = LoadCausalLM(TinyLlama(), &hip);
    const std::vector<double> want = CompletionLogProbs(*on_cpu, kExamples, &cpu, 3);
    const std::vector<double> got = CompletionLogProbs(*on_hip, kExamples, &hip, 3);
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < want.size(); ++i) EXPECT_NEAR(got[i], want[i], 1e-4) << "example " << i;
}

TEST(LmScoringHipTest, ScoreChoicesMatchTheCpu) {
    CPUBackend cpu;
    HIPBackend hip;
    auto on_cpu = LoadCausalLM(TinyLlama(), &cpu);
    auto on_hip = LoadCausalLM(TinyLlama(), &hip);
    const std::vector<std::vector<int64_t>> prompts = {{1, 5, 9}, {3, 4, 5, 6, 7, 8}, {2}};
    const ChoiceScores want = ScoreChoices(*on_cpu, prompts, {10, 20}, &cpu, 2);
    const ChoiceScores got = ScoreChoices(*on_hip, prompts, {10, 20}, &hip, 2);
    for (size_t i = 0; i < prompts.size(); ++i) {
        for (size_t c = 0; c < 2; ++c) {
            EXPECT_NEAR(got.log_probs[i][c], want.log_probs[i][c], 1e-4);
            EXPECT_NEAR(got.choice_log_probs[i][c], want.choice_log_probs[i][c], 1e-4);
        }
    }
}

TEST(LmScoringHipTest, CollatedLossMatchesTheCpu) {
    CPUBackend cpu;
    HIPBackend hip;
    auto on_cpu = LoadCausalLM(TinyLlama(), &cpu);
    auto on_hip = LoadCausalLM(TinyLlama(), &hip);
    const CausalLmBatch a = CollatePromptCompletions(kExamples, &cpu);
    const CausalLmBatch b = CollatePromptCompletions(kExamples, &hip);
    TokenCrossEntropyLoss loss_cpu(&cpu), loss_hip(&hip);
    EXPECT_NEAR(loss_hip.forward(on_hip->forward(b.ids), b.targets), loss_cpu.forward(on_cpu->forward(a.ids), a.targets), 1e-4);
}

}  // namespace
}  // namespace pulsatrix
