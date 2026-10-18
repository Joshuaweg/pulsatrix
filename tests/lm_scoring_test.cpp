#include "pulsatrix/lm_scoring.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"
#include "pulsatrix/tokenizer_json.hpp"

namespace pulsatrix {
namespace {

std::string Fixture(const std::string& path) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/" + path; }

std::unique_ptr<CausalLM> TinyLlama(CPUBackend* cpu) { return LoadCausalLM(Fixture("hf_tiny/llama"), cpu); }

// log P(token | ids[0..position]) from one unpadded forward pass, in double.
double ReferenceLogProb(CausalLM& model, CPUBackend* cpu, const std::vector<int64_t>& ids, size_t position, int64_t token) {
    std::vector<float> f(ids.begin(), ids.end());
    const Tensor logits = model.forward(Tensor(Shape({1, static_cast<int64_t>(ids.size())}), cpu, f));
    const std::vector<float> v = logits.to_host_vector();
    const int64_t vocab = model.config().vocab_size;
    const float* row = &v[position * static_cast<size_t>(vocab)];
    double top = row[0];
    for (int64_t k = 1; k < vocab; ++k) top = std::max(top, static_cast<double>(row[k]));
    double sum = 0.0;
    for (int64_t k = 0; k < vocab; ++k) sum += std::exp(row[k] - top);
    return row[token] - top - std::log(sum);
}

std::vector<PromptCompletion> Examples() {
    return {{{1, 5, 9}, {12}}, {{3, 4, 5, 6, 7, 8}, {20, 21}}, {{2}, {30, 31, 32}}, {{11, 13, 17, 19}, {40}}};
}

TEST(LmScoringTest, ReadJsonlSkipsBlankLinesAndNamesTheBadOne) {
    const std::string path = ::testing::TempDir() + "pulsatrix_lm_scoring.jsonl";
    {
        std::ofstream out(path, std::ios::binary);
        out << "{\"prompt\": \"a\", \"completion\": \" b\"}\r\n\n  \n[1, 2]\n";
    }
    const std::vector<JsonValue> rows = ReadJsonl(path);
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_EQ(rows[0].find("prompt")->as_string(), "a");
    EXPECT_EQ(rows[1].as_array().size(), 2u);
    {
        std::ofstream out(path, std::ios::binary);
        out << "{}\n{\"prompt\": \n";
    }
    try {
        (void)ReadJsonl(path);
        FAIL() << "a bad line should throw";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("line 2"), std::string::npos) << e.what();
    }
    std::remove(path.c_str());
    EXPECT_THROW((void)ReadJsonl(path), std::runtime_error);
}

TEST(LmScoringTest, TokenizesPromptAndCompletionApart) {
    const TextTokenizer tok = LoadTokenizerJson(Fixture("tokenizers/smollm2/tokenizer.json"));
    const std::vector<JsonValue> rows = {ParseJson(R"({"prompt": "Is the sky blue? A:", "completion": " Yes"})")};
    const std::vector<PromptCompletion> e = TokenizePromptCompletions(rows, tok);
    ASSERT_EQ(e.size(), 1u);
    EXPECT_EQ(e[0].prompt, tok.encode("Is the sky blue? A:", false).ids);
    EXPECT_EQ(e[0].completion, tok.encode(" Yes", false).ids);
    // A token's own text is that one token. (The fixture's vocabulary is cut down, so which
    // words are single tokens depends on it.)
    const int64_t the = tok.encode(" the", false).ids.at(0);
    EXPECT_EQ(SingleTokenId(tok, tok.decode({the})), the);
    EXPECT_THROW((void)TokenizePromptCompletions({ParseJson(R"({"prompt": "x"})")}, tok), std::invalid_argument);
    EXPECT_THROW((void)TokenizePromptCompletions({ParseJson(R"({"prompt": "x", "completion": ""})")}, tok), std::invalid_argument);
    EXPECT_THROW((void)SingleTokenId(tok, " Yes and no"), std::invalid_argument);
}

TEST(LmScoringTest, CollateMasksThePromptAndShiftsTheTargets) {
    CPUBackend cpu;
    const CausalLmBatch batch = CollatePromptCompletions({{{1, 2, 3}, {7, 8}}, {{4}, {9}}}, &cpu, 0);
    EXPECT_EQ(batch.lengths, (std::vector<int64_t>{4, 1}));
    EXPECT_EQ(batch.target_tokens, 3);
    EXPECT_EQ(batch.ids.shape(), Shape({2, 4}));
    const float x = static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex);
    EXPECT_EQ(batch.ids.to_host_vector(), (std::vector<float>{1, 2, 3, 7, 4, 0, 0, 0}));
    EXPECT_EQ(batch.targets.to_host_vector(), (std::vector<float>{x, x, 7, 8, 9, x, x, x}));
    EXPECT_THROW((void)CollatePromptCompletions({}, &cpu), std::invalid_argument);
    EXPECT_THROW((void)CollatePromptCompletions({{{}, {1}}}, &cpu), std::invalid_argument);
    EXPECT_THROW((void)CollatePromptCompletions({{{1}, {}}}, &cpu), std::invalid_argument);
}

// Padding and batching change nothing: each example scores as it does alone, unpadded.
TEST(LmScoringTest, CompletionLogProbsMatchEachExampleAlone) {
    CPUBackend cpu;
    auto model = TinyLlama(&cpu);
    const std::vector<PromptCompletion> examples = Examples();
    for (const int64_t batch_size : {1, 3, 8}) {
        const std::vector<double> scores = CompletionLogProbs(*model, examples, &cpu, batch_size);
        ASSERT_EQ(scores.size(), examples.size());
        for (size_t i = 0; i < examples.size(); ++i) {
            std::vector<int64_t> ids = examples[i].prompt;
            double reference = 0.0;
            for (const int64_t c : examples[i].completion) {
                reference += ReferenceLogProb(*model, &cpu, ids, ids.size() - 1, c);
                ids.push_back(c);
            }
            EXPECT_NEAR(scores[i], reference, 1e-4) << "example " << i << ", batch size " << batch_size;
        }
    }
}

// The training loss on a collated batch is the completions' mean negative log-likelihood.
TEST(LmScoringTest, TrainingLossIsTheCompletionLogLoss) {
    CPUBackend cpu;
    auto model = TinyLlama(&cpu);
    const std::vector<PromptCompletion> examples = Examples();
    const CausalLmBatch batch = CollatePromptCompletions(examples, &cpu);
    TokenCrossEntropyLoss loss(&cpu);
    const float value = loss.forward(model->forward(batch.ids), batch.targets);
    EXPECT_EQ(loss.num_tokens(), batch.target_tokens);
    double total = 0.0;
    for (const double s : CompletionLogProbs(*model, examples, &cpu)) total -= s;
    EXPECT_NEAR(value, total / static_cast<double>(batch.target_tokens), 1e-4);
}

TEST(LmScoringTest, ScoreChoicesRenormalizesOverTheChoices) {
    CPUBackend cpu;
    auto model = TinyLlama(&cpu);
    const std::vector<std::vector<int64_t>> prompts = {{1, 5, 9}, {3, 4, 5, 6, 7, 8}, {2}};
    const std::vector<int64_t> choices = {10, 20, 33};
    const ChoiceScores s = ScoreChoices(*model, prompts, choices, &cpu, 2);
    ASSERT_EQ(s.log_probs.size(), prompts.size());
    for (size_t i = 0; i < prompts.size(); ++i) {
        double mass = 0.0, renormalized = 0.0;
        for (size_t c = 0; c < choices.size(); ++c) {
            EXPECT_NEAR(s.log_probs[i][c], ReferenceLogProb(*model, &cpu, prompts[i], prompts[i].size() - 1, choices[c]), 1e-4);
            mass += std::exp(s.log_probs[i][c]);
            renormalized += std::exp(s.choice_log_probs[i][c]);
        }
        EXPECT_NEAR(s.choice_mass[i], mass, 1e-9);
        EXPECT_NEAR(renormalized, 1.0, 1e-12);
        // Log-odds between two choices are the same either way.
        EXPECT_NEAR(s.choice_log_probs[i][0] - s.choice_log_probs[i][1], s.log_probs[i][0] - s.log_probs[i][1], 1e-12);
    }
    EXPECT_THROW((void)ScoreChoices(*model, {}, choices, &cpu), std::invalid_argument);
    EXPECT_THROW((void)ScoreChoices(*model, prompts, {}, &cpu), std::invalid_argument);
    EXPECT_THROW((void)ScoreChoices(*model, prompts, {64}, &cpu), std::invalid_argument);
    EXPECT_THROW((void)ScoreChoices(*model, {{}}, choices, &cpu), std::invalid_argument);
    EXPECT_THROW((void)ScoreChoices(*model, prompts, choices, &cpu, 0), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
