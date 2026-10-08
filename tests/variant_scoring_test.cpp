// PLM-3: variant parsing, ProteinGym's windowing and files, and VariantScorer (scores, scans and its
// memory budget) on the tiny random ESM-2 in tests/fixtures/hf_tiny/esm.
#include "pulsatrix/variant_scoring.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/proteingym.hpp"

namespace pulsatrix {
namespace {

std::string TinyEsm() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm"; }

TEST(ScoringPassBytes, GrowsWithTheSquareOfTheLength) {
    EncoderLMConfig c;
    c.num_attention_heads = 20;
    c.hidden_size = 1280;
    c.intermediate_size = 5120;
    c.vocab_size = 33;
    // ESM-2 650M on a full 1024-token window: the six (20, 1024, 1024) attention copies dominate.
    EXPECT_GT(ScoringPassBytes(c, 1024), int64_t{6} * 20 * 1024 * 1024 * 4);
    EXPECT_LT(ScoringPassBytes(c, 1024), int64_t{1} << 30);
    EXPECT_GT(ScoringPassBytes(c, 1024), 3 * ScoringPassBytes(c, 512));
}

TEST(VariantScorer, BatchesFewerUnderAMemoryBudgetWithTheSameResults) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    const std::string seq = "MKTAYIAKQRQISFVKSHFSRQ";
    VariantScorer wide(*model, tok, &cpu);
    const ResidueLogProbs expected = wide.masked_marginals(seq);

    VariantScoringOptions one_at_a_time;
    one_at_a_time.max_pass_bytes = ScoringPassBytes(model->config(), static_cast<int64_t>(seq.size()) + 2);
    VariantScorer narrow(*model, tok, &cpu, one_at_a_time);
    const ResidueLogProbs got = narrow.masked_marginals(seq);
    ASSERT_EQ(got.values.size(), expected.values.size());
    for (size_t i = 0; i < got.values.size(); ++i) EXPECT_NEAR(got.values[i], expected.values[i], 1e-5) << i;
    // Scoring leaves the model keeping activations, as it found it.
    EXPECT_TRUE(model->keep_activations());
}

TEST(VariantScorer, RefusesAPassOverItsMemoryBudget) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    VariantScoringOptions tight;
    tight.max_pass_bytes = 1024;
    VariantScorer scorer(*model, tok, &cpu, tight);
    EXPECT_THROW((void)scorer.masked_marginals("MKTAYIAKQR"), std::invalid_argument);
    EXPECT_THROW((void)scorer.wild_type_marginals("MKTAYIAKQR"), std::invalid_argument);
    tight.max_pass_bytes = 0;
    EXPECT_THROW(VariantScorer(*model, tok, &cpu, tight), std::invalid_argument);
}

TEST(Mutations, ParseAndApply) {
    const std::vector<Mutation> m = ParseMutations("A2G:K10P");
    ASSERT_EQ(m.size(), 2U);
    EXPECT_EQ(m[0].wild_type, 'A');
    EXPECT_EQ(m[0].position, 2);
    EXPECT_EQ(m[0].mutant, 'G');
    EXPECT_EQ(m[1].position, 10);
    for (const char* bad : {"", "A2", "2G", "A2G:", "AxG", "A2G::K10P"}) EXPECT_THROW((void)ParseMutations(bad), std::invalid_argument) << bad;

    EXPECT_EQ(ApplyMutations("MAKLV", ParseMutations("A2G:V5W")), "MGKLW");
    EXPECT_EQ(ApplyMutations("MAKLV", ParseMutations("A12G"), 11), "MGKLV");  // numbered from 11
    EXPECT_THROW((void)ApplyMutations("MAKLV", ParseMutations("K2G")), std::invalid_argument);  // wrong wild type
    EXPECT_THROW((void)ApplyMutations("MAKLV", ParseMutations("A6G")), std::invalid_argument);  // past the end
}

TEST(OptimalWindow, MatchesProteinGym) {
    using W = std::pair<int64_t, int64_t>;
    EXPECT_EQ(OptimalWindow(10, 20, 1024), W(0, 20));  // fits whole
    EXPECT_EQ(OptimalWindow(500, 1865, 1024), W(0, 1024));
    EXPECT_EQ(OptimalWindow(1000, 1865, 1024), W(488, 1512));
    EXPECT_EQ(OptimalWindow(1500, 1865, 1024), W(841, 1865));
}

TEST(VariantScorer, ScoresAreLogRatiosThatAddUp) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    const std::string seq = "MKTAYIAKQR";
    VariantScorer scorer(*model, tok, &cpu);
    for (const ResidueLogProbs& m : {scorer.masked_marginals(seq), scorer.wild_type_marginals(seq)}) {
        ASSERT_EQ(m.length, 10);
        const std::vector<float> scan = scorer.single_mutant_scan(m, seq);
        ASSERT_EQ(scan.size(), 200U);
        const auto col = [](char aa) { return static_cast<size_t>(std::string(kAminoAcids).find(aa)); };
        EXPECT_EQ(scan[1 * 20 + col('K')], 0.0f);  // the wild type's own column
        const double k2a = scorer.score(m, seq, ParseMutations("K2A")), t3w = scorer.score(m, seq, ParseMutations("T3W"));
        EXPECT_NEAR(k2a, scan[1 * 20 + col('A')], 1e-6);
        EXPECT_NEAR(k2a, m.at(1, scorer.token_of('A')) - m.at(1, scorer.token_of('K')), 1e-6);
        EXPECT_NEAR(scorer.score(m, seq, ParseMutations("K2A:T3W")), k2a + t3w, 1e-6);
        EXPECT_THROW((void)scorer.score(m, seq, ParseMutations("A2K")), std::invalid_argument);
        EXPECT_THROW((void)scorer.score(m, "MKTAYIAKQ", ParseMutations("K2A")), std::invalid_argument);  // other sequence
    }
    double pll = 0;
    const ResidueLogProbs masked = scorer.masked_marginals(seq);
    for (int64_t i = 0; i < 10; ++i) pll += masked.at(i, scorer.token_of(seq[static_cast<size_t>(i)]));
    EXPECT_NEAR(scorer.pseudo_log_likelihood(seq), pll, 1e-9);
    EXPECT_THROW((void)scorer.masked_marginals(""), std::invalid_argument);
    EXPECT_THROW((void)scorer.masked_marginals("MKXZ1"), std::invalid_argument);
}

TEST(ProteinGym, ReadsFilesAndEvaluatesAnAssay) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "pulsatrix_proteingym_test";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "ref.csv") << "DMS_id,DMS_filename,target_seq,start_idx\nTOY_1,toy.csv,mktayiakqr,\nTOY_2,toy2.csv,MKT,5\n";
    std::ofstream(dir / "toy.csv") << "mutant,mutated_sequence,DMS_score,DMS_score_bin\n"
                                      "K2A,x,0.5,1\nT3W,x,-1.0,0\nK2A:T3W,x,-0.4,0\nQ9E,x,1.2,1\nM1L,x,0.1,0\n";
    std::ofstream(dir / "bad.csv") << "mutant,DMS_score\nK2A,0.5\n";

    const std::vector<ProteinGymAssay> ref = ReadProteinGymReference((dir / "ref.csv").string());
    ASSERT_EQ(ref.size(), 2U);
    EXPECT_EQ(ref[0].target_seq, "MKTAYIAKQR");  // upper-cased
    EXPECT_EQ(ref[0].offset, 1);
    EXPECT_EQ(ref[1].offset, 5);
    const DmsVariants v = ReadDmsVariants((dir / "toy.csv").string());
    ASSERT_EQ(v.mutants.size(), 5U);
    EXPECT_EQ(v.mutants[2], "K2A:T3W");
    EXPECT_EQ(v.bins[3], 1);
    EXPECT_THROW((void)ReadDmsVariants((dir / "bad.csv").string()), std::invalid_argument);

    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    VariantScorer scorer(*model, tok, &cpu);
    const AssayResult r = EvaluateAssay(scorer, ref[0], v);
    EXPECT_EQ(r.id, "TOY_1");
    EXPECT_EQ(r.length, 10);
    ASSERT_EQ(r.scores.size(), 5U);
    EXPECT_NEAR(r.scores[0], scorer.score(scorer.masked_marginals(ref[0].target_seq), ref[0].target_seq, ParseMutations("K2A")), 1e-9);
    EXPECT_EQ(r.metrics.spearman, SpearmanCorrelation(v.scores, r.scores));
    std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace pulsatrix
