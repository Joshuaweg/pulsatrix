// PLM-6: EncoderExplainer against LXT's AttnLRP rules applied to transformers' ESM-2
// (tools/golden/make_esm_attnlrp_reference.py) on the tiny random ESM-2, and with
// PULSATRIX_GOLDEN_DIR on downloaded models; the plain-gradient negative control; and the checks:
// agreement, DMS sensitivity, alignment conservation, deletion and randomization.
#include "pulsatrix/protein_explanations.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string TinyEsm() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm"; }

std::vector<int64_t> Ints(const SafetensorsFile& f, const std::string& name) {
    auto [ptr, size] = f.bytes(name);
    std::vector<int64_t> out(size / sizeof(int64_t));
    std::memcpy(out.data(), ptr, size);
    return out;
}

std::vector<float> Floats(const SafetensorsFile& f, const std::string& name) {
    CPUBackend cpu;
    return f.tensor(name, &cpu).to_host_vector();
}

/** @brief max |a - b| over the largest |b|. */
double RelativeDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double diff = 0, scale = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        diff = std::max(diff, std::abs(static_cast<double>(a[i]) - b[i]));
        scale = std::max(scale, std::abs(static_cast<double>(b[i])));
    }
    return scale > 0 ? diff / scale : diff;
}

std::vector<float> Tokens(const ResidueRelevance& r) {
    std::vector<float> t{r.cls};
    t.insert(t.end(), r.residues.begin(), r.residues.end());
    t.push_back(r.eos);
    return t;
}

struct Comparison {
    double worst_relative = 0;
    double worst_correlation = 1;
    int checked = 0;
};

/** @brief Every target in @p dir's @p file against pulsatrix's AttnLRP. */
Comparison CompareToReference(const std::string& dir, const std::string& file) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(dir, &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(dir + "/vocab.txt");
    EncoderExplainer explainer(*model, tok, &cpu);
    const SafetensorsFile ref = SafetensorsFile::Map(dir + "/" + file);
    const JsonValue sequences = ParseJson(ref.metadata().at("sequences"));
    const LinearHead head{Floats(ref, "head_weight"), Floats(ref, "head_bias")[0]};
    Comparison c;
    auto check = [&](const ResidueRelevance& got, const std::vector<float>& want, const std::string& what) {
        const std::vector<float> ours = Tokens(got);
        EXPECT_EQ(ours.size(), want.size()) << what;
        if (ours.size() != want.size()) return;
        c.worst_relative = std::max(c.worst_relative, RelativeDiff(ours, want));
        c.worst_correlation = std::min(c.worst_correlation, PearsonCorrelation(ours, want));
        ++c.checked;
    };
    for (size_t k = 0; k < sequences.as_array().size(); ++k) {
        const std::string s = std::to_string(k), seq = sequences.as_array()[k].as_string();
        const std::vector<int64_t> target = Ints(ref, "target." + s);
        const int64_t residue = target[0] - 1;
        const char token = tok.id_to_token(target[1])->front(), wild = tok.id_to_token(target[2])->front();

        const ResidueRelevance masked = explainer.explain(seq, EncoderTarget::MaskedToken(residue, token));
        EXPECT_NEAR(masked.value, Floats(ref, "mask_value." + s)[0], 1e-3 * std::max(1.0f, std::abs(masked.value))) << dir << " " << s;
        check(masked, Floats(ref, "mask_relevance." + s), "masked " + s);
        const std::vector<float> layers = Floats(ref, "mask_layers." + s);
        const size_t T = seq.size() + 2;
        for (size_t l = 0; l < masked.layers.size(); ++l) {
            const std::vector<float> want(layers.begin() + static_cast<std::ptrdiff_t>(l * T), layers.begin() + static_cast<std::ptrdiff_t>((l + 1) * T));
            c.worst_relative = std::max(c.worst_relative, RelativeDiff(masked.layers[l], want));
        }
        check(explainer.explain(seq, EncoderTarget::ForMutation({wild, residue + 1, token})), Floats(ref, "mutation_relevance." + s),
              "mutation " + s);
        check(explainer.explain(seq, EncoderTarget::ProteinHead(head)), Floats(ref, "protein_head_relevance." + s), "protein head " + s);
        check(explainer.explain(seq, EncoderTarget::ResidueHead(head, target[3] - 1)), Floats(ref, "residue_head_relevance." + s),
              "residue head " + s);
    }
    return c;
}

TEST(EncoderExplainer, TinyEsmMatchesLxtAttnLrp) {
    const Comparison c = CompareToReference(TinyEsm(), "esm_attnlrp.safetensors");
    EXPECT_EQ(c.checked, 12);
    EXPECT_LT(c.worst_relative, 1e-4);
    EXPECT_GT(c.worst_correlation, 0.9999);
}

TEST(EncoderExplainer, TinyEsmDoesNotMatchPlainGradientTimesInput) {
    // The negative control: without AttnLRP's rules the reference differs, so matching it above
    // says something.
    const Comparison c = CompareToReference(TinyEsm(), "esm_plain_gradient.safetensors");
    EXPECT_GT(c.worst_relative, 0.05);
}

TEST(EncoderExplainer, RealModelsInPulsatrixGoldenDirMatchLxtAttnLrp) {
    const char* dir = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set PULSATRIX_GOLDEN_DIR to check downloaded ESM-2 models";
    int checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!std::filesystem::exists(entry.path() / "esm_attnlrp.safetensors")) continue;
        const Comparison c = CompareToReference(entry.path().string(), "esm_attnlrp.safetensors");
        std::printf("%s: worst relative difference %.3g, lowest correlation %.6f over %d explanations\n", entry.path().filename().c_str(),
                    c.worst_relative, c.worst_correlation, c.checked);
        EXPECT_GT(c.worst_correlation, 0.999) << entry.path();
        EXPECT_LT(c.worst_relative, 1e-2) << entry.path();
        if (std::filesystem::exists(entry.path() / "esm_plain_gradient.safetensors")) {
            EXPECT_GT(CompareToReference(entry.path().string(), "esm_plain_gradient.safetensors").worst_relative, 0.05) << entry.path();
        }
        ++checked;
    }
    if (checked == 0) GTEST_SKIP() << "no ESM model with an esm_attnlrp.safetensors in " << dir;
}

TEST(EncoderExplainer, ExplainsWhatItEvaluatesAndRefusesBadTargets) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    EncoderExplainer explainer(*model, tok, &cpu);
    const std::string seq = "MKTAYIAKQRQISF";
    const EncoderTarget t = EncoderTarget::ForMutation(ParseMutations("K2R")[0]);
    EXPECT_EQ(t.residue, 1);
    const ResidueRelevance r = explainer.explain(seq, t);
    EXPECT_EQ(r.target, "K2R log-odds");
    EXPECT_EQ(r.residues.size(), seq.size());
    EXPECT_EQ(r.layers.size(), 3u);
    // evaluate() masks the target's residue itself, from the unmasked tokens.
    EXPECT_NEAR(explainer.evaluate(explainer.tokens(seq), t), r.value, 1e-5);
    // The masked residue's own token carries no relevance: its embedding is zeroed.
    EXPECT_EQ(r.residues[1], 0.0f);

    EXPECT_THROW((void)explainer.explain(seq, EncoderTarget::ForMutation(ParseMutations("A2R")[0])), std::invalid_argument);
    EXPECT_THROW((void)explainer.explain(seq, EncoderTarget::MaskedToken(14, 'A')), std::invalid_argument);
    EXPECT_THROW((void)explainer.explain(seq, EncoderTarget::ProteinHead(LinearHead{{1.0f, 2.0f}, 0.0f})), std::invalid_argument);
    EXPECT_THROW((void)explainer.explain("", EncoderTarget::MaskedToken(0, 'A')), std::invalid_argument);
    EXPECT_THROW((void)explainer.explain("MK1", EncoderTarget::MaskedToken(0, 'A')), std::invalid_argument);
}

TEST(EncoderExplainer, RelevanceGraphCarriesTheExplanationThroughEveryLayer) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    EncoderExplainer explainer(*model, tok, &cpu);
    const std::string seq = "MKTAYIAKQR";
    const EncoderTarget t = EncoderTarget::MaskedToken(4, 'L');
    const ResidueRelevance r = explainer.explain(seq, t);
    RelevanceGraphOptions o;
    o.edge_threshold = 1.0;  // no pruning, so every link is there
    const AttributionGraph g = explainer.relevance_graph(seq, t, o);
    std::map<std::string, double> embedding, outgoing;
    for (const auto& n : g.nodes) {
        if (n.layer == "E") embedding[n.node_id] = *n.activation;
    }
    ASSERT_EQ(embedding.size(), seq.size() + 2);
    for (const auto& l : g.links) {
        if (embedding.count(l.source)) outgoing[l.source] += l.weight;
    }
    // Embedding node i holds token i's relevance, and passes exactly that on through layer 0.
    const std::vector<float> tokens = Tokens(r);
    std::vector<int64_t> ids = explainer.tokens(seq);
    ids[5] = explainer.mask_id();  // the explained residue is masked
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string id = "E_" + std::to_string(ids[i]) + "_" + std::to_string(i);
        ASSERT_TRUE(embedding.count(id)) << id;
        EXPECT_NEAR(embedding[id], tokens[i], 1e-5) << id;
        EXPECT_NEAR(outgoing[id], tokens[i], 1e-4) << id;
    }
    const auto out = std::find_if(g.nodes.begin(), g.nodes.end(), [](const auto& n) { return n.is_target_logit; });
    ASSERT_NE(out, g.nodes.end());
    EXPECT_NE(out->clerp.find("Output \"Y5 masked: L\" (p="), std::string::npos) << out->clerp;
    EXPECT_EQ(out->ctx_idx, 5);
    o.max_tokens = 5;
    EXPECT_THROW((void)explainer.relevance_graph(seq, t, o), std::invalid_argument);
    // The graph leaves the model as explain() does: activations freed, the same answers after.
    EXPECT_EQ(explainer.explain(seq, t).residues, r.residues);
}

// ---- checks ----------------------------------------------------------------------------------

TEST(ResidueChecks, AgreementCountsRanksAndTopResidues) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // |relevance| ranks residues exactly as the signal does, except the one without a signal.
    const std::vector<float> relevance = {-5, 4, 3, 2, 1, 0.5f, 0.25f, 0.1f, 0.05f, 0.01f, 9};
    const std::vector<float> signal = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1, nan};
    const ResidueAgreement a = CompareResidueSignals(relevance, signal, 0.2);
    EXPECT_EQ(a.compared, 10);
    EXPECT_NEAR(a.spearman, 1.0, 1e-6);
    EXPECT_EQ(a.top, 2);
    EXPECT_DOUBLE_EQ(a.top_overlap, 1.0);
    EXPECT_DOUBLE_EQ(a.chance_overlap, 0.2);
    std::vector<float> reversed(signal.rbegin() + 1, signal.rend());
    reversed.push_back(nan);
    EXPECT_NEAR(CompareResidueSignals(relevance, reversed, 0.2).spearman, -1.0, 1e-6);
    EXPECT_THROW((void)CompareResidueSignals({1, 2}, {1, 2}), std::invalid_argument);
    EXPECT_THROW((void)CompareResidueSignals(relevance, signal, 0.0), std::invalid_argument);
}

TEST(ResidueChecks, DmsSensitivityIsMinusTheMeanSingleMutantScore) {
    DmsVariants v;
    v.mutants = {"M1A", "M1C", "K2A", "M1A:K2A"};
    v.scores = {-1.0, -3.0, 0.5, -10.0};
    const std::vector<float> s = DmsPositionSensitivity(v, "MKT");
    EXPECT_FLOAT_EQ(s[0], 2.0f);
    EXPECT_FLOAT_EQ(s[1], -0.5f);
    EXPECT_TRUE(std::isnan(s[2]));
    v.mutants = {"A1C"};
    v.scores = {0.0};
    EXPECT_THROW((void)DmsPositionSensitivity(v, "MKT"), std::invalid_argument);
}

TEST(ResidueChecks, ConservationReadsBothA2mLayouts) {
    // Column-aligned (ProteinGym): every row as long as the query; lowercase columns aren't focus.
    const std::vector<FastaRecord> aligned = {{"q", "", "mACD"}, {"a", "", "kAC-"}, {"b", "", "lAE-"}, {"c", "", "iAEE"}};
    const std::vector<float> c = AlignmentConservation(aligned);
    ASSERT_EQ(c.size(), 4u);
    EXPECT_TRUE(std::isnan(c[0]));
    EXPECT_NEAR(c[1], std::log2(20.0), 1e-5);                  // all A
    EXPECT_NEAR(c[2], std::log2(20.0) - 1.0, 1e-5);            // C C E E: one bit of entropy
    EXPECT_NEAR(c[3], (std::log2(20.0) - 1.0) * 0.5, 1e-5);    // D and E in half the rows
    EXPECT_EQ(AlignmentQuery(aligned), "MACD");

    // Standard A2M: lowercase and '.' are insertions; match columns line up.
    const std::vector<FastaRecord> a2m = {{"q", "", "ACD"}, {"a", "", "AxxC-"}, {"b", "", "A.CE"}};
    const std::vector<float> s = AlignmentConservation(a2m);
    ASSERT_EQ(s.size(), 3u);
    EXPECT_NEAR(s[0], std::log2(20.0), 1e-5);
    EXPECT_NEAR(s[1], std::log2(20.0), 1e-5);
    EXPECT_THROW((void)AlignmentConservation({{"q", "", "ACD"}, {"a", "", "AC"}}), std::invalid_argument);
    EXPECT_THROW((void)AlignmentConservation({}), std::invalid_argument);
}

TEST(ResidueChecks, DeletionAndRandomizationRunOnTheModel) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    EncoderExplainer explainer(*model, tok, &cpu);
    const std::string seq = "MKTAYIAKQRQISFVKSH";
    const EncoderTarget t = EncoderTarget::MaskedToken(5, 'L');
    const ResidueRelevance r = explainer.explain(seq, t);

    const ResidueDeletionCurve d = DeletionCheck(explainer, seq, t, r.residues, 4, 3, 1);
    ASSERT_EQ(d.scores.size(), 5u);
    ASSERT_EQ(d.random_scores.size(), 5u);
    EXPECT_FLOAT_EQ(d.fractions.back(), 1.0f);
    // Nothing masked yet, and everything masked, are the same in either order.
    EXPECT_NEAR(d.scores.front(), r.value, 1e-5);
    EXPECT_NEAR(d.random_scores.front(), r.value, 1e-5);
    EXPECT_NEAR(d.scores.back(), d.random_scores.back(), 1e-5);
    EXPECT_THROW((void)DeletionCheck(explainer, seq, t, {1.0f}, 4), std::invalid_argument);

    const RandomizationCheckResult z = RandomizationCheck(explainer, seq, t, 3);
    EXPECT_EQ(z.layers, (std::vector<std::string>{"lm_head", "layers.1", "layers.0", "embed_tokens"}));
    ASSERT_EQ(z.similarity.size(), 4u);
    EXPECT_LT(z.similarity.back(), 0.9f);
    // The model is restored: the same explanation again.
    const ResidueRelevance again = explainer.explain(seq, t);
    for (size_t i = 0; i < r.residues.size(); ++i) EXPECT_EQ(again.residues[i], r.residues[i]) << i;
}

}  // namespace
}  // namespace pulsatrix
