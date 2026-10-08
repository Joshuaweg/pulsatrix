// PLM-8: Swiss-Prot annotations as per-residue concepts, residue embeddings, linear probes and the
// local-sequence control, sparse codes and InterPLM-style concept matching, and the feature
// dashboard's structure panel.
#include "pulsatrix/protein_concepts.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"
#include "pulsatrix/viz/protein_views.hpp"

namespace pulsatrix {
namespace {

std::string TinyEsm() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm"; }

const char* const kJsonl =
    R"({"accession": "P1", "sequence": "MCKCAHHHLL", "features": [{"type": "Helix", "start": 5, "end": 8}, {"type": "Disulfide bond", "start": 2, "end": 4}, {"type": "Binding site", "start": 9, "end": 9}]})"
    "\n\n"
    R"({"accession": "P2", "sequence": "GSEEV", "features": [{"type": "Beta strand", "start": 2, "end": 4}]})"
    "\n";

TEST(ProteinConcepts, AnnotationsBecomePerResidueLabels) {
    const std::vector<AnnotatedProtein> proteins = ParseAnnotatedProteins(kJsonl);
    ASSERT_EQ(proteins.size(), 2u);
    EXPECT_EQ(proteins[0].accession, "P1");
    EXPECT_EQ(proteins[0].features.size(), 3u);
    // A disulfide bond marks its two cysteines only.
    EXPECT_EQ(ConceptLabels(proteins[0], "Disulfide bond"), (std::vector<int>{0, 1, 0, 1, 0, 0, 0, 0, 0, 0}));
    EXPECT_EQ(ConceptLabels(proteins[0], "Helix"), (std::vector<int>{0, 0, 0, 0, 1, 1, 1, 1, 0, 0}));
    EXPECT_EQ(SecondaryStructureLabels(proteins[0]), (std::vector<int>{2, 2, 2, 2, 0, 0, 0, 0, 2, 2}));
    EXPECT_EQ(SecondaryStructureLabels(proteins[1]), (std::vector<int>{2, 1, 1, 1, 2}));
    const auto counts = ConceptCounts(proteins);
    ASSERT_EQ(counts.size(), 4u);
    EXPECT_EQ(counts[0], (std::pair<std::string, int64_t>{"Helix", 4}));
    EXPECT_EQ(counts.back().second, 1);  // the binding site

    EXPECT_THROW((void)ParseAnnotatedProteins("{\"accession\": \"X\"}\n"), std::invalid_argument);
    EXPECT_THROW((void)ParseAnnotatedProteins(R"({"accession": "X", "sequence": "MK", "features": [{"type": "Helix", "start": 2, "end": 3}]})"),
                 std::invalid_argument);
    EXPECT_THROW((void)ParseAnnotatedProteins("not json\n"), std::invalid_argument);
}

TEST(ProteinConcepts, SequenceWindowIsOneHotNeighbours) {
    const std::vector<float> f = SequenceWindowFeatures({"AC", "W"}, 1);
    ASSERT_EQ(f.size(), 3u * 60u);
    auto at = [&](size_t residue, int offset, char aa) {
        return f[residue * 60 + static_cast<size_t>(offset + 1) * 20 + std::string("ACDEFGHIKLMNPQRSTVWY").find(aa)];
    };
    EXPECT_EQ(at(0, 0, 'A'), 1.0f);
    EXPECT_EQ(at(0, 1, 'C'), 1.0f);
    EXPECT_EQ(at(1, -1, 'A'), 1.0f);
    // The next sequence doesn't leak into the window.
    EXPECT_EQ(at(1, 1, 'W'), 0.0f);
    EXPECT_EQ(at(2, 0, 'W'), 1.0f);
    double total = 0;
    for (float v : f) total += v;
    EXPECT_EQ(total, 2.0 + 2.0 + 1.0);
}

TEST(ProteinConcepts, EmbeddingsKeepResiduesAtTheAskedLayers) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    const ResidueEmbeddings e = EmbedResidues(*model, tok, &cpu, {"MKTAY", "GS"}, {0, 2});
    EXPECT_EQ(e.residues, 7);
    EXPECT_EQ(e.offsets, (std::vector<int64_t>{0, 5}));
    ASSERT_EQ(e.values.size(), 2u);
    EXPECT_EQ(e.values[0].size(), 7u * 32u);
    // Layer num_layers is the final representation, after the final LayerNorm.
    (void)model->forward(Tensor(Shape({1, 4}), &cpu, {0, 6, 8, 2}));
    const std::vector<float> last = model->last_hidden_state().to_host_vector();
    for (int64_t j = 0; j < 32; ++j) EXPECT_NEAR(e.row(1, 5)[j], last[static_cast<size_t>(32 + j)], 1e-5) << j;
    EXPECT_THROW((void)EmbedResidues(*model, tok, &cpu, {"MK"}, {3}), std::invalid_argument);
    EXPECT_THROW((void)EmbedResidues(*model, tok, &cpu, {"MKTAY"}, {0}, 6), std::invalid_argument);
    EXPECT_TRUE(model->keep_activations());
}

TEST(ProteinConcepts, ProbeLearnsSeparableLabelsAndNotNoise) {
    CPUBackend cpu;
    const int64_t n = 2000, dim = 4;
    std::vector<float> x(static_cast<size_t>(n * dim));
    std::vector<int> separable(static_cast<size_t>(n)), noise(static_cast<size_t>(n));
    uint64_t s = 7;
    auto u = [&] {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<float>(static_cast<uint32_t>(s >> 32)) / 4294967296.0f;
    };
    std::vector<bool> train(static_cast<size_t>(n)), test(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < dim; ++j) x[static_cast<size_t>(i * dim + j)] = u() * 2 - 1;
        separable[static_cast<size_t>(i)] = x[static_cast<size_t>(i * dim)] + 0.5f * x[static_cast<size_t>(i * dim + 1)] > 0 ? 1 : 0;
        noise[static_cast<size_t>(i)] = u() < 0.5f ? 1 : 0;
        train[static_cast<size_t>(i)] = i % 5 != 0;
        test[static_cast<size_t>(i)] = i % 5 == 0;
    }
    ProbeOptions o;
    o.epochs = 40;
    o.batch = 128;
    const ProbeResult good = TrainLinearProbe(x, dim, separable, 2, train, test, &cpu, o);
    EXPECT_GT(good.accuracy, 0.95);
    EXPECT_GT(good.auc, 0.98);
    EXPECT_EQ(good.train, 1600);
    EXPECT_EQ(good.test, 400);
    const ProbeResult bad = TrainLinearProbe(x, dim, noise, 2, train, test, &cpu, o);
    EXPECT_LT(bad.auc, 0.6);
    // Three classes: balanced accuracy, no AUC.
    std::vector<int> three(separable);
    for (size_t i = 0; i < three.size(); ++i) three[i] = x[i * dim + 2] > 0.5f ? 2 : separable[i];
    const ProbeResult multi = TrainLinearProbe(x, dim, three, 3, train, test, &cpu, o);
    EXPECT_GT(multi.balanced_accuracy, 0.85);
    EXPECT_TRUE(std::isnan(multi.auc));
    EXPECT_THROW((void)TrainLinearProbe(x, dim, separable, 1, train, test, &cpu), std::invalid_argument);
    EXPECT_THROW((void)TrainLinearProbe(x, dim + 1, separable, 2, train, test, &cpu), std::invalid_argument);
}

TEST(ProteinConcepts, SparseCodesMatchTheDenseEncoding) {
    CPUBackend cpu;
    SparseAutoencoder sae(4, 16, 0.01f, &cpu, 2u);
    std::vector<float> rows(40);
    for (size_t i = 0; i < rows.size(); ++i) rows[i] = std::sin(static_cast<float>(i) * 0.7f);
    const SparseCodes c = EncodeSparse(sae, rows, &cpu, 3);  // batches that don't divide the 10 rows
    ASSERT_EQ(c.residues(), 10);
    const std::vector<float> dense = sae.encode(Tensor(Shape({10, 4}), &cpu, rows)).to_host_vector();
    for (int64_t r = 0; r < 10; ++r) {
        std::vector<float> back(16, 0.0f);
        for (int64_t e = c.row_start[static_cast<size_t>(r)]; e < c.row_start[static_cast<size_t>(r) + 1]; ++e) back[static_cast<size_t>(c.feature[static_cast<size_t>(e)])] = c.value[static_cast<size_t>(e)];
        for (size_t j = 0; j < 16; ++j) EXPECT_FLOAT_EQ(back[j], dense[static_cast<size_t>(r) * 16 + j]);
    }
    const SparseCodes n = NeuronCodes({1.0f, -2.0f, 0.0f, 3.0f}, 2);
    EXPECT_EQ(n.num_features, 4);
    EXPECT_EQ(n.feature, (std::vector<int32_t>{0, 3, 2}));
    EXPECT_EQ(n.max_value, (std::vector<float>{1.0f, 0.0f, 3.0f, 2.0f}));
}

TEST(ProteinConcepts, MatchingFindsTheFeatureThatTracksTheConcept) {
    // Feature 1 fires exactly on the concept's residues; feature 0 fires everywhere.
    SparseCodes c;
    c.num_features = 2;
    c.max_value = {1.0f, 2.0f};
    const std::vector<int> labels = {1, 0, 1, 0, 0, 1};
    c.row_start = {0};
    for (int label : labels) {
        c.feature.push_back(0);
        c.value.push_back(1.0f);
        if (label) {
            c.feature.push_back(1);
            c.value.push_back(2.0f);
        }
        c.row_start.push_back(static_cast<int64_t>(c.feature.size()));
    }
    const ConceptMatch m = MatchConcept(c, labels, "site");
    EXPECT_EQ(m.feature, 1);
    EXPECT_DOUBLE_EQ(m.f1, 1.0);
    EXPECT_EQ(m.positives, 3);
    EXPECT_EQ(m.features_above_half, 2);  // feature 0: precision 0.5, recall 1, F1 0.67
    EXPECT_THROW((void)MatchConcept(c, {1, 0}, "site"), std::invalid_argument);
}

TEST(ProteinConcepts, DashboardCarriesAStructurePanel) {
    const std::string pdb =
        "ATOM      1  CA  MET A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  LYS A   2       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  THR A   3       3.000   0.000   0.000  1.00  0.00           C\n";
    FeatureDashboardDocument doc;
    doc.source = "sae";
    doc.feature_index = 5;
    doc.max_activation = 2.0f;
    doc.top_examples.push_back({"P1", {"M", "K", "T"}, {0.0f, 2.0f, 1.0f}});
    const std::string html = RenderFeatureDashboardHtml(doc, pdb, "A", 0);
    EXPECT_NE(html.find("id=\"panel-viewer\""), std::string::npos);
    EXPECT_NE(html.find("id=\"panel-data\""), std::string::npos);
    EXPECT_NE(html.find("feature 5 activation"), std::string::npos);
    EXPECT_NE(html.find("('panel');"), std::string::npos);
    EXPECT_THROW((void)RenderFeatureDashboardHtml(doc, pdb, "A", 1), std::invalid_argument);
    doc.top_examples[0].tokens = {"M", "K", "W"};  // not the chain's residues
    EXPECT_THROW((void)RenderFeatureDashboardHtml(doc, pdb, "A", 0), std::invalid_argument);
    ResidueTracksDocument t;
    t.sequence = "MKT";
    t.tracks.push_back({"x", {1, 2, 3}, false});
    EXPECT_THROW((void)StructurePanelHtml(pdb, "A", t, {}, "Bad Id"), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
