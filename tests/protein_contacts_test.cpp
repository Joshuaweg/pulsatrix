// PLM-4: ESM's contact head against transformers on the tiny random ESM-2 in
// tests/fixtures/hf_tiny/esm (and on downloaded ESM-2 models with PULSATRIX_GOLDEN_DIR), the
// top-K head average, head ranking, and precision at L.
#include "pulsatrix/protein_contacts.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string TinyEsm() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm"; }

const char* const kUbiquitin = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

TEST(ContactFeatures, SymmetrizesTheResidueBlockAndSubtractsTheAverageProduct) {
    // <cls>, two residues, <eos>: only the middle 2 x 2 block counts.
    const std::vector<float> a = {9, 9, 9, 9,  //
                                  9, 1, 2, 9,  //
                                  9, 4, 3, 9,  //
                                  9, 9, 9, 9};
    const ContactMap f = ContactFeatures(a.data(), 4);
    // Symmetric block {{2, 6}, {6, 6}}: row sums 8 and 12, total 20.
    ASSERT_EQ(f.length, 2);
    EXPECT_NEAR(f.at(0, 0), 2 - 8.0 * 8 / 20, 1e-6);
    EXPECT_NEAR(f.at(0, 1), 6 - 8.0 * 12 / 20, 1e-6);
    EXPECT_NEAR(f.at(1, 0), f.at(0, 1), 1e-6);
    EXPECT_NEAR(f.at(1, 1), 6 - 12.0 * 12 / 20, 1e-6);
    EXPECT_THROW((void)ContactFeatures(a.data(), 2), std::invalid_argument);
}

TEST(EsmContactHead, LoadsOneWeightPerHead) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const EsmContactHead head = LoadEsmContactHead(TinyEsm(), model->config());
    EXPECT_EQ(head.layers, 2);
    EXPECT_EQ(head.heads, 4);
    EXPECT_EQ(head.weight.size(), 8u);
    EncoderLMConfig wrong = model->config();
    wrong.num_attention_heads = 2;
    EXPECT_THROW((void)LoadEsmContactHead(TinyEsm(), wrong), std::invalid_argument);
}

/** @brief Every contacts.<k> in esm_golden.safetensors, from transformers' predict_contacts. */
void ExpectContactsMatchGolden(const std::string& dir, double tolerance) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(dir, &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(dir + "/vocab.txt");
    const EsmContactHead head = LoadEsmContactHead(dir, model->config());
    ContactPredictor predictor(*model, tok, &cpu);
    const SafetensorsFile golden = SafetensorsFile::Map(dir + "/esm_golden.safetensors");
    const JsonValue sequences = ParseJson(golden.metadata().at("sequences"));
    int checked = 0;
    for (size_t k = 0; k < sequences.as_array().size(); ++k) {
        const std::string name = "contacts." + std::to_string(k);
        if (!golden.contains(name)) continue;
        const ContactMap got = predictor.predict(sequences.as_array()[k].as_string(), head);
        const std::vector<float> want = golden.tensor(name, &cpu).to_host_vector();
        ASSERT_EQ(got.values.size(), want.size()) << dir << " " << name;
        for (size_t i = 0; i < want.size(); ++i) EXPECT_NEAR(got.values[i], want[i], tolerance) << dir << " " << name << " " << i;
        ++checked;
    }
    EXPECT_GE(checked, 2) << dir;
    // Predicting leaves the model as it found it: keeping activations, with no observer.
    EXPECT_TRUE(model->keep_activations());
    (void)model->forward(Tensor(Shape({1, 3}), &cpu, std::vector<float>{0, 5, 2}));
    EXPECT_EQ(model->hidden_states().size(), 3u);
}

TEST(ContactPredictor, TinyEsmMatchesTransformersPredictContacts) { ExpectContactsMatchGolden(TinyEsm(), 1e-5); }

TEST(ContactPredictor, HeadAverageAndContactHeadAgreeWithEachHeadsFeatures) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    const EsmContactHead head = LoadEsmContactHead(TinyEsm(), model->config());
    ContactPredictor predictor(*model, tok, &cpu);
    const std::string seq = "MKTAYIAKQRQISFVKSHFSRQ";
    std::vector<ContactMap> features;
    predictor.for_each_head(seq, [&](AttentionHead h, const ContactMap& f) {
        EXPECT_EQ(h.layer * 4 + h.head, static_cast<int64_t>(features.size()));
        features.push_back(f);
    });
    ASSERT_EQ(features.size(), 8u);

    const ContactMap probs = predictor.predict(seq, head);
    const ContactMap one = predictor.average_heads(seq, {{1, 2}});
    const ContactMap two = predictor.average_heads(seq, {{0, 1}, {1, 3}});
    for (size_t k = 0; k < probs.values.size(); ++k) {
        double logit = head.bias;
        for (size_t c = 0; c < 8; ++c) logit += head.weight[c] * features[c].values[k];
        EXPECT_NEAR(probs.values[k], 1.0 / (1.0 + std::exp(-logit)), 1e-6) << k;
        EXPECT_FLOAT_EQ(one.values[k], features[6].values[k]) << k;
        EXPECT_NEAR(two.values[k], (features[1].values[k] + features[7].values[k]) / 2, 1e-6) << k;
    }
    // ESM's maps are symmetric.
    EXPECT_FLOAT_EQ(probs.at(3, 11), probs.at(11, 3));

    EXPECT_THROW((void)predictor.average_heads(seq, {}), std::invalid_argument);
    EXPECT_THROW((void)predictor.average_heads(seq, {{2, 0}}), std::invalid_argument);
    EXPECT_THROW((void)predictor.average_heads(seq, {{0, 4}}), std::invalid_argument);
    EXPECT_THROW((void)predictor.predict("", head), std::invalid_argument);
    EXPECT_THROW((void)predictor.predict("MK1", head), std::invalid_argument);
    EsmContactHead short_head = head;
    short_head.weight.pop_back();
    EXPECT_THROW((void)predictor.predict(seq, short_head), std::invalid_argument);
}

TEST(ContactPredictor, RefusesAPassOverItsMemoryBudget) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    ContactOptions tight;
    tight.max_pass_bytes = 1024;
    ContactPredictor predictor(*model, tok, &cpu, tight);
    EXPECT_THROW((void)predictor.average_heads(kUbiquitin, {{0, 0}}), std::invalid_argument);
    EXPECT_TRUE(model->keep_activations());
}

ContactMap Square(int64_t L, float value) { return {L, std::vector<float>(static_cast<size_t>(L * L), value)}; }

TEST(ContactPrecision, CountsTheTopPairsInRange) {
    const int64_t L = 40;
    ContactMap pred = Square(L, 0.0f), truth = Square(L, 0.0f);
    auto set = [&](ContactMap& m, int64_t i, int64_t j, float v) {
        m.values[static_cast<size_t>(i * L + j)] = v;
        m.values[static_cast<size_t>(j * L + i)] = v;
    };
    // Long range: the three best pairs, two of them contacts.
    set(pred, 0, 30, 0.9f);
    set(truth, 0, 30, 1.0f);
    set(pred, 1, 35, 0.8f);
    set(pred, 2, 39, 0.7f);
    set(truth, 2, 39, 1.0f);
    // Short range: a contact the long range never sees, and one too close to count.
    set(pred, 5, 12, 0.95f);
    set(truth, 5, 12, 1.0f);
    set(pred, 7, 10, 0.99f);
    set(truth, 7, 10, 1.0f);

    EXPECT_DOUBLE_EQ(ContactPrecision(pred, truth, kLongRange, 1), 1.0);
    EXPECT_DOUBLE_EQ(ContactPrecision(pred, truth, kLongRange, 2), 0.5);
    EXPECT_DOUBLE_EQ(ContactPrecision(pred, truth, kLongRange, 3), 2.0 / 3);
    EXPECT_DOUBLE_EQ(ContactPrecision(pred, truth, kShortRange, 1), 1.0);
    // Pairs closer than 6 apart never count, whatever their score.
    EXPECT_DOUBLE_EQ(ContactPrecision(pred, truth, {6, 0}, 1), 1.0);
    EXPECT_DOUBLE_EQ(ContactPrecision(pred, truth, {3, 0}, 1), 1.0);  // (7, 10) is now in range
    EXPECT_TRUE(std::isnan(ContactPrecision(pred, truth, kLongRange, 0)));

    // An unknown truth drops its pair from the competition: (1, 35) no longer takes a slot.
    set(truth, 1, 35, std::numeric_limits<float>::quiet_NaN());
    EXPECT_DOUBLE_EQ(ContactPrecision(pred, truth, kLongRange, 2), 1.0);

    const ContactPrecisions p = PrecisionAtL(pred, truth, kLongRange);
    EXPECT_DOUBLE_EQ(p.at_l5, 2.0 / 8);  // the top 8, two of them contacts
    EXPECT_DOUBLE_EQ(p.at_l2, 2.0 / 20);
    EXPECT_DOUBLE_EQ(p.at_l, 2.0 / 40);
    const ContactEvaluation e = EvaluateContacts(pred, truth);
    EXPECT_DOUBLE_EQ(e.long_range.at_l5, p.at_l5);
    EXPECT_DOUBLE_EQ(e.short_range.at_l5, 1.0 / 8);
    EXPECT_DOUBLE_EQ(e.medium_range.at_l5, 0.0);

    EXPECT_THROW((void)ContactPrecision(pred, Square(L - 1, 0.0f), kLongRange, 1), std::invalid_argument);
    EXPECT_THROW((void)ContactPrecision(pred, truth, kLongRange, -1), std::invalid_argument);
}

TEST(ContactPrecision, CountsMissingPairsAsWrongAndBreaksTiesInOrder) {
    // L = 8 has no pair 24 apart: every slot is missing.
    EXPECT_DOUBLE_EQ(ContactPrecision(Square(8, 1.0f), Square(8, 1.0f), kLongRange, 8), 0.0);
    // Pairs 6 apart in L = 8: (0, 6), (0, 7), (1, 7). All score the same, so (0, 6) comes first.
    ContactMap truth = Square(8, 0.0f);
    truth.values[0 * 8 + 6] = 1.0f;
    EXPECT_DOUBLE_EQ(ContactPrecision(Square(8, 0.5f), truth, {6, 0}, 1), 1.0);
    EXPECT_DOUBLE_EQ(ContactPrecision(Square(8, 0.5f), truth, {6, 0}, 4), 0.25);
}

TEST(RankContactHeads, PutsTheHeadThatDefinesTheContactsFirst) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    ContactPredictor predictor(*model, tok, &cpu);
    // Contacts: head 1.2's top long-range pairs.
    const int64_t L = static_cast<int64_t>(std::string(kUbiquitin).size());
    const ContactMap f = predictor.average_heads(kUbiquitin, {{1, 2}});
    std::vector<std::pair<float, int64_t>> pairs;
    for (int64_t i = 0; i < L; ++i) {
        for (int64_t j = i + 24; j < L; ++j) pairs.emplace_back(f.at(i, j), i * L + j);
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    ContactMap truth = Square(L, 0.0f);
    for (int64_t n = 0; n < L; ++n) truth.values[static_cast<size_t>(pairs[static_cast<size_t>(n)].second)] = 1.0f;

    const std::vector<HeadPrecision> ranked = RankContactHeads(predictor, {{kUbiquitin, truth}});
    ASSERT_EQ(ranked.size(), 8u);
    EXPECT_EQ(ranked[0].head.layer, 1);
    EXPECT_EQ(ranked[0].head.head, 2);
    EXPECT_DOUBLE_EQ(ranked[0].precision, 1.0);
    for (size_t k = 1; k < ranked.size(); ++k) EXPECT_LE(ranked[k].precision, ranked[k - 1].precision);

    EXPECT_THROW((void)RankContactHeads(predictor, {}), std::invalid_argument);
    EXPECT_THROW((void)RankContactHeads(predictor, {{"MKT", truth}}), std::invalid_argument);
}

/** @brief A downloaded model's contacts_golden.safetensors: its predictions and precision. */
void ExpectStructureContactsMatchGolden(const std::string& dir, const std::string& structures) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(dir, &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(dir + "/vocab.txt");
    const EsmContactHead head = LoadEsmContactHead(dir, model->config());
    ContactPredictor predictor(*model, tok, &cpu);
    const SafetensorsFile golden = SafetensorsFile::Map(dir + "/contacts_golden.safetensors");
    const JsonValue proteins = ParseJson(golden.metadata().at("proteins"));
    for (const JsonValue& protein : proteins.as_array()) {
        const std::string id = protein.as_string(), entry = id.substr(0, id.find(':')), chain_id = id.substr(id.find(':') + 1);
        const std::string key = entry + "." + chain_id;
        const StructureChain chain = ReadStructure(structures + "/" + entry + ".cif").chain(chain_id);
        const ContactMap truth = TrueContacts(chain);
        const ContactMap got = predictor.predict(chain.sequence(), head);
        const std::vector<float> want = golden.tensor("contacts." + key, &cpu).to_host_vector();
        ASSERT_EQ(got.values.size(), want.size()) << dir << " " << key;
        double worst = 0;
        for (size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::abs(static_cast<double>(got.values[i]) - want[i]));
        EXPECT_LT(worst, 1e-4) << dir << " " << key;
        // The metric itself, on transformers' predictions, is exact; ours may swap near-ties.
        const ContactMap theirs{got.length, want};
        const std::vector<float> precision = golden.tensor("precision." + key, &cpu).to_host_vector();
        const ContactEvaluation exact = EvaluateContacts(theirs, truth), ours = EvaluateContacts(got, truth);
        const ContactPrecisions* rows[] = {&exact.short_range, &exact.medium_range, &exact.long_range};
        const ContactPrecisions* our_rows[] = {&ours.short_range, &ours.medium_range, &ours.long_range};
        for (size_t r = 0; r < 3; ++r) {
            const double e[] = {rows[r]->at_l, rows[r]->at_l2, rows[r]->at_l5}, o[] = {our_rows[r]->at_l, our_rows[r]->at_l2, our_rows[r]->at_l5};
            for (size_t c = 0; c < 3; ++c) {
                EXPECT_NEAR(e[c], precision[r * 3 + c], 1e-6) << dir << " " << key << " range " << r << " column " << c;
                EXPECT_NEAR(o[c], precision[r * 3 + c], 2.0 / static_cast<double>(got.length)) << dir << " " << key;
            }
        }
    }
}

TEST(ContactPredictor, RealModelsInPulsatrixGoldenDirMatchTransformersOnStructures) {
    const char* dir = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set PULSATRIX_GOLDEN_DIR to check downloaded ESM-2 models";
    int checked = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!std::filesystem::exists(entry.path() / "contacts_golden.safetensors")) continue;
        ExpectStructureContactsMatchGolden(entry.path().string(), std::string(dir) + "/structures");
        ++checked;
    }
    if (checked == 0) GTEST_SKIP() << "no ESM model with a contacts_golden.safetensors in " << dir;
}

}  // namespace
}  // namespace pulsatrix
