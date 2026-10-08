// PLM-5: the protein documents (mutation map, sequence logo, contact map, residue tracks), their
// builders from the tiny random ESM-2, their SVG figures (golden files in
// tests/fixtures/viz/svg, checked by eye) and HTML pages, and the 3D structure page.
#include "pulsatrix/viz/protein_views.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/protein_contacts.hpp"
#include "pulsatrix/protein_sequences.hpp"

namespace pulsatrix {
namespace {

std::string ReadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string Fixture(const std::string& name) { return ReadFile(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/" + name); }
std::string TinyEsm() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/esm"; }

size_t Count(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (size_t at = haystack.find(needle); at != std::string::npos; at = haystack.find(needle, at + needle.size())) ++n;
    return n;
}

/** @brief Equal, with NaN equal to NaN. */
void ExpectSameValues(const std::vector<float>& a, const std::vector<float>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i])) {
            EXPECT_TRUE(std::isnan(b[i])) << i;
        } else {
            EXPECT_EQ(a[i], b[i]) << i;
        }
    }
}

// ---- documents -------------------------------------------------------------------------------

TEST(ProteinDocuments, FixturesRoundTrip) {
    const MutationMapDocument m = ParseMutationMapDocument(Fixture("mutation_map.v1.json"));
    EXPECT_EQ(m.first_position, 24);
    EXPECT_TRUE(std::isnan(m.values[65]));
    const MutationMapDocument m2 = ParseMutationMapDocument(ToJson(m));
    EXPECT_EQ(m2.sequence, m.sequence);
    ExpectSameValues(m2.values, m.values);

    const SequenceLogoDocument l = ParseSequenceLogoDocument(Fixture("sequence_logo.v1.json"));
    EXPECT_EQ(l.positions(), 14);
    ExpectSameValues(ParseSequenceLogoDocument(ToJson(l)).probabilities, l.probabilities);

    const ContactMapDocument c = ParseContactMapDocument(Fixture("contact_map.v1.json"));
    EXPECT_EQ(c.length(), 30);
    EXPECT_TRUE(std::isnan(c.truth[5 * 30 + 9]));
    ExpectSameValues(ParseContactMapDocument(ToJson(c)).truth, c.truth);

    const ResidueTracksDocument t = ParseResidueTracksDocument(Fixture("residue_tracks.v1.json"));
    ASSERT_EQ(t.tracks.size(), 2u);
    EXPECT_TRUE(t.tracks[0].is_signed);
    EXPECT_FALSE(t.tracks[1].is_signed);
    ASSERT_EQ(t.features.size(), 3u);
    EXPECT_EQ(t.features[1].category, "secondary structure");
    const ResidueTracksDocument t2 = ParseResidueTracksDocument(ToJson(t));
    ExpectSameValues(t2.tracks[1].values, t.tracks[1].values);
    EXPECT_EQ(t2.residue_ids, t.residue_ids);
    EXPECT_EQ(t2.features[2].start, 122);
}

TEST(ProteinDocuments, ReadersNameTheProblem) {
    auto expect_error = [](auto parse, const std::string& json, const std::string& pointer) {
        try {
            (void)parse(json);
            ADD_FAILURE() << "accepted " << json;
        } catch (const std::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find(pointer), std::string::npos) << e.what();
        }
    };
    expect_error(ParseMutationMapDocument,
                 R"({"schema":"pulsatrix.mutation_map.v1","title":"","sequence":"MK","first_position":1,"method":"","alphabet":"AC","values":[1,2,3]})",
                 "/values");
    expect_error(ParseMutationMapDocument,
                 R"({"schema":"pulsatrix.mutation_map.v1","title":"","sequence":"M","first_position":1,"method":"","alphabet":"AA","values":[1,2]})",
                 "/alphabet");
    expect_error(ParseSequenceLogoDocument,
                 R"({"schema":"pulsatrix.sequence_logo.v1","title":"","sequence":"","first_position":1,"method":"","alphabet":"AC","probabilities":[0.5,0.4]})",
                 "/probabilities");
    expect_error(ParseSequenceLogoDocument,
                 R"({"schema":"pulsatrix.sequence_logo.v1","title":"","sequence":"","first_position":1,"method":"","alphabet":"AC","probabilities":[1.5,-0.5]})",
                 "/probabilities/0");
    expect_error(ParseContactMapDocument,
                 R"({"schema":"pulsatrix.contact_map.v1","title":"","sequence":"MK","first_position":1,"method":"","predicted":[0,1,1,0],"truth":[0,2,1,0]})",
                 "/truth/1");
    expect_error(ParseResidueTracksDocument,
                 R"({"schema":"pulsatrix.residue_tracks.v1","title":"","sequence":"MK","first_position":1,"tracks":[],"features":[{"name":"x","start":2,"end":3,"category":""}]})",
                 "/features/0");
    expect_error(ParseResidueTracksDocument,
                 R"({"schema":"pulsatrix.residue_tracks.v1","title":"","sequence":"MK","first_position":1,"tracks":[{"name":"t","values":[1]}],"features":[]})",
                 "/tracks/0/values");
    expect_error(ParseContactMapDocument,
                 R"({"schema":"pulsatrix.contact_map.v1","title":"","sequence":"M","first_position":9000000000000000000,"method":"","predicted":[1],"truth":[]})",
                 "/first_position");
}

TEST(ProteinDocuments, BuildersReadTheModelsOutputs) {
    CPUBackend cpu;
    std::unique_ptr<EncoderLM> model = LoadEncoderLM(TinyEsm(), &cpu);
    const TextTokenizer tok = LoadEsmTokenizer(TinyEsm() + "/vocab.txt");
    VariantScorer scorer(*model, tok, &cpu);
    const std::string seq = "MKTAYIAKQR";
    const ResidueLogProbs marginals = scorer.masked_marginals(seq);

    const MutationMapDocument m = MakeMutationMapDocument(seq, scorer.single_mutant_scan(marginals, seq), "masked_marginals", 5);
    EXPECT_EQ(m.first_position, 5);
    EXPECT_EQ(m.values.size(), 200u);
    const std::vector<float> mean = MeanSubstitutionScore(m);
    double want = 0;
    for (size_t a = 0; a < 20; ++a) want += kAminoAcids[a] == 'K' ? 0.0 : m.values[20 + a];
    EXPECT_NEAR(mean[1], want / 19, 1e-6);
    EXPECT_THROW((void)MakeMutationMapDocument(seq, std::vector<float>(199), ""), std::invalid_argument);

    const SequenceLogoDocument l = MakeSequenceLogoDocument(marginals, tok, seq);
    EXPECT_EQ(l.positions(), 10);
    for (int64_t i = 0; i < 10; ++i) {
        double sum = 0;
        for (int64_t a = 0; a < 20; ++a) sum += l.probabilities[static_cast<size_t>(i * 20 + a)];
        EXPECT_NEAR(sum, 1.0, 1e-5);
    }
    // Renormalized over the amino acids: the ratio of two stays the model's.
    const double ratio = std::exp(marginals.at(0, *tok.token_to_id("A")) - marginals.at(0, *tok.token_to_id("C")));
    EXPECT_NEAR(l.probabilities[0] / l.probabilities[1], ratio, 1e-4 * ratio);
    for (float bits : InformationContent(l)) {
        EXPECT_GE(bits, 0.0f);
        EXPECT_LE(bits, std::log2(20.0f) + 1e-5f);
    }
    EXPECT_THROW((void)MakeSequenceLogoDocument(marginals, tok, "MKT"), std::invalid_argument);

    const ContactMap predicted = ContactPredictor(*model, tok, &cpu).predict(seq, LoadEsmContactHead(TinyEsm(), model->config()));
    ContactMap truth{10, std::vector<float>(100, 0.0f)};
    truth.values[3] = std::numeric_limits<float>::quiet_NaN();
    const ContactMapDocument c = MakeContactMapDocument(seq, predicted, &truth, "contact head");
    EXPECT_EQ(c.predicted, predicted.values);
    EXPECT_NO_THROW((void)ToJson(c));
    ContactMap short_truth{9, std::vector<float>(81, 0.0f)};
    EXPECT_THROW((void)MakeContactMapDocument(seq, predicted, &short_truth), std::invalid_argument);
    ContactMapDocument bad = c;
    bad.truth[0] = 0.5f;
    EXPECT_THROW((void)ToJson(bad), std::invalid_argument);
}

TEST(ProteinDocuments, InformationContentRunsFromUniformToCertain) {
    SequenceLogoDocument l;
    l.probabilities.assign(40, 0.05f);  // uniform
    l.probabilities[20] = 0.0f;
    for (size_t a = 21; a < 40; ++a) l.probabilities[a] = 0.0f;
    l.probabilities[25] = 1.0f;  // certain
    const std::vector<float> bits = InformationContent(l);
    EXPECT_NEAR(bits[0], 0.0, 1e-5);
    EXPECT_NEAR(bits[1], std::log2(20.0), 1e-5);
}

TEST(ProteinDocuments, ResidueIdsCarryInsertionCodes) {
    const ProteinStructure s = ParsePdb(
        "ATOM      1  CA  GLY A  52       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A  52A      2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  SER A  53       3.000   0.000   0.000  1.00  0.00           C\n");
    EXPECT_EQ(ResidueIds(s.chain("A")), (std::vector<std::string>{"52", "52A", "53"}));
    ResidueTracksDocument t;
    t.sequence = "GAS";
    t.residue_ids = {"52", "52A"};
    EXPECT_THROW((void)ToJson(t), std::invalid_argument);
}

// ---- figures ---------------------------------------------------------------------------------

// Each figure must match tests/fixtures/viz/svg/ byte for byte. After an intended change, rerun
// with PULSATRIX_UPDATE_GOLDEN=1, then look at the new figures before committing.
void ExpectGolden(const std::string& name, const std::string& svg) {
    const std::string path = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/viz/svg/" + name;
    const char* update = std::getenv("PULSATRIX_UPDATE_GOLDEN");
    if (update != nullptr && std::string(update) == "1") {
        std::ofstream(path, std::ios::binary) << svg;
        return;
    }
    const std::string golden = ReadFile(path);
    ASSERT_FALSE(golden.empty()) << "missing golden file " << path;
    EXPECT_EQ(svg, golden) << name << " changed; see the comment above ExpectGolden";
}

TEST(ProteinViewsGolden, EveryFigure) {
    SvgOptions narrow;
    narrow.width = 420;
    ExpectGolden("mutation_map.svg", RenderMutationMapSvg(ParseMutationMapDocument(Fixture("mutation_map.v1.json")), narrow));
    ExpectGolden("sequence_logo.svg", RenderSequenceLogoSvg(ParseSequenceLogoDocument(Fixture("sequence_logo.v1.json"))));
    const ContactMapDocument c = ParseContactMapDocument(Fixture("contact_map.v1.json"));
    ExpectGolden("contact_map.svg", RenderContactMapSvg(c, {}, narrow));
    ContactMapDocument predicted_only = c;
    predicted_only.truth.clear();
    ExpectGolden("contact_map_predicted.svg", RenderContactMapSvg(predicted_only, {12}, narrow));
    ExpectGolden("residue_tracks.svg", RenderResidueTracksSvg(ParseResidueTracksDocument(Fixture("residue_tracks.v1.json")), narrow));
}

TEST(ProteinViews, MutationMapMarksTheWildTypeAndWrapsLongSequences) {
    const MutationMapDocument m = ParseMutationMapDocument(Fixture("mutation_map.v1.json"));
    const std::string svg = RenderMutationMapSvg(m);
    EXPECT_EQ(Count(svg, "class=\"cell\""), 12u * 20u);
    EXPECT_EQ(Count(svg, "class=\"wild-type\""), 12u);
    EXPECT_NE(svg.find("K25R: "), std::string::npos);        // numbered from first_position
    EXPECT_NE(svg.find("A27G: not scored"), std::string::npos);  // the NaN cell
    EXPECT_NE(svg.find("K25 (wild type): 0"), std::string::npos);

    // 700 residues: 14,000 cells, drawn as images, one per block.
    MutationMapDocument big;
    big.sequence = std::string(700, 'A');
    big.values.assign(700 * 20, -1.0f);
    const std::string img = RenderMutationMapSvg(big);
    EXPECT_EQ(Count(img, "class=\"cell\""), 0u);
    EXPECT_GT(Count(img, "class=\"mutation-image\""), 1u);
    EXPECT_EQ(Count(img, "class=\"wild-type\""), 700u);

    SvgOptions tiny;
    tiny.width = 100;
    EXPECT_THROW((void)RenderMutationMapSvg(m, tiny), std::invalid_argument);
    big.values.pop_back();
    EXPECT_THROW((void)RenderMutationMapSvg(big), std::invalid_argument);
}

TEST(ProteinViews, LogoStacksLettersToTheirInformation) {
    SequenceLogoDocument l;
    l.sequence = "WA";
    l.probabilities.assign(40, 0.0f);
    l.probabilities[18] = 1.0f;  // position 1: certainly W
    for (size_t a = 20; a < 40; ++a) l.probabilities[a] = 0.05f;  // position 2: uniform, 0 bits
    const std::string svg = RenderSequenceLogoSvg(l);
    // One letter, the full height: 7 font sizes (84 px) over the glyph's 71.6-unit capitals.
    ASSERT_EQ(Count(svg, "class=\"glyph\""), 1u);
    EXPECT_NE(svg.find(">W</text>"), std::string::npos);
    EXPECT_NE(svg.find(" 0 0 " + std::string("1.1732") + " "), std::string::npos) << svg;
    EXPECT_NE(svg.find("W1: W 100%"), std::string::npos);
    EXPECT_NE(svg.find("(0 bits)"), std::string::npos);
}

TEST(ProteinViews, ContactMapMarksTheTopLPairsAndReportsPrecision) {
    const ContactMapDocument c = ParseContactMapDocument(Fixture("contact_map.v1.json"));
    const std::string svg = RenderContactMapSvg(c);
    EXPECT_EQ(Count(svg, "class=\"top-pair\""), 30u);
    // Dots are colored by the truth: as many blue as ContactPrecision counts hits.
    const ContactMap predicted{30, c.predicted}, truth{30, c.truth};
    const double p = ContactPrecision(predicted, truth, {6, 0}, 30);
    EXPECT_EQ(Count(svg, "fill=\"#2166ac\""), static_cast<size_t>(std::lround(p * 30)));
    EXPECT_NE(svg.find("Precision at L: " + std::string(p == 0 ? "0" : "")), std::string::npos);
    EXPECT_NE(svg.find("T3-L27 (#"), std::string::npos);

    // A long sequence is drawn as an image.
    ContactMapDocument big;
    big.sequence = std::string(120, 'A');
    big.predicted.assign(120 * 120, 0.5f);
    const std::string img = RenderContactMapSvg(big);
    EXPECT_EQ(Count(img, "class=\"contact-image\""), 1u);
    EXPECT_EQ(Count(img, "class=\"top-pair\""), 120u);
    EXPECT_EQ(img.find("Precision at L"), std::string::npos);  // no structure, no precision

    EXPECT_THROW((void)RenderContactMapSvg(c, {0}), std::invalid_argument);
}

TEST(ProteinViews, TracksDrawEveryResidueAndClipFeaturesToBlocks) {
    const ResidueTracksDocument t = ParseResidueTracksDocument(Fixture("residue_tracks.v1.json"));
    SvgOptions narrow;
    narrow.width = 300;  // a few residues per block
    const std::string svg = RenderResidueTracksSvg(t, narrow);
    EXPECT_EQ(Count(svg, "class=\"track-cell\""), 50u);
    EXPECT_GT(Count(svg, "class=\"feature\""), 3u);  // a feature spans blocks
    EXPECT_NE(svg.find("K108 information (bits): none"), std::string::npos);
    EXPECT_EQ(Count(svg, "class=\"legend\""), 2u);
}

TEST(ProteinViews, PagesHoldTheFigureWithoutScripts) {
    const MutationMapDocument m = ParseMutationMapDocument(Fixture("mutation_map.v1.json"));
    HtmlOptions o;
    o.title = "A <scan>";
    const std::string html = RenderMutationMapHtml(m, o);
    EXPECT_EQ(html.find("<script"), std::string::npos);
    EXPECT_EQ(html.find("<?xml"), std::string::npos);
    EXPECT_NE(html.find("<figure>\n<svg"), std::string::npos);
    EXPECT_NE(html.find("<h1>A &lt;scan&gt;</h1>"), std::string::npos);
    EXPECT_NE(RenderSequenceLogoHtml(ParseSequenceLogoDocument(Fixture("sequence_logo.v1.json"))).find("class=\"glyph\""), std::string::npos);
    EXPECT_NE(RenderContactMapHtml(ParseContactMapDocument(Fixture("contact_map.v1.json"))).find("class=\"top-pair\""), std::string::npos);
    EXPECT_NE(RenderResidueTracksHtml(ParseResidueTracksDocument(Fixture("residue_tracks.v1.json"))).find("class=\"track-cell\""),
              std::string::npos);
}

// ---- structure page --------------------------------------------------------------------------

/** @brief The JSON the structure page's script reads. */
JsonValue StructureData(const std::string& html) {
    const std::string open = "<script type=\"application/json\" id=\"structure-data\">";
    const size_t begin = html.find(open);
    if (begin == std::string::npos) throw std::runtime_error("no structure data");
    const size_t start = begin + open.size();
    return ParseJson(html.substr(start, html.find("</script>", start) - start));
}

ResidueTracksDocument CrambinTracks(const std::string& sequence) {
    ResidueTracksDocument t;
    t.sequence = sequence;
    std::vector<float> v(sequence.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i) - 20.0f;
    v[3] = std::numeric_limits<float>::quiet_NaN();
    t.tracks.push_back({"index", v, true});
    t.tracks.push_back({"constant", std::vector<float>(sequence.size(), 2.0f), false});
    return t;
}

TEST(StructurePage, EmbedsTheStructureAndAColorPerResidue) {
    const std::string cif = ReadFile(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/structures/1CRN.cif");
    const StructureChain chain = ParseMmcif(cif).chain("A");
    const std::string html = RenderStructureHtml(cif, "A", CrambinTracks(chain.sequence()));
    EXPECT_NE(html.find("https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js\" integrity=\"sha384-"), std::string::npos);
    const JsonValue data = StructureData(html);
    EXPECT_EQ(data.find("format")->as_string(), "cif");
    EXPECT_EQ(data.find("structure")->as_string(), cif);
    const JsonValue& tracks = *data.find("tracks");
    ASSERT_EQ(tracks.as_array().size(), 2u);
    // Residue 4 has no value: no color, so the page draws it gray.
    const auto& colors = tracks.as_array()[0].find("colors")->as_object();
    EXPECT_EQ(colors.size(), chain.residues.size() - 1);
    EXPECT_EQ(tracks.as_array()[0].find("colors")->find("4"), nullptr);
    EXPECT_NE(tracks.as_array()[0].find("colors")->find("1"), nullptr);
    EXPECT_EQ(tracks.as_array()[0].find("values")->find("2")->as_string(), "-19");

    // The same structure as a PDB file.
    const std::string pdb = ReadFile(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/structures/1CRN.pdb");
    EXPECT_EQ(StructureData(RenderStructureHtml(pdb, "A", CrambinTracks(chain.sequence()))).find("format")->as_string(), "pdb");
}

TEST(StructurePage, MapsResiduesByIdAndRefusesWhatDoesntFit) {
    const std::string pdb =
        "ATOM      1  CA  GLY A  52       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A  52A      2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  SER A  53       3.000   0.000   0.000  1.00  0.00           C\n";
    ResidueTracksDocument t;
    t.sequence = "AS";
    t.tracks.push_back({"x", {1.0f, 2.0f}, false});
    t.residue_ids = {"52A", "53"};
    const JsonValue data = StructureData(RenderStructureHtml(pdb, "A", t));
    const JsonValue& colors = *data.find("tracks")->as_array()[0].find("colors");
    EXPECT_NE(colors.find("52A"), nullptr);
    EXPECT_EQ(colors.find("52"), nullptr);

    t.residue_ids = {"52A", "54"};
    EXPECT_THROW((void)RenderStructureHtml(pdb, "A", t), std::invalid_argument);  // no residue 54
    t.residue_ids.clear();
    EXPECT_THROW((void)RenderStructureHtml(pdb, "A", t), std::invalid_argument);  // GAS isn't AS
    EXPECT_THROW((void)RenderStructureHtml(pdb, "B", t), std::invalid_argument);  // no chain B
    t.tracks.clear();
    EXPECT_THROW((void)RenderStructureHtml(pdb, "A", t), std::invalid_argument);  // nothing to color by
}

TEST(StructurePage, InlineScriptIsCopiedInAndEscaped) {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "pulsatrix_3dmol_test";
    std::filesystem::create_directories(dir);
    std::ofstream(dir / "3Dmol-min.js") << "var x = '</script>';";
    const std::string cif = ReadFile(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/structures/1CRN.cif");
    HtmlOptions o;
    o.scripts = HtmlScripts::Inline;
    o.script_dir = dir.string();
    const std::string html = RenderStructureHtml(cif, "A", CrambinTracks(ParseMmcif(cif).chain("A").sequence()), o);
    EXPECT_EQ(html.find("cdn.jsdelivr.net"), std::string::npos);
    EXPECT_NE(html.find("var x = '<\\/script>';"), std::string::npos);
    o.script_dir = (dir / "missing").string();
    EXPECT_THROW((void)RenderStructureHtml(cif, "A", CrambinTracks(ParseMmcif(cif).chain("A").sequence()), o), std::runtime_error);
    std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace pulsatrix
