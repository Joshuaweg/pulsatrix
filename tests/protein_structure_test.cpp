// PLM-4: PDB and mmCIF reading, residue distances and true contacts, against biotite's reading of
// crambin (tests/fixtures/structures) and hand-made files with the awkward cases.
#include "pulsatrix/protein_structure.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::string Fixtures() { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/structures"; }

/** @brief Every chain of @p dir's structures_golden.safetensors, read from both formats. */
int ExpectMatchesBiotite(const std::string& dir) {
    CPUBackend cpu;
    const SafetensorsFile golden = SafetensorsFile::Map(dir + "/structures_golden.safetensors");
    const JsonValue sequences = ParseJson(golden.metadata().at("sequences"));
    int checked = 0;
    for (const auto& [key, sequence] : sequences.as_object()) {
        const std::string entry = key.substr(0, key.find('.')), chain_id = key.substr(key.find('.') + 1);
        for (const std::string ext : {".cif", ".pdb"}) {
            const ProteinStructure s = ReadStructure(dir + "/" + entry + ext);
            const StructureChain& chain = s.chain(chain_id);
            EXPECT_EQ(chain.sequence(), sequence.as_string()) << key << ext;
            if (chain.sequence() != sequence.as_string()) continue;
            for (const auto& [kind, atom] : {std::pair<std::string, ContactAtom>{"beta", ContactAtom::Beta},
                                             {"alpha", ContactAtom::Alpha}, {"virtual", ContactAtom::VirtualBeta}}) {
                const std::vector<float> want = golden.tensor(kind + "." + key, &cpu).to_host_vector();
                const ContactMap got = ResidueDistances(chain, atom);
                EXPECT_EQ(got.values.size(), want.size()) << key << ext << " " << kind;
                if (got.values.size() != want.size()) continue;
                for (size_t k = 0; k < want.size(); ++k) {
                    if (std::isnan(want[k])) {
                        EXPECT_TRUE(std::isnan(got.values[k])) << key << ext << " " << kind << " " << k;
                    } else {
                        EXPECT_NEAR(got.values[k], want[k], 1e-4) << key << ext << " " << kind << " " << k;
                    }
                }
            }
            ++checked;
        }
    }
    return checked;
}

TEST(ProteinStructure, CrambinMatchesBiotiteInBothFormats) { EXPECT_EQ(ExpectMatchesBiotite(Fixtures()), 2); }

TEST(ProteinStructure, DownloadedStructuresInPulsatrixGoldenDirMatchBiotite) {
    const char* dir = std::getenv("PULSATRIX_GOLDEN_DIR");
    if (dir == nullptr) GTEST_SKIP() << "set PULSATRIX_GOLDEN_DIR to check downloaded structures";
    const std::string structures = std::string(dir) + "/structures";
    if (!std::filesystem::exists(structures + "/structures_golden.safetensors")) GTEST_SKIP() << "no structures_golden in " << structures;
    EXPECT_GT(ExpectMatchesBiotite(structures), 0);
}

// Glycine, an alternate location, an insertion code, selenomethionine, a water, a residue
// without Cα, chain B, then a second model that must be ignored.
const char* const kPdb =
    "HEADER    TEST\n"
    "ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00  0.00           N\n"
    "ATOM      2  CA  GLY A   1       1.000   0.000   0.000  1.00  0.00           C\n"
    "ATOM      3  C   GLY A   1       1.500   1.000   0.000  1.00  0.00           C\n"
    "ATOM      4  N   ALA A   2       2.500   1.000   0.000  1.00  0.00           N\n"
    "ATOM      5  CA AALA A   2       3.000   2.000   0.000  0.60  0.00           C\n"
    "ATOM      6  CA BALA A   2       9.000   9.000   9.000  0.40  0.00           C\n"
    "ATOM      7  C   ALA A   2       4.000   2.000   1.000  1.00  0.00           C\n"
    "ATOM      8  CB  ALA A   2       3.000   3.000  -1.000  1.00  0.00           C\n"
    "ATOM      9  CA  SER A   2A      6.000   2.000   0.000  1.00  0.00           C\n"
    "ATOM     10  CB  SER A   2A      6.000   3.000   0.000  1.00  0.00           C\n"
    "HETATM   11  CA  MSE A   3       9.000   2.000   0.000  1.00  0.00           C\n"
    "HETATM   12  CB  MSE A   3       9.000   3.000   0.000  1.00  0.00           C\n"
    "ATOM     13  N   LYS A   4      12.000   2.000   0.000  1.00  0.00           N\n"
    "TER\n"
    "HETATM   14  O   HOH A 101      20.000  20.000  20.000  1.00  0.00           O\n"
    "ATOM     15  CA  TRP B   1       0.000   5.000   0.000  1.00  0.00           C\n"
    "ENDMDL\n"
    "MODEL        2\n"
    "ATOM     16  CA  VAL A   9      50.000  50.000  50.000  1.00  0.00           C\n"
    "ENDMDL\n";

const char* const kMmcif =
    "data_TEST\n"
    "_struct.title 'a title, with a quote''s apostrophe'\n"
    "_struct.pdbx_descriptor\n"
    ";a text field\n"
    "over two lines\n"
    ";\n"
    "loop_\n"
    "_entity.id\n"
    "_entity.type\n"
    "1 polymer\n"
    "#\n"
    "loop_\n"
    "_atom_site.group_PDB\n"
    "_atom_site.id\n"
    "_atom_site.label_atom_id\n"
    "_atom_site.label_alt_id\n"
    "_atom_site.label_comp_id\n"
    "_atom_site.label_asym_id\n"
    "_atom_site.label_seq_id\n"
    "_atom_site.pdbx_PDB_ins_code\n"
    "_atom_site.Cartn_x\n"
    "_atom_site.Cartn_y\n"
    "_atom_site.Cartn_z\n"
    "_atom_site.auth_seq_id\n"
    "_atom_site.auth_comp_id\n"
    "_atom_site.auth_asym_id\n"
    "_atom_site.auth_atom_id\n"
    "_atom_site.pdbx_PDB_model_num\n"
    "ATOM 1 N . GLY C 1 ? 0.000 0.000 0.000 1 GLY A N 1\n"
    "ATOM 2 CA . GLY C 1 ? 1.000 0.000 0.000 1 GLY A CA 1\n"
    "ATOM 3 C . GLY C 1 ? 1.500 1.000 0.000 1 GLY A C 1\n"
    "ATOM 4 N . ALA C 2 ? 2.500 1.000 0.000 2 ALA A N 1\n"
    "ATOM 5 CA A ALA C 2 ? 3.000 2.000 0.000 2 ALA A CA 1\n"
    "ATOM 6 CA B ALA C 2 ? 9.000 9.000 9.000 2 ALA A CA 1\n"
    "ATOM 7 C . ALA C 2 ? 4.000 2.000 1.000 2 ALA A C 1\n"
    "ATOM 8 CB . ALA C 2 ? 3.000 3.000 -1.000 2 ALA A CB 1\n"
    "ATOM 9 CA . SER C 3 A 6.000 2.000 0.000 2 SER A CA 1\n"
    "ATOM 10 CB . SER C 3 A 6.000 3.000 0.000 2 SER A CB 1\n"
    "HETATM 11 CA . MSE C 4 ? 9.000 2.000 0.000 3 MSE A CA 1\n"
    "HETATM 12 CB . MSE C 4 ? 9.000 3.000 0.000 3 MSE A CB 1\n"
    "ATOM 13 N . LYS C 5 ? 12.000 2.000 0.000 4 LYS A N 1\n"
    "HETATM 14 O . HOH D . ? 20.000 20.000 20.000 101 HOH A O 1\n"
    "ATOM 15 CA . TRP E 1 ? 0.000 5.000 0.000 1 TRP B CA 1\n"
    "ATOM 16 CA . VAL C 9 ? 50.000 50.000 50.000 9 VAL A CA 2\n"
    "#\n";

void ExpectHandMadeStructure(const ProteinStructure& s) {
    ASSERT_EQ(s.chains.size(), 2u);
    const StructureChain& a = s.chain("A");
    // LYS has no Cα and HOH is a water: both left out. The second model's VAL is ignored.
    EXPECT_EQ(a.sequence(), "GASM");
    EXPECT_EQ(s.chain("B").sequence(), "W");
    ASSERT_EQ(a.residues.size(), 4u);
    EXPECT_EQ(a.residues[1].number, 2);
    EXPECT_EQ(a.residues[1].insertion_code, ' ');
    EXPECT_EQ(a.residues[2].number, 2);
    EXPECT_EQ(a.residues[2].insertion_code, 'A');
    EXPECT_EQ(a.residues[3].name, "MSE");
    // The first alternate location wins.
    EXPECT_DOUBLE_EQ(a.residues[1].atom("CA")->x, 3.0);
    EXPECT_FALSE(a.residues[2].atom("N").has_value());
    EXPECT_THROW((void)s.chain("Z"), std::invalid_argument);

    // Glycine's Cβ distance uses its Cα; serine's virtual Cβ needs its missing N and C.
    const ContactMap beta = ResidueDistances(a, ContactAtom::Beta);
    EXPECT_NEAR(beta.at(0, 1), std::sqrt(2.0 * 2.0 + 3.0 * 3.0 + 1.0), 1e-6);
    EXPECT_NEAR(beta.at(2, 3), 3.0, 1e-6);
    EXPECT_EQ(beta.at(1, 1), 0.0f);
    const ContactMap virt = ResidueDistances(a, ContactAtom::VirtualBeta);
    EXPECT_FALSE(std::isnan(virt.at(0, 1)));
    EXPECT_TRUE(std::isnan(virt.at(0, 2)));
    EXPECT_TRUE(std::isnan(virt.at(2, 2)));

    const ContactMap contacts = TrueContacts(a, 3.5);
    EXPECT_EQ(contacts.at(2, 3), 1.0f);
    EXPECT_EQ(contacts.at(0, 3), 0.0f);
    EXPECT_TRUE(std::isnan(TrueContacts(a, 8.0, ContactAtom::VirtualBeta).at(0, 2)));
}

TEST(ProteinStructure, PdbKeepsProteinResiduesOfTheFirstModel) { ExpectHandMadeStructure(ParsePdb(kPdb)); }

TEST(ProteinStructure, MmcifReadsTheAtomSiteLoopWithAuthorNumbering) { ExpectHandMadeStructure(ParseMmcif(kMmcif)); }

TEST(ProteinStructure, RejectsWhatItCantRead) {
    EXPECT_THROW((void)ParsePdb("HEADER only\n"), std::invalid_argument);
    EXPECT_THROW((void)ParsePdb("ATOM      2  CA  GLY A   1       1.0x0   0.000   0.000  1.00  0.00           C\n"),
                 std::invalid_argument);
    EXPECT_THROW((void)ParseMmcif("data_X\n_entry.id X\n"), std::invalid_argument);
    EXPECT_THROW((void)ParseMmcif("data_X\nloop_\n_atom_site.group_PDB\n_atom_site.Cartn_x\nATOM\n"), std::invalid_argument);
    EXPECT_THROW((void)ParseMmcif("data_X\n_struct.title 'open\n"), std::invalid_argument);
    EXPECT_THROW((void)ParseMmcif("data_X\n_struct.title\n;never closed\n"), std::invalid_argument);
    EXPECT_THROW((void)ReadStructure(Fixtures() + "/missing.pdb"), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
