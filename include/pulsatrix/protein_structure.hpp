/** @file protein_structure.hpp
 *  @brief Protein structures (PLM-4): PDB and mmCIF coordinates, residue distances and true
 *         contacts, to check a protein language model's predicted contacts against.
 *  @ingroup data_pipeline
 */
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace pulsatrix {

/** @brief A position in Ångström. */
struct Point3 {
    double x = 0, y = 0, z = 0;
};

/** @brief One residue of a chain and its atoms. */
struct StructureResidue {
    /** @brief The three-letter name, such as `ALA` or `MSE`. */
    std::string name;
    /** @brief The one-letter code: the 20 standard amino acids, `M` for selenomethionine (`MSE`),
     *         `U` and `O` for selenocysteine and pyrrolysine, and `X` for anything else. */
    char code = 'X';
    /** @brief The author's residue number (`auth_seq_id`), and its insertion code, or ' '. */
    int64_t number = 0;
    char insertion_code = ' ';
    /** @brief Atoms by name (`CA`, `CB`, ...). With alternate locations, the first one listed. */
    std::map<std::string, Point3> atoms;

    [[nodiscard]] std::optional<Point3> atom(const std::string& atom_name) const;
};

/** @brief A chain's residues, in file order. */
struct StructureChain {
    /** @brief The author's chain id (`auth_asym_id`), as in a PDB file. */
    std::string id;
    std::vector<StructureResidue> residues;

    /** @brief The residues' one-letter codes. Residues without coordinates in the file aren't here,
     *         so this is the observed sequence, which may skip unresolved stretches. */
    [[nodiscard]] std::string sequence() const;
};

/** @brief The protein chains of a structure's first model. */
struct ProteinStructure {
    std::vector<StructureChain> chains;

    /** @throws std::invalid_argument if there's no chain @p id. */
    [[nodiscard]] const StructureChain& chain(std::string_view id) const;
};

/**
 * @brief Reads PDB-format text: the `ATOM` records, and the `HETATM` records of selenomethionine,
 *        of the first model.
 * - A residue is kept when it has a Cα atom, so nucleic acids and stray atoms are left out.
 * - Waters, ions and ligands (other `HETATM` records) are left out.
 * - Of alternate locations, each atom's first is kept.
 * @throws std::invalid_argument for a malformed coordinate, or no protein residues.
 */
[[nodiscard]] ProteinStructure ParsePdb(std::string_view text);

/**
 * @brief Reads mmCIF text: the `_atom_site` loop, with the same choices as ParsePdb. Chains and
 *        residue numbers are the author's (`auth_asym_id`, `auth_seq_id`), as in PDB files, so
 *        both formats of one entry give the same structure.
 * @throws std::invalid_argument for malformed CIF, an `_atom_site` loop missing a needed column,
 *         or no protein residues.
 */
[[nodiscard]] ProteinStructure ParseMmcif(std::string_view text);

/**
 * @brief Reads a structure file: mmCIF for `.cif` or `.mmcif`, PDB format otherwise (`.pdb`,
 *        `.ent`).
 * @throws std::runtime_error if it can't be read; what ParsePdb or ParseMmcif throws.
 */
[[nodiscard]] ProteinStructure ReadStructure(const std::string& path);

/** @brief A square matrix over a sequence's residue pairs, row-major: predicted contact
 *         probabilities, distances, or true contacts. */
struct ContactMap {
    int64_t length = 0;
    std::vector<float> values;

    [[nodiscard]] float at(int64_t i, int64_t j) const { return values[static_cast<size_t>(i * length + j)]; }
};

/** @brief Which atom stands for a residue in residue distances. */
enum class ContactAtom {
    /** @brief Cβ, or Cα for glycine (CASP's definition). */
    Beta,
    /** @brief Cα. */
    Alpha,
    /**
     * @brief A Cβ placed from the backbone (N, Cα, C) with ideal geometry, for every residue,
     *        glycine included. ESM's contact-prediction example and trRosetta use it.
     */
    VirtualBeta,
};

/**
 * @brief Distances between the residues of @p chain, in Ångström, `L x L` for its L residues.
 *        A residue missing the atoms @p atom needs gives NaN in its row and column.
 */
[[nodiscard]] ContactMap ResidueDistances(const StructureChain& chain, ContactAtom atom = ContactAtom::Beta);

/**
 * @brief True contacts: 1 where two residues are closer than @p threshold (8 Å by convention),
 *        0 where they aren't, and NaN where a distance is unknown.
 */
[[nodiscard]] ContactMap TrueContacts(const StructureChain& chain, double threshold = 8.0, ContactAtom atom = ContactAtom::Beta);

}  // namespace pulsatrix
