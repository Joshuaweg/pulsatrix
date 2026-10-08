/** @file protein_documents.hpp
 *  @brief Viz documents for proteins (PLM-5): the mutation map, the sequence logo, the contact
 *         map and residue tracks, with builders from a protein language model's outputs.
 *  @ingroup visualization
 *
 *  They follow the rules of every viz document (document.hpp): a `"schema"` of
 *  `pulsatrix.<kind>.v1`, non-finite numbers as null plus a `"nonfinite"` entry, and readers that
 *  name the JSON Pointer of the first problem. Positions are residues of one sequence, numbered
 *  from `first_position` (1 unless the sequence is a fragment), as variant tables number them.
 */
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/protein_structure.hpp"
#include "pulsatrix/text_tokenizer.hpp"
#include "pulsatrix/variant_scoring.hpp"

namespace pulsatrix {

// ---- mutation map --------------------------------------------------------------------------

/**
 * @brief `pulsatrix.mutation_map.v1`: a score for every substitution, residue by residue, such
 *        as a model's log-odds `log p(mutant) - log p(wild type)` or a deep mutational scan's
 *        measurements.
 */
struct MutationMapDocument {
    std::string title;
    /** @brief How the scores were made, for example `"masked_marginals"`. Optional. */
    std::string method;
    /** @brief The wild-type sequence, one letter per residue. */
    std::string sequence;
    /** @brief The number of the sequence's first residue. */
    int64_t first_position = 1;
    /** @brief The substitutions scored at each residue, one letter each: the columns of values. */
    std::string alphabet = kAminoAcids;
    /** @brief Row-major `(residues, alphabet)`. NaN marks a substitution that wasn't scored. */
    std::vector<float> values;
};

/**
 * @brief A mutation map from VariantScorer::single_mutant_scan (`L x 20`, in kAminoAcids order).
 * @throws std::invalid_argument if @p scan isn't 20 values per residue of @p sequence.
 */
[[nodiscard]] MutationMapDocument MakeMutationMapDocument(std::string_view sequence, const std::vector<float>& scan,
                                                          std::string method = "", int64_t first_position = 1);
/** @throws std::invalid_argument if the sequence or alphabet is empty, a letter repeats in the
 *          alphabet, or values isn't residues x alphabet. */
[[nodiscard]] std::string ToJson(const MutationMapDocument& doc);
[[nodiscard]] MutationMapDocument ParseMutationMapDocument(std::string_view json);

// ---- sequence logo -------------------------------------------------------------------------

/**
 * @brief `pulsatrix.sequence_logo.v1`: a distribution over letters at each residue, such as what
 *        a masked language model predicts there, drawn as a sequence logo.
 */
struct SequenceLogoDocument {
    std::string title;
    std::string method;
    /** @brief The sequence the distributions belong to (its letters are marked), or empty. */
    std::string sequence;
    int64_t first_position = 1;
    std::string alphabet = kAminoAcids;
    /** @brief Row-major `(positions, alphabet)`; each row is a distribution (non-negative, summing
     *         to 1 within 1e-3). */
    std::vector<float> probabilities;

    [[nodiscard]] int64_t positions() const;
};

/**
 * @brief A logo from a model's log-probabilities (VariantScorer::masked_marginals or
 *        wild_type_marginals): each residue's probabilities over the 20 amino acids, renormalized
 *        to sum to 1 (special and rare tokens dropped).
 * @throws std::invalid_argument if @p log_probs doesn't have one row per residue of @p sequence,
 *         or the tokenizer lacks an amino acid.
 */
[[nodiscard]] SequenceLogoDocument MakeSequenceLogoDocument(const ResidueLogProbs& log_probs, const TextTokenizer& tokenizer,
                                                            std::string_view sequence, std::string method = "",
                                                            int64_t first_position = 1);
/**
 * @brief Each position's information content in bits, `log2(letters) - entropy`: 0 when every
 *        letter is equally likely, `log2(20)` (4.32) when one is certain. A logo stacks the
 *        letters to this height, each in proportion to its probability.
 */
[[nodiscard]] std::vector<float> InformationContent(const SequenceLogoDocument& doc);
/** @throws std::invalid_argument if the alphabet is empty or repeats a letter, the probabilities
 *          aren't positions x alphabet, a row isn't a distribution, or a non-empty sequence has
 *          another length. */
[[nodiscard]] std::string ToJson(const SequenceLogoDocument& doc);
[[nodiscard]] SequenceLogoDocument ParseSequenceLogoDocument(std::string_view json);

// ---- contact map ---------------------------------------------------------------------------

/**
 * @brief `pulsatrix.contact_map.v1`: predicted contacts between a sequence's residues, and the
 *        true ones when a structure is known. Drawn as one square: predictions above the
 *        diagonal, the structure below it, with the best predictions marked right or wrong.
 */
struct ContactMapDocument {
    std::string title;
    std::string method;
    std::string sequence;
    int64_t first_position = 1;
    /** @brief `L x L` scores, higher meaning more likely in contact, such as probabilities. */
    std::vector<float> predicted;
    /** @brief Empty, or `L x L`: 1 for a contact, 0 for none, NaN where unknown. */
    std::vector<float> truth;

    [[nodiscard]] int64_t length() const { return static_cast<int64_t>(sequence.size()); }
};

/**
 * @brief Pairs a prediction with the structure's contacts (TrueContacts), if any.
 * @throws std::invalid_argument if a map isn't `L x L` for the sequence's L residues.
 */
[[nodiscard]] ContactMapDocument MakeContactMapDocument(std::string_view sequence, const ContactMap& predicted,
                                                        const ContactMap* truth = nullptr, std::string method = "",
                                                        int64_t first_position = 1);
/** @throws std::invalid_argument if the sequence is empty, predicted isn't L x L, or truth is
 *          neither empty nor L x L of 0, 1 and NaN. */
[[nodiscard]] std::string ToJson(const ContactMapDocument& doc);
[[nodiscard]] ContactMapDocument ParseContactMapDocument(std::string_view json);

// ---- residue tracks ------------------------------------------------------------------------

/**
 * @brief `pulsatrix.residue_tracks.v1`: per-residue signals stacked under a sequence, and
 *        annotated stretches (domains, sites), as in a genome or protein browser. The same
 *        document colors a 3D structure (RenderStructureHtml).
 */
struct ResidueTracksDocument {
    std::string title;
    std::string sequence;
    int64_t first_position = 1;
    struct Track {
        std::string name;
        /** @brief One value per residue; NaN where there is none. */
        std::vector<float> values;
        /** @brief Signed values are drawn on a diverging scale around zero, others from zero up.
         *         A reader treats a missing member as false. */
        bool is_signed = false;
    };
    std::vector<Track> tracks;
    /** @brief An annotated stretch: positions `start` to `end` inclusive, in the document's
     *         numbering. */
    struct Feature {
        std::string name;
        int64_t start = 0;
        int64_t end = 0;
        /** @brief What kind of feature, such as `"domain"` or `"site"`; one row per category. */
        std::string category;
    };
    std::vector<Feature> features;
    /**
     * @brief Empty, or each residue's id in a structure: the author's number and insertion code,
     *        `"52"` or `"52A"`. RenderStructureHtml colors the residue with that id; without them,
     *        the chain's residues must match the sequence one for one.
     */
    std::vector<std::string> residue_ids;
};

/** @brief Each residue's mean score over its substitutions (the wild type left out): how much the
 *         model minds a change there. NaN where none was scored. */
[[nodiscard]] std::vector<float> MeanSubstitutionScore(const MutationMapDocument& doc);
/** @brief @p chain's residue ids, for ResidueTracksDocument::residue_ids. */
[[nodiscard]] std::vector<std::string> ResidueIds(const StructureChain& chain);
/** @throws std::invalid_argument if the sequence is empty, a track's length isn't the sequence's,
 *          a name is empty, a feature's start is after its end or outside the sequence, or
 *          residue_ids is neither empty nor one per residue. */
[[nodiscard]] std::string ToJson(const ResidueTracksDocument& doc);
[[nodiscard]] ResidueTracksDocument ParseResidueTracksDocument(std::string_view json);

}  // namespace pulsatrix
