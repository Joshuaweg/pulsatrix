/** @file protein_concepts.hpp
 *  @brief What a protein language model's representations encode (PLM-8): per-residue concepts
 *         from Swiss-Prot annotations, per-layer linear probes with control tasks, and sparse
 *         autoencoder features matched to the concepts, as InterPLM does.
 *  @ingroup interpretability
 *
 *  Probes and features answer different questions. A probe asks whether a concept can be read
 *  out of a layer linearly; a high score can also mean the probe learned the task from residue
 *  identity alone, so it comes with a local-sequence control and a randomly initialized model. An SAE asks
 *  which directions the model uses unprompted; a feature matching a concept is a lead, not proof.
 */
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/text_tokenizer.hpp"

namespace pulsatrix {

// ---- annotations -----------------------------------------------------------------------------

/** @brief One annotated stretch, numbered from 1, inclusive, as UniProt numbers it. */
struct ProteinFeature {
    std::string type;  ///< UniProt's feature type: `"Helix"`, `"Binding site"`, `"Disulfide bond"`, ...
    int64_t start = 0;
    int64_t end = 0;
};

struct AnnotatedProtein {
    std::string accession;
    std::string sequence;
    std::vector<ProteinFeature> features;
};

/**
 * @brief Reads JSON lines, one protein per line: `{"accession", "sequence", "features": [{"type",
 *        "start", "end"}]}` (tools/plm/fetch_swissprot_annotations.py writes them). Blank lines are
 *        skipped.
 * @throws std::invalid_argument for a line that isn't such an object, or a feature outside its
 *         sequence.
 */
[[nodiscard]] std::vector<AnnotatedProtein> ParseAnnotatedProteins(std::string_view jsonl);
/** @brief ParseAnnotatedProteins on a file. @throws std::runtime_error if it can't be read. */
[[nodiscard]] std::vector<AnnotatedProtein> ReadAnnotatedProteins(const std::string& path);

/**
 * @brief 1 for each residue a feature of @p type covers, 0 elsewhere. A disulfide bond marks only
 *        its two cysteines (UniProt gives them as start and end), not the stretch between.
 */
[[nodiscard]] std::vector<int> ConceptLabels(const AnnotatedProtein& protein, const std::string& type);
/** @brief Three-state secondary structure per residue: 0 helix (`Helix`), 1 strand (`Beta
 *         strand`), 2 anything else (turns and coil). */
[[nodiscard]] std::vector<int> SecondaryStructureLabels(const AnnotatedProtein& protein);
/** @brief Each feature type, with the residues it covers over @p proteins, most first. */
[[nodiscard]] std::vector<std::pair<std::string, int64_t>> ConceptCounts(const std::vector<AnnotatedProtein>& proteins);

// ---- representations ---------------------------------------------------------------------------

/** @brief Per-residue representations at some layers: for each layer, `residues x hidden` values,
 *         the proteins' residues one after another. */
struct ResidueEmbeddings {
    std::vector<int64_t> layers;
    int64_t hidden = 0;
    int64_t residues = 0;
    /** @brief Where each protein's residues start. */
    std::vector<int64_t> offsets;
    std::vector<std::vector<float>> values;

    [[nodiscard]] const float* row(size_t layer_index, int64_t residue) const {
        return values[layer_index].data() + residue * hidden;
    }
};

/**
 * @brief Runs @p sequences through the model one at a time and keeps each residue's representation
 *        (not `<cls>` or `<eos>`) at @p layers: 0 is the embeddings, i the output of layer i, and
 *        `num_layers()` the final representation, after the final LayerNorm, as ESM embeddings
 *        are usually taken.
 * @throws std::invalid_argument for a layer outside [0, num_layers()], or a sequence longer than
 *         @p max_tokens - 2 or holding a character the tokenizer doesn't know.
 */
[[nodiscard]] ResidueEmbeddings EmbedResidues(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend,
                                              const std::vector<std::string>& sequences, const std::vector<int64_t>& layers,
                                              int64_t max_tokens = 1024);

// ---- probes ------------------------------------------------------------------------------------

/**
 * @brief The control a residue probe has to beat: each residue's local sequence, one-hot over the
 *        20 amino acids at every position within @p window of it (`20 * (2 window + 1)` values,
 *        zeros past the ends and for other letters). A linear probe that reads a concept from a
 *        layer no better than from this hasn't shown the model adds anything beyond nearby
 *        sequence.
 * @note Hewitt and Liang's control task, labels fixed per token type, is no control here: with
 *       only 20 amino acids, a linear probe on any representation that encodes identity learns
 *       it perfectly. Local sequence plays its role instead.
 * @return `residues x 20 (2 window + 1)`, the sequences' residues one after another.
 */
[[nodiscard]] std::vector<float> SequenceWindowFeatures(const std::vector<std::string>& sequences, int64_t window);

struct ProbeOptions {
    int64_t epochs = 20;
    int64_t batch = 1024;
    float learning_rate = 1e-2f;
    float weight_decay = 1e-4f;
    uint64_t seed = 0;
};

/** @brief A probe's quality on held-out residues. */
struct ProbeResult {
    double accuracy = 0;
    /** @brief The mean recall over classes, so a rare class counts as much as a common one. */
    double balanced_accuracy = 0;
    /** @brief For two classes, the ROC AUC of class 1's probability; NaN otherwise. */
    double auc = 0;
    int64_t train = 0;
    int64_t test = 0;
};

/**
 * @brief Trains a linear softmax probe (standardized features, AdamW, cross-entropy) on the
 *        residues marked in @p train and scores it on those marked in @p test.
 * @param features `residues x dim`, row-major. @param labels one class in [0, classes) per residue.
 * @throws std::invalid_argument if the sizes don't agree, classes < 2, or either set is empty.
 */
[[nodiscard]] ProbeResult TrainLinearProbe(const std::vector<float>& features, int64_t dim, const std::vector<int>& labels, int classes,
                                           const std::vector<bool>& train, const std::vector<bool>& test, DeviceBackend* backend,
                                           const ProbeOptions& options = {});

// ---- features ----------------------------------------------------------------------------------

/** @brief Sparse codes: for each residue, its active features and their values. */
struct SparseCodes {
    int64_t num_features = 0;
    std::vector<int64_t> row_start;  ///< residues + 1 entries
    std::vector<int32_t> feature;
    std::vector<float> value;
    /** @brief Each feature's largest value. */
    std::vector<float> max_value;

    [[nodiscard]] int64_t residues() const { return static_cast<int64_t>(row_start.size()) - 1; }
};

/** @brief @p featurizer's codes for every residue in @p rows (`residues x input_dim`), encoded in
 *         batches of @p batch and kept where above zero. */
[[nodiscard]] SparseCodes EncodeSparse(Featurizer& featurizer, const std::vector<float>& rows, DeviceBackend* backend,
                                       int64_t batch = 4096);
/** @brief Dense values as codes, each dimension `d` twice: as feature `2d` (its positive part) and
 *         `2d + 1` (its negative part, flipped), so single neurons can be matched to concepts the
 *         same way as features. */
[[nodiscard]] SparseCodes NeuronCodes(const std::vector<float>& rows, int64_t dim);

/** @brief The feature that best matches one concept. */
struct ConceptMatch {
    std::string concept_name;
    int64_t positives = 0;  ///< residues with the concept
    int64_t feature = -1;
    double threshold = 0;   ///< the share of the feature's largest value it must exceed
    double precision = 0;
    double recall = 0;
    double f1 = 0;
    /** @brief Features whose best F1 for this concept is above 0.5. */
    int64_t features_above_half = 0;
};

/**
 * @brief InterPLM's matching: a feature predicts the concept on a residue when its value exceeds a
 *        threshold, a share of its largest value (@p thresholds), and is scored by F1 over the
 *        residues. Returns the best feature and threshold.
 * @throws std::invalid_argument if @p labels doesn't have one value per residue.
 */
[[nodiscard]] ConceptMatch MatchConcept(const SparseCodes& codes, const std::vector<int>& labels, const std::string& concept_name,
                                        const std::vector<double>& thresholds = {0.0, 0.15, 0.5, 0.6, 0.8});

}  // namespace pulsatrix
