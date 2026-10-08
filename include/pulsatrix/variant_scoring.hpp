/** @file variant_scoring.hpp
 *  @brief Zero-shot variant effects from a protein language model (PLM-3): how much the model
 *         prefers a mutant residue over the wild type, with no training on the protein.
 *  @ingroup interpretability
 */
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/text_tokenizer.hpp"

namespace pulsatrix {

/** @brief One substitution, numbered from 1 as variant tables write it: `A23G`. */
struct Mutation {
    char wild_type = 0;
    int64_t position = 0;
    char mutant = 0;
};

/**
 * @brief Parses a variant, `A23G`, or several joined by `:`, `A23G:L45P` (ProteinGym's format).
 * @throws std::invalid_argument for anything else.
 */
[[nodiscard]] std::vector<Mutation> ParseMutations(std::string_view text);

/**
 * @brief @p sequence with @p mutations applied; position p is `sequence[p - offset]`.
 * @throws std::invalid_argument if a position is outside the sequence or its residue isn't the
 *         listed wild type.
 */
[[nodiscard]] std::string ApplyMutations(std::string_view sequence, const std::vector<Mutation>& mutations, int64_t offset = 1);

/** @brief Log-probabilities of every token at each residue: `length` rows of `vocab` values. */
struct ResidueLogProbs {
    int64_t length = 0;
    int64_t vocab = 0;
    std::vector<float> values;
    [[nodiscard]] float at(int64_t residue, int64_t token) const { return values[static_cast<size_t>(residue * vocab + token)]; }
};

/** @brief The 20 standard amino acids, in the order SingleMutantScan's columns use. */
inline constexpr const char* kAminoAcids = "ACDEFGHIKLMNPQRSTVWY";

struct VariantScoringOptions {
    /**
     * @brief The longest token window the model sees. A longer sequence is cut, for each masked
     *        position, to the window ProteinGym's `get_optimal_window` picks around it. ESM-2 uses 1024.
     */
    int64_t window = 1024;
    /** @brief Masked copies run together in one forward pass, at most. It changes speed, not results. */
    int64_t batch_size = 8;
    /**
     * @brief The memory one forward pass may use, by ScoringPassBytes. A batch of long windows
     *        shrinks to fit; a single sequence that doesn't fit is refused rather than risking
     *        the machine running out of memory.
     */
    int64_t max_pass_bytes = int64_t{4} << 30;
};

/**
 * @brief Scores variants with a masked language model and its tokenizer, the way ProteinGym scores
 *        ESM models. A higher score means the model finds the variant more plausible.
 *
 * - **Masked marginals** (ProteinGym's choice for ESM-2) mask each residue in turn. A substitution
 *   scores `log p(mutant) - log p(wild type)` at its masked position, and a multiple mutant the sum
 *   of its substitutions. The cost is one forward pass per residue, whatever the number of variants.
 * - **Wild-type marginals** read the same log-ratio from one unmasked pass. They are faster but
 *   less accurate.
 * - **Pseudo-log-likelihood** sums `log p(residue)` with each residue masked in turn: one pass per
 *   residue, for each sequence scored.
 */
class VariantScorer {
public:
    /** @param backend Where the model runs. The model, tokenizer and backend must outlive this. */
    VariantScorer(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend, VariantScoringOptions options = {});

    /** @brief Each residue's distribution with that residue masked. @throws std::invalid_argument for
     *         an empty sequence or a character the tokenizer doesn't know. */
    [[nodiscard]] ResidueLogProbs masked_marginals(std::string_view sequence);
    /** @brief Each residue's distribution from one pass over the unmasked sequence. */
    [[nodiscard]] ResidueLogProbs wild_type_marginals(std::string_view sequence);
    /** @brief The sum over residues of `log p(residue)` with it masked. Every residue is scored. */
    [[nodiscard]] double pseudo_log_likelihood(std::string_view sequence);

    /**
     * @brief The summed log-ratio of @p mutations in marginals computed on @p sequence.
     * @throws std::invalid_argument if a mutation doesn't match the sequence, or names a residue
     *         that isn't a single vocabulary token.
     */
    [[nodiscard]] double score(const ResidueLogProbs& marginals, std::string_view sequence, const std::vector<Mutation>& mutations,
                               int64_t offset = 1) const;

    /**
     * @brief Every single substitution: row i, column j is `log p(kAminoAcids[j]) - log p(wild
     *        type)` at residue i, row-major, `length x 20`. The wild type's own column is 0.
     */
    [[nodiscard]] std::vector<float> single_mutant_scan(const ResidueLogProbs& marginals, std::string_view sequence) const;

    [[nodiscard]] int64_t token_of(char residue) const;

private:
    /** @brief `<cls>` + residues + `<eos>` as token ids. */
    [[nodiscard]] std::vector<int64_t> Tokens(std::string_view sequence) const;
    /** @brief How many @p length-token sequences fit in one pass under max_pass_bytes, at most
     *         batch_size. @throws std::invalid_argument if not even one does. */
    [[nodiscard]] int64_t BatchSize(int64_t length) const;
    /** @brief Log-softmax rows of a batch of equal-length token sequences, `(N, L, vocab)` flattened. */
    [[nodiscard]] std::vector<float> LogSoftmax(const std::vector<std::vector<int64_t>>& batch);

    EncoderLM& model_;
    const TextTokenizer& tokenizer_;
    DeviceBackend* backend_;
    VariantScoringOptions options_;
    int64_t mask_id_ = 0;
};

/**
 * @brief Roughly the most memory one sequence of @p length tokens takes in a forward pass of
 *        @p config's model that keeps no activations (EncoderLM::set_keep_activations(false)):
 *        one layer's attention scores and their softmax copies, its projections and MLP, and
 *        the logits. The weights aren't counted.
 */
[[nodiscard]] int64_t ScoringPassBytes(const EncoderLMConfig& config, int64_t length);

/** @brief ProteinGym's `get_optimal_window`: the token range `[begin, end)` around @p position in a
 *         sequence of @p length tokens, for a model that sees @p window tokens. */
[[nodiscard]] std::pair<int64_t, int64_t> OptimalWindow(int64_t position, int64_t length, int64_t window);

}  // namespace pulsatrix
