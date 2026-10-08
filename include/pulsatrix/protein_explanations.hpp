/** @file protein_explanations.hpp
 *  @brief Explaining a protein language model residue by residue (PLM-6): AttnLRP from a masked
 *         residue's prediction, a mutation's log-odds or a head on the representations back to
 *         the input residues, and the checks that keep such explanations honest.
 *  @ingroup interpretability
 *
 *  Residue-level explanations can look plausible and still not explain the model: a 2026 study
 *  (arXiv 2606.22181) found integrated gradients on a well-performing ESM-2 classifier missed the
 *  annotated epitopes. So every explanation here comes with checks against something independent
 *  of how it looks:
 *  - **Randomization** (Adebayo et al. 2018): randomize the model's layers from the top down;
 *    an explanation that barely changes isn't explaining the model.
 *  - **Deletion**: mask residues most relevant first; a faithful explanation makes the
 *    explained value fall faster than masking them in random order.
 *  - **Agreement** with an independent per-residue signal: a deep mutational scan's per-position
 *    sensitivity, or conservation in a multiple sequence alignment.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/attnlrp_parity.hpp"  // LxtAttnLrpConfig
#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_sequences.hpp"
#include "pulsatrix/proteingym.hpp"
#include "pulsatrix/relevance_graph.hpp"
#include "pulsatrix/text_tokenizer.hpp"
#include "pulsatrix/variant_scoring.hpp"

namespace pulsatrix {

/** @brief A linear head on the model's representations (last_hidden_state): `weight . h + bias`. */
struct LinearHead {
    std::vector<float> weight;  ///< hidden_size values
    float bias = 0.0f;
};

/** @brief What an explanation explains. Residues are numbered from 0 here. */
struct EncoderTarget {
    enum class Kind {
        /** @brief The logit of `token` at `residue`, with that residue masked. */
        MaskedToken,
        /** @brief `logit(token) - logit(wild_type)` at `residue`, masked: a mutation's log-odds,
         *         the masked-marginal score (VariantScorer) before log-softmax, which cancels. */
        Mutation,
        /** @brief A per-protein head on the mean of the residues' representations (`<cls>` and
         *         `<eos>` left out, as ESM embeddings are usually pooled), unmasked. */
        ProteinHead,
        /** @brief A per-residue head on `residue`'s representation, unmasked. */
        ResidueHead,
    };
    Kind kind = Kind::MaskedToken;
    int64_t residue = 0;
    char token = 0;
    char wild_type = 0;
    LinearHead head;

    [[nodiscard]] static EncoderTarget MaskedToken(int64_t residue, char token);
    /** @brief The mutation @p m; its position is numbered from @p offset, as variant tables do. */
    [[nodiscard]] static EncoderTarget ForMutation(const Mutation& m, int64_t offset = 1);
    [[nodiscard]] static EncoderTarget ProteinHead(LinearHead head);
    [[nodiscard]] static EncoderTarget ResidueHead(LinearHead head, int64_t residue);
    /** @brief For labels: `"K11 masked: R"`, `"K11R log-odds"`, `"protein head"`, `"head at K11"`. */
    [[nodiscard]] std::string describe(std::string_view sequence) const;
};

/** @brief Relevance per residue for one explained value. */
struct ResidueRelevance {
    std::string sequence;
    std::string target;
    /** @brief The explained value: a logit, a log-odds or a head's output. */
    float value = 0.0f;
    /** @brief One value per residue (the input tokens between `<cls>` and `<eos>`). */
    std::vector<float> residues;
    float cls = 0.0f;
    float eos = 0.0f;
    /** @brief Relevance per token (`<cls>`, residues, `<eos>`) at the embeddings and after each
     *         layer: `(layers + 1)` rows of `L + 2`. */
    std::vector<std::vector<float>> layers;

    /** @brief The relevance of every token, `<cls>` and `<eos>` included. */
    [[nodiscard]] double total() const;
};

/**
 * @brief AttnLRP for an ESM-2 EncoderLM: explains a target (EncoderTarget) as relevance on the
 *        input residues, with LXT's rules by default (the identity rule on LayerNorm and GELU, the
 *        uniform rule at attention's matmuls, epsilon elsewhere), so it matches LXT's AttnLRP.
 *
 * Relevance at the output is the explained value itself, as LXT seeds it (`value.backward()`):
 * the residues' relevance, with `<cls>`'s and `<eos>`'s, adds up to roughly that value; LRP through
 * biases and norms isn't exactly conservative.
 */
class EncoderExplainer {
public:
    /** @param backend Where the model runs. The model, tokenizer and backend must outlive this. */
    EncoderExplainer(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend,
                     LRPRuleConfig config = LxtAttnLrpConfig());

    /**
     * @brief Explains @p target on @p sequence.
     * @throws std::invalid_argument for an empty sequence, a character the tokenizer doesn't
     *         know, a residue outside the sequence, a mutation whose wild type isn't the
     *         sequence's, or a head of the wrong size.
     */
    [[nodiscard]] ResidueRelevance explain(std::string_view sequence, const EncoderTarget& target);

    /**
     * @brief The explanation as an attribution graph over residues (VIZ-4's format, for
     *        Neuronpedia's and circuit-tracer's viewers): a node per layer and token, links carrying
     *        relevance between adjacent layers, and the output. The output's label is
     *        EncoderTarget::describe(), with the predicted token's probability for a masked or
     *        mutation target (1 for a head). Only `config` among @p options' rules is ignored:
     *        the explainer's own rules apply.
     * @note Cost: explain(), then one layer LRP pass per layer and token.
     * @throws std::invalid_argument as explain(), or for a sequence longer than max_tokens - 2.
     */
    [[nodiscard]] AttributionGraph relevance_graph(std::string_view sequence, const EncoderTarget& target,
                                                   const RelevanceGraphOptions& options);

    /** @brief The explained value on @p tokens (`<cls>` + residues + `<eos>`, some perhaps masked):
     *         the model run without keeping activations. The residue a masked target names is
     *         masked first. */
    [[nodiscard]] float evaluate(const std::vector<int64_t>& tokens, const EncoderTarget& target);

    /** @brief `<cls>` + @p sequence + `<eos>` as token ids. */
    [[nodiscard]] std::vector<int64_t> tokens(std::string_view sequence) const;
    [[nodiscard]] int64_t mask_id() const { return mask_id_; }
    [[nodiscard]] EncoderLM& model() { return model_; }

private:
    void Check(std::string_view sequence, const EncoderTarget& target) const;
    /** @brief explain(), keeping the model's activations when @p keep (for the graph). */
    ResidueRelevance Explain(std::string_view sequence, const EncoderTarget& target, bool keep, std::vector<std::vector<float>>* boundaries,
                             std::vector<float>* logits);
    /** @brief Runs the model; returns the logits on the host. */
    std::vector<float> Forward(const std::vector<int64_t>& tokens, bool keep);
    /** @brief The value, given the logits and last hidden state of a forward pass. */
    float Value(const std::vector<float>& logits, const std::vector<float>& hidden, int64_t T, const EncoderTarget& target) const;

    EncoderLM& model_;
    const TextTokenizer& tokenizer_;
    DeviceBackend* backend_;
    LRPRuleConfig config_;
    int64_t mask_id_ = 0;
};

// ---- checks ----------------------------------------------------------------------------------

/** @brief How a per-residue explanation agrees with an independent per-residue signal. */
struct ResidueAgreement {
    /** @brief Residues where both are finite. */
    int64_t compared = 0;
    /** @brief Spearman correlation of |relevance| with the signal. */
    double spearman = 0;
    /** @brief The share of the top-k residues by |relevance| that are also top-k by the signal. */
    double top_overlap = 0;
    int64_t top = 0;
    /** @brief The share expected by chance, `k / compared`. */
    double chance_overlap = 0;
};

/**
 * @brief Compares |relevance| with @p signal over the residues where both are finite; @p
 *        top_fraction picks k (at least 1).
 * @throws std::invalid_argument if the lengths differ, fewer than 3 residues compare, or
 *         top_fraction isn't in (0, 1].
 */
[[nodiscard]] ResidueAgreement CompareResidueSignals(const std::vector<float>& relevance, const std::vector<float>& signal,
                                                     double top_fraction = 0.1);

/**
 * @brief Each residue's sensitivity in a deep mutational scan: minus the mean score of its single
 *        mutants (higher scores being fitter, as in ProteinGym), so residues that tolerate no change
 *        rank highest. NaN where no single mutant was measured. Multiple mutants are skipped.
 * @throws std::invalid_argument if a mutant doesn't parse or doesn't match @p sequence.
 */
[[nodiscard]] std::vector<float> DmsPositionSensitivity(const DmsVariants& variants, std::string_view sequence, int64_t offset = 1);

/**
 * @brief Conservation of each query residue in a multiple sequence alignment whose first record
 *        is the query: `log2(20) - entropy` of the column's amino acids, in bits, times the share
 *        of sequences without a gap there. NaN for query residues outside the alignment's focus
 *        columns (lowercase).
 *
 * Reads both A2M layouts: column-aligned files, where every record has the query's length and
 * lowercase marks non-focus columns (ProteinGym's and EVE's), and standard A2M, where lowercase
 * letters and '.' are insertions and the match columns (uppercase and '-') line up.
 * @note Sequences are counted unweighted; tools that reweight by similarity give somewhat
 *       different numbers for families with many near-duplicates.
 * @return One value per query residue (its letters, any case).
 * @throws std::invalid_argument for no records, or records whose match columns don't line up.
 */
[[nodiscard]] std::vector<float> AlignmentConservation(const std::vector<FastaRecord>& alignment);

/** @brief The query's residues in @p alignment's first record: its letters, uppercased. */
[[nodiscard]] std::string AlignmentQuery(const std::vector<FastaRecord>& alignment);

/**
 * @brief How much the model draws on each residue when it predicts the others: the mean, over
 *        every residue's masked wild-type logit (EncoderTarget::MaskedToken), of each residue's
 *        share of the explanation's total |relevance|. A per-protein explanation to compare with
 *        per-residue signals such as DMS sensitivity or conservation.
 * @note Cost: one explanation per residue. @p progress, if set, is called after each.
 */
[[nodiscard]] std::vector<float> RelevanceProfile(EncoderExplainer& explainer, std::string_view sequence,
                                                  const std::function<void(int64_t done, int64_t total)>& progress = {});

/** @brief A deletion curve against its random baseline. */
struct ResidueDeletionCurve {
    /** @brief Fractions of residues masked: 0, 1/steps, ..., 1. */
    std::vector<float> fractions;
    /** @brief The explained value after masking that share, most relevant first. */
    std::vector<float> scores;
    /** @brief The same, in random orders (their mean). */
    std::vector<float> random_scores;
    /** @brief The sign of the unmasked value: +1, or -1 for a negative one such as most
     *         mutations' log-odds. */
    float sign = 1.0f;
    /** @brief Trapezoid areas under each curve times `sign`, so both measure how much of the value
     *         is left; a faithful explanation removes it faster: auc < random_auc. */
    double auc = 0;
    double random_auc = 0;
};

/**
 * @brief Masks residues (`<mask>`) in order of how much they support the explained value
 *        (relevance of the value's own sign, largest first) and records the value, beside
 *        @p random_orders random orders. A faithful explanation moves the value toward zero
 *        faster than random. The explained residue itself is never masked twice: a masked
 *        target's residue stays masked throughout, and isn't counted.
 * @throws std::invalid_argument if @p relevance doesn't have one value per residue, steps < 1 or
 *         random_orders < 1.
 */
[[nodiscard]] ResidueDeletionCurve DeletionCheck(EncoderExplainer& explainer, std::string_view sequence, const EncoderTarget& target,
                                                 const std::vector<float>& relevance, int64_t steps = 20, int64_t random_orders = 5,
                                                 uint64_t seed = 0);

/** @brief The randomization check's result: after randomizing `layers[0..i]`, how similar the
 *         explanation stays (Spearman of |relevance| per residue with the original). */
struct RandomizationCheckResult {
    std::vector<std::string> layers;
    std::vector<float> similarity;
};

/**
 * @brief Cascading model randomization (Adebayo et al. 2018) over the encoder's own layers, top
 *        down: the LM head, then each layer from the last to the first, then the embeddings.
 *        Each step redraws the group's weight matrices (Gaussian, each with its
 *        own standard deviation) and re-explains. Norm gains and biases are kept: redrawn from
 *        their own small spread, a pre-LN block's gains shrink toward zero and the block becomes
 *        its residual connection, which would hide the randomization. The model's parameters are
 *        restored before returning.
 * @note A copy of every parameter is held while it runs.
 */
[[nodiscard]] RandomizationCheckResult RandomizationCheck(EncoderExplainer& explainer, std::string_view sequence,
                                                          const EncoderTarget& target, uint64_t seed = 0);

}  // namespace pulsatrix
