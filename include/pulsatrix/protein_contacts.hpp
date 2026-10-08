/** @file protein_contacts.hpp
 *  @brief Residue contacts from a protein language model's attention (PLM-4): ESM's contact head,
 *         the top-K head average, and precision at L, L/2 and L/5 against a structure's contacts.
 *  @ingroup interpretability
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/protein_structure.hpp"
#include "pulsatrix/text_tokenizer.hpp"

namespace pulsatrix {

/** @brief One attention head: `layer` from 0, `head` within it from 0. */
struct AttentionHead {
    int64_t layer = 0;
    int64_t head = 0;
};

/**
 * @brief ESM's contact head (`EsmContactPredictionHead`): a logistic regression over every head's
 *        corrected attention map, `sigmoid(sum_c weight[c] * f_c(i, j) + bias)`. Channel
 *        `c = layer * heads + head`, as transformers stacks them.
 */
struct EsmContactHead {
    int64_t layers = 0;
    int64_t heads = 0;
    std::vector<float> weight;  ///< `layers * heads` values
    float bias = 0.0f;
};

/**
 * @brief Reads the contact head from an ESM-2 checkpoint directory: `esm.contact_head.regression.*`
 *        (`EsmForMaskedLM`) or `contact_head.regression.*` (`EsmModel`).
 * @throws std::invalid_argument if the checkpoint has no contact head, or its size isn't
 *         `layers * heads` of @p config.
 */
[[nodiscard]] EsmContactHead LoadEsmContactHead(const std::string& directory, const EncoderLMConfig& config);

/**
 * @brief One head's contact features from its attention map over `<cls>`, L residues and `<eos>`
 *        (`tokens x tokens`, row-major, `tokens = L + 2`): the residue block, made symmetric
 *        (`A + Aᵀ`), then corrected by the average product (APC): `F - rowsum * colsum / total`.
 * @return The `L x L` features, the input to ESM's regression.
 * @throws std::invalid_argument if @p tokens is under 3.
 */
[[nodiscard]] ContactMap ContactFeatures(const float* attention, int64_t tokens);

struct ContactOptions {
    /** @brief The memory the forward pass may use, by ScoringPassBytes. A longer sequence is
     *         refused rather than risking the machine running out of memory. */
    int64_t max_pass_bytes = int64_t{4} << 30;
};

/**
 * @brief Contacts from a masked LM's attention, the way ESM predicts them: one forward pass over
 *        `<cls>` + the sequence + `<eos>`, every head's map turned into ContactFeatures, then
 *        combined. Each map is used as its layer finishes, so a pass holds one layer's at a time.
 *        Sequences are run one at a time, with no padding.
 */
class ContactPredictor {
public:
    /** @param backend Where the model runs. The model, tokenizer and backend must outlive this. */
    ContactPredictor(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend, ContactOptions options = {});

    /**
     * @brief ESM's contact head: contact probabilities, `L x L`, symmetric. Matches transformers'
     *        `EsmForMaskedLM.predict_contacts`.
     * @throws std::invalid_argument for an empty sequence, a character the tokenizer doesn't know,
     *         a head that doesn't fit the model, or a pass over max_pass_bytes.
     */
    [[nodiscard]] ContactMap predict(std::string_view sequence, const EsmContactHead& head);

    /**
     * @brief The mean of @p heads' ContactFeatures: an unsupervised predictor with no regression.
     *        Scores rank pairs; they aren't probabilities. Pick the heads with RankContactHeads.
     * @throws std::invalid_argument as predict(), or for no heads or one outside the model.
     */
    [[nodiscard]] ContactMap average_heads(std::string_view sequence, const std::vector<AttentionHead>& heads);

    /** @brief Calls @p visit with every head's ContactFeatures, layer by layer, from one pass. */
    void for_each_head(std::string_view sequence, const std::function<void(AttentionHead, const ContactMap&)>& visit);

private:
    /** @brief Runs the model, calling @p on_layer with each layer's index and its attention
     *         `(heads, T, T)` on the host. */
    void Run(std::string_view sequence, const std::function<void(int64_t, const std::vector<float>&, int64_t)>& on_layer);

    EncoderLM& model_;
    const TextTokenizer& tokenizer_;
    DeviceBackend* backend_;
    ContactOptions options_;
};

/** @brief Pairs `(i, j)` counted by separation `j - i`: at least `min`, and under `max` unless
 *         `max` is 0. */
struct SeparationRange {
    int64_t min = 6;
    int64_t max = 0;
};
/** @brief CASP's and ESM's ranges: short 6–11, medium 12–23, long 24 and up. */
inline constexpr SeparationRange kShortRange{6, 12};
inline constexpr SeparationRange kMediumRange{12, 24};
inline constexpr SeparationRange kLongRange{24, 0};

/**
 * @brief The fraction of the @p top highest-scoring pairs in @p range that are true contacts.
 *
 * - Only pairs `i < j` whose truth is known (not NaN) compete.
 * - Ties keep the order `(i, j)`, row by row.
 * - With fewer than @p top such pairs, the missing ones count as wrong, as in ESM's
 *   `compute_precisions`.
 * @return NaN if @p top is 0.
 * @throws std::invalid_argument if the maps' sizes differ, or @p top is negative.
 */
[[nodiscard]] double ContactPrecision(const ContactMap& predicted, const ContactMap& truth, SeparationRange range, int64_t top);

/** @brief Precision of the top L, L/2 and L/5 pairs, L the sequence length (rounded down). */
struct ContactPrecisions {
    double at_l = 0;
    double at_l2 = 0;
    double at_l5 = 0;
};
[[nodiscard]] ContactPrecisions PrecisionAtL(const ContactMap& predicted, const ContactMap& truth, SeparationRange range);

/** @brief PrecisionAtL in each of the short, medium and long ranges. */
struct ContactEvaluation {
    ContactPrecisions short_range;
    ContactPrecisions medium_range;
    ContactPrecisions long_range;
};
[[nodiscard]] ContactEvaluation EvaluateContacts(const ContactMap& predicted, const ContactMap& truth);

/** @brief A protein with known contacts, to choose heads with. */
struct LabeledProtein {
    std::string sequence;
    ContactMap contacts;  ///< TrueContacts, `L x L` for the sequence's L residues
};

struct HeadPrecision {
    AttentionHead head;
    double precision = 0;  ///< the mean over the proteins
};

/**
 * @brief Every head, best first, by its features' mean precision at L in @p range over
 *        @p proteins. The top K make an unsupervised predictor with
 *        ContactPredictor::average_heads (arXiv 2606.21876 chooses them with 10 labeled proteins).
 *        Choose heads on proteins you won't evaluate on.
 * @throws std::invalid_argument for no proteins, or a contact map that doesn't fit its sequence.
 */
[[nodiscard]] std::vector<HeadPrecision> RankContactHeads(ContactPredictor& predictor, const std::vector<LabeledProtein>& proteins,
                                                          SeparationRange range = kLongRange);

}  // namespace pulsatrix
