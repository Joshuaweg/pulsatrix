/** @file fine_tune_diff.hpp
 *  @brief What did a fine-tune change? The fine-tuned model's relevance minus the base model's for
 *         the same prediction, input by input and token by token across a dataset, and intruder
 *         dimensions: new top singular directions in the fine-tuned weights (TRN-11).
 *  @ingroup interpretability_lrp
 *
 * Works for a fine-tune held as adapters on the base model (LoRA, IA³, LoReFT), which
 * DiffAdapterRelevance() switches off and on, and for two separate models of the same architecture,
 * such as a full fine-tune and its base, with DiffRelevance().
 */
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief Which output both models are asked to explain. */
struct DiffTarget {
    /** @brief For outputs shaped `(N, L, C)`, such as a language model's logits: the position, with
     *         negative values counted from the end. -1, the default, is the last. Ignored for `(N, C)`. */
    int64_t position = -1;
    /** @brief One class or token per sample. Empty: the fine-tuned model's top prediction, so both
     *         models explain the choice the fine-tune makes. */
    std::vector<int64_t> classes;
};

/** @brief The two explanations of the same prediction, and their difference. */
struct RelevanceDiff {
    /** @brief The fine-tuned model's input relevance, as its propagate_relevance() returns it
     *         (`(N, L)` per token for a language model). */
    Tensor tuned;
    /** @brief The base model's, for the same class at the same position. */
    Tensor base;
    /** @brief tuned - base. */
    Tensor difference;
    /** @brief The class explained for each sample. */
    std::vector<int64_t> classes;
    /** @brief Each model's output at that class, per sample: the relevance each one distributes. */
    std::vector<double> tuned_scores;
    std::vector<double> base_scores;
};

/**
 * @brief Explains the same prediction in both models: forward() on @p input, then relevance seeded
 *        with each model's own score for the class, at the class only.
 * @throws std::invalid_argument if the models' outputs differ in shape, the output isn't `(N, C)` or
 *         `(N, L, C)`, or a class or the position is out of range; as the models' own calls throw.
 */
[[nodiscard]] RelevanceDiff DiffRelevance(Module& base, Module& tuned, const Tensor& input, const DiffTarget& target = {},
                                          const LRPRuleConfig& config = {});

/**
 * @brief DiffRelevance() for a fine-tune held as adapters on @p model: the tuned run with its LoRA
 *        adapters, IA³ vectors and (for a CausalLM) interventions on, the base run with them off.
 *        They are back on afterwards, even if a call throws.
 * @throws std::invalid_argument if the model has no enabled adapters, or one is merged (its update
 *         is in the weights and can't be switched off).
 */
[[nodiscard]] RelevanceDiff DiffAdapterRelevance(Module& model, const Tensor& input, const DiffTarget& target = {},
                                                 const LRPRuleConfig& config = {});

/** @brief One token's relevance shift, averaged over every occurrence seen. */
struct TokenShift {
    int64_t token = 0;
    int64_t count = 0;
    double tuned = 0;
    double base = 0;
    /** @brief The mean of tuned - base. */
    double difference = 0;
};

/**
 * @brief Collects RelevanceDiff results over a dataset by input token: which tokens a fine-tune made
 *        the model rely on more, and which less.
 */
class TokenRelevanceShift {
public:
    /** @brief Adds one batch: @p token_ids `(N, L)` and a diff whose relevance is `(N, L)`. Each score
     *         is divided by its sample's tuned or base score, so samples weigh alike.
     *  @throws std::invalid_argument if the shapes differ. */
    void add(const Tensor& token_ids, const RelevanceDiff& diff);
    /** @brief The @p n tokens with the largest mean increase (or, with @p decrease, the largest mean
     *         decrease), among those seen at least @p min_count times. */
    [[nodiscard]] std::vector<TokenShift> top(int64_t n, bool decrease = false, int64_t min_count = 1) const;
    /** @brief Every token seen, by id. */
    [[nodiscard]] std::vector<TokenShift> all() const;

private:
    struct Sum {
        int64_t count = 0;
        double tuned = 0, base = 0;
    };
    std::map<int64_t, Sum> sums_;
};

/** @brief Intruder dimensions of one weight matrix (Shuttleworth et al., 2024, Definition 3.1). */
struct IntruderResult {
    /** @brief For each of the tuned matrix's top-k singular vectors, the largest |cosine similarity|
     *         with any of the base matrix's singular vectors. */
    std::vector<double> max_similarity;
    /** @brief Their singular values, largest first. */
    std::vector<double> singular_values;
    /** @brief How many fall below epsilon: the intruder dimensions. */
    int64_t count = 0;
};

/**
 * @brief Counts the top-k singular vectors of @p tuned that no singular vector of @p base resembles:
 *        |cos| < @p epsilon against every one of them.
 * @param base, tuned Weights of the same shape in pulsatrix's `(in, out)` layout. The vectors compared
 *        are the output-space ones, PyTorch's left singular vectors of its `(out, in)` weight, as in the
 *        paper.
 * @param k The paper's figures use 10.
 * @param epsilon The paper sweeps it (it calls 0.6 to 0.9 typical for full fine-tuning); 0.6 is a
 *        starting point, not a standard. max_similarity lets a caller apply any other threshold.
 * @note The paper's Algorithm 1 compares cosines without an absolute value. A singular vector's sign
 *       is arbitrary, so this uses |cos|. Two O(out^3) eigendecompositions of W^T W, on the CPU in
 *       double (SymmetricEigen takes about 0.8 s at n = 512 in a Release build).
 * @throws std::invalid_argument if the shapes differ or aren't rank 2, k < 1, or epsilon isn't in (0, 1].
 */
[[nodiscard]] IntruderResult IntruderDimensions(const Tensor& base, const Tensor& tuned, int64_t k = 10, double epsilon = 0.6);

/** @brief IntruderDimensions() for one named layer. */
struct LayerIntruders {
    std::string name;
    IntruderResult result;
};

/**
 * @brief Intruder dimensions of every LoRA layer: its base weight W against W + (alpha / r) A B.
 * @throws std::invalid_argument if the model has no unmerged LoRA adapters.
 */
[[nodiscard]] std::vector<LayerIntruders> LoraIntruderDimensions(Module& model, int64_t k = 10, double epsilon = 0.6);

/**
 * @brief Intruder dimensions of every LinearModule the two models share by name and shape: a full
 *        fine-tune against its base.
 * @throws std::invalid_argument if they share none.
 */
[[nodiscard]] std::vector<LayerIntruders> IntruderDimensions(Module& base, Module& tuned, int64_t k = 10, double epsilon = 0.6);

}  // namespace pulsatrix
