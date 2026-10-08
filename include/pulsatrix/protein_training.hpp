/** @file protein_training.hpp
 *  @brief Training protein language models (PLM-7): ESM-2's masked-LM data pipeline (masking,
 *         cropping, cluster-weighted sampling), its initialization, a training step and
 *         evaluation, and heads for fine-tuning on per-protein and per-residue labels.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/encoder_lm.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/text_tokenizer.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {

/** @brief ESM-2's masking (Lin et al. 2023; BERT's 80/10/10). */
struct MaskedLMOptions {
    /** @brief Each residue is picked for prediction with this probability. */
    double mask_probability = 0.15;
    /** @brief Of the picked residues, the share replaced by `<mask>`... */
    double mask_fraction = 0.8;
    /** @brief ...and the share replaced by a random amino acid; the rest stay as they are. */
    double random_fraction = 0.1;
    /** @brief The longest a sequence may be, in tokens with `<cls>` and `<eos>` (ESM-2's 1024):
     *         longer ones are cropped to a random window of `max_tokens - 2` residues. */
    int64_t max_tokens = 1024;
};

/** @brief One collated batch, on the collator's backend. */
struct MaskedLMBatch {
    /** @brief `(N, L)` token ids: masked inputs, padded with `<pad>`. */
    Tensor ids;
    /** @brief `(N, L)`: 1 for a real token, 0 for padding (EncoderLM::set_padding_mask). */
    Tensor keep;
    /** @brief `(N, L)`: the original token where one was picked, -100 (ignored) elsewhere. */
    Tensor targets;
    /** @brief Picked tokens: the loss's normalizer. */
    int64_t num_targets = 0;
};

/**
 * @brief Turns sequences into masked-LM batches as ESM-2 was trained: `<cls>` + residues + `<eos>`,
 *        cropped to a random window when too long, then each residue picked with probability
 *        0.15; of those, 80% become `<mask>`, 10% a random amino acid (uniform over the 20 standard
 *        ones) and 10% stay. `<cls>`, `<eos>` and padding are never picked. Draws come from a
 *        seeded portable generator, so a seed gives the same batches on every platform.
 * @note Token dropout (ESM-2 zeroes `<mask>` embeddings and rescales the rest) is EncoderLM's
 *       forward(), in training as in inference.
 */
class MaskedLMCollator {
public:
    /** @throws std::invalid_argument if the tokenizer lacks a special token or an amino acid, or
     *          the options are out of range (probabilities outside [0, 1], fractions summing over
     *          1, max_tokens < 3). */
    MaskedLMCollator(const TextTokenizer& tokenizer, DeviceBackend* backend, MaskedLMOptions options = {}, uint64_t seed = 0);

    /** @throws std::invalid_argument for no sequences, an empty one, or a character the tokenizer
     *          doesn't know. */
    [[nodiscard]] MaskedLMBatch collate(const std::vector<std::string>& sequences);

    /** @brief `<cls>` + @p sequence (cropped as collate() would) + `<eos>`, unmasked. */
    [[nodiscard]] std::vector<int64_t> tokens(const std::string& sequence);

private:
    const TextTokenizer& tokenizer_;
    DeviceBackend* backend_;
    MaskedLMOptions options_;
    uint64_t state_;
    int64_t cls_ = 0, eos_ = 0, pad_ = 0, mask_ = 0;
    std::vector<int64_t> amino_acids_;
    uint64_t Next();
    double Uniform();
};

/**
 * @brief Cluster-weighted sampling, as ESM-2 drew UniRef50: a cluster uniformly, then one of its
 *        members uniformly, so large families don't dominate. With every sequence its own
 *        cluster, it is uniform sampling with replacement.
 */
class ClusterSampler {
public:
    /** @param cluster_of Each sequence's cluster, any ids. @throws std::invalid_argument if empty. */
    ClusterSampler(const std::vector<std::string>& cluster_of, uint64_t seed = 0);
    /** @brief The next sequence's index. */
    [[nodiscard]] int64_t next();
    [[nodiscard]] int64_t num_clusters() const { return static_cast<int64_t>(members_.size()); }

private:
    std::vector<std::vector<int64_t>> members_;
    uint64_t state_;
};

/**
 * @brief Initializes a model for training from scratch as transformers' `EsmPreTrainedModel`
 *        does: linear weights and the embedding table from N(0, std²) (the `<pad>` row zeroed),
 *        biases 0, LayerNorm gains 1 and offsets 0. Deterministic for a seed.
 */
void InitializeEsm(EncoderLM& model, uint64_t seed, float std = 0.02f);

/**
 * @brief One masked-LM forward and backward: sets the batch's padding mask, runs the model, and
 *        accumulates the gradient of the cross-entropy over the picked tokens, divided by
 *        @p normalizer (0: the batch's own num_targets, i.e. the mean). Gradients add to what's
 *        there; zero them between optimizer steps.
 * @return The loss (mean over picked tokens unless a normalizer is given).
 */
float MaskedLMForwardBackward(EncoderLM& model, const MaskedLMBatch& batch, TokenCrossEntropyLoss& loss, float normalizer = 0.0f);

/** @brief Masked-LM quality on held-out sequences. */
struct MaskedLMEvaluation {
    double loss = 0;        ///< mean cross-entropy over picked tokens, in nats
    double perplexity = 0;  ///< exp(loss)
    double accuracy = 0;    ///< share of picked tokens whose argmax is the original
    int64_t tokens = 0;
};

/**
 * @brief Masks @p sequences with a collator seeded by @p seed (so every model sees the same masks)
 *        and measures the loss over the picked tokens, in batches of @p batch_size. The model keeps
 *        no activations while it runs.
 */
[[nodiscard]] MaskedLMEvaluation EvaluateMaskedLM(EncoderLM& model, const TextTokenizer& tokenizer, DeviceBackend* backend,
                                                  const std::vector<std::string>& sequences, int64_t batch_size = 8,
                                                  uint64_t seed = 0, MaskedLMOptions options = {});

/**
 * @brief A head on an encoder's representations (last_hidden_state), for fine-tuning or probing.
 *        `Pooling::Mean` averages each sequence's residues (not `<cls>`, `<eos>` or padding) and
 *        applies a linear layer: `(N, L, hidden)` to `(N, outputs)`, for per-protein labels.
 *        `Pooling::PerResidue` applies it to every token: `(N, L, hidden)` to `(N, L, outputs)`,
 *        for per-residue labels (use -100 targets at `<cls>`, `<eos>` and padding).
 *
 * Train it alone on a frozen encoder (a probe), or with it: pass backward()'s result to
 * EncoderLM::backward_hidden(). Parameters are named `linear.weight` and `linear.bias`.
 */
class SequenceHead : public Module {
public:
    enum class Pooling { Mean, PerResidue };

    /** @brief The linear layer starts as transformers' heads do: weights from N(0, 0.02²), drawn
     *         from @p seed, and zero bias. */
    SequenceHead(int64_t hidden_size, int64_t outputs, Pooling pooling, DeviceBackend* backend, uint64_t seed = 0);

    /** @brief Which tokens Pooling::Mean averages: `(N, L)`, 1 for a residue. Set it before every
     *         forward() whose shape changes; ResidueMask() builds it. */
    void set_residue_mask(const Tensor& residues);

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    /** @brief Relevance at the representations, `(N, L, hidden)`: the linear layer's rule, then
     *         (for Mean) each residue's share of the average. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    void set_training(bool training) override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void release_activations() override;

    [[nodiscard]] LinearModule& linear() { return linear_; }
    [[nodiscard]] Pooling pooling() const { return pooling_; }

protected:
    /** @throws std::invalid_argument unless the input is `(N, L, hidden)`, or for Mean, the
     *          residue mask is set and matches `(N, L)`. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t hidden_, outputs_;
    Pooling pooling_;
    DeviceBackend* backend_;
    LinearModule linear_;
    std::vector<float> residues_;  ///< (N * L), Mean only
    Shape residues_shape_ = Shape({0});
    std::vector<float> last_input_;  ///< (N, L, hidden) on the host, Mean only
    Shape last_shape_ = Shape({0});
};

/** @brief The residue mask for SequenceHead::Pooling::Mean from token ids and a padding mask (or
 *         none): 1 for every real token that isn't `<cls>` or `<eos>`. */
[[nodiscard]] Tensor ResidueMask(const Tensor& ids, const Tensor* keep, const TextTokenizer& tokenizer, DeviceBackend* backend);

}  // namespace pulsatrix
