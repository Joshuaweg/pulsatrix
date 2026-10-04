/** @file tagger_finetune_example.hpp
 *  @brief A full fine-tuning recipe (roadmap TRN-6): pretrain a tiny transformer tagger on one
 *         rule, save it, reload it and fine-tune every parameter on a related rule.
 *  @ingroup dl_modules
 *  @note Real pretrained models (SmolLM2, ResNet18) arrive with v1.2's import work (IO-3 to
 *        IO-5). Until then the "pretrained" model is one this recipe trains itself; the
 *        fine-tuning half is unchanged when a real checkpoint replaces it.
 *  @note It is per-token tagging rather than next-token prediction because attention has no
 *        causal mask yet; with bidirectional attention a language-model target would be visible
 *        in the input.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/embedding_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {

/** @brief Embedding -> TransformerBlock -> per-token Linear head: (N, L) tokens to (N, L, C) logits. */
class TinyTagger : public Module {
public:
    static constexpr int64_t kVocab = 8;  ///< token 0 is padding
    static constexpr int64_t kSeqLen = 8;
    static constexpr int64_t kClasses = 4;
    static constexpr int64_t kDim = 16;
    static constexpr int64_t kHeads = 2;
    static constexpr int64_t kFeedForward = 32;

    explicit TinyTagger(DeviceBackend* backend);

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    void set_training(bool training) override;

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    EmbeddingModule embed_;
    TransformerBlock block_;
    LinearModule head_;
    int64_t last_n_ = 0;
};

/** @brief Fills the tagger's parameters reproducibly on every platform: matrices uniform in
 *         +-1/sqrt(fan_in), normalization gains 1, biases 0. */
void InitTagger(TinyTagger& tagger, uint64_t seed);

/** @brief How a token's tag follows from it and the token before it (the first token's
 *         predecessor is 0). Tags are modulo kClasses. */
enum class TaggingRule { SumWithPrevious, DifferenceWithPrevious };

/** @brief Ragged sequences: real tokens 1..kVocab-1 followed by padding (input 0, target -100). */
struct TaggingBatch {
    Tensor inputs;   ///< (N, kSeqLen) token ids as whole-number floats
    Tensor targets;  ///< (N, kSeqLen) tags, -100 at padding
};

/** @brief A deterministic, platform-independent batch; lengths are 5 to kSeqLen. */
[[nodiscard]] TaggingBatch MakeTaggingBatch(TaggingRule rule, int64_t batch_size, uint64_t seed,
                                            DeviceBackend* backend = nullptr);

/** @brief Training settings. The defaults are the recipe's. */
struct FineTuneConfig {
    int64_t steps = 300;
    int64_t micro_batches = 2;        ///< accumulated per optimizer step (TRN-5)
    int64_t micro_batch_size = 8;
    float learning_rate = 1e-2f;
    int64_t warmup = 10;              ///< then cosine decay to step `steps` (TRN-4)
    float weight_decay = 0.01f;       ///< AdamW, none on biases and norms (TRN-1, TRN-2)
    float max_grad_norm = 1.0f;       ///< (TRN-3)
    uint64_t data_seed = 1000;
    int64_t stop_after = -1;          ///< stop after this many steps and checkpoint to checkpoint_path
    std::string checkpoint_path;
    std::string resume_from;          ///< resume model, optimizer and schedule from this checkpoint (IO-2)
};

/**
 * @brief Trains every parameter of `tagger` on `rule` with the whole v1.1 training stack: AdamW
 *        with parameter groups, warmup plus cosine, gradient accumulation normalized by the
 *        window's token count, and gradient clipping.
 * @return The loss of each step run (the window mean over its real tokens).
 * @note Each step's data depends only on (data_seed, step), so a resumed run sees exactly the
 *       batches an uninterrupted one would.
 */
[[nodiscard]] std::vector<float> TrainTagger(TinyTagger& tagger, TaggingRule rule, const FineTuneConfig& config,
                                             DeviceBackend* backend);

/** @brief Fraction of real tokens tagged correctly on `num_sequences` fresh sequences. */
[[nodiscard]] float TaggingAccuracy(TinyTagger& tagger, TaggingRule rule, int64_t num_sequences);

}  // namespace pulsatrix
