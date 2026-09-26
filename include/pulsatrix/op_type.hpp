/** @file op_type.hpp
 *  @brief Closed set of operation categories every graph Node is tagged with.
 *  @ingroup dl_modules
 */
#pragma once

namespace pulsatrix {

/**
 * @brief The op-type tag a Node carries. Charter Part 2 §3: nodes are tagged by a small
 *        closed set of op types rather than by concrete layer class, so a query like
 *        "find the last conv layer's activations" (Grad-CAM) works by querying op type,
 *        not by string-matching a layer name.
 * @note This enum grows only when a genuinely new operation category is needed -- it is
 *       not meant to enumerate every concrete Module subtype (LinearModule and a future
 *       EmbeddingModule might both tag their nodes Linear if they're mathematically the
 *       same operation).
 */
enum class OpType {
    Linear,
    Conv,
    Activation,
    Elementwise,
    Reduction,
    Normalization,
    Pooling,
    Embedding,
    Composite,
    Recurrent,
    /**
     * @brief Multi-head (scaled dot-product) attention -- Phase 3's MultiHeadAttentionModule.
     * @note Justification for growing the enum here (contrast RoPEModule, which deliberately
     *       reused `Elementwise`): attention is a genuinely new *operation category*, not a
     *       new spelling of an existing one. It contracts two learned projections against
     *       each other (`Q @ K^T`) and re-mixes a third along the sequence axis -- an
     *       input-dependent, content-addressed mixing across positions that no existing
     *       category describes. It is not `Linear` (the mixing weights are computed from the
     *       input, not stored), not `Composite` (`Composite` means "a container of other
     *       modules with no math of its own", which SequentialModule is and this is not --
     *       this module owns the two batched matmuls, the scale and the bilinear LRP rule),
     *       and not `Recurrent` (no state carried across steps; all positions are attended
     *       in parallel). The charter's stated reason for the enum -- "find the last conv
     *       layer"-style graph queries -- is exactly the use case that needs
     *       "find the attention blocks" to be answerable.
     */
    Attention
};

}  // namespace pulsatrix
