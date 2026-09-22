/** @file op_type.hpp
 *  @brief Closed set of operation categories every graph Node is tagged with.
 */
#pragma once

namespace exai {

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
    Composite
};

}  // namespace exai
