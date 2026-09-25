/** @file embedding_module.hpp
 *  @brief Lookup-table (row-select) layer -- y = W[index], batched (N, L) -> (N, L, embedding_dim).
 */
#pragma once

#include <initializer_list>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Embedding lookup table, rank-2 input (N, L) of float-encoded indices -> rank-3
 *        output (N, L, embedding_dim). Structurally unlike every other module in this
 *        codebase: forward is a pure selection (row copy), with no arithmetic mixing
 *        across input features.
 * @note **Float-indices design decision**: Tensor is float-only (no integer tensor type
 *       exists anywhere in this codebase). Each input element is resolved to an index via
 *       round-to-nearest (std::llround, not truncation -- a caller passing 2.9999999f due
 *       to float round-trip from an integer source should not silently land on index 2),
 *       then bounds-checked against [0, num_embeddings). This is a genuinely new pattern
 *       in this codebase, not reused from any existing module.
 * @note propagate_relevance sums relevance over the embedding dimension per (n, l)
 *       position (Arras et al. 2017, "Explaining Recurrent Neural Network Predictions in
 *       Sentiment Analysis" -- already cited in this charter for RNN/LSTM): a token's total
 *       relevance is the sum of its embedding vector's per-dimension relevance values --
 *       there is no further "input" beneath a discrete token id to redistribute to.
 *       Conserves exactly by construction.
 * @note backward() uses scatter-add gradient accumulation (the first module in this
 *       codebase needing it -- LinearModule/Conv2DModule's gradients are dense-matmul
 *       sums, not index-selected accumulation): multiple (n, l) positions referencing the
 *       same row each contribute additively into that row of weight_grad_. Gradient w.r.t.
 *       the input indices themselves is undefined (discrete, non-differentiable) --
 *       backward() returns an all-zero tensor matching the input shape, matching every
 *       mainstream framework's nn.Embedding behavior.
 */
class EmbeddingModule : public Module {
public:
    /**
     * @brief Constructs an embedding table with a zero-initialized weight matrix.
     * @param num_embeddings Number of rows (vocabulary size).
     * @param embedding_dim Row width.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this module.
     * @throws std::invalid_argument if num_embeddings <= 0 or embedding_dim <= 0 --
     *         external boundary (construction arguments can originate from Phase 5's
     *         Python bindings with no upstream validation).
     */
    EmbeddingModule(int64_t num_embeddings, int64_t embedding_dim, DeviceBackend* backend);

    /**
     * @brief Scatter-adds grad_output into the corresponding rows of weight_grad_.
     * @param grad_output Gradient w.r.t. this module's output. Must be (N, L, embedding_dim)
     *        matching the most recent forward() call's output shape.
     * @return An all-zero tensor matching the cached input shape (N, L) -- gradient w.r.t.
     *         discrete indices is undefined; this module never claims otherwise.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if grad_output's shape doesn't match the cached
     *         forward output shape.
     * @note Not yet backend-generic -- raw host loop. PULSATRIX_ASSERT(grad_output.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;

    /** @brief Embedding per charter's closed OpType set. */
    [[nodiscard]] OpType op_type() const override { return OpType::Embedding; }

    /** @brief Overwrites the weight buffer -- test/initialization use only. */
    void set_weight(std::initializer_list<float> values);
    /** @brief Vector overload for runtime-sized sources -- see Tensor's own vector ctor. */
    void set_weight(const std::vector<float>& values);

    [[nodiscard]] const Tensor& weight() const { return weight_; }
    [[nodiscard]] const Tensor& weight_grad() const { return weight_grad_; }

    /**
     * @brief Sum-over-embedding-dimension LRP relevance aggregation (Arras et al. 2017).
     * @param relevance_out Relevance at this module's output. Must be (N, L, embedding_dim)
     *        matching the most recent forward() call's output shape.
     * @param config Unused -- this rule has no tunable parameter.
     * @return Relevance at this module's input, shape (N, L): relevance_out summed over
     *         the embedding dimension at each (n, l) position. Conserves exactly by
     *         construction.
     * @throws std::logic_error if forward() has never been called.
     * @throws std::invalid_argument if relevance_out's shape doesn't match the cached
     *         forward output shape.
     */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;

    [[nodiscard]] std::vector<ParamRef> parameters() override { return {{&weight_, &weight_grad_}}; }

protected:
    /**
     * @brief The actual forward computation -- per-position row copy from weight_.
     * @throws std::invalid_argument if input isn't rank-2 (N, L), or any element
     *         round-resolves to an index outside [0, num_embeddings).
     * @note Not yet backend-generic -- raw host loop. PULSATRIX_ASSERT(input.device() ==
     *       DeviceType::Cpu) guards against silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    DeviceBackend* backend_;
    Tensor weight_;       // shape (num_embeddings, embedding_dim)
    Tensor weight_grad_;
    Shape last_input_shape_ = Shape({0});
    std::vector<int64_t> last_indices_;  // flat, N*L entries, resolved+bounds-checked
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
