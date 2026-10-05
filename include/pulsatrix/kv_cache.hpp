/** @file kv_cache.hpp
 *  @brief Preallocated key/value cache for incremental attention (LLM-5).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief One attention layer's keys and values for the positions processed so far, in buffers
 *        allocated once for @p max_length positions, so generation never reallocates or copies
 *        the past.
 *
 * Keys are stored after QK-Norm and RoPE, at the K/V head count (not repeated for grouped-query
 * attention). MultiHeadAttentionModule::forward_cached writes new positions and advances
 * length(); everything before length() is attended to.
 */
class KVCache {
public:
    /**
     * @throws std::invalid_argument if any size is not positive.
     */
    KVCache(int64_t batch, int64_t num_kv_heads, int64_t head_dim, int64_t max_length, DeviceBackend* backend);

    [[nodiscard]] int64_t batch() const { return batch_; }
    [[nodiscard]] int64_t num_kv_heads() const { return num_kv_heads_; }
    [[nodiscard]] int64_t head_dim() const { return head_dim_; }
    [[nodiscard]] int64_t max_length() const { return max_length_; }
    /** @brief Positions cached so far; the next forward_cached input starts at this position. */
    [[nodiscard]] int64_t length() const { return length_; }

    /** @brief Forgets every position (the buffers stay allocated). */
    void reset() { length_ = 0; }
    /**
     * @brief Keeps only the first @p length positions, so a sequence can continue from a shared
     *        prefix.
     * @throws std::invalid_argument if length is negative or longer than what is cached.
     */
    void truncate(int64_t length);

    /** @brief `(batch, num_kv_heads, max_length, head_dim)`; positions from length() on are
     *         unused. */
    [[nodiscard]] Tensor& keys() { return keys_; }
    [[nodiscard]] Tensor& values() { return values_; }
    [[nodiscard]] const Tensor& keys() const { return keys_; }
    [[nodiscard]] const Tensor& values() const { return values_; }

    /** @brief Records @p count more positions as written. Called by forward_cached.
     *  @throws std::invalid_argument if that would pass max_length(). */
    void advance(int64_t count);

private:
    int64_t batch_;
    int64_t num_kv_heads_;
    int64_t head_dim_;
    int64_t max_length_;
    int64_t length_ = 0;
    Tensor keys_;
    Tensor values_;
};

}  // namespace pulsatrix
