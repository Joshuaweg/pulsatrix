#include "pulsatrix/kv_cache.hpp"

#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

int64_t Positive(int64_t v, const char* what) {
    if (v <= 0) {
        throw std::invalid_argument(std::string("KVCache: ") + what + " must be positive");
    }
    return v;
}

}  // namespace

KVCache::KVCache(int64_t batch, int64_t num_kv_heads, int64_t head_dim, int64_t max_length, DeviceBackend* backend)
    : batch_(Positive(batch, "batch")),
      num_kv_heads_(Positive(num_kv_heads, "num_kv_heads")),
      head_dim_(Positive(head_dim, "head_dim")),
      max_length_(Positive(max_length, "max_length")),
      keys_(Shape({batch, num_kv_heads, max_length, head_dim}), backend),
      values_(Shape({batch, num_kv_heads, max_length, head_dim}), backend) {}

void KVCache::truncate(int64_t length) {
    if (length < 0 || length > length_) {
        throw std::invalid_argument("KVCache::truncate: length must be in [0, length()]");
    }
    length_ = length;
}

void KVCache::advance(int64_t count) {
    if (count < 0 || length_ + count > max_length_) {
        throw std::invalid_argument("KVCache: " + std::to_string(length_ + count) +
                                    " positions don't fit in a cache of " + std::to_string(max_length_));
    }
    length_ += count;
}

}  // namespace pulsatrix
