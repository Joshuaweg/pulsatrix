#include "pulsatrix/sampler.hpp"

#include <algorithm>
#include <numeric>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

void SequentialSampler::reset(int64_t dataset_size) {
    PULSATRIX_ASSERT(dataset_size >= 0);
    size_ = dataset_size;
    position_ = 0;
}

std::optional<int64_t> SequentialSampler::next() {
    if (position_ >= size_) {
        return std::nullopt;
    }
    return position_++;
}

ShuffleSampler::ShuffleSampler(unsigned seed) : seed_(seed), rng_(seed) {}

void ShuffleSampler::reset(int64_t dataset_size) {
    PULSATRIX_ASSERT(dataset_size >= 0);
    indices_.resize(static_cast<size_t>(dataset_size));
    std::iota(indices_.begin(), indices_.end(), 0);
    std::shuffle(indices_.begin(), indices_.end(), rng_);
    position_ = 0;
}

std::optional<int64_t> ShuffleSampler::next() {
    if (position_ >= indices_.size()) {
        return std::nullopt;
    }
    return indices_[position_++];
}

}  // namespace pulsatrix
