#include "pulsatrix/caching_allocator.hpp"

#include <new>
#include <utility>

namespace pulsatrix {

CachingAllocator::CachingAllocator(RawAlloc raw_alloc, RawFree raw_free, Synchronize synchronize, size_t budget_bytes)
    : raw_alloc_(std::move(raw_alloc)),
      raw_free_(std::move(raw_free)),
      synchronize_(std::move(synchronize)),
      budget_(budget_bytes) {}

CachingAllocator::~CachingAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    synchronize_();
    for (auto& [cls, blocks] : free_blocks_) {
        for (void* p : blocks) {
            raw_free_(p);
        }
    }
    for (auto& [p, cls] : in_use_) {
        raw_free_(p);
    }
}

size_t CachingAllocator::round_size(size_t bytes) {
    constexpr size_t kSmallStep = 512;
    constexpr size_t kLarge = size_t{1} << 20;
    if (bytes < kLarge) {
        return (bytes + kSmallStep - 1) / kSmallStep * kSmallStep;
    }
    size_t power = kLarge;
    while (power <= bytes / 2) {
        power *= 2;
    }
    const size_t step = power / 8;
    return (bytes + step - 1) / step * step;
}

void CachingAllocator::release_cache_locked() {
    if (stats_.cached_bytes == 0) {
        return;
    }
    synchronize_();
    for (auto& [cls, blocks] : free_blocks_) {
        for (void* p : blocks) {
            raw_free_(p);
            ++stats_.raw_frees;
        }
        blocks.clear();
    }
    stats_.cached_bytes = 0;
}

void* CachingAllocator::allocate(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    const size_t cls = round_size(bytes);
    std::lock_guard<std::mutex> lock(mutex_);

    auto hit = free_blocks_.find(cls);
    void* p = nullptr;
    if (hit != free_blocks_.end() && !hit->second.empty()) {
        p = hit->second.back();
        hit->second.pop_back();
        stats_.cached_bytes -= cls;
        ++stats_.cache_hits;
    } else {
        if (budget_ != 0 && stats_.in_use_bytes + stats_.cached_bytes + cls > budget_) {
            release_cache_locked();
            if (stats_.in_use_bytes + cls > budget_) {
                throw std::bad_alloc();
            }
        }
        try {
            p = raw_alloc_(cls);
        } catch (const std::bad_alloc&) {
            if (stats_.cached_bytes == 0) {
                throw;
            }
            release_cache_locked();
            p = raw_alloc_(cls);  // a second failure propagates
        }
        ++stats_.raw_allocs;
    }
    in_use_.emplace(p, cls);
    stats_.in_use_bytes += cls;
    if (stats_.in_use_bytes > stats_.peak_in_use_bytes) {
        stats_.peak_in_use_bytes = stats_.in_use_bytes;
    }
    return p;
}

void CachingAllocator::free(void* ptr) noexcept {
    if (ptr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = in_use_.find(ptr);
    if (it == in_use_.end()) {
        return;  // not ours (or already freed): nothing to do
    }
    const size_t cls = it->second;
    in_use_.erase(it);
    stats_.in_use_bytes -= cls;
    free_blocks_[cls].push_back(ptr);
    stats_.cached_bytes += cls;
}

void CachingAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_cache_locked();
}

void CachingAllocator::set_budget(size_t budget_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    budget_ = budget_bytes;
}

CachingAllocator::Stats CachingAllocator::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}  // namespace pulsatrix
