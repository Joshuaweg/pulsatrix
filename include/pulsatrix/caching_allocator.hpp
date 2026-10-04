/** @file caching_allocator.hpp
 *  @brief A caching device-memory allocator (roadmap HIP-3): freed blocks are kept and reused
 *         for later requests of the same size class instead of going back to the driver.
 *  @ingroup dl_modules
 *  @note Device-independent: it is given the raw allocate, free and synchronize operations, so
 *        HIPBackend can use it and its logic is tested in CPU-only builds.
 *  @note Safe on a single in-order stream: a block freed by the host was last used by work
 *        already queued on that stream, and its next user's work is queued after it. Memory is
 *        only returned to the driver after synchronizing.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace pulsatrix {

class CachingAllocator {
public:
    using RawAlloc = std::function<void*(size_t bytes)>;  ///< throws std::bad_alloc on failure
    using RawFree = std::function<void(void* ptr)>;
    using Synchronize = std::function<void()>;  ///< waits until queued device work is done

    struct Stats {
        size_t in_use_bytes = 0;
        size_t cached_bytes = 0;
        size_t peak_in_use_bytes = 0;
        size_t raw_allocs = 0;
        size_t raw_frees = 0;
        size_t cache_hits = 0;
    };

    /**
     * @param budget_bytes Most memory to hold, in use plus cached; 0 means no limit beyond what
     *        the device grants. A configured budget replaces asking the driver, which on APUs
     *        reports more memory than can actually be allocated.
     */
    CachingAllocator(RawAlloc raw_alloc, RawFree raw_free, Synchronize synchronize, size_t budget_bytes = 0);
    /** @brief Synchronizes, then returns every block to the device, cached or still in use. */
    ~CachingAllocator();
    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;

    /**
     * @brief A block of at least `bytes`, reused from the cache when one of its size class is free.
     * @return nullptr for 0 bytes.
     * @throws std::bad_alloc if the request doesn't fit the budget even with the cache released,
     *         or the device refuses it twice, once more after the cache was released.
     */
    [[nodiscard]] void* allocate(size_t bytes);

    /** @brief Returns a block to the cache. nullptr is ignored. */
    void free(void* ptr) noexcept;

    /** @brief Synchronizes, then returns every cached block to the device. */
    void empty_cache();

    /** @brief Changes the budget (0: unlimited). Takes effect at the next allocation. */
    void set_budget(size_t budget_bytes);

    [[nodiscard]] Stats stats() const;

    /** @brief The size class: 512-byte steps below 1 MiB, then steps of an eighth of the power of
     *         two below the size, so rounding wastes at most 12.5%. */
    [[nodiscard]] static size_t round_size(size_t bytes);

private:
    void release_cache_locked();

    RawAlloc raw_alloc_;
    RawFree raw_free_;
    Synchronize synchronize_;
    size_t budget_;
    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::vector<void*>> free_blocks_;  // size class -> blocks
    std::unordered_map<void*, size_t> in_use_;                     // block -> size class
    Stats stats_;
};

}  // namespace pulsatrix
