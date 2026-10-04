#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <new>
#include <set>
#include <stdexcept>

#include "pulsatrix/caching_allocator.hpp"

namespace pulsatrix {
namespace {

// Host memory standing in for the device, with counters and an optional capacity, so the
// allocator's logic runs in CPU-only builds.
struct FakeDevice {
    size_t capacity = 0;  // 0 = unlimited
    size_t held = 0;
    int mallocs = 0, frees = 0, syncs = 0;
    std::set<void*> live;

    CachingAllocator make(size_t budget = 0) {
        return CachingAllocator(
            [this](size_t bytes) -> void* {
                if (capacity != 0 && held + bytes > capacity) throw std::bad_alloc();
                void* p = std::malloc(bytes);
                ++mallocs;
                held += bytes;
                live.insert(p);
                sizes[p] = bytes;
                return p;
            },
            [this](void* p) {
                ++frees;
                held -= sizes[p];
                live.erase(p);
                std::free(p);
            },
            [this] { ++syncs; }, budget);
    }
    std::map<void*, size_t> sizes;
};

TEST(CachingAllocatorTest, SizeClasses) {
    EXPECT_EQ(CachingAllocator::round_size(1), 512u);
    EXPECT_EQ(CachingAllocator::round_size(512), 512u);
    EXPECT_EQ(CachingAllocator::round_size(513), 1024u);
    EXPECT_EQ(CachingAllocator::round_size((1u << 20) - 1), 1u << 20);
    // From 1 MiB, steps of one eighth of the power of two below: waste is at most 12.5%.
    EXPECT_EQ(CachingAllocator::round_size((1u << 20) + 1), (1u << 20) + (1u << 17));
    EXPECT_EQ(CachingAllocator::round_size(3u << 20), 3u << 20);
    EXPECT_EQ(CachingAllocator::round_size((3u << 20) + 1), (3u << 20) + (1u << 18));
}

TEST(CachingAllocatorTest, AFreedBlockIsReusedForTheSameSizeClass) {
    FakeDevice dev;
    CachingAllocator a = dev.make();
    void* p = a.allocate(100);
    a.free(p);
    EXPECT_EQ(a.allocate(400), p);  // 100 and 400 bytes share the 512-byte class
    EXPECT_EQ(dev.mallocs, 1);
    EXPECT_EQ(a.stats().cache_hits, 1u);
    void* q = a.allocate(600);  // a different class: a new block
    EXPECT_NE(q, p);
    EXPECT_EQ(dev.mallocs, 2);
}

TEST(CachingAllocatorTest, TracksInUseCachedAndPeakBytes) {
    FakeDevice dev;
    CachingAllocator a = dev.make();
    void* p = a.allocate(1000);  // class 1024
    void* q = a.allocate(2000);  // class 2048
    EXPECT_EQ(a.stats().in_use_bytes, 3072u);
    a.free(p);
    EXPECT_EQ(a.stats().in_use_bytes, 2048u);
    EXPECT_EQ(a.stats().cached_bytes, 1024u);
    EXPECT_EQ(a.stats().peak_in_use_bytes, 3072u);
    a.free(q);
    EXPECT_EQ(a.stats().cached_bytes, 3072u);
}

TEST(CachingAllocatorTest, EmptyCacheSynchronizesThenReleasesOnlyCachedBlocks) {
    FakeDevice dev;
    CachingAllocator a = dev.make();
    void* kept = a.allocate(512);
    a.free(a.allocate(1024));
    const int syncs = dev.syncs;
    a.empty_cache();
    EXPECT_GT(dev.syncs, syncs);  // never release memory a queued kernel may still use
    EXPECT_EQ(dev.frees, 1);
    EXPECT_EQ(a.stats().cached_bytes, 0u);
    EXPECT_TRUE(dev.live.count(kept));
}

TEST(CachingAllocatorTest, GoingOverTheBudgetReleasesTheCacheFirst) {
    FakeDevice dev;
    CachingAllocator a = dev.make(/*budget=*/4096);
    a.free(a.allocate(2048));     // 2048 cached
    void* big = a.allocate(3000);  // class 3072: 2048 + 3072 > 4096, so the cache is released first
    EXPECT_NE(big, nullptr);
    EXPECT_EQ(dev.frees, 1);
    EXPECT_EQ(a.stats().cached_bytes, 0u);
    EXPECT_EQ(a.stats().in_use_bytes, 3072u);
}

TEST(CachingAllocatorTest, ARequestThatCannotFitTheBudgetThrowsAndChangesNothing) {
    FakeDevice dev;
    CachingAllocator a = dev.make(/*budget=*/4096);
    void* p = a.allocate(3000);
    EXPECT_THROW((void)a.allocate(2000), std::bad_alloc);
    EXPECT_EQ(a.stats().in_use_bytes, 3072u);
    a.free(p);
}

TEST(CachingAllocatorTest, ADeviceAllocationFailureReleasesTheCacheAndRetries) {
    FakeDevice dev;
    dev.capacity = 4096;
    CachingAllocator a = dev.make();  // no budget: only the device says no
    a.free(a.allocate(2048));
    void* p = a.allocate(3000);  // fails once (2048 held + 3072 > 4096), succeeds after release
    EXPECT_NE(p, nullptr);
    EXPECT_EQ(dev.frees, 1);
    dev.capacity = 1;
    EXPECT_THROW((void)a.allocate(8000), std::bad_alloc);  // nothing left to release
}

TEST(CachingAllocatorTest, ZeroBytesAndNull) {
    FakeDevice dev;
    CachingAllocator a = dev.make();
    EXPECT_EQ(a.allocate(0), nullptr);
    a.free(nullptr);
    EXPECT_EQ(dev.mallocs, 0);
}

TEST(CachingAllocatorTest, DestructionReleasesEverything) {
    FakeDevice dev;
    {
        CachingAllocator a = dev.make();
        (void)a.allocate(100);  // still in use: released too, since the device goes away
        a.free(a.allocate(5000));
    }
    EXPECT_TRUE(dev.live.empty());
    EXPECT_EQ(dev.mallocs, dev.frees);
}

}  // namespace
}  // namespace pulsatrix
