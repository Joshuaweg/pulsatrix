#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "pulsatrix/data_thread_pool.hpp"

namespace pulsatrix {
namespace {

TEST(DataThreadPoolTest, SubmittedTasksAllResolveCorrectly) {
    DataThreadPool pool(4);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 20; ++i) {
        futures.push_back(pool.submit([i] { return i * i; }));
    }
    for (int i = 0; i < 20; ++i) {
        EXPECT_EQ(futures[static_cast<size_t>(i)].get(), i * i);
    }
}

TEST(DataThreadPoolTest, SingleThreadPoolStillExecutesAllTasks) {
    DataThreadPool pool(1);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.submit([&counter] { counter.fetch_add(1); }));
    }
    for (auto& f : futures) {
        f.get();
    }
    EXPECT_EQ(counter.load(), 10);
}

TEST(DataThreadPoolTest, TaskExceptionPropagatesThroughFuture) {
    DataThreadPool pool(2);
    auto future = pool.submit([]() -> int { throw std::runtime_error("boom"); });
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(DataThreadPoolTest, DestructorJoinsCleanlyAfterPendingWorkCompletes) {
    std::atomic<int> counter{0};
    {
        DataThreadPool pool(3);
        for (int i = 0; i < 30; ++i) {
            pool.submit([&counter] { counter.fetch_add(1); });
        }
    }  // destructor runs here -- must wait for all queued tasks to complete
    EXPECT_EQ(counter.load(), 30);
}

}  // namespace
}  // namespace pulsatrix
