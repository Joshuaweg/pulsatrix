#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "pulsatrix/bounded_queue.hpp"

namespace pulsatrix {
namespace {

TEST(BoundedQueueTest, PushThenPopReturnsSameItem) {
    BoundedQueue<int> queue(4);
    queue.push(42);
    auto item = queue.pop();
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(*item, 42);
}

TEST(BoundedQueueTest, PreservesFifoOrder) {
    BoundedQueue<int> queue(4);
    queue.push(1);
    queue.push(2);
    queue.push(3);
    EXPECT_EQ(*queue.pop(), 1);
    EXPECT_EQ(*queue.pop(), 2);
    EXPECT_EQ(*queue.pop(), 3);
}

TEST(BoundedQueueTest, PopBlocksUntilPushFromAnotherThread) {
    BoundedQueue<int> queue(4);
    std::atomic<bool> popped{false};

    std::thread consumer([&] {
        auto item = queue.pop();
        ASSERT_TRUE(item.has_value());
        EXPECT_EQ(*item, 7);
        popped = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(popped.load());
    queue.push(7);
    consumer.join();
    EXPECT_TRUE(popped.load());
}

TEST(BoundedQueueTest, PushBlocksWhenAtCapacity) {
    BoundedQueue<int> queue(1);
    queue.push(1);
    std::atomic<bool> pushed_second{false};

    std::thread producer([&] {
        queue.push(2);
        pushed_second = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(pushed_second.load());
    EXPECT_EQ(*queue.pop(), 1);
    producer.join();
    EXPECT_TRUE(pushed_second.load());
    EXPECT_EQ(*queue.pop(), 2);
}

TEST(BoundedQueueTest, CloseUnblocksWaitingPopWithNullopt) {
    BoundedQueue<int> queue(4);
    std::atomic<bool> finished{false};
    std::optional<int> result = 999;

    std::thread consumer([&] {
        result = queue.pop();
        finished = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_FALSE(finished.load());
    queue.close();
    consumer.join();
    EXPECT_TRUE(finished.load());
    EXPECT_FALSE(result.has_value());
}

TEST(BoundedQueueTest, PopDrainsRemainingItemsAfterClose) {
    BoundedQueue<int> queue(4);
    queue.push(1);
    queue.push(2);
    queue.close();

    EXPECT_EQ(*queue.pop(), 1);
    EXPECT_EQ(*queue.pop(), 2);
    EXPECT_FALSE(queue.pop().has_value());
}

}  // namespace
}  // namespace pulsatrix
