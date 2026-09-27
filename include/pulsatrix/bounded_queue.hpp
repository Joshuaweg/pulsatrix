/** @file bounded_queue.hpp
 *  @brief Fixed-capacity thread-safe blocking queue -- the staged-pipeline backbone.
 *  @ingroup data_pipeline
 */
#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <utility>

namespace pulsatrix {

/**
 * @brief Fixed-capacity thread-safe queue with blocking push/pop -- the bounded buffer
 *        between pulsatrix data-pipeline stages (campaign_exai_dl_library_data_pipeline).
 *        Not tied to any Dataset/DataLoader type; deliberately generic.
 * @note This is pulsatrix's first concurrent code path (data_pipeline campaign, Risk
 *       Register: "Threading model correctness"). No production DataLoader path exercises
 *       this class yet in Phase 1 -- DataLoader defaults to num_workers=0 (fully
 *       synchronous); wiring this into DataLoader's fetch/collate stages is a later,
 *       separately-tested mission (Decision Point 7).
 */
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity) {}

    /**
     * @brief Pushes an item, blocking while the queue is full.
     * @note A push after close() is silently dropped -- close() means "no more producers",
     *       so a caller racing a push against its own close() call has a logic error, not
     *       something this queue can meaningfully recover from.
     */
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return closed_ || items_.size() < capacity_; });
        if (closed_) {
            return;
        }
        items_.push_back(std::move(item));
        lock.unlock();
        not_empty_.notify_one();
    }

    /**
     * @brief Pops the next item, blocking while the queue is empty and not closed.
     * @return The next item, or std::nullopt once closed AND drained.
     */
    [[nodiscard]] std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
        if (items_.empty()) {
            return std::nullopt;
        }
        T item = std::move(items_.front());
        items_.pop_front();
        lock.unlock();
        not_full_.notify_one();
        return item;
    }

    /** @brief Signals no more pushes will occur; unblocks every waiting push()/pop(). */
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::deque<T> items_;
    size_t capacity_;
    std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    bool closed_ = false;
};

}  // namespace pulsatrix
