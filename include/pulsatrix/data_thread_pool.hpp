/** @file data_thread_pool.hpp
 *  @brief Minimal generic thread pool for CPU-side data pipeline work.
 *  @ingroup dl_modules
 */
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace pulsatrix {

/**
 * @brief Minimal, generic thread pool for CPU-side data pipeline work (fetch, decode,
 *        transform, collate stages) -- campaign_exai_dl_library_data_pipeline's staged-
 *        pipeline backbone alongside BoundedQueue. Deliberately not tied to
 *        Dataset/DataLoader types.
 * @note This is pulsatrix's first concurrent code path. No production DataLoader path
 *       exercises this class yet in Phase 1 -- see bounded_queue.hpp's matching note.
 */
class DataThreadPool {
public:
    explicit DataThreadPool(unsigned num_threads);
    ~DataThreadPool();

    DataThreadPool(const DataThreadPool&) = delete;
    DataThreadPool& operator=(const DataThreadPool&) = delete;

    /**
     * @brief Enqueues a task for execution by a worker thread.
     * @return A future resolving to the task's return value (or exception, if it throws).
     * @throws std::runtime_error if the pool has already been stopped (destructor
     *         entered) -- external boundary: a caller submitting after shutdown has begun
     *         is a real usage error a well-formed caller can still trigger under a race.
     */
    template <typename F>
    auto submit(F&& task) -> std::future<std::invoke_result_t<F>> {
        using ReturnType = std::invoke_result_t<F>;
        auto packaged = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(task));
        std::future<ReturnType> result = packaged->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("DataThreadPool::submit: pool is stopped");
            }
            tasks_.emplace([packaged]() { (*packaged)(); });
        }
        cv_.notify_one();
        return result;
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

}  // namespace pulsatrix
