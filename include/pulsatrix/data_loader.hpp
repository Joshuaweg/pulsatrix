/** @file data_loader.hpp
 *  @brief Orchestrates sampling, fetch, and collation into batches.
 *  @ingroup dl_modules
 */
#pragma once

#include <memory>
#include <optional>

#include "pulsatrix/collate.hpp"
#include "pulsatrix/dataset.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/iterable_dataset.hpp"
#include "pulsatrix/sampler.hpp"

namespace pulsatrix {

/** @brief Configuration for a DataLoader. */
struct DataLoaderOptions {
    int64_t batch_size = 1;
    bool shuffle = false;
    unsigned shuffle_seed = 42;
    /** @brief 0 = fully synchronous, no threads spawned (this phase's only exercised path). */
    int num_workers = 0;
    int64_t prefetch_batches = 2;
    bool drop_last = false;
    CollateFn collate_fn = DefaultCollate;
};

/**
 * @brief Orchestrates sampling and collation into batches -- pulsatrix's DataLoader
 *        (PyTorch DataLoader / torch::data::DataLoader analogue).
 * @note Phase 1 ships a fully synchronous implementation (num_workers=0 is the only
 *       exercised path): DataThreadPool/BoundedQueue exist (see those headers) but are not
 *       yet wired into this class's fetch/collate path -- wiring num_workers > 0 through a
 *       real staged pipeline is a dedicated, separately-tested later mission (Decision
 *       Point 7 in campaign_exai_dl_library_data_pipeline).
 */
class DataLoader {
public:
    /**
     * @brief Constructs a DataLoader over a random-access Dataset.
     * @throws std::invalid_argument if dataset is null or options.batch_size <= 0.
     */
    DataLoader(std::shared_ptr<Dataset> dataset, DeviceBackend* backend, DataLoaderOptions options = {});

    /**
     * @brief Constructs a DataLoader over a streaming IterableDataset.
     * @throws std::invalid_argument if dataset is null, options.batch_size <= 0, or
     *         options.num_workers > 1 -- sharded multi-worker streaming is out of scope
     *         for this phase (a single worker can safely consume the whole stream).
     */
    DataLoader(std::shared_ptr<IterableDataset> dataset, DeviceBackend* backend, DataLoaderOptions options = {});

    /** @brief Resets to the start of a new epoch (re-seeds/reshuffles the sampler, or resets the stream). */
    void reset_epoch();

    /**
     * @brief Fetches the next batch.
     * @return The next Batch, or std::nullopt once the epoch is exhausted (or, with
     *         drop_last=true, once fewer than batch_size samples remain).
     */
    [[nodiscard]] std::optional<Batch> next_batch();

    /**
     * @brief Number of batches per epoch.
     * @throws std::logic_error if this DataLoader was constructed from an IterableDataset
     *         (its length is unknown ahead of iteration).
     */
    [[nodiscard]] int64_t num_batches() const;

private:
    [[nodiscard]] std::optional<Sample> fetch_next_sample();

    std::shared_ptr<Dataset> dataset_;                  // non-null iff constructed from a Dataset
    std::shared_ptr<IterableDataset> iterable_dataset_;  // non-null iff constructed from an IterableDataset
    DeviceBackend* backend_;
    DataLoaderOptions options_;
    std::unique_ptr<Sampler> sampler_;  // only used when dataset_ is set
};

}  // namespace pulsatrix
