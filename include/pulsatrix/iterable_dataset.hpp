/** @file iterable_dataset.hpp
 *  @brief Streaming dataset abstraction -- reset()/next() for sources with no random access.
 *  @ingroup dl_modules
 */
#pragma once

#include <optional>

#include "pulsatrix/dataset.hpp"

namespace pulsatrix {

/**
 * @brief Streaming dataset abstraction for sources with no random access or no known length
 *        (sharded files, generators) -- pulsatrix's analogue of PyTorch's IterableDataset /
 *        tf.data's source-op model. DataLoader treats this and Dataset via a common internal
 *        adapter (data_loader.hpp) so both share one fetch/collate path.
 * @note Sharding across multiple DataLoader workers (each worker owning a distinct stream
 *       shard) is out of scope for campaign_exai_dl_library_data_pipeline's Phase 1 --
 *       DataLoader guards num_workers > 1 against an IterableDataset rather than silently
 *       producing duplicate/missing records.
 */
class IterableDataset {
public:
    virtual ~IterableDataset() = default;

    /** @brief Resets iteration to the beginning. Called once per epoch by DataLoader. */
    virtual void reset() = 0;

    /**
     * @brief Fetches the next sample.
     * @return The next Sample, or std::nullopt once the stream is exhausted.
     */
    [[nodiscard]] virtual std::optional<Sample> next() = 0;
};

}  // namespace pulsatrix
