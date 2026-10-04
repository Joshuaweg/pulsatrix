#include "pulsatrix/data_loader.hpp"

#include "pulsatrix/determinism.hpp"

#include <stdexcept>

namespace pulsatrix {

namespace {
void ValidateCommonOptions(const DataLoaderOptions& options) {
    if (options.batch_size <= 0) {
        throw std::invalid_argument("DataLoader: batch_size must be positive");
    }
}
}  // namespace

DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, DeviceBackend* backend, DataLoaderOptions options)
    : dataset_(std::move(dataset)), backend_(backend), options_(std::move(options)) {
    if (!dataset_) {
        throw std::invalid_argument("DataLoader: dataset must not be null");
    }
    ValidateCommonOptions(options_);
    if (options_.shuffle) {
        const unsigned seed =
            options_.shuffle_seed ? *options_.shuffle_seed : static_cast<unsigned>(next_seed());
        sampler_ = std::make_unique<ShuffleSampler>(seed);
    } else {
        sampler_ = std::make_unique<SequentialSampler>();
    }
    reset_epoch();
}

DataLoader::DataLoader(std::shared_ptr<IterableDataset> dataset, DeviceBackend* backend, DataLoaderOptions options)
    : iterable_dataset_(std::move(dataset)), backend_(backend), options_(std::move(options)) {
    if (!iterable_dataset_) {
        throw std::invalid_argument("DataLoader: dataset must not be null");
    }
    ValidateCommonOptions(options_);
    if (options_.num_workers > 1) {
        throw std::invalid_argument(
            "DataLoader: num_workers > 1 is not supported for an IterableDataset in this phase "
            "(sharded streaming is out of scope for campaign_exai_dl_library_data_pipeline Phase 1)");
    }
    reset_epoch();
}

void DataLoader::reset_epoch() {
    if (dataset_) {
        sampler_->reset(dataset_->size());
    } else {
        iterable_dataset_->reset();
    }
}

std::optional<Sample> DataLoader::fetch_next_sample() {
    if (dataset_) {
        std::optional<int64_t> index = sampler_->next();
        if (!index.has_value()) {
            return std::nullopt;
        }
        return dataset_->get(*index);
    }
    return iterable_dataset_->next();
}

std::optional<Batch> DataLoader::next_batch() {
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(options_.batch_size));
    for (int64_t i = 0; i < options_.batch_size; ++i) {
        std::optional<Sample> sample = fetch_next_sample();
        if (!sample.has_value()) {
            break;
        }
        samples.push_back(std::move(*sample));
    }
    if (samples.empty()) {
        return std::nullopt;
    }
    if (options_.drop_last && static_cast<int64_t>(samples.size()) < options_.batch_size) {
        return std::nullopt;
    }
    return options_.collate_fn(std::move(samples), backend_);
}

int64_t DataLoader::num_batches() const {
    if (!dataset_) {
        throw std::logic_error("DataLoader::num_batches: undefined for an IterableDataset source (unknown length)");
    }
    int64_t total = dataset_->size();
    int64_t full_batches = total / options_.batch_size;
    int64_t remainder = total % options_.batch_size;
    if (remainder == 0 || options_.drop_last) {
        return full_batches;
    }
    return full_batches + 1;
}

}  // namespace pulsatrix
