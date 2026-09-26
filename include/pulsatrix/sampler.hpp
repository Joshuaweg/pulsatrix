/** @file sampler.hpp
 *  @brief Index-order abstraction for DataLoader -- SequentialSampler, ShuffleSampler.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace pulsatrix {

/**
 * @brief Produces the order in which a DataLoader visits a Dataset's indices for one epoch.
 * @note reset(dataset_size) is called once per epoch (DataLoader::reset_epoch()); next()
 *       is called once per sample within that epoch, returning std::nullopt once every
 *       index has been produced.
 */
class Sampler {
public:
    virtual ~Sampler() = default;

    /**
     * @brief Begins a new epoch over a dataset of the given size.
     * @param dataset_size Number of samples to produce indices for. Must be >= 0 --
     *        internal invariant (PULSATRIX_ASSERT, not throw): DataLoader always calls
     *        this with Dataset::size()'s return value directly, never unvalidated
     *        external input.
     */
    virtual void reset(int64_t dataset_size) = 0;

    /**
     * @brief Fetches the next index in this epoch's order.
     * @return The next index, or std::nullopt once every index has been produced.
     */
    [[nodiscard]] virtual std::optional<int64_t> next() = 0;
};

/** @brief Visits indices [0, dataset_size) in ascending order. */
class SequentialSampler : public Sampler {
public:
    void reset(int64_t dataset_size) override;
    [[nodiscard]] std::optional<int64_t> next() override;

private:
    int64_t size_ = 0;
    int64_t position_ = 0;
};

/**
 * @brief Visits indices [0, dataset_size) in a seeded pseudo-random permutation --
 *        reproducible across runs given the same seed (std::mt19937), re-shuffled fresh
 *        each reset() call (a new epoch is a new permutation, not the same one repeated).
 */
class ShuffleSampler : public Sampler {
public:
    explicit ShuffleSampler(unsigned seed);
    void reset(int64_t dataset_size) override;
    [[nodiscard]] std::optional<int64_t> next() override;

private:
    unsigned seed_;
    std::mt19937 rng_;
    std::vector<int64_t> indices_;
    size_t position_ = 0;
};

}  // namespace pulsatrix
