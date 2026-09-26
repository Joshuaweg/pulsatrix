/** @file transform.hpp
 *  @brief Sample-level preprocessing abstraction -- Transform, Compose, TransformDataset.
 *  @ingroup dl_modules
 */
#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "pulsatrix/dataset.hpp"

namespace pulsatrix {

/**
 * @brief A single sample-level preprocessing step (normalize, augment, tokenize, ...).
 * @note Plain pure-virtual functor, not NVI (unlike Module) -- there is no shared
 *       precondition worth centralizing yet for a sample-in/sample-out transform.
 */
class Transform {
public:
    virtual ~Transform() = default;

    /**
     * @brief Applies this transform to a sample.
     * @param sample Input sample.
     * @return The transformed sample.
     */
    [[nodiscard]] virtual Sample apply(Sample sample) const = 0;
};

/**
 * @brief Eager, ordered list of Transforms applied in sequence -- pulsatrix's analogue of
 *        torchvision.transforms.Compose. Deliberately simple: no fusion/graph, matching
 *        Compose's own upstream design; unlike Python, there is no GIL here for that
 *        simplicity to cost anything.
 */
class Compose : public Transform {
public:
    explicit Compose(std::vector<std::shared_ptr<Transform>> steps) : steps_(std::move(steps)) {}

    [[nodiscard]] Sample apply(Sample sample) const override {
        for (const auto& step : steps_) {
            sample = step->apply(std::move(sample));
        }
        return sample;
    }

private:
    std::vector<std::shared_ptr<Transform>> steps_;
};

/**
 * @brief Decorates a Dataset with a Transform, applied to every sample get() returns --
 *        lets Transform/Compose compose with any Dataset without modifying it or DataLoader.
 */
class TransformDataset : public Dataset {
public:
    TransformDataset(std::shared_ptr<Dataset> base, std::shared_ptr<Transform> transform)
        : base_(std::move(base)), transform_(std::move(transform)) {}

    [[nodiscard]] int64_t size() const override { return base_->size(); }

    [[nodiscard]] Sample get(int64_t index) const override { return transform_->apply(base_->get(index)); }

private:
    std::shared_ptr<Dataset> base_;
    std::shared_ptr<Transform> transform_;
};

}  // namespace pulsatrix
