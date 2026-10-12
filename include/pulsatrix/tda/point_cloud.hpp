/** @file point_cloud.hpp
 *  @brief The input to topological data analysis: a layer's activations as a set of points, one
 *         row per input or per position, gathered across batches.
 *  @ingroup tda
 *
 * Persistent homology measures the shape of a finite set of points. A layer's activations come as
 * batches of tensors, (batch, ...), from ActivationSnapshot or a model's hooks. PointCloud turns
 * those batches into one (n, dimension) set of points, and records where each point came from, so
 * a loop or a cluster found later can be traced back to the inputs and positions that form it.
 *
 * How a sample becomes points is the one choice to make (PointPooling):
 * - Flatten (the default): one point per sample, every dimension after the batch flattened. Right
 *   for an MLP layer or a classifier's logits.
 * - PerPosition: one point per token or pixel, with the feature axis as its coordinates. Right for
 *   the geometry of token representations.
 * - Mean, Max or Last: one point per sample, pooled over its positions. Last is a causal language
 *   model's summary of the whole input.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/activation_snapshot.hpp"
#include "pulsatrix/node.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief How PointCloud turns one sample's activations into points. */
enum class PointPooling {
    /** @brief One point per sample: every dimension after the batch, flattened. */
    Flatten,
    /** @brief One point per position (token, pixel): the feature axis is its coordinates. */
    PerPosition,
    /** @brief One point per sample: the mean over its positions. */
    Mean,
    /** @brief One point per sample: the maximum over its positions, per feature. */
    Max,
    /** @brief One point per sample: its last position, as a causal language model reads it. */
    Last,
};

/** @brief PointCloud's settings. */
struct PointCloudOptions {
    /** @brief How each sample becomes points. */
    PointPooling pooling = PointPooling::Flatten;
    /**
     * @brief The axis of an added tensor that holds the features, for every pooling but Flatten.
     *        Every other axis after the batch is a position. -1, the last axis, suits a sequence
     *        model's (batch, tokens, width); 1 suits a convolution's (batch, channels, height,
     *        width), whose positions are then its pixels in row-major order.
     */
    int64_t feature_axis = -1;
};

/** @brief Where one point of a PointCloud came from. */
struct PointOrigin {
    /** @brief The sample: its index over every batch added, in the order they were added. */
    int64_t sample = 0;
    /** @brief The position within the sample (row-major over the position axes) for PerPosition
     *         and Last; -1 when the point stands for the whole sample (Flatten, Mean, Max). */
    int64_t position = -1;
};

/**
 * @brief A set of points in R^dimension, built from activation batches.
 *
 * Add batches with add(); each is a tensor with the batch on its first axis. The first batch fixes
 * the dimension, and every later one must give points of the same dimension. Batches may differ in
 * their number of positions, so texts of different lengths can be added one by one.
 *
 * Points are stored on the host as float32, as the activations were; ToTensor() places them on a
 * backend for the distance computation.
 *
 * @code
 * PointCloud cloud({PointPooling::PerPosition});
 * model->set_hidden_state_hook([&](int64_t position, const Tensor& h) {
 *     if (position == 6) cloud.add(h);  // (1, tokens, width): one point per token
 *     return h;
 * });
 * for (const auto& text : corpus) (void)model->forward(Encode(text));
 * // cloud.size() points of cloud.dimension() = width; cloud.origins()[i] says which text and token.
 * @endcode
 */
class PointCloud {
public:
    /**
     * @brief An empty cloud.
     * @throws std::invalid_argument if options.pooling is out of range.
     */
    explicit PointCloud(PointCloudOptions options = {});

    /**
     * @brief Adds one batch.
     * @param batch shape (batch, ...). For every pooling but Flatten it needs a feature axis:
     *        rank 2 or more, with options().feature_axis not the batch axis. A rank-2 tensor has
     *        one position per sample.
     * @throws std::invalid_argument if batch has rank below 2, an empty dimension, a NaN or
     *         infinity, a feature axis out of range or on the batch axis, or points of a different
     *         dimension from the batches before it. A batch that throws adds nothing.
     */
    void add(const Tensor& batch);

    /**
     * @brief Adds one forward pass's activations at a node.
     * @param node One of snapshot.node_ids().
     * @throws std::invalid_argument as add(const Tensor&) does.
     */
    void add(const ActivationSnapshot& snapshot, NodeId node);

    /** @brief The number of points. */
    [[nodiscard]] int64_t size() const { return static_cast<int64_t>(origins_.size()); }

    /** @brief Each point's number of coordinates; 0 until the first batch. */
    [[nodiscard]] int64_t dimension() const { return dimension_; }

    /** @brief The number of samples added, over every batch. */
    [[nodiscard]] int64_t samples() const { return samples_; }

    /** @brief The settings it was built with. */
    [[nodiscard]] const PointCloudOptions& options() const { return options_; }

    /** @brief The points, row-major: size() rows of dimension() values. */
    [[nodiscard]] const std::vector<float>& values() const { return values_; }

    /** @brief Where each point came from, in the same order as values(). */
    [[nodiscard]] const std::vector<PointOrigin>& origins() const { return origins_; }

    /**
     * @brief The points as a (size(), dimension()) tensor.
     * @throws std::invalid_argument if the cloud is empty.
     */
    [[nodiscard]] Tensor ToTensor(DeviceBackend* backend) const;

private:
    PointCloudOptions options_;
    int64_t dimension_ = 0;
    int64_t samples_ = 0;
    std::vector<float> values_;
    std::vector<PointOrigin> origins_;
};

/**
 * @brief A cloud from one node of several forward passes, such as one snapshot per batch.
 * @throws std::invalid_argument as PointCloud::add does, or if snapshots is empty.
 */
[[nodiscard]] PointCloud MakePointCloud(const std::vector<ActivationSnapshot>& snapshots, NodeId node,
                                        PointCloudOptions options = {});

}  // namespace pulsatrix
