/** @file feature_distillation.hpp
 *  @brief Distilling a teacher's intermediate features, not only its outputs (roadmap KD-3):
 *         FitNets' hint loss (Romero et al., ICLR 2015) and attention transfer (Zagoruyko and
 *         Komodakis, ICLR 2017).
 *  @ingroup dl_modules
 *
 * DistillationLoss matches a student's outputs to its teacher's. These losses match what happens
 * on the way:
 *
 *  - **HintLoss:** the mean squared error between a student layer's features, mapped to the
 *    teacher's width by a learned linear regressor, and a teacher layer's features. FitNets trains
 *    the student up to that layer this way first, then the whole student with DistillationLoss.
 *  - **AttentionTransferLoss:** for convolutional features (N, C, H, W), each layer's spatial
 *    attention map Q = mean over channels of |A|^p, flattened and L2-normalized per example; the
 *    loss is the mean squared difference of the student's and teacher's maps. The channel counts
 *    may differ; the spatial sizes must match.
 *
 * Both losses return the gradient with respect to the student's features, at the layer they come
 * from. To train through it, run the student in two parts (two SequentialModules over its
 * layers, say) and add this gradient to the one the second part's backward() returns before
 * calling the first part's backward().
 *
 * tests/feature_distillation_test.cpp checks both against
 * tools/generate_feature_distillation_reference_values.py, which writes attention transfer the way
 * the authors' code does and takes gradients by central differences.
 *
 * @note Not here: MiniLM's distillation of self-attention distributions. It needs the gradient of
 *       the loss to flow back through MultiHeadAttentionModule's attention probabilities, which
 *       the module doesn't take.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief FitNets' hint loss: mean((r(student) − teacher)²), with r a learned linear map from the
 *        student's feature width to the teacher's.
 * @note Features are (..., width): every row along the last dimension is one feature vector, and
 *       the leading shapes must match.
 * @note The regressor is a LinearModule with a bias, started the way PyTorch starts nn.Linear
 *       (weights and bias uniform in ±1/√student_width) from `seed`: LinearModule itself starts at
 *       zero, which would pass no gradient back to the student. Step it with the student's
 *       optimizer, or its own: backward() accumulates its gradients.
 */
class HintLoss {
public:
    /** @brief With a regressor from @p student_width to @p teacher_width. */
    HintLoss(int64_t student_width, int64_t teacher_width, DeviceBackend* backend, uint64_t seed = 0);
    /** @brief Without a regressor: the student's features are compared as they are, so the widths
     *         must match. */
    explicit HintLoss(DeviceBackend* backend);

    /**
     * @brief Computes the loss and caches what backward() needs.
     * @throws std::invalid_argument if a tensor's rank is below 2, the leading shapes differ, a
     *         width doesn't match the regressor's (or, without one, the two widths differ), or
     *         the tensors are on different devices.
     */
    [[nodiscard]] float forward(const Tensor& student_features, const Tensor& teacher_features);

    /** @brief Gradient w.r.t. the student's features, their shape; also accumulates the
     *         regressor's parameter gradients. @throws std::logic_error before forward(). */
    [[nodiscard]] Tensor backward();

    /** @brief The regressor, or nullptr without one. */
    [[nodiscard]] LinearModule* regressor() { return regressor_.get(); }

private:
    DeviceBackend* backend_;
    std::unique_ptr<LinearModule> regressor_;
    MSELoss mse_;
    Shape student_shape_ = Shape({0});
    bool has_forwarded_ = false;
};

/**
 * @brief Attention transfer: mean over examples and positions of (q_s − q_t)², where q is the
 *        L2-normalized map mean_c |A_c|^p of each example's features.
 * @note Features are (N, C, ...): the dimensions after the channels are flattened into positions.
 *       Computed on the host in double precision, whatever the tensors' device; the gradient is
 *       returned on the student features' device.
 * @note p = 2, the default and the paper's choice, is the authors' `x.pow(2).mean(1)`; other p
 *       use |A|^p, the paper's definition.
 */
class AttentionTransferLoss {
public:
    explicit AttentionTransferLoss(DeviceBackend* backend, double p = 2.0);

    /**
     * @throws std::invalid_argument if a tensor's rank is below 3, the batch sizes or numbers of
     *         positions differ, or the tensors are on different devices.
     */
    [[nodiscard]] float forward(const Tensor& student_features, const Tensor& teacher_features);

    /** @brief Gradient w.r.t. the student's features, their shape. @throws std::logic_error
     *         before forward(). */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    double p_;
    Tensor student_;
    std::vector<double> q_s_, q_t_, norms_;
    int64_t batch_ = 0, channels_ = 0, positions_ = 0;
    bool has_forwarded_ = false;
};

/**
 * @brief The attention map AttentionTransferLoss compares: (N, positions), each row
 *        L2-normalized, on @p features' device.
 * @throws std::invalid_argument if the rank is below 3.
 */
[[nodiscard]] Tensor AttentionMap(const Tensor& features, DeviceBackend* backend, double p = 2.0);

}  // namespace pulsatrix
