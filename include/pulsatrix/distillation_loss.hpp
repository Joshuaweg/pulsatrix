/** @file distillation_loss.hpp
 *  @brief Knowledge distillation (roadmap KD-1): a student learns from a teacher's softened
 *         outputs (Hinton, Vinyals and Dean, 2015), optionally mixed with the hard labels.
 *  @ingroup dl_modules
 *
 * With temperature T, student logits s, teacher logits t and p = softmax(·/T), each row (one
 * example, or one token) contributes
 *
 *   soft = T² · KL(p_t ‖ p_s)        (forward KL, Hinton's), or
 *   soft = T² · KL(p_s ‖ p_t)        (reverse KL, MiniLLM's mode-seeking choice for LLMs),
 *
 * and with targets the loss is alpha · soft + (1 − alpha) · cross-entropy(s, target) at T = 1.
 * The T² keeps the soft term's gradient the same size as the hard term's as T changes: the
 * gradient of KL with respect to s is (p_s − p_t) / T, so T² · KL's is T · (p_s − p_t).
 *
 * The forward soft term with the mean over rows is PyTorch's
 * `F.kl_div(F.log_softmax(s / T, -1), F.softmax(t / T, -1), reduction="batchmean") * T * T` for
 * (N, classes) logits. For (batch, length, classes) logits the mean is over tokens, not
 * sequences, which is what distillation of language models does (batchmean would divide by the
 * batch size alone). tests/distillation_loss_test.cpp checks values and gradients against
 * tools/generate_distillation_reference_values.py.
 */
#pragma once

#include <cstdint>

#include "pulsatrix/tensor.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {

/** @brief Which way round the soft term's KL divergence goes. */
enum class DistillationDivergence {
    Forward,  ///< KL(teacher ‖ student): the student covers every mode the teacher puts mass on
    Reverse,  ///< KL(student ‖ teacher): the student concentrates on the teacher's main modes
};

/** @brief DistillationLoss's settings. */
struct DistillationOptions {
    /** @brief Softens both distributions; 1 compares them as they are. Must be positive. */
    float temperature = 2.0f;
    /** @brief The soft term's weight when there are targets; the hard term gets 1 − alpha.
     *         Without targets the loss is the soft term alone. Must be in [0, 1]. */
    float alpha = 0.5f;
    DistillationDivergence divergence = DistillationDivergence::Forward;
    /** @brief Rows whose target is this are left out of both terms (padding). */
    int64_t ignore_index = TokenCrossEntropyLoss::kIgnoreIndex;
};

/**
 * @brief Distillation loss between a student's and a teacher's logits, (..., classes): every
 *        row along the last dimension is one distribution.
 * @note Device-generic: everything runs on the logits' device through softmax_rows,
 *       logsumexp_rows and axpby. forward() reads back the loss's parts; with targets it also reads
 *       the targets, one float per row, to find the rows to keep.
 * @note The teacher's logits get no gradient: the teacher is fixed.
 */
class DistillationLoss {
public:
    explicit DistillationLoss(DeviceBackend* backend, DistillationOptions options = {});

    /**
     * @brief The soft term alone, over every row.
     * @param normalizer What the summed loss is divided by; 0 (the default) means the number of
     *        rows. Pass the whole window's row count to accumulate gradients over micro-batches.
     * @throws std::invalid_argument if the logits aren't (..., classes) with rank >= 2, the
     *         teacher's shape isn't the student's, they're on different devices, or the
     *         normalizer is negative or not finite.
     */
    [[nodiscard]] float forward(const Tensor& student_logits, const Tensor& teacher_logits, float normalizer = 0.0f);

    /**
     * @brief alpha · soft + (1 − alpha) · hard over the rows whose target isn't the ignore index.
     * @param targets The logits' leading shape, whole-number floats, as for TokenCrossEntropyLoss.
     * @param normalizer 0 (the default) means the number of kept rows.
     * @throws std::invalid_argument as the other overload, or for targets TokenCrossEntropyLoss
     *         rejects.
     */
    [[nodiscard]] float forward(const Tensor& student_logits, const Tensor& teacher_logits, const Tensor& targets,
                                float normalizer = 0.0f);

    /** @brief Gradient w.r.t. the student's logits, same shape; zero on ignored rows.
     *  @throws std::logic_error before forward(). */
    [[nodiscard]] Tensor backward() const;

    /** @brief The last forward()'s parts, each divided by the normalizer and unweighted: the
     *         soft term (T² · KL) and the hard term (0 without targets). */
    [[nodiscard]] float soft_loss() const { return soft_loss_; }
    [[nodiscard]] float hard_loss() const { return hard_loss_; }
    /** @brief Rows the last forward() kept. */
    [[nodiscard]] int64_t num_rows() const { return num_rows_; }

    [[nodiscard]] const DistillationOptions& options() const { return options_; }

private:
    float Run(const Tensor& student_logits, const Tensor& teacher_logits, const Tensor* targets, float normalizer);

    DeviceBackend* backend_;
    DistillationOptions options_;
    TokenCrossEntropyLoss hard_;
    Tensor soft_grad_;  // (kept rows, classes), already scaled by the normalizer
    Tensor kept_;       // (kept rows, 1) row indices, when rows were dropped
    Shape logits_shape_ = Shape({0});
    float soft_loss_ = 0.0f;
    float hard_loss_ = 0.0f;
    int64_t num_rows_ = 0;
    bool all_rows_ = true;
    bool has_targets_ = false;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
