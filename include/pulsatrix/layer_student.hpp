/** @file layer_student.hpp
 *  @brief DistilBERT's recipe for a smaller language model (Sanh et al., 2019; roadmap KD-4): a
 *         student with some of the teacher's layers, started from those layers' weights, and a
 *         cosine loss that aligns the two models' hidden states.
 *  @ingroup dl_modules
 *
 * DistilBERT keeps BERT's width and halves its depth. The student starts as a copy of the teacher
 * with one layer out of two, so it begins close to the teacher, and trains on three losses:
 *
 *  - the distillation loss on the teacher's softened outputs (DistillationLoss);
 *  - the usual language-modeling loss on the data (TokenCrossEntropyLoss, or DistillationLoss's
 *    hard term);
 *  - a cosine loss, 1 − cos(h_s, h_t), between the two models' last hidden states, averaged over
 *    tokens (HiddenStateCosineLoss), which pulls the student's states toward the teacher's.
 *
 * DistilBERT weights them 5 : 2 : 1. In its ablation, the teacher's initialization and the
 * distillation loss mattered most, and the language-modeling loss least.
 *
 * CausalLM's backward() starts at the logits. To add the cosine loss's gradient at a layer's
 * output, put a HiddenStateTap there with CausalLM::set_intervention(): it passes the hidden
 * states through unchanged, keeps a copy of them, and adds the gradient it is given to the one
 * backward() brings down.
 *
 * In a pre-norm model such as Llama, the last layer's output is the residual stream before the
 * final RMSNorm, not the normalized states DistilBERT's BERT returns. The cosine ignores each
 * vector's length, but the stream's few large dimensions weigh on its direction.
 *
 * tests/layer_student_test.cpp checks the cosine loss against
 * tools/generate_layer_student_reference_values.py (numpy, gradients by central differences), and
 * that a student with every layer computes the teacher's logits.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief Layers 0, 2, 4, ... of a @p num_layers model: one out of two, DistilBERT's choice. */
[[nodiscard]] std::vector<int64_t> EveryOtherLayer(int64_t num_layers);

/**
 * @brief A CausalLM with @p teacher's config and weights but only the blocks in @p layers, in
 *        that order: the student's block j is a copy of the teacher's block layers[j]. The
 *        embedding, final norm and head are copied too; `layer_types` keeps the chosen layers'.
 * @note The student is a separate model on @p backend's device: training it leaves the teacher
 *       alone. The teacher's interventions aren't copied.
 * @throws std::invalid_argument if @p layers is empty, has an index out of range or repeats one,
 *         or a teacher parameter has no counterpart of the same shape (a teacher with LoRA
 *         adapters, say).
 */
[[nodiscard]] std::unique_ptr<CausalLM> MakeStudentFromLayers(CausalLM& teacher, const std::vector<int64_t>& layers,
                                                              DeviceBackend* backend);

/**
 * @brief An identity intervention that keeps the hidden states it sees and adds a gradient to the
 *        one flowing back through it: the way to train a CausalLM on a loss of a layer's output.
 * @note Each forward() replaces the kept states; add_gradient() applies to the next backward()
 *       only. Relevance passes through unchanged.
 */
class HiddenStateTap : public ResidualIntervention {
public:
    explicit HiddenStateTap(DeviceBackend* backend);

    [[nodiscard]] Tensor forward_at(const Tensor& hidden, int64_t start, int64_t prompt_length) override;
    /** @brief @p grad_output plus the gradient given to add_gradient(), if any, which is then
     *         dropped. */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] bool supports_lrp_rule(LRPRule) const override { return true; }
    [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void release_activations() override;

    /** @brief The last forward()'s hidden states, (N, L, hidden_size). @throws std::logic_error
     *         before a forward(). */
    [[nodiscard]] const Tensor& hidden() const;
    /** @brief A gradient for the next backward() to add, the hidden states' shape. @throws
     *         std::invalid_argument for another shape or device, std::logic_error before a forward(). */
    void add_gradient(const Tensor& gradient);

private:
    DeviceBackend* backend_;
    Tensor hidden_;
    Tensor extra_;
    bool has_hidden_ = false;
    bool has_extra_ = false;
};

/**
 * @brief DistilBERT's cosine loss: the mean over kept rows of 1 − cos(s, t), with s and t the
 *        student's and teacher's vectors, (..., width); PyTorch's CosineEmbeddingLoss with target 1.
 * @note Computed on the host in double precision; the gradient comes back on the student's device.
 *       A row whose student or teacher vector is zero counts as cos = 0, with no gradient.
 */
class HiddenStateCosineLoss {
public:
    explicit HiddenStateCosineLoss(DeviceBackend* backend);
    /**
     * @param mask One value per row, 0 to leave it out (padding) and anything else to keep it;
     *        empty keeps every row.
     * @throws std::invalid_argument if the shapes differ, the rank is below 2, the tensors are on
     *         different devices, the mask has the wrong length, or it keeps no row.
     */
    [[nodiscard]] float forward(const Tensor& student, const Tensor& teacher, const std::vector<float>& mask = {});
    /** @brief Gradient with respect to the student's vectors, their shape. @throws
     *         std::logic_error before forward(). */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    Shape shape_ = Shape({0});
    DeviceType device_ = DeviceType::Cpu;
    std::vector<double> grad_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
