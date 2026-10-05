/** @file linear_probe.hpp
 *  @brief Linear probe -- trains one linear classifier to decode a binary concept from a
 *         layer's activations, answering "is this concept linearly represented here?".
 *  @ingroup mech_interp
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/determinism.hpp"
#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {

/**
 * @brief A linear probe: `LinearModule(activation_dim, 1)` + `BCEWithLogitsLoss`, trained
 *        on `(activation, binary concept label)` pairs. High post-training accuracy means
 *        the concept is linearly decodable from those activations; chance-level accuracy
 *        means it is not (at least not linearly).
 *
 * The activations come from an ActivationSnapshot (this campaign's Phase 1 output) in
 * production use, but nothing here depends on that: a probe consumes a plain
 * `(N, activation_dim)` batch, so it is equally usable against hand-built synthetic data --
 * which is exactly what the positive/negative-control verification in
 * `tests/linear_probe_test.cpp` needs.
 *
 * @note Mirrors XorNetwork/MnistConvNet's established shape rather than inventing a
 *       training abstraction: constructor takes a backend, `train_step()` drives one
 *       loss/backward/optimizer-step cycle, and the epoch loop is the caller's. The probe
 *       deliberately does **not** own its optimizer -- same division of responsibility
 *       XorNetwork::train_step already uses, which is what lets a caller choose SGD vs.
 *       Adam and keep one optimizer's state across a whole training run.
 * @note `train_step` is a template on the optimizer type rather than taking an
 *       `Optimizer&`: this codebase has no `Optimizer` base class. `SGDOptimizer`,
 *       `AdamOptimizer` and `AdamWOptimizer` share the `step(Module&)` /
 *       `zero_grad(Module&)` shape (a compile-time, duck-typed contract), so the template
 *       accepts any of them.
 * @note Small **seeded random** weight init, not zero init. Unlike XorNetwork there is no
 *       symmetry to break here -- a single linear layer does receive non-zero gradients
 *       from all-zero weights (`grad_W = X^T (sigmoid(0) - y)`), so zero init would train
 *       fine. The reason is accuracy(): with all-zero weight and bias every logit is
 *       exactly 0, so every example's predicted probability is exactly 0.5, landing the
 *       entire batch precisely on the decision threshold. Small random init removes that
 *       degenerate tie at step 0 while keeping the probe fully reproducible.
 */
class LinearProbe {
public:
    /**
     * @brief Constructs a probe over activations of a given dimension.
     * @param activation_dim Width of the activation vectors this probe reads. Must be > 0.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this probe.
     * @param seed RNG seed for weight initialization -- same seed gives the same probe.
     * @throws std::invalid_argument if activation_dim <= 0. External boundary: a probe's
     *         dimension is caller-supplied (typically read off a snapshot tensor's shape),
     *         and nothing downstream rejects it -- `LinearModule(0, 1, ...)` builds a
     *         well-formed zero-element weight and fails only later, confusingly.
     */
    /** @brief Seeded from the global seed stream (next_seed(), FND-7). */
    LinearProbe(int64_t activation_dim, DeviceBackend* backend)
        : LinearProbe(activation_dim, backend, static_cast<unsigned>(next_seed())) {}

    LinearProbe(int64_t activation_dim, DeviceBackend* backend, unsigned seed)
        : activation_dim_(activation_dim),
          // Clamped only so a rejected dimension can't reach LinearModule/Shape and throw
          // *their* message before the check below throws this class's own, clearer one --
          // member initialization necessarily runs before the constructor body.
          classifier_(activation_dim > 0 ? activation_dim : 1, 1, backend),
          loss_(backend) {
        if (activation_dim <= 0) {
            throw std::invalid_argument("LinearProbe: activation_dim must be positive");
        }
        std::mt19937 rng(seed);
        std::vector<float> weights(static_cast<size_t>(activation_dim));
        for (float& w : weights) {
            w = uniform_symmetric(rng, 0.01f);
        }
        classifier_.set_weight(weights);
        classifier_.set_bias(std::vector<float>{uniform_symmetric(rng, 0.01f)});
    }

    /**
     * @brief Runs one training step: forward, BCE-with-logits loss, backward, one optimizer
     *        update of the probe's weight/bias.
     * @tparam OptimizerT Any type exposing `step(Module&)` and `zero_grad(Module&)` --
     *         SGDOptimizer, AdamOptimizer or AdamWOptimizer (see the class-level note on why this is a
     *         template rather than an `Optimizer&`).
     * @param activation_batch Shape (N, activation_dim), N > 0.
     * @param label_batch Shape (N, 1), values in {0, 1}, same N.
     * @param optimizer Optimizer to update this probe's parameters with. Not owned.
     * @return The loss for this batch, measured *before* the update (mirroring
     *         XorNetwork::train_step's return contract).
     * @throws std::invalid_argument on any malformed batch -- see validate_batch().
     * @note Zeroes gradients before accumulating, for the exact reason XorNetwork::
     *       train_step documents: this codebase's modules accumulate parameter gradients
     *       across backward() calls until something resets them, so without this every
     *       step would silently sum onto every previous step's gradient.
     */
    template <typename OptimizerT>
    float train_step(const Tensor& activation_batch, const Tensor& label_batch, OptimizerT& optimizer) {
        validate_batch(activation_batch, label_batch, "LinearProbe::train_step");

        optimizer.zero_grad(classifier_);

        Tensor logits = classifier_.forward(activation_batch);
        const float loss_value = loss_.forward(logits, label_batch);

        Tensor grad_logits = loss_.backward();
        (void)classifier_.backward(grad_logits);  // the activations have no upstream to receive this

        optimizer.step(classifier_);
        return loss_value;
    }

    /**
     * @brief Fraction of the batch the probe classifies correctly.
     * @param activation_batch Shape (N, activation_dim), N > 0.
     * @param label_batch Shape (N, 1), values in {0, 1}, same N.
     * @return Correct predictions / N, in [0, 1].
     * @throws std::invalid_argument on any malformed batch -- see validate_batch().
     * @note Threshold convention: predicted class is 1 iff `sigmoid(logit) >= 0.5`, i.e.
     *       iff `logit >= 0`. Evaluated on the logit directly -- the sigmoid is monotonic,
     *       so the comparison is exactly equivalent and avoids an unnecessary exp() plus
     *       the rounding question of whether `sigmoid(0)` lands on 0.5f exactly. A label is
     *       read as positive iff it is >= 0.5, the same threshold, so soft/smoothed targets
     *       score sensibly rather than counting as neither class.
     * @note `const` although LinearModule::forward() is not: the probe's *logical* state is
     *       its learned parameters, which a forward-only scoring pass cannot change. The
     *       classifier is `mutable` purely so forward()'s internal input/pre-bias caches
     *       (backward()'s working state, not the probe's observable value) can be written.
     * @note Raw host loop over `Tensor::data()` -- the same convention every loss class in
     *       this codebase already uses internally, since Tensor has no elementwise
     *       comparison or reduction primitive. Adding one is out of this mission's scope.
     */
    [[nodiscard]] float accuracy(const Tensor& activation_batch, const Tensor& label_batch) const {
        validate_batch(activation_batch, label_batch, "LinearProbe::accuracy");

        const Tensor logits = classifier_.forward(activation_batch);
        const int64_t n = logits.numel();
        int64_t correct = 0;
        for (int64_t i = 0; i < n; ++i) {
            const bool predicted_positive = logits.data()[i] >= 0.0f;
            const bool labeled_positive = label_batch.data()[i] >= 0.5f;
            if (predicted_positive == labeled_positive) {
                ++correct;
            }
        }
        return static_cast<float>(correct) / static_cast<float>(n);
    }

    /** @brief Width of the activation vectors this probe reads. */
    [[nodiscard]] int64_t activation_dim() const { return activation_dim_; }

    /**
     * @brief The underlying linear classifier -- inspection (learned weights are the whole
     *        point of a probe: their direction *is* the decoded concept vector) and
     *        test-time weight injection.
     */
    [[nodiscard]] LinearModule& classifier() { return classifier_; }

    /** @brief Const overload of classifier(). */
    [[nodiscard]] const LinearModule& classifier() const { return classifier_; }

private:
    /** @brief Uniform in [-scale, scale), from a seeded engine, with no std distribution. */
    static float uniform_symmetric(std::mt19937& rng, float scale) {
        // std::uniform_real_distribution's mapping is implementation-defined, so identical
        // seeds would not give identical weights across standard libraries. This mapping is
        // fixed here, making a seeded probe reproducible everywhere.
        const float unit = static_cast<float>(rng() - std::mt19937::min()) /
                            static_cast<float>(std::mt19937::max() - std::mt19937::min() + 1ull);
        return (unit * 2.0f - 1.0f) * scale;
    }

    /**
     * @brief Validates an (activations, labels) batch pair at the probe's own entry point.
     * @throws std::invalid_argument if either operand is not rank-2, if the activation
     *         width is not activation_dim(), if the labels are not a single column, if the
     *         two batch sizes differ, or if the batch is empty.
     * @note Every one of these is classified **external boundary -> throw** per
     *       cpp_tdd/context_tdd_adversarial_boundary_testing.md, not PULSATRIX_ASSERT: a probe
     *       is driven by caller-assembled batches (an analyst's, ultimately a Python
     *       caller's), not by an already-validated internal call chain -- and per that
     *       file's "when genuinely unsure, default to external boundary" rule. Checked here
     *       rather than left to LinearModule::forward/BCEWithLogitsLoss::forward because a
     *       batch-size mismatch is invisible to both (each operand is individually
     *       well-formed) and an empty batch would otherwise report Module::forward's
     *       generic message, naming neither the probe nor which operand was wrong.
     */
    void validate_batch(const Tensor& activation_batch, const Tensor& label_batch, const char* method) const {
        const std::string where(method);
        if (activation_batch.rank() != 2) {
            throw std::invalid_argument(where + ": activation_batch must be rank-2 (N, activation_dim)");
        }
        if (label_batch.rank() != 2 || label_batch.shape().dim(1) != 1) {
            throw std::invalid_argument(where + ": label_batch must be rank-2 (N, 1)");
        }
        if (activation_batch.shape().dim(1) != activation_dim_) {
            throw std::invalid_argument(where + ": activation_batch width must equal activation_dim");
        }
        if (activation_batch.shape().dim(0) != label_batch.shape().dim(0)) {
            throw std::invalid_argument(where + ": activation_batch and label_batch must have the same batch size");
        }
        if (activation_batch.shape().dim(0) <= 0) {
            throw std::invalid_argument(where + ": batch must not be empty");
        }
    }

    int64_t activation_dim_;
    mutable LinearModule classifier_;
    BCEWithLogitsLoss loss_;
};

}  // namespace pulsatrix
