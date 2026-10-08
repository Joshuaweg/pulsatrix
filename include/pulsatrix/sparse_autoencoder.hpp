/** @file sparse_autoencoder.hpp
 *  @brief Sparse autoencoder -- reconstructs an activation through an overcomplete,
 *         L1-penalized hidden layer, decomposing it into a larger, sparser basis.
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
#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/relu_module.hpp"

namespace pulsatrix {

/**
 * @brief A sparse autoencoder (SAE): `LinearModule(dim, hidden_dim)` -> `ReluModule` ->
 *        `LinearModule(hidden_dim, dim)`, trained with `MSELoss` to reconstruct its own
 *        input while an L1 penalty on the hidden ReLU activation pushes most hidden units
 *        to zero on any given example.
 *
 * `hidden_dim` is normally chosen **larger** than `dim` -- an overcomplete basis, more
 * directions than the activation space being decomposed, which is what distinguishes an
 * SAE from a compressing autoencoder. The activations come from an ActivationSnapshot
 * (this campaign's Phase 1 output) in production use, but nothing here depends on that: an
 * SAE consumes a plain `(N, dim)` batch, so it is equally usable against hand-built
 * synthetic data -- which is exactly what the paired penalty/no-penalty control in
 * `tests/sparse_autoencoder_test.cpp` needs.
 *
 * @note **Bounded claim** (campaign Decision Point 3): this class measures and reports
 *       reconstruction fidelity (reconstruction_error) and sparsity (mean_hidden_activation)
 *       as numbers. It does **not** establish that the hidden directions it learns are
 *       semantically meaningful or causally compositional features -- that is a live
 *       debate in the field, and Phase 4 (activation patching) evidence is the minimum
 *       prerequisite for even arguing it. Do not let a low reconstruction error be read as
 *       a claim about feature semantics.
 * @note Mirrors LinearProbe/XorNetwork's established shape rather than inventing a
 *       training abstraction: the constructor takes a backend, `train_step()` drives one
 *       loss/backward/optimizer-step cycle, and the epoch loop is the caller's. The SAE
 *       deliberately does **not** own its optimizer -- same division of responsibility
 *       LinearProbe::train_step already uses, which is what lets a caller choose SGD vs.
 *       Adam and keep one optimizer's state across a whole training run.
 * @note `train_step` is a template on the optimizer type rather than taking an
 *       `Optimizer&`, for exactly the reason LinearProbe documents: this codebase has no
 *       `Optimizer` base class. `SGDOptimizer`, `AdamOptimizer` and `AdamWOptimizer` share
 *       the `step(Module&)` / `zero_grad(Module&)` shape (a compile-time, duck-typed
 *       contract), so the template accepts any of them.
 * @note Small **seeded random** weight init, not zero init -- and here, unlike LinearProbe,
 *       zero init is not merely degenerate but dead: with an all-zero decoder weight the
 *       gradient reaching the hidden layer is identically zero, so the encoder never
 *       receives any gradient and the autoencoder cannot leave the origin.
 * @note **Featurizer (FEAT-1).** It implements the Featurizer interface: encode() gives the
 *       ReLU codes, decode() the reconstruction, and TrainFeaturizer() trains it like any other
 *       featurizer. It is also a Module from input to reconstruction, with backward() and LRP,
 *       and parameters named `encoder.*` and `decoder.*`.
 * @note **Unit-norm decoders.** By default every feature's decoder direction (a row of the
 *       decoder's weight) has unit L2 norm: at construction and after every train_step(). The
 *       rescaling moves the norm into the encoder's row and bias, which a ReLU passes through
 *       unchanged, so reconstructions don't move. Without it an L1 penalty can be dodged by
 *       shrinking codes and growing directions. set_unit_norm_decoder(false) turns it off.
 * @note `hidden_dim > dim` is **not** enforced. An undercomplete or square autoencoder is
 *       still a well-formed object with well-defined training behavior; rejecting it would
 *       turn a modeling choice into a hard error for no correctness gain. Explicit
 *       decision (mission exit gate asked for it to be made, not assumed), pinned by
 *       ConstructorAcceptsAnUndercompleteHiddenDimension.
 */
class SparseAutoencoder : public Module, public Featurizer {
public:
    /**
     * @brief Constructs a sparse autoencoder over activations of a given dimension.
     * @param dim Width of the activation vectors this SAE reconstructs. Must be > 0.
     * @param hidden_dim Width of the sparse hidden basis. Must be > 0; normally > dim.
     * @param l1_lambda Coefficient of the L1 penalty on the hidden activation. Must be
     *        >= 0; 0 is a valid degenerate configuration (a plain autoencoder), which is
     *        precisely what the mission's no-penalty control trains.
     * @param backend Backend to allocate/compute through. Not owned; must outlive this SAE.
     * @param seed RNG seed for weight initialization -- same seed gives the same SAE.
     * @throws std::invalid_argument if dim <= 0, hidden_dim <= 0, or l1_lambda < 0.
     *         External boundary: all three are caller-supplied (the dimensions typically
     *         read off a snapshot tensor's shape) and nothing downstream rejects them --
     *         `LinearModule(0, h, ...)` builds a well-formed zero-element weight and fails
     *         only later, confusingly, and a negative lambda never fails at all: it
     *         *rewards* hidden activation without bound, so the run diverges silently
     *         rather than erroring.
     */
    /** @brief Seeded from the global seed stream (next_seed(), FND-7). */
    SparseAutoencoder(int64_t dim, int64_t hidden_dim, float l1_lambda, DeviceBackend* backend)
        : SparseAutoencoder(dim, hidden_dim, l1_lambda, backend, static_cast<unsigned>(next_seed())) {}

    SparseAutoencoder(int64_t dim, int64_t hidden_dim, float l1_lambda, DeviceBackend* backend, unsigned seed)
        : dim_(dim),
          hidden_dim_(hidden_dim),
          l1_lambda_(l1_lambda),
          backend_(backend),
          // Clamped only so a rejected dimension can't reach LinearModule/Shape and throw
          // *their* message before the check below throws this class's own, clearer one --
          // member initialization necessarily runs before the constructor body.
          encoder_(dim > 0 ? dim : 1, hidden_dim > 0 ? hidden_dim : 1, backend),
          relu_(backend),
          decoder_(hidden_dim > 0 ? hidden_dim : 1, dim > 0 ? dim : 1, backend),
          loss_(backend) {
        if (dim <= 0) {
            throw std::invalid_argument("SparseAutoencoder: dim must be positive");
        }
        if (hidden_dim <= 0) {
            throw std::invalid_argument("SparseAutoencoder: hidden_dim must be positive");
        }
        if (l1_lambda < 0.0f) {
            throw std::invalid_argument("SparseAutoencoder: l1_lambda must be non-negative");
        }

        std::mt19937 rng(seed);
        // Fan-in scaling: each layer's init range shrinks with its own input width, so the
        // pre-activation magnitude at step 0 stays O(1) regardless of dim/hidden_dim rather
        // than growing with the layer's width (which for a deliberately overcomplete SAE
        // would otherwise saturate the decoder's input from the first step).
        init_layer(encoder_, rng, 1.0f / std::sqrt(static_cast<float>(dim)));
        init_layer(decoder_, rng, 1.0f / std::sqrt(static_cast<float>(hidden_dim)));
        normalize_decoder();
    }

    /**
     * @brief Runs one training step: forward, MSE reconstruction loss, backward with the
     *        L1 penalty gradient injected at the hidden layer, one optimizer update of the
     *        encoder's and decoder's parameters.
     * @tparam OptimizerT Any type exposing `step(Module&)` and `zero_grad(Module&)` --
     *         SGDOptimizer, AdamOptimizer or AdamWOptimizer (see the class-level note on why this is a
     *         template rather than an `Optimizer&`).
     * @param input_batch Shape (N, dim), N > 0. Both the input and the reconstruction
     *        target -- an autoencoder's target *is* its input.
     * @param optimizer Optimizer to update this SAE's encoder/decoder parameters with. Not
     *        owned; its state persists across calls, which is the point of not owning it.
     * @return The **reconstruction** loss for this batch, measured *before* the update
     *         (mirroring LinearProbe/XorNetwork's train_step return contract). The L1
     *         penalty term is deliberately not folded into this number: the two quantities
     *         trade off against each other, and reporting their sum would hide which of
     *         them a change in the total came from. The sparsity side is observed via
     *         mean_hidden_activation().
     * @throws std::invalid_argument on any malformed batch -- see validate_batch().
     * @note Zeroes gradients before accumulating, for the exact reason XorNetwork::
     *       train_step documents: this codebase's modules accumulate parameter gradients
     *       across backward() calls until something resets them, so without this every
     *       step would silently sum onto every previous step's gradient.
     * @note The L1 penalty is `(l1_lambda / N) * sum_over(batch, hidden) h_ij` -- a
     *       per-example sum of hidden activations, averaged over the batch. `abs()` is
     *       unnecessary: h is the output of a ReLU and therefore already >= 0. Its
     *       gradient w.r.t. every hidden element is the constant `l1_lambda / N`, which is
     *       why a uniform fill()+accumulate() onto the incoming hidden gradient is exactly
     *       correct and not an approximation. Adding it *before* relu_.backward() is also
     *       what makes it correct for the clamped-off units: ReLU's own backward zeroes
     *       the contribution wherever the pre-activation was <= 0, which is precisely where
     *       the penalty has no gradient to give (an already-inactive unit is not pushed
     *       further down).
     * @note The penalty gradient is built and accumulated unconditionally, with no
     *       `l1_lambda > 0` fast path. At lambda = 0 it is a tensor of zeros and
     *       accumulating it is a no-op, so the no-penalty configuration exercises exactly
     *       the same instructions the penalized one does -- which is what makes the
     *       mission's paired control a controlled comparison rather than a comparison of
     *       two code paths.
     */
    template <typename OptimizerT>
    float train_step(const Tensor& input_batch, OptimizerT& optimizer) {
        validate_batch(input_batch, "SparseAutoencoder::train_step");
        optimizer.zero_grad(*this);
        const FeaturizerLoss loss = loss_and_backward(input_batch);
        optimizer.step(*this);
        if (unit_norm_decoder_) normalize_decoder();
        return loss.reconstruction;
    }

    // ---- Featurizer ----------------------------------------------------------------------

    [[nodiscard]] int64_t input_dim() const override { return dim_; }
    [[nodiscard]] int64_t num_features() const override { return hidden_dim_; }
    /** @brief The ReLU codes, `(N, hidden_dim)`. */
    [[nodiscard]] Tensor encode(const Tensor& x) override {
        validate_batch(x, "SparseAutoencoder::encode");
        return hidden_codes(x);
    }
    /** @brief The reconstruction from codes, `(N, dim)`. */
    [[nodiscard]] Tensor decode(const Tensor& codes) override {
        if (codes.rank() != 2 || codes.shape().dim(1) != hidden_dim_ || codes.shape().dim(0) <= 0) {
            throw std::invalid_argument("SparseAutoencoder::decode: codes must be (N, hidden_dim) with N > 0");
        }
        return decoder_.forward(codes);
    }
    /**
     * @brief Mean squared reconstruction error plus `l1_lambda * mean_over_inputs(sum_i f_i)`,
     *        with both gradients added to the parameters'. See train_step() for the L1 term's
     *        gradient.
     */
    FeaturizerLoss loss_and_backward(const Tensor& input_batch, std::vector<float>* codes = nullptr) override {
        validate_batch(input_batch, "SparseAutoencoder::loss_and_backward");
        const Tensor hidden = relu_.forward(encoder_.forward(input_batch));
        const Tensor reconstruction = decoder_.forward(hidden);
        FeaturizerLoss loss;
        loss.reconstruction = loss_.forward(reconstruction, input_batch);
        const Tensor grad_reconstruction = loss_.backward();
        Tensor grad_hidden = decoder_.backward(grad_reconstruction);
        const float batch_size = static_cast<float>(input_batch.shape().dim(0));
        Tensor l1_grad(grad_hidden.shape(), backend_, grad_hidden.device());
        l1_grad.fill(l1_lambda_ / batch_size);
        grad_hidden.accumulate(l1_grad);
        const Tensor grad_pre_activation = relu_.backward(grad_hidden);
        (void)encoder_.backward(grad_pre_activation);  // the input has no upstream to receive this
        const std::vector<float> h = hidden.to_host_vector();
        double sum = 0.0;
        for (float v : h) sum += v;
        loss.sparsity = static_cast<float>(l1_lambda_ * sum / batch_size);
        loss.total = loss.reconstruction + loss.sparsity;
        if (codes != nullptr) *codes = h;
        return loss;
    }
    /** @brief Unit-norm decoder rows, the norm moved into the encoder's row and bias: the ReLU
     *         passes a positive scale through, so reconstructions don't change. A feature whose
     *         direction is zero is left alone. */
    void normalize_decoder() override {
        std::vector<float> dec = decoder_.weight().to_host_vector(), enc = encoder_.weight().to_host_vector(),
                           bias = encoder_.bias().to_host_vector();
        for (int64_t i = 0; i < hidden_dim_; ++i) {
            double sq = 0.0;
            for (int64_t j = 0; j < dim_; ++j) sq += static_cast<double>(dec[static_cast<size_t>(i * dim_ + j)]) * dec[static_cast<size_t>(i * dim_ + j)];
            const double norm = std::sqrt(sq);
            if (norm == 0.0) continue;
            for (int64_t j = 0; j < dim_; ++j) dec[static_cast<size_t>(i * dim_ + j)] = static_cast<float>(dec[static_cast<size_t>(i * dim_ + j)] / norm);
            for (int64_t j = 0; j < dim_; ++j) enc[static_cast<size_t>(j * hidden_dim_ + i)] = static_cast<float>(enc[static_cast<size_t>(j * hidden_dim_ + i)] * norm);
            bias[static_cast<size_t>(i)] = static_cast<float>(bias[static_cast<size_t>(i)] * norm);
        }
        decoder_.set_weight(dec);
        encoder_.set_weight(enc);
        encoder_.set_bias(bias);
    }
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i) override {
        if (i < 0 || i >= hidden_dim_) throw std::invalid_argument("SparseAutoencoder::decoder_direction: no such feature");
        const std::vector<float> dec = decoder_.weight().to_host_vector();
        return {dec.begin() + static_cast<std::ptrdiff_t>(i * dim_), dec.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim_)};
    }
    [[nodiscard]] Module& parameters_module() override { return *this; }

    /** @brief Whether train_step() keeps decoder directions at unit norm (the default). */
    [[nodiscard]] bool unit_norm_decoder() const { return unit_norm_decoder_; }
    void set_unit_norm_decoder(bool on) { unit_norm_decoder_ = on; }

    // ---- Module: input to reconstruction --------------------------------------------------

    /** @brief Backward from the reconstruction's gradient to the input's, through the decoder,
     *         the ReLU and the encoder, adding to their parameters' gradients. */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override {
        return encoder_.backward(relu_.backward(decoder_.backward(grad_output)));
    }
    /** @brief Each layer's own rule, from the reconstruction back to the input. */
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override {
        return encoder_.propagate_relevance(relu_.propagate_relevance(decoder_.propagate_relevance(relevance_out, config), config), config);
    }
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override {
        std::vector<NamedParamRef> out;
        append_named_parameters(out, "encoder", encoder_);
        append_named_parameters(out, "decoder", decoder_);
        return out;
    }
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override {
        Module::set_training(training);
        encoder_.set_training(training);
        decoder_.set_training(training);
    }
    void release_activations() override {
        encoder_.release_activations();
        relu_.release_activations();
        decoder_.release_activations();
    }

    /**
     * @brief The SAE's reconstruction of a batch -- the encoder->ReLU->decoder forward path,
     *        forward-only: no loss, no backward, no parameter update.
     * @param input_batch Shape (N, dim), N > 0.
     * @return Shape (N, dim) -- x_hat, the same tensor reconstruction_error() scores against
     *         the input. Exposed as a value rather than only as a scalar error because the
     *         reconstruction *itself* is what an activation-patching experiment substitutes
     *         (campaign_exai_dl_library_mechanistic_interpretability, Phase 4's exit gate);
     *         before this method the only way to obtain one was train_step(), which also
     *         mutates the parameters -- an observation that changes what it observes.
     * @throws std::invalid_argument on any malformed batch -- see validate_batch().
     * @note `const` for the same reason reconstruction_error() is -- see its note. Third
     *       instance of the mutable-members-for-a-const-scoring-path pattern.
     * @note No new computation: reconstruction_error() is defined in terms of *this* method,
     *       so the two can never report a reconstruction and an error computed from
     *       different forward paths. Pinned by
     *       ReconstructOutputAgreesWithReconstructionErrorsInternalComputation, which
     *       recomputes the MSE by hand from this method's output and compares.
     */
    [[nodiscard]] Tensor reconstruct(const Tensor& input_batch) const {
        validate_batch(input_batch, "SparseAutoencoder::reconstruct");
        return decoder_.forward(hidden_codes(input_batch));
    }

    /**
     * @brief Mean squared reconstruction error over the batch -- forward-only, no backward,
     *        no parameter update.
     * @param input_batch Shape (N, dim), N > 0.
     * @return mean over all N * dim elements of (reconstruction - input)^2, i.e. exactly
     *         MSELoss's own convention, so this number is directly comparable to the loss
     *         train_step() returns.
     * @throws std::invalid_argument on any malformed batch -- see validate_batch().
     * @note `const` although Module::forward() is not: the SAE's *logical* state is its
     *       learned parameters, which a forward-only scoring pass cannot change. The
     *       modules are `mutable` purely so forward()'s internal caches (backward()'s
     *       working state, not the SAE's observable value) can be written. Flagged as a
     *       recurring pattern by LinearProbe's AAR; this is its second instance.
     * @note Computed in a raw host loop rather than through the member `loss_`, and
     *       deliberately so: MSELoss::forward caches its operands to arm a subsequent
     *       backward(), and a scoring pass has no business arming a backward it never
     *       performs -- doing so would leave the loss primed with a batch the next
     *       train_step() did not produce. The loop itself is the established convention
     *       (MSELoss/CrossEntropyLoss/LinearProbe::accuracy all read Tensor::data()
     *       directly, since Tensor has no reduction primitive).
     */
    [[nodiscard]] float reconstruction_error(const Tensor& input_batch) const {
        validate_batch(input_batch, "SparseAutoencoder::reconstruction_error");

        const Tensor reconstruction = reconstruct(input_batch);
        const int64_t n = reconstruction.numel();
        float sum_squared = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            const float diff = reconstruction.data()[i] - input_batch.data()[i];
            sum_squared += diff * diff;
        }
        return sum_squared / static_cast<float>(n);
    }

    /**
     * @brief Mean hidden (post-ReLU) activation over the batch -- the sparsity metric.
     * @param input_batch Shape (N, dim), N > 0.
     * @return mean over all N * hidden_dim hidden elements. Always >= 0.
     * @throws std::invalid_argument on any malformed batch -- see validate_batch().
     * @note Post-ReLU, so every element is already >= 0 and the mean is a direct, valid
     *       proxy for the L1 norm this class penalizes -- no abs() needed. Lower means
     *       sparser: the floor, 0, is every hidden unit clamped off for every example.
     * @note Reports mean *magnitude*, not a count of active units. A fraction-nonzero
     *       metric (L0) would be the other natural choice and is not offered here: it is
     *       not what the penalty term optimizes, and it would be discontinuous in the
     *       parameters, making it a poor thing to compare two training runs by. This
     *       metric is the one the penalty actually targets, which is the property the
     *       mission's paired control tests.
     * @note `const` for the same reason reconstruction_error() is -- see its note.
     */
    [[nodiscard]] float mean_hidden_activation(const Tensor& input_batch) const {
        validate_batch(input_batch, "SparseAutoencoder::mean_hidden_activation");

        const Tensor hidden = hidden_codes(input_batch);
        const int64_t n = hidden.numel();
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += hidden.data()[i];
        }
        return sum / static_cast<float>(n);
    }

    /** @brief Width of the activation vectors this SAE reconstructs. */
    [[nodiscard]] int64_t dim() const { return dim_; }

    /** @brief Width of the sparse hidden basis. */
    [[nodiscard]] int64_t hidden_dim() const { return hidden_dim_; }

    /** @brief Coefficient of the L1 penalty on the hidden activation. */
    [[nodiscard]] float l1_lambda() const { return l1_lambda_; }

    /**
     * @brief The encoder -- inspection (its columns are the learned feature directions,
     *        which is the whole point of an SAE) and test-time weight injection.
     */
    [[nodiscard]] LinearModule& encoder() { return encoder_; }

    /** @brief Const overload of encoder(). */
    [[nodiscard]] const LinearModule& encoder() const { return encoder_; }

    /** @brief The decoder -- same rationale as encoder(). */
    [[nodiscard]] LinearModule& decoder() { return decoder_; }

    /** @brief Const overload of decoder(). */
    [[nodiscard]] const LinearModule& decoder() const { return decoder_; }

protected:
    /** @brief The reconstruction, `(N, dim)`. @throws std::invalid_argument for a malformed batch. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override {
        validate_batch(input, "SparseAutoencoder::forward");
        return decoder_.forward(relu_.forward(encoder_.forward(input)));
    }

private:
    /** @brief Post-ReLU hidden activation for a batch. Forward-only, unvalidated (callers
     *         are this class's own already-validated entry points). */
    [[nodiscard]] Tensor hidden_codes(const Tensor& input_batch) const { return relu_.forward(encoder_.forward(input_batch)); }

    /** @brief Fills one layer's weight and bias with uniform values in [-scale, scale). */
    static void init_layer(LinearModule& layer, std::mt19937& rng, float scale) {
        std::vector<float> weights(static_cast<size_t>(layer.weight().numel()));
        for (float& w : weights) {
            w = uniform_symmetric(rng, scale);
        }
        layer.set_weight(weights);
        layer.set_bias(std::vector<float>(static_cast<size_t>(layer.bias().numel()), 0.0f));
    }

    /** @brief Uniform in [-scale, scale), from a seeded engine, with no std distribution. */
    static float uniform_symmetric(std::mt19937& rng, float scale) {
        // std::uniform_real_distribution's mapping is implementation-defined, so identical
        // seeds would not give identical weights across standard libraries. This mapping is
        // fixed here, making a seeded SAE reproducible everywhere -- the same convention
        // LinearProbe already uses.
        const float unit = static_cast<float>(rng() - std::mt19937::min()) /
                           static_cast<float>(std::mt19937::max() - std::mt19937::min() + 1ull);
        return (unit * 2.0f - 1.0f) * scale;
    }

    /**
     * @brief Validates an input batch at the SAE's own entry point.
     * @throws std::invalid_argument if the batch is not rank-2, if its width is not dim(),
     *         or if it is empty.
     * @note Classified **external boundary -> throw** per
     *       cpp_tdd/context_tdd_adversarial_boundary_testing.md, not PULSATRIX_ASSERT: an SAE is
     *       driven by caller-assembled batches (an analyst's, ultimately a Python
     *       caller's), not by an already-validated internal call chain -- and per that
     *       file's "when genuinely unsure, default to external boundary" rule. Checked here
     *       rather than left to LinearModule::forward because an empty batch would
     *       otherwise report Module::forward's generic message naming neither the
     *       autoencoder nor which operand was wrong, and a width mismatch would be reported
     *       against the encoder's in_features rather than against this class's documented
     *       `dim`.
     */
    void validate_batch(const Tensor& input_batch, const char* method) const {
        const std::string where(method);
        if (input_batch.rank() != 2) {
            throw std::invalid_argument(where + ": input_batch must be rank-2 (N, dim)");
        }
        if (input_batch.shape().dim(1) != dim_) {
            throw std::invalid_argument(where + ": input_batch width must equal dim");
        }
        if (input_batch.shape().dim(0) <= 0) {
            throw std::invalid_argument(where + ": batch must not be empty");
        }
    }

    int64_t dim_;
    int64_t hidden_dim_;
    float l1_lambda_;
    DeviceBackend* backend_;
    mutable LinearModule encoder_;
    mutable ReluModule relu_;
    mutable LinearModule decoder_;
    MSELoss loss_;
    bool unit_norm_decoder_ = true;
};

}  // namespace pulsatrix
