#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/sparse_autoencoder.hpp"

// SparseAutoencoder (campaign_exai_dl_library_mechanistic_interpretability, Phase 3
// Mission 1) is the sparse-autoencoder training utility: LinearModule(dim, hidden_dim) ->
// ReluModule -> LinearModule(hidden_dim, dim) + MSELoss, with an L1-style penalty on the
// hidden ReLU activation, trained to reconstruct its own input.
//
// This file covers both mission objectives:
//   Objective 1 -- the class itself (construction, train_step, reconstruction_error,
//                  mean_hidden_activation) plus its adversarial/boundary set.
//   Objective 2 -- the mission's actual acceptance criterion: a paired penalty /
//                  no-penalty control, two instances identical in every respect except
//                  l1_lambda, proving the penalty is load-bearing on the sparsity metric
//                  it targets rather than merely present and inert.
//
// Bounded claim (campaign Decision Point 3): these tests measure *reconstruction fidelity*
// and *mean hidden activation* as numbers. They do not claim -- and are not evidence for --
// that the hidden units the penalty produces correspond to semantically meaningful or
// causally compositional features. That is a live debate in the field, and Phase 4
// (activation patching) evidence would be the minimum prerequisite for even arguing it.
namespace pulsatrix {
namespace {

class SparseAutoencoderTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(SparseAutoencoderTest, ConstructorBuildsAnEncoderAndDecoderOfTheRequestedDimensions) {
    SparseAutoencoder sae(4, 16, 0.1f, &backend);

    EXPECT_EQ(sae.dim(), 4);
    EXPECT_EQ(sae.hidden_dim(), 16);
    EXPECT_FLOAT_EQ(sae.l1_lambda(), 0.1f);
    EXPECT_EQ(sae.encoder().weight().shape(), Shape({4, 16}));
    EXPECT_EQ(sae.encoder().bias().shape(), Shape({16}));
    EXPECT_EQ(sae.decoder().weight().shape(), Shape({16, 4}));
    EXPECT_EQ(sae.decoder().bias().shape(), Shape({4}));
}

// Zero-initialized weights are not merely a degenerate starting point here, they are a
// dead one: with the decoder weight all zero the gradient reaching the hidden layer is
// identically zero, so the encoder receives no gradient at all and the autoencoder can
// never leave the origin. Small seeded random init breaks that, deterministically.
TEST_F(SparseAutoencoderTest, ConstructorInitializesWeightsDeterministicallyAndNonZero) {
    SparseAutoencoder a(3, 6, 0.0f, &backend, /*seed=*/123);
    SparseAutoencoder b(3, 6, 0.0f, &backend, /*seed=*/123);

    bool any_non_zero = false;
    for (int64_t i = 0; i < a.encoder().weight().numel(); ++i) {
        EXPECT_FLOAT_EQ(a.encoder().weight()[i], b.encoder().weight()[i]);
        if (a.encoder().weight()[i] != 0.0f) {
            any_non_zero = true;
        }
    }
    for (int64_t i = 0; i < a.decoder().weight().numel(); ++i) {
        EXPECT_FLOAT_EQ(a.decoder().weight()[i], b.decoder().weight()[i]);
        if (a.decoder().weight()[i] != 0.0f) {
            any_non_zero = true;
        }
    }
    EXPECT_TRUE(any_non_zero);
}

TEST_F(SparseAutoencoderTest, ConstructorGivesDifferentSeedsDifferentWeights) {
    SparseAutoencoder a(3, 6, 0.0f, &backend, /*seed=*/1);
    SparseAutoencoder b(3, 6, 0.0f, &backend, /*seed=*/2);

    bool any_differ = false;
    for (int64_t i = 0; i < a.encoder().weight().numel(); ++i) {
        if (a.encoder().weight()[i] != b.encoder().weight()[i]) {
            any_differ = true;
        }
    }
    EXPECT_TRUE(any_differ);
}

// hidden_dim > dim (an overcomplete basis) is the *intended* use and what makes this an
// SAE rather than a compressing autoencoder -- but it is deliberately not enforced: an
// undercomplete or square autoencoder is still a well-formed object with well-defined
// training behavior, and rejecting it would turn a modeling choice into a hard error for
// no correctness gain. Documented on the class; asserted here so the decision is explicit
// in the test record rather than implicit in the absence of a check.
TEST_F(SparseAutoencoderTest, ConstructorAcceptsAnUndercompleteHiddenDimension) {
    EXPECT_NO_THROW(SparseAutoencoder(8, 2, 0.1f, &backend));
}

TEST_F(SparseAutoencoderTest, TrainStepReturnsAFiniteReconstructionLoss) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    AdamOptimizer optimizer(0.05f, &backend);
    Tensor batch(Shape({2, 2}), &backend, {1.0f, -1.0f, 0.5f, 2.0f});

    const float loss = sae.train_step(batch, optimizer);

    EXPECT_TRUE(std::isfinite(loss));
    EXPECT_GT(loss, 0.0f);
}

TEST_F(SparseAutoencoderTest, TrainStepDecreasesReconstructionLossOverRepeatedCallsWithoutAPenalty) {
    SparseAutoencoder sae(2, 4, 0.0f, &backend);
    AdamOptimizer optimizer(0.05f, &backend);
    Tensor batch(Shape({2, 2}), &backend, {1.0f, -1.0f, 0.5f, 2.0f});

    const float first = sae.train_step(batch, optimizer);
    float last = first;
    for (int i = 0; i < 60; ++i) {
        last = sae.train_step(batch, optimizer);
    }

    EXPECT_LT(last, first);
}

// l1_lambda = 0 is a valid degenerate configuration, not an error and not a separate code
// path -- the penalty gradient is built and accumulated unconditionally, and at lambda = 0
// it is a tensor of zeros, so the no-penalty case exercises exactly the same instructions
// the penalized case does. This test and the one above are the pair that pins that down.
TEST_F(SparseAutoencoderTest, TrainStepDecreasesReconstructionLossOverRepeatedCallsWithAPenalty) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    AdamOptimizer optimizer(0.05f, &backend);
    Tensor batch(Shape({2, 2}), &backend, {1.0f, -1.0f, 0.5f, 2.0f});

    const float first = sae.train_step(batch, optimizer);
    float last = first;
    for (int i = 0; i < 60; ++i) {
        last = sae.train_step(batch, optimizer);
    }

    EXPECT_LT(last, first);
}

TEST_F(SparseAutoencoderTest, TrainStepChangesBothTheEncoderAndTheDecoderParameters) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    AdamOptimizer optimizer(0.05f, &backend);
    Tensor batch(Shape({2, 2}), &backend, {1.0f, -1.0f, 0.5f, 2.0f});
    const std::vector<float> encoder_before(sae.encoder().weight().data(),
                                            sae.encoder().weight().data() + sae.encoder().weight().numel());
    const std::vector<float> decoder_before(sae.decoder().weight().data(),
                                            sae.decoder().weight().data() + sae.decoder().weight().numel());

    (void)sae.train_step(batch, optimizer);

    bool encoder_changed = false;
    for (size_t i = 0; i < encoder_before.size(); ++i) {
        if (sae.encoder().weight()[static_cast<int64_t>(i)] != encoder_before[i]) {
            encoder_changed = true;
        }
    }
    bool decoder_changed = false;
    for (size_t i = 0; i < decoder_before.size(); ++i) {
        if (sae.decoder().weight()[static_cast<int64_t>(i)] != decoder_before[i]) {
            decoder_changed = true;
        }
    }
    EXPECT_TRUE(encoder_changed);
    EXPECT_TRUE(decoder_changed);
}

// ---------------------------------------------------------------------------------------
// Hand-computed scoring cases.
//
// dim = 2, hidden_dim = 3, one example x = (1, -2).
//   encoder W1 = [[1,0,0],[0,1,0]], b1 = 0  ->  pre-activation = (1, -2, 0)
//   ReLU                                     ->  hidden        = (1,  0, 0)
//   decoder W2 = [[1,0],[0,1],[0,0]], b2 = 0 ->  reconstruction = (1,  0)
// mean_hidden_activation = (1 + 0 + 0) / 3          = 1/3
// reconstruction_error   = ((1-1)^2 + (0-(-2))^2)/2 = 2
// (MSELoss's own convention: mean over *every* element, i.e. over N * dim, which for N = 1
// is just dim -- matched exactly here so the two numbers are directly comparable.)
// ---------------------------------------------------------------------------------------

TEST_F(SparseAutoencoderTest, ReconstructionErrorMatchesAHandComputedValue) {
    SparseAutoencoder sae(2, 3, 0.0f, &backend);
    sae.encoder().set_weight({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    sae.encoder().set_bias({0.0f, 0.0f, 0.0f});
    sae.decoder().set_weight({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
    sae.decoder().set_bias({0.0f, 0.0f});
    Tensor batch(Shape({1, 2}), &backend, {1.0f, -2.0f});

    EXPECT_FLOAT_EQ(sae.reconstruction_error(batch), 2.0f);
}

TEST_F(SparseAutoencoderTest, MeanHiddenActivationMatchesAHandComputedValue) {
    SparseAutoencoder sae(2, 3, 0.0f, &backend);
    sae.encoder().set_weight({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    sae.encoder().set_bias({0.0f, 0.0f, 0.0f});
    Tensor batch(Shape({1, 2}), &backend, {1.0f, -2.0f});

    EXPECT_FLOAT_EQ(sae.mean_hidden_activation(batch), 1.0f / 3.0f);
}

TEST_F(SparseAutoencoderTest, ReconstructionErrorIsZeroForAnExactlyReconstructingAutoencoder) {
    SparseAutoencoder sae(2, 2, 0.0f, &backend);
    sae.encoder().set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    sae.encoder().set_bias({0.0f, 0.0f});
    sae.decoder().set_weight({1.0f, 0.0f, 0.0f, 1.0f});
    sae.decoder().set_bias({0.0f, 0.0f});
    Tensor batch(Shape({2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f});  // all non-negative -> ReLU is identity

    EXPECT_FLOAT_EQ(sae.reconstruction_error(batch), 0.0f);
}

// Boundary: every pre-activation strictly negative -> ReLU clamps the entire hidden layer
// to exactly zero, so the sparsity metric must read exactly 0, its floor. This also pins
// down that mean_hidden_activation reads the *post*-ReLU hidden layer (a pre-activation
// mean here would be -10, not 0).
TEST_F(SparseAutoencoderTest, MeanHiddenActivationIsZeroWhenEveryHiddenUnitIsClampedOff) {
    SparseAutoencoder sae(2, 3, 0.0f, &backend);
    sae.encoder().set_weight({0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    sae.encoder().set_bias({-10.0f, -10.0f, -10.0f});
    Tensor batch(Shape({2, 2}), &backend, {1.0f, -2.0f, 3.0f, 4.0f});

    EXPECT_FLOAT_EQ(sae.mean_hidden_activation(batch), 0.0f);
}

TEST_F(SparseAutoencoderTest, MeanHiddenActivationIsNeverNegative) {
    SparseAutoencoder sae(2, 5, 0.0f, &backend, /*seed=*/99);
    Tensor batch(Shape({3, 2}), &backend, {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, -6.0f});

    EXPECT_GE(sae.mean_hidden_activation(batch), 0.0f);
}

// Both scoring methods are forward-only: they must not disturb the learned parameters.
TEST_F(SparseAutoencoderTest, ScoringMethodsDoNotChangeTheParameters) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({2, 2}), &backend, {1.0f, -1.0f, 0.5f, 2.0f});
    const std::vector<float> before(sae.encoder().weight().data(),
                                    sae.encoder().weight().data() + sae.encoder().weight().numel());

    (void)sae.reconstruction_error(batch);
    (void)sae.mean_hidden_activation(batch);

    for (size_t i = 0; i < before.size(); ++i) {
        EXPECT_FLOAT_EQ(sae.encoder().weight()[static_cast<int64_t>(i)], before[i]);
    }
}

// ---------------------------------------------------------------------------------------
// reconstruct() -- the forward-only reconstruction accessor (Phase 4 Mission 2).
//
// reconstruction_error() reports a *scalar* against the input, which is enough to score an
// SAE but not enough to use one: the campaign's Phase 4 exit-gate wording ("substitute an
// SAE reconstruction" into a patched forward pass) needs the reconstruction Tensor itself.
// Before this method the only way to obtain one was train_step(), which also mutates the
// parameters -- an observation pass that changes what it observes.
// ---------------------------------------------------------------------------------------

TEST_F(SparseAutoencoderTest, ReconstructReturnsATensorShapedLikeItsInput) {
    SparseAutoencoder sae(2, 5, 0.01f, &backend);
    Tensor batch(Shape({3, 2}), &backend, {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, -6.0f});

    EXPECT_EQ(sae.reconstruct(batch).shape(), Shape({3, 2}));
}

// Same hand-computed fixture as ReconstructionErrorMatchesAHandComputedValue above, read one
// level earlier: that test asserts the scalar 2; this one asserts the (1, 0) it is computed
// from, so a reconstruction that happened to be wrong in a norm-preserving way could not
// pass both.
TEST_F(SparseAutoencoderTest, ReconstructMatchesAHandComputedValue) {
    SparseAutoencoder sae(2, 3, 0.0f, &backend);
    sae.encoder().set_weight({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
    sae.encoder().set_bias({0.0f, 0.0f, 0.0f});
    sae.decoder().set_weight({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f});
    sae.decoder().set_bias({0.0f, 0.0f});
    Tensor batch(Shape({1, 2}), &backend, {1.0f, -2.0f});

    const Tensor reconstruction = sae.reconstruct(batch);

    ASSERT_EQ(reconstruction.numel(), 2);
    EXPECT_FLOAT_EQ(reconstruction.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(reconstruction.data()[1], 0.0f);
}

// Cross-check, not a duplicate assertion of one code path: the MSE is recomputed *here*,
// by hand, from reconstruct()'s returned Tensor, and compared against the number
// reconstruction_error() arrives at internally. If reconstruct() ever diverged from the
// forward path reconstruction_error() scores -- a different ReLU placement, a stale cache, a
// missing bias -- the two numbers would separate and this test would catch it. Run on an
// un-hand-set, seeded SAE and a multi-example batch so the agreement isn't an artifact of a
// degenerate identity-weight fixture.
TEST_F(SparseAutoencoderTest, ReconstructOutputAgreesWithReconstructionErrorsInternalComputation) {
    SparseAutoencoder sae(2, 6, 0.01f, &backend, /*seed=*/7);
    Tensor batch(Shape({3, 2}), &backend, {1.0f, -2.0f, 3.0f, 4.0f, -5.0f, -6.0f});

    const Tensor reconstruction = sae.reconstruct(batch);
    ASSERT_EQ(reconstruction.numel(), batch.numel());

    float sum_squared = 0.0f;
    for (int64_t i = 0; i < reconstruction.numel(); ++i) {
        const float diff = reconstruction.data()[i] - batch.data()[i];
        sum_squared += diff * diff;
    }
    const float hand_computed_mse = sum_squared / static_cast<float>(reconstruction.numel());

    EXPECT_FLOAT_EQ(hand_computed_mse, sae.reconstruction_error(batch));
}

// reconstruct() must reflect *learned* state, not be a fixed transform of its input: a
// trained instance's reconstruction has to sit measurably closer to the data than a freshly
// constructed one's, on the same data. Both start from the same seed, so the only difference
// between them is the training.
TEST_F(SparseAutoencoderTest, ReconstructIsCloserToTheInputAfterTrainingThanForAnUntrainedInstance) {
    Tensor batch(Shape({2, 2}), &backend, {1.0f, -1.0f, 0.5f, 2.0f});
    const SparseAutoencoder untrained(2, 8, 0.0f, &backend, /*seed=*/5);
    SparseAutoencoder trained(2, 8, 0.0f, &backend, /*seed=*/5);
    AdamOptimizer optimizer(0.05f, &backend);
    for (int i = 0; i < 200; ++i) {
        (void)trained.train_step(batch, optimizer);
    }

    const Tensor untrained_reconstruction = untrained.reconstruct(batch);
    const Tensor trained_reconstruction = trained.reconstruct(batch);

    float untrained_sse = 0.0f;
    float trained_sse = 0.0f;
    for (int64_t i = 0; i < batch.numel(); ++i) {
        const float untrained_diff = untrained_reconstruction.data()[i] - batch.data()[i];
        const float trained_diff = trained_reconstruction.data()[i] - batch.data()[i];
        untrained_sse += untrained_diff * untrained_diff;
        trained_sse += trained_diff * trained_diff;
    }

    EXPECT_LT(trained_sse, untrained_sse);
}

// Forward-only, like the other two scoring methods: observing a reconstruction must not
// disturb the parameters that produced it. (This is what the `const` on the method claims;
// this test is what makes the claim checkable rather than a comment.)
TEST_F(SparseAutoencoderTest, ReconstructDoesNotChangeTheParameters) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({2, 2}), &backend, {1.0f, -1.0f, 0.5f, 2.0f});
    const std::vector<float> before(sae.encoder().weight().data(),
                                    sae.encoder().weight().data() + sae.encoder().weight().numel());

    (void)sae.reconstruct(batch);

    for (size_t i = 0; i < before.size(); ++i) {
        EXPECT_FLOAT_EQ(sae.encoder().weight()[static_cast<int64_t>(i)], before[i]);
    }
}

// ---------------------------------------------------------------------------------------
// Adversarial / boundary conditions.
//
// Every SparseAutoencoder entry point is classified **external boundary -> throw** per
// cpp_tdd/context_tdd_adversarial_boundary_testing.md, for the same reason LinearProbe's
// are: an SAE is driven by an analyst with caller-assembled activation batches (ultimately,
// via Phase 5's bindings, from Python), not by an already-validated internal call chain --
// and per that file's "when genuinely unsure, default to external boundary" rule. These are
// real throws; every one of these tests must pass identically in Debug and Release, with no
// NDEBUG skip.
//
// The checks live on SparseAutoencoder itself rather than being delegated to
// LinearModule::forward / Module::forward, per the same file's "validate at the *true*
// entry point" rule: an empty batch would otherwise surface as Module::forward's generic
// message naming neither the autoencoder nor which operand was wrong, and a width mismatch
// would be reported against the encoder's in_features rather than against the SAE's own
// documented `dim`.
// ---------------------------------------------------------------------------------------

TEST_F(SparseAutoencoderTest, ConstructorThrowsOnNonPositiveDim) {
    EXPECT_THROW(SparseAutoencoder(0, 4, 0.1f, &backend), std::invalid_argument);
    EXPECT_THROW(SparseAutoencoder(-1, 4, 0.1f, &backend), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, ConstructorThrowsOnNonPositiveHiddenDim) {
    EXPECT_THROW(SparseAutoencoder(4, 0, 0.1f, &backend), std::invalid_argument);
    EXPECT_THROW(SparseAutoencoder(4, -1, 0.1f, &backend), std::invalid_argument);
}

// A negative penalty coefficient is not a degenerate-but-meaningful setting the way zero
// is: it *rewards* activation without bound, and the resulting training run diverges
// silently rather than failing. Rejected at construction.
TEST_F(SparseAutoencoderTest, ConstructorThrowsOnANegativeL1Lambda) {
    EXPECT_THROW(SparseAutoencoder(4, 8, -0.1f, &backend), std::invalid_argument);
}

// Zero is explicitly *not* an error -- it is the no-penalty control's configuration, and
// Objective 2 depends on it being an ordinary, fully-supported value.
TEST_F(SparseAutoencoderTest, ConstructorAcceptsAZeroL1Lambda) {
    EXPECT_NO_THROW(SparseAutoencoder(4, 8, 0.0f, &backend));
}

TEST_F(SparseAutoencoderTest, TrainStepThrowsWhenTheBatchWidthDiffersFromDim) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    AdamOptimizer optimizer(0.05f, &backend);
    Tensor batch(Shape({2, 5}), &backend);

    EXPECT_THROW((void)sae.train_step(batch, optimizer), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, ReconstructionErrorThrowsWhenTheBatchWidthDiffersFromDim) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({2, 5}), &backend);

    EXPECT_THROW((void)sae.reconstruction_error(batch), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, MeanHiddenActivationThrowsWhenTheBatchWidthDiffersFromDim) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({2, 5}), &backend);

    EXPECT_THROW((void)sae.mean_hidden_activation(batch), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, TrainStepThrowsOnAnEmptyBatch) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    AdamOptimizer optimizer(0.05f, &backend);
    Tensor batch(Shape({0, 2}), &backend);

    EXPECT_THROW((void)sae.train_step(batch, optimizer), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, ReconstructionErrorThrowsOnAnEmptyBatch) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({0, 2}), &backend);

    EXPECT_THROW((void)sae.reconstruction_error(batch), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, MeanHiddenActivationThrowsOnAnEmptyBatch) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({0, 2}), &backend);

    EXPECT_THROW((void)sae.mean_hidden_activation(batch), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, TrainStepThrowsOnARankOneBatch) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    AdamOptimizer optimizer(0.05f, &backend);
    Tensor batch(Shape({2}), &backend);

    EXPECT_THROW((void)sae.train_step(batch, optimizer), std::invalid_argument);
}

// reconstruct() is a public entry point like the other two, and validated at the same
// boundary for the same reason -- a caller-assembled batch, ultimately from Python.
TEST_F(SparseAutoencoderTest, ReconstructThrowsWhenTheBatchWidthDiffersFromDim) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({2, 5}), &backend);

    EXPECT_THROW((void)sae.reconstruct(batch), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, ReconstructThrowsOnAnEmptyBatch) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({0, 2}), &backend);

    EXPECT_THROW((void)sae.reconstruct(batch), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, ReconstructThrowsOnARankOneBatch) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({2}), &backend);

    EXPECT_THROW((void)sae.reconstruct(batch), std::invalid_argument);
}

TEST_F(SparseAutoencoderTest, ReconstructionErrorThrowsOnARankThreeBatch) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({2, 2, 2}), &backend);

    EXPECT_THROW((void)sae.reconstruction_error(batch), std::invalid_argument);
}

// =======================================================================================
// Objective 2 -- paired penalty / no-penalty control.
//
// The mission's actual acceptance criterion. A trained SAE reporting a low mean hidden
// activation proves nothing on its own: the penalty term could be numerically inert (wrong
// sign, wrong scale, dropped by a `lambda > 0` branch, cancelled by the optimizer) and the
// hidden layer would still land wherever reconstruction alone put it. "The SAE is sparse"
// and "the L1 penalty made it sparse" are only the same statement if the *identical*
// procedure, run with the penalty switched off, is shown to be measurably less sparse.
//
// Hence two SparseAutoencoders over the same data, the same dim/hidden_dim, the same weight
// init seed, the same optimizer, the same learning rate and the same epoch count --
// differing in exactly one thing: l1_lambda.
//
// Both are scored on a **held-out** split drawn from the same generator but never trained
// on, following LinearProbe's precedent. For the fidelity half this is the meaningful
// choice: an overcomplete autoencoder (32 hidden units for 8 input dimensions) has ample
// freedom to memorize a training split, so a *training* reconstruction error would be
// partly an artifact of capacity rather than a statement about the learned basis.
//
// Bounded claim, restated (campaign Decision Point 3): what this pair establishes is that
// the penalty moves the metric it targets, at a quantified cost in reconstruction fidelity.
// It establishes nothing about whether the sparser hidden units correspond to meaningful
// features -- that would need Phase 4 (activation patching) evidence at minimum, and this
// test is deliberately not built on ground-truth-recoverable synthetic data, which would
// have smuggled in that stronger claim by construction.
// =======================================================================================

// Deterministic generator -- hand-rolled rather than std::normal_distribution for the same
// reason linear_probe_test.cpp's copy is: the standard distributions' mappings from the
// engine's output are implementation-defined, so identical seeds would not give identical
// data across standard libraries, and the thresholds below are asserted on specific data.
class TestRng {
public:
    explicit TestRng(uint64_t seed) : state_(seed) {}

    /** @brief Uniform in [0, 1). */
    float uniform() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        const uint32_t bits = static_cast<uint32_t>(state_ >> 32);
        return static_cast<float>(bits) / 4294967296.0f;
    }

    /** @brief Standard normal, via Box-Muller. */
    float gaussian() {
        const float u1 = std::fmax(uniform(), 1e-7f);
        const float u2 = uniform();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
    }

private:
    uint64_t state_;
};

constexpr int64_t kControlDim = 8;
constexpr int64_t kControlHiddenDim = 32;  // overcomplete: 4x the input dimension
constexpr int64_t kControlTrainSize = 256;
constexpr int64_t kControlHeldoutSize = 256;
constexpr int kControlEpochs = 400;
constexpr float kControlLearningRate = 0.01f;
constexpr unsigned kControlSeed = 11;
// With unit-norm decoders (FEAT-1) the penalty can't be dodged by shrinking codes and growing
// directions, so it bites harder than it used to: 0.05 now collapses reconstruction on this
// data (0.54), and 0.01 is the comparable setting.
constexpr float kControlPenalty = 0.01f;

// Reconstruction-fidelity bound. The data is standard Gaussian, so the variance per element
// is ~1 and the error a trivial "reconstruct the mean" baseline would achieve is ~1.0 --
// this bound therefore says both runs explain at least 95% of the input variance. Measured:
// the penalized run 0.0090, the unpenalized one 0.0026. The point of the bound is that the
// penalty costs *some* fidelity but does not collapse reconstruction altogether, which would
// make its sparsity vacuous: an SAE that outputs zeros is perfectly sparse and perfectly useless.
constexpr float kMaxReconstructionError = 0.05f;

// Sparsity-gap bound, on L0 (active features per input), which the codes' scale can't game.
// Measured: 10.5 penalized against 18.3 unpenalized, a ratio of 0.57; everything but
// l1_lambda is held identical (same seed, data, optimizer and epochs).
constexpr double kMaxL0Ratio = 0.7;

std::vector<float> make_gaussian_batch(TestRng& rng, int64_t n) {
    std::vector<float> values;
    values.reserve(static_cast<size_t>(n * kControlDim));
    for (int64_t i = 0; i < n * kControlDim; ++i) {
        values.push_back(rng.gaussian());
    }
    return values;
}

struct ControlResult {
    float reconstruction_error = 0.0f;
    float mean_hidden_activation = 0.0f;
    double l0 = 0.0;
};

// The single shared training procedure. Both controls call exactly this, and its only
// parameter beyond the data is l1_lambda -- precisely so neither run can quietly be given
// an advantage the other didn't get.
ControlResult train_and_score_heldout(DeviceBackend* backend, const std::vector<float>& train_values,
                                      const std::vector<float>& heldout_values, float l1_lambda, bool unit_norm = true) {
    SparseAutoencoder sae(kControlDim, kControlHiddenDim, l1_lambda, backend, kControlSeed);
    sae.set_unit_norm_decoder(unit_norm);
    AdamOptimizer optimizer(kControlLearningRate, backend);

    const Tensor train_batch(Shape({kControlTrainSize, kControlDim}), backend, train_values);
    for (int epoch = 0; epoch < kControlEpochs; ++epoch) {
        (void)sae.train_step(train_batch, optimizer);
    }

    const Tensor heldout_batch(Shape({kControlHeldoutSize, kControlDim}), backend, heldout_values);
    return {sae.reconstruction_error(heldout_batch), sae.mean_hidden_activation(heldout_batch), MeanL0(sae, heldout_batch)};
}

class SparseAutoencoderControlTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(SparseAutoencoderControlTest, TheL1PenaltyLowersHiddenActivationWithoutCollapsingReconstruction) {
    TestRng rng(20260924u);
    const std::vector<float> train_values = make_gaussian_batch(rng, kControlTrainSize);
    const std::vector<float> heldout_values = make_gaussian_batch(rng, kControlHeldoutSize);

    const ControlResult penalized = train_and_score_heldout(&backend, train_values, heldout_values, kControlPenalty);
    const ControlResult unpenalized = train_and_score_heldout(&backend, train_values, heldout_values, 0.0f);

    std::cout << "[SparseAutoencoder] penalized   (l1_lambda=" << kControlPenalty
              << "): held-out reconstruction_error=" << penalized.reconstruction_error
              << ", mean_hidden_activation=" << penalized.mean_hidden_activation << ", L0=" << penalized.l0 << std::endl;
    std::cout << "[SparseAutoencoder] unpenalized (l1_lambda=0): held-out reconstruction_error="
              << unpenalized.reconstruction_error
              << ", mean_hidden_activation=" << unpenalized.mean_hidden_activation << ", L0=" << unpenalized.l0 << std::endl;

    // (a) Neither run may collapse reconstruction -- the penalized one is the one at risk.
    EXPECT_LT(unpenalized.reconstruction_error, kMaxReconstructionError)
        << "held-out reconstruction error of the no-penalty control";
    EXPECT_LT(penalized.reconstruction_error, kMaxReconstructionError)
        << "held-out reconstruction error of the penalized run -- a penalty that destroys "
           "reconstruction makes its own sparsity vacuous";

    // (b) The penalty must make the codes sparser, not merely present.
    EXPECT_LT(penalized.l0, unpenalized.l0 * kMaxL0Ratio) << "penalized L0 vs. the identical run with l1_lambda = 0";
    EXPECT_LT(penalized.mean_hidden_activation, unpenalized.mean_hidden_activation);
}

// Why decoders are kept at unit norm (FEAT-1): without it, an L1 penalty is largely paid by
// shrinking every code and growing the decoder to compensate, so the codes' mean falls much
// more than the number of active features. At l1_lambda = 0.05 here, free decoders cut the mean
// activation to 0.19 of the unpenalized run's but L0 only to 0.58 of it; unit-norm decoders
// cut L0 to 0.21.
TEST_F(SparseAutoencoderControlTest, WithFreeDecodersTheL1PenaltyShrinksCodesMoreThanItSparsifiesThem) {
    TestRng rng(20260924u);
    const std::vector<float> train_values = make_gaussian_batch(rng, kControlTrainSize);
    const std::vector<float> heldout_values = make_gaussian_batch(rng, kControlHeldoutSize);
    const ControlResult free_none = train_and_score_heldout(&backend, train_values, heldout_values, 0.0f, false);
    const ControlResult free_l1 = train_and_score_heldout(&backend, train_values, heldout_values, 0.05f, false);
    const ControlResult unit_none = train_and_score_heldout(&backend, train_values, heldout_values, 0.0f, true);
    const ControlResult unit_l1 = train_and_score_heldout(&backend, train_values, heldout_values, 0.05f, true);
    const double free_activation_ratio = free_l1.mean_hidden_activation / free_none.mean_hidden_activation;
    const double free_l0_ratio = free_l1.l0 / free_none.l0, unit_l0_ratio = unit_l1.l0 / unit_none.l0;
    std::cout << "[SparseAutoencoder] l1_lambda 0.05, free decoders: activation x" << free_activation_ratio << ", L0 x" << free_l0_ratio
              << "; unit-norm decoders: L0 x" << unit_l0_ratio << std::endl;
    EXPECT_LT(free_activation_ratio, 0.5 * free_l0_ratio);  // mostly shrinking, not sparsifying
    EXPECT_LT(unit_l0_ratio, 0.5 * free_l0_ratio);           // unit norm turns the penalty into sparsity
}

// The control above compares two *different* lambdas at one point. This one pins the
// direction of the effect across a third setting, so the comparison cannot be an accident
// of the single pair of values chosen: a larger penalty must not be less sparse than a
// smaller one.
TEST_F(SparseAutoencoderControlTest, SparsityIncreasesMonotonicallyWithThePenaltyCoefficient) {
    TestRng rng(20260924u);
    const std::vector<float> train_values = make_gaussian_batch(rng, kControlTrainSize);
    const std::vector<float> heldout_values = make_gaussian_batch(rng, kControlHeldoutSize);

    const ControlResult none = train_and_score_heldout(&backend, train_values, heldout_values, 0.0f);
    const ControlResult small = train_and_score_heldout(&backend, train_values, heldout_values, 0.01f);
    const ControlResult large = train_and_score_heldout(&backend, train_values, heldout_values, 0.05f);

    std::cout << "[SparseAutoencoder] mean_hidden_activation by l1_lambda: 0 -> " << none.mean_hidden_activation
              << ", 0.01 -> " << small.mean_hidden_activation << ", 0.05 -> " << large.mean_hidden_activation
              << std::endl;

    EXPECT_LT(small.mean_hidden_activation, none.mean_hidden_activation);
    EXPECT_LT(large.mean_hidden_activation, small.mean_hidden_activation);
}

}  // namespace
}  // namespace pulsatrix
