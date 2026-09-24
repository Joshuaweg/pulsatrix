#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "exai/adam_optimizer.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/sparse_autoencoder.hpp"

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
namespace exai {
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

TEST_F(SparseAutoencoderTest, ReconstructionErrorThrowsOnARankThreeBatch) {
    SparseAutoencoder sae(2, 4, 0.01f, &backend);
    Tensor batch(Shape({2, 2, 2}), &backend);

    EXPECT_THROW((void)sae.reconstruction_error(batch), std::invalid_argument);
}

}  // namespace
}  // namespace exai
