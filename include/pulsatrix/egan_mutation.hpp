/** @file egan_mutation.hpp
 *  @brief E-GAN's own three named "mutation" objectives (Wang et al. 2019, "Evolutionary
 *         Generative Adversarial Networks") plus its sample-quality/diversity fitness metric.
 *         In E-GAN, "mutation" means training a copy of a generator against a *different*
 *         loss function, not perturbing its weights directly -- a structurally different kind
 *         of mutation than this campaign's own Phase 3 NEAT/ES operators, which do perturb
 *         weights.
 *  @ingroup evolutionary
 *  @note Reuses this codebase's own already-shipped, tested loss primitives
 *        (BCEWithLogitsLoss, MSELoss) rather than reimplementing softplus/MSE math by hand:
 *        - Heuristic (HG, Goodfellow's non-saturating form): BCEWithLogitsLoss(D(fake), 1) --
 *          identical to gan_integration_test.cpp's own ToyGAN::generator_step objective.
 *        - Least-squares (LS, Mao et al. 2017's LSGAN, adapted onto a logit-output
 *          discriminator exactly as E-GAN's own paper does): MSELoss(D(fake), 1).
 *        - Minimax (MM, the original saturating GAN objective, Goodfellow et al. 2014):
 *          `log(1 - sigmoid(x))` -- algebraically exactly `-BCEWithLogitsLoss(x, 0)`
 *          (`BCEWithLogitsLoss(x, 0) = softplus(x) = -log(1-sigmoid(x))`), computed as the
 *          negation of BCEWithLogitsLoss's own numerically-stable value/gradient rather than a
 *          separate hand-rolled softplus implementation.
 */
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pulsatrix/bce_with_logits_loss.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/mse_loss.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief E-GAN's three named mutation objectives. */
enum class MutationObjective { Minimax, Heuristic, LeastSquares };

/**
 * @brief Computes one of E-GAN's three named mutation objectives against a discriminator's
 *        own raw logit output, and its gradient w.r.t. those logits. Every objective trains
 *        the generator to make the discriminator's output move toward the "real" (1) class --
 *        they differ only in *how* that pressure is shaped (saturating vs. non-saturating vs.
 *        quadratic).
 */
class MutationLoss {
public:
    explicit MutationLoss(DeviceBackend* backend) : backend_(backend), bce_(backend), mse_(backend) {}

    /** @brief Computes the chosen objective's scalar value against logits. */
    float Forward(MutationObjective objective, const Tensor& logits) {
        objective_ = objective;
        Tensor ones(logits.shape(), backend_);
        ones.fill(1.0f);
        switch (objective) {
            case MutationObjective::Heuristic:
                return bce_.forward(logits, ones);
            case MutationObjective::LeastSquares:
                return mse_.forward(logits, ones);
            case MutationObjective::Minimax: {
                Tensor zeros(logits.shape(), backend_);
                zeros.fill(0.0f);
                return -bce_.forward(logits, zeros);
            }
        }
        throw std::invalid_argument("MutationLoss::Forward: unknown MutationObjective");
    }

    /**
     * @brief Gradient w.r.t. the logits passed to the most recent Forward() call.
     * @throws std::logic_error if Forward() has never been called (delegated from the
     *         underlying loss's own backward()).
     */
    [[nodiscard]] Tensor Backward() const {
        switch (objective_) {
            case MutationObjective::Heuristic:
                return bce_.backward();
            case MutationObjective::LeastSquares:
                return mse_.backward();
            case MutationObjective::Minimax:
                return Negate(bce_.backward());
        }
        throw std::invalid_argument("MutationLoss::Backward: unknown MutationObjective");
    }

private:
    [[nodiscard]] Tensor Negate(const Tensor& t) const {
        std::vector<float> values(static_cast<size_t>(t.numel()));
        for (int64_t i = 0; i < t.numel(); ++i) {
            values[static_cast<size_t>(i)] = -t.data()[i];
        }
        return Tensor(t.shape(), backend_, values, t.device());
    }

    DeviceBackend* backend_;
    BCEWithLogitsLoss bce_;
    MSELoss mse_;
    MutationObjective objective_ = MutationObjective::Heuristic;
};

/**
 * @brief E-GAN's own quality fitness Fq: mean sigmoid(D(fake)) over the batch -- how
 *        convincingly "real" the discriminator currently rates these samples. Higher is
 *        better (the discriminator being fooled more).
 */
inline float QualityFitness(const Tensor& logits) {
    double total = 0.0;
    for (int64_t i = 0; i < logits.numel(); ++i) {
        total += 1.0 / (1.0 + std::exp(-static_cast<double>(logits.data()[i])));
    }
    return static_cast<float>(total / static_cast<double>(logits.numel()));
}

/**
 * @brief E-GAN's own diversity fitness Fd = -log(||grad||): the negative log of the L2 norm
 *        of the discriminator's own parameter gradient from its fake-recognition loss term
 *        (BCEWithLogitsLoss(D(fake), 0)), evaluated on fake. A *smaller* discriminator
 *        gradient here means the discriminator is already close to a local optimum against
 *        these particular samples -- the paper's own signal that this offspring is
 *        contributing mode coverage the discriminator can't easily exploit further
 *        (discourages mode collapse).
 * @note This function's own backward() pass is a fitness-evaluation probe, not a real training
 *       step -- its gradient accumulates into discriminator's own parameter-gradient buffers
 *       exactly like any other backward() call would, and the caller must zero_grad
 *       discriminator's optimizer immediately after calling this function, every time (the
 *       same gradient-contamination discipline BCEWithLogitsLoss's own @warning already
 *       establishes for the generator step).
 * @throws std::invalid_argument if discriminator has no parameters.
 */
inline float DiversityFitness(Module& discriminator, const Tensor& fake, DeviceBackend* backend) {
    auto params = discriminator.parameters();
    if (params.empty()) {
        throw std::invalid_argument("DiversityFitness: discriminator must have at least one parameter");
    }

    BCEWithLogitsLoss bce(backend);
    Tensor logits = discriminator.forward(fake);
    Tensor zeros(logits.shape(), backend);
    zeros.fill(0.0f);
    (void)bce.forward(logits, zeros);
    (void)discriminator.backward(bce.backward());

    double sum_squares = 0.0;
    for (const auto& p : params) {
        for (int64_t i = 0; i < p.grad->numel(); ++i) {
            double g = p.grad->data()[i];
            sum_squares += g * g;
        }
    }
    double norm = std::sqrt(sum_squares);
    // An exactly-zero gradient (e.g. against a freshly-constructed, all-zero-weight
    // discriminator) maps to +infinity fitness -- treated as "maximally diverse" by explicit
    // convention rather than left as an undefined log(0), a documented edge case, not a
    // silent bug.
    if (norm <= 0.0) {
        return std::numeric_limits<float>::infinity();
    }
    return static_cast<float>(-std::log(norm));
}

/**
 * @brief Combined E-GAN fitness: Fq + gamma*Fd (Wang et al. 2019's own weighted combination).
 *        gamma's default (0.05) is this mission's own reasonable working value, not a literal
 *        reproduction of the paper's own tuned constant (never stated precisely enough there
 *        to reproduce exactly) -- a deliberate, documented choice, not an assumed one.
 */
inline float CombinedFitness(float quality, float diversity, float gamma = 0.05f) {
    return quality + gamma * diversity;
}

/**
 * @brief Runs one E-GAN mutation training step: trains offspring (an already-independent
 *        Module instance -- typically initialized as a copy of some parent's current weights,
 *        which this function does not itself construct or assume anything about) for one step
 *        against discriminator using the given objective, then scores the mutated result via
 *        CombinedFitness on offspring's own post-mutation samples.
 * @note Caller must zero_grad discriminator's own optimizer immediately after calling this
 *       function -- both the objective's own backward-through-discriminator call and
 *       DiversityFitness's own probe backward accumulate into discriminator's gradient
 *       buffers, exactly like a generator step in this codebase's existing GAN training loop.
 */
template <typename OptimizerT>
float RunMutationStep(Module& offspring, Module& discriminator, const Tensor& noise, MutationObjective objective,
                       OptimizerT& g_optimizer, DeviceBackend* backend, float gamma = 0.05f) {
    Tensor fake = offspring.forward(noise);
    Tensor logits = discriminator.forward(fake);
    MutationLoss loss(backend);
    (void)loss.Forward(objective, logits);
    Tensor grad_fake = discriminator.backward(loss.Backward());
    (void)offspring.backward(grad_fake);
    g_optimizer.step(offspring);
    g_optimizer.zero_grad(offspring);

    Tensor post_fake = offspring.forward(noise);
    Tensor post_logits = discriminator.forward(post_fake);
    float quality = QualityFitness(post_logits);
    float diversity = DiversityFitness(discriminator, post_fake, backend);
    return CombinedFitness(quality, diversity, gamma);
}

}  // namespace pulsatrix
