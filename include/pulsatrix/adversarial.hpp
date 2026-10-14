/** @file adversarial.hpp
 *  @brief Adversarial attacks on a classifier (KS-4): the fast gradient sign method (Goodfellow,
 *         Shlens and Szegedy, ICLR 2015) and projected gradient descent (Madry et al., ICLR 2018),
 *         under the L∞ or L2 norm, untargeted or targeted, and the robust accuracy they leave.
 *  @ingroup interpretability
 *
 * An adversarial example is an input changed by less than ε, in some norm, that the model
 * classifies differently. How small an ε suffices says how robust the model is; which inputs fall
 * first says where it is fragile. That answers the Control reason's question, what could make
 * this model fail, with a number.
 *
 * Both attacks follow the gradient of the cross-entropy loss with respect to the input:
 *
 *  - **FGSM** takes one step of size ε: x + ε sign(∇ₓL) under L∞, x + ε ∇ₓL / ‖∇ₓL‖₂ under L2
 *    (the fast gradient method).
 *  - **PGD** starts at a random point of the ε-ball, takes `steps` steps of size α the same way,
 *    and after each one projects back onto the ball and into the input's valid range.
 *
 * A targeted attack descends the loss of a chosen class instead of ascending the true one's.
 * The steps, the projection and the random start follow torchattacks' FGSM, PGD and PGDL2;
 * tests/adversarial_test.cpp checks them against tools/generate_adversarial_reference_values.py,
 * a numpy implementation of the same updates.
 *
 * PGD is a lower bound on what an attacker can do: a model that resists it may fall to a stronger
 * attack. Robust accuracy measured with it is an upper bound on the true robust accuracy.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

class DeviceBackend;
class ExplainerContext;
class Module;

/**
 * @brief A classifier the attacks can differentiate: `forward` maps an (N, ...) input to (N, C)
 *        logits; `backward` maps a gradient at those logits to the gradient at the input, for the
 *        input of the last forward.
 */
struct DifferentiableClassifier {
    std::function<Tensor(const Tensor&)> forward;
    std::function<Tensor(const Tensor&)> backward;
};

/** @brief Differentiates through an ExplainerContext's autograd, as the gradient explainers do;
 *         the model's parameter gradients are left alone. The context must outlive the result. */
[[nodiscard]] DifferentiableClassifier ClassifierFrom(ExplainerContext& context);

/** @brief Differentiates through Module::forward and Module::backward. backward() also
 *         accumulates the parameters' gradients, so zero them before training again. The module
 *         must outlive the result, and should be in eval mode, so dropout doesn't change between
 *         the steps. */
[[nodiscard]] DifferentiableClassifier ClassifierFrom(Module& model);

/** @brief The norm an attack's budget is measured in. */
enum class AttackNorm { Linf, L2 };

/** @brief Settings for Fgsm() and Pgd(). */
struct AttackOptions {
    /** @brief The budget: every adversarial input is within epsilon of its original. */
    double epsilon = 0.03;
    AttackNorm norm = AttackNorm::Linf;
    /** @brief PGD: the number of steps. */
    int64_t steps = 10;
    /** @brief PGD: the step size; 0 means 2.5 epsilon / steps, Madry et al.'s choice. */
    double step_size = 0.0;
    /** @brief PGD: start at a uniformly random point of the ball (the L2 ball for L2), as Madry
     *         et al. do; without it PGD starts at the input. */
    bool random_start = true;
    uint64_t seed = 0;
    /** @brief The valid range of each input element, such as [0, 1] for pixels; every adversarial
     *         input is clipped into it. Unbounded by default. */
    double clip_min = -std::numeric_limits<double>::infinity();
    double clip_max = std::numeric_limits<double>::infinity();
};

/** @brief One attack's result. */
struct AttackResult {
    /** @brief The adversarial inputs, the input's shape, on the input's device. */
    Tensor adversarial;
    /** @brief Each input's predicted class before and after the attack. */
    std::vector<int64_t> clean_predicted;
    std::vector<int64_t> adversarial_predicted;
    /** @brief Untargeted: the adversarial input isn't classified as its label. Targeted: it is
     *         classified as its target. */
    std::vector<bool> success;
    /** @brief The cross-entropy of each label (the target's, when targeted) before and after. */
    std::vector<double> clean_loss;
    std::vector<double> adversarial_loss;
    /** @brief Each perturbation's size in the attack's norm: at most epsilon, up to rounding. */
    std::vector<double> perturbation;
    /** @brief The fraction of inputs the attack succeeded on. */
    double success_rate = 0.0;
    /** @brief Untargeted: the fraction classified correctly after the attack. */
    double robust_accuracy = 0.0;
    /** @brief Untargeted: the fraction classified correctly before it. */
    double clean_accuracy = 0.0;
    bool targeted = false;
};

/**
 * @brief The fast gradient sign method: one step of size epsilon (options.steps, step_size and
 *        random_start are ignored).
 * @param input (N, ...): N inputs, any shape after the first dimension.
 * @param labels The true class of each input, or with @p targeted the class to push it to.
 * @param backend Backend to put the perturbed inputs on, on the input's device.
 * @throws std::invalid_argument if labels doesn't have N entries, a label is outside the logits'
 *         classes, the logits aren't (N, C) with C >= 2, epsilon isn't finite and positive, or
 *         clip_min >= clip_max.
 */
[[nodiscard]] AttackResult Fgsm(const DifferentiableClassifier& model, const Tensor& input, const std::vector<int64_t>& labels,
                                DeviceBackend* backend, const AttackOptions& options = {}, bool targeted = false);

/**
 * @brief Projected gradient descent: random start, then `steps` steps, each projected onto the
 *        epsilon-ball around the input and into [clip_min, clip_max]. Returns the last iterate.
 * @throws std::invalid_argument as Fgsm() does, or if steps < 1 or step_size is negative or not
 *         finite.
 */
[[nodiscard]] AttackResult Pgd(const DifferentiableClassifier& model, const Tensor& input, const std::vector<int64_t>& labels,
                               DeviceBackend* backend, const AttackOptions& options = {}, bool targeted = false);

/** @brief One budget's result on a robustness curve. */
struct RobustnessPoint {
    double epsilon = 0.0;
    /** @brief The fraction of inputs still classified correctly after PGD. */
    double robust_accuracy = 0.0;
    /** @brief Of the inputs classified correctly before the attack, the fraction it flipped. */
    double flip_rate = 0.0;
    /** @brief The mean perturbation size of the successful attacks; 0 if there were none. */
    double mean_perturbation = 0.0;
};

/** @brief RobustAccuracyCurve()'s result. */
struct RobustnessCurve {
    std::string attack;
    AttackNorm norm = AttackNorm::Linf;
    int64_t n = 0;
    int64_t steps = 0;
    double clean_accuracy = 0.0;
    std::vector<RobustnessPoint> points;
};

/**
 * @brief Robust accuracy against untargeted PGD at each budget: how fast the accuracy falls as the
 *        attacker gets more room. options.epsilon is ignored; options.step_size, when set, is a
 *        fraction of each epsilon (0 keeps 2.5 / steps).
 * @throws std::invalid_argument as Pgd() does, or if epsilons is empty or holds a value that isn't
 *         finite and positive.
 */
[[nodiscard]] RobustnessCurve RobustAccuracyCurve(const DifferentiableClassifier& model, const Tensor& input,
                                                  const std::vector<int64_t>& labels, const std::vector<double>& epsilons,
                                                  DeviceBackend* backend, const AttackOptions& options = {});

/** @brief The result as a `pulsatrix.attack.v1` document, without the adversarial inputs. */
[[nodiscard]] JsonValue ToJson(const AttackResult& result, const AttackOptions& options);

/** @brief The curve as a `pulsatrix.robustness_curve.v1` document. */
[[nodiscard]] JsonValue ToJson(const RobustnessCurve& curve);

}  // namespace pulsatrix
