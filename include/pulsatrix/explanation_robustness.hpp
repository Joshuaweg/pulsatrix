/** @file explanation_robustness.hpp
 *  @brief How much an explanation changes when its input barely does (KS-4): the max- and
 *         average-sensitivity of Yeh et al. (NeurIPS 2019), the local Lipschitz estimate of
 *         Alvarez-Melis and Jaakkola (2018), and the sensitivity at the perturbation an attacker
 *         would try first, FGSM's.
 *  @ingroup interpretability
 *
 * An explanation that changes a lot under a change to the input too small to change the
 * prediction says more about the explainer than the model: Ghorbani, Abid and Zou (AAAI 2019)
 * changed saliency maps beyond recognition with imperceptible perturbations. These are the
 * robustness metrics of Quantus (Hedström et al., JMLR 2023), computed the same way:
 *
 *  - **max- and average-sensitivity:** ‖E(x') − E(x)‖₂ / ‖E(x)‖₂ over sampled x' near x, the
 *    largest and the mean;
 *  - **local Lipschitz estimate:** the largest ‖E(x') − E(x)‖₂ / ‖x' − x‖₂;
 *  - **adversarial sensitivity:** ‖E(x') − E(x)‖₂ / ‖E(x)‖₂ at x' = FGSM's step from x, the
 *    direction that changes the model's loss fastest.
 *
 * The samples are uniform in the L∞ ball of radius `radius` times the input's standard deviation,
 * and only those that keep the predicted class count: an explanation should change when the
 * prediction does. tests/explanation_robustness_test.cpp checks the metrics against
 * tools/generate_explanation_robustness_reference_values.py, which draws the same samples.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "pulsatrix/adversarial.hpp"
#include "pulsatrix/explanation_metrics.hpp"
#include "pulsatrix/json.hpp"

namespace pulsatrix {

/** @brief ExplanationSensitivity()'s settings. */
struct SensitivityOptions {
    /** @brief The perturbations' L∞ radius, as a fraction of the input's standard deviation (its
     *         largest absolute value when the input is constant). Quantus's default is an absolute
     *         0.2. */
    double radius = 0.1;
    /** @brief Perturbations to sample; Quantus uses 200, which is slow for sampling explainers. */
    int64_t samples = 20;
    uint64_t seed = 0;
    /** @brief The adversarial point: the FGSM step is halved until the prediction stays, at most
     *         this many times. */
    int64_t max_halvings = 4;
};

/** @brief ExplanationSensitivity()'s result. */
struct SensitivityReport {
    /** @brief The largest and the mean relative change of the explanation over the kept samples;
     *         0 when none were kept. */
    double max_sensitivity = 0.0;
    double avg_sensitivity = 0.0;
    /** @brief The largest change of the explanation per unit change of the input. */
    double local_lipschitz = 0.0;
    /** @brief The relative change at FGSM's point; empty without a classifier, or when every
     *         halving still changed the prediction. */
    std::optional<double> adversarial_sensitivity;
    /** @brief FGSM's L∞ step at the adversarial point. */
    double adversarial_epsilon = 0.0;
    /** @brief The samples drawn, and those that kept the predicted class. */
    int64_t samples = 0;
    int64_t kept = 0;
    /** @brief The L∞ radius the samples were drawn in, in the input's units. */
    double radius = 0.0;
};

/**
 * @brief Measures how much @p explain's explanation of @p input changes under small perturbations.
 * @param input One input, `(1, ...)`.
 * @param explain The explainer; its attribution's values are compared as one flat vector.
 * @param predict The model; the index of its largest output is the predicted class.
 * @param classifier The model again as a DifferentiableClassifier, for the adversarial point; nullptr
 *        to skip it.
 * @param backend Backend to put the perturbed inputs on, on the input's device.
 * @throws std::invalid_argument if the input's first dimension isn't 1, radius isn't finite and
 *         positive, samples < 1, max_halvings < 0, or an explanation's size changes between calls.
 */
[[nodiscard]] SensitivityReport ExplanationSensitivity(const Tensor& input, const ExplainFn& explain, const PredictFn& predict,
                                                       const DifferentiableClassifier* classifier, DeviceBackend* backend,
                                                       const SensitivityOptions& options = {});

/** @brief The report as a `pulsatrix.explanation_sensitivity.v1` document. */
[[nodiscard]] JsonValue ToJson(const SensitivityReport& report);

}  // namespace pulsatrix
