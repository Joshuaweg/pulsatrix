/** @file born_again.hpp
 *  @brief Born-again networks (Furlanello et al., ICML 2018; roadmap KD-4): a student the size of
 *         its teacher, trained on the teacher's outputs, then a teacher for the next generation.
 *  @ingroup dl_modules
 *
 * Train a model on the labels; train a second with the same architecture on the first's softened
 * outputs (DistillationLoss); repeat. Furlanello et al. found the students often beat their
 * teachers, and an ensemble of the generations did better still. Later work found the gains small
 * and not always there, so measure them over several seeds before believing them.
 *
 * Two pieces here; the training loop is yours:
 *
 *  - **EnsembleLogProbabilities:** the generations' mean distribution, as log-probabilities. It is
 *    the born-again ensemble's prediction, and, passed to DistillationLoss as the teacher's logits
 *    (softmax(log p) = p), a teacher made of several generations.
 *  - **PermuteDarkKnowledge:** Furlanello et al.'s control (DKPP). Each row keeps its largest
 *    logit where it is and shuffles the others among the other classes. The teacher's top class
 *    and its confidence stay; what it says about the other classes, the dark knowledge, is
 *    scrambled. If a student taught this way does as well, the gain didn't come from the dark
 *    knowledge.
 *
 * Not here: their other control, CWTM, which weights each example's label loss by the teacher's
 * confidence. It needs per-example weights in the cross-entropy loss.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief log((1/G) Σ_g softmax(logits_g)) row by row, (..., classes), on the first tensor's device.
 * @note Computed on the host in double precision, with each softmax shifted by its row's maximum.
 * @throws std::invalid_argument if @p logits is empty, a rank is below 2, or the shapes or devices
 *         differ.
 */
[[nodiscard]] Tensor EnsembleLogProbabilities(const std::vector<Tensor>& logits, DeviceBackend* backend);

/**
 * @brief The teacher's logits with each row's non-maximal entries shuffled among themselves by a
 *        portable generator seeded with @p seed: the same result on every platform. The largest
 *        entry (the first, if tied) stays in place.
 * @throws std::invalid_argument if the rank is below 2.
 */
[[nodiscard]] Tensor PermuteDarkKnowledge(const Tensor& teacher_logits, uint64_t seed, DeviceBackend* backend);

}  // namespace pulsatrix
