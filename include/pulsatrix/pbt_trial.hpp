/** @file pbt_trial.hpp
 *  @brief Extends the sibling HPO campaign's own `ResumableTrial` (successive_halving.hpp)
 *         with the weight/hyperparameter read-write capability Population Based Training's own
 *         exploit/explore step needs.
 *  @ingroup evolutionary
 *  @note Recon finding (this mission's own activation check, 2026-09-27): `ResumableTrial`
 *        exists and is real (built for HPO Phase 3's Successive Halving/Hyperband/ASHA), but
 *        exposes only `TrainForEpochs(int) -> double` -- exactly as
 *        `successive_halving.hpp`'s own note predicted, it does not suffice for PBT's
 *        exploit step (copying a top performer's live weights and hyperparameters onto a
 *        bottom performer mid-training) as-is. Per this project's own precedent (RL's Agent
 *        interface extended per-consumer -- DQNAgent's epsilon-greedy methods,
 *        CategoricalPolicyAgent's log-probability accessor -- rather than speculatively
 *        upfront), this is a new, purely-additive derived interface. Zero changes to
 *        ResumableTrial itself: Successive Halving/Hyperband/ASHA are completely unaffected.
 */
#pragma once

#include <vector>

#include "pulsatrix/search_space.hpp"
#include "pulsatrix/successive_halving.hpp"

namespace pulsatrix {

/**
 * @brief A ResumableTrial that additionally exposes its live weights (a flat vector) and its
 *        current hyperparameter Configuration, both readable and replaceable mid-training.
 */
class PBTResumableTrial : public ResumableTrial {
public:
    ~PBTResumableTrial() override = default;

    /** @brief This trial's current live weights, flattened to a single vector. */
    [[nodiscard]] virtual std::vector<double> GetWeights() const = 0;

    /**
     * @brief Overwrites this trial's live weights.
     * @throws Whatever the concrete implementation throws on a size mismatch.
     */
    virtual void SetWeights(const std::vector<double>& weights) = 0;

    /** @brief This trial's current hyperparameter configuration. */
    [[nodiscard]] virtual Configuration GetHyperparameters() const = 0;

    /** @brief Overwrites this trial's hyperparameter configuration (e.g. learning rate). */
    virtual void SetHyperparameters(const Configuration& config) = 0;
};

}  // namespace pulsatrix
