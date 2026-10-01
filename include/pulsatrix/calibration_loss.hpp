/** @file calibration_loss.hpp
 *  @brief Brier score -- a proper-scoring-rule calibration loss (Brier, 1950).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief `BS = mean_n( Σ_k (p[n,k] − y[n,k])² )`, `y` one-hot at `target_class[n]` -- Brier's
 *        original multi-class proper scoring rule.
 *
 * @note Takes probabilities directly, not raw logits -- deliberately **not** fused with
 *       softmax the way `CrossEntropyLoss` is. `CrossEntropyLoss` fuses softmax specifically
 *       because `log(softmax(x))` needs the log-sum-exp stabilization trick to avoid
 *       `log(0)`; the Brier score has no log anywhere, so there is no numerical-stability
 *       reason to fuse softmax the same way. A caller with raw logits composes the existing
 *       `SoftmaxModule` first. See `mission_calibration_loss.md`'s Design section.
 * @note **Does not decompose into aleatoric/epistemic uncertainty.** That decomposition
 *       structurally requires a credal set or second-order distribution -- genuinely multiple
 *       forward passes / an ensemble (Hofman, Sale & Hüllermeier, arXiv:2404.12215) -- which
 *       a single deterministic prediction cannot produce. This class computes the base
 *       scoring-rule loss only; the decomposition is a distinct, not-yet-built feature
 *       requiring its own multi-forward-pass design, not something this single-pass API
 *       silently omits or approximates.
 * @note Averaged over the batch of `N` examples (not over `N·K` elements) -- mirrors
 *       `DQNLoss`/`PolicyGradientLoss`'s "mean over examples" convention: each example
 *       contributes one Brier score (itself already summed over `K` classes), so the batch
 *       mean divides by `N`, not `N·K`.
 * @note Not a Module subclass, for `MSELoss`'s own reason: a loss is the seed point relevance
 *       propagation starts from, not something `propagate_relevance` is defined for.
 */
class CalibrationLoss {
public:
    /**
     * @brief Constructs a calibration loss.
     * @param backend Backend to allocate the gradient tensor through. Not owned; must outlive
     *        this loss.
     */
    explicit CalibrationLoss(DeviceBackend* backend);

    /**
     * @brief Computes the Brier score and caches what backward() needs.
     * @param probs Predicted probabilities, shape (N, num_classes), N >= 1, num_classes >= 1.
     *        Not validated to sum to 1 per row -- an internal numerical invariant of whatever
     *        produced them (e.g. `SoftmaxModule`), not an external boundary this loss owns.
     * @param target_class Ground-truth class index per example, shape (N, 1) -- the same
     *        float-encoded discrete-index convention `DQNLoss`/`PolicyGradientLoss` use.
     * @return The scalar mean Brier score.
     * @throws std::invalid_argument if either tensor has the wrong rank, if the batch
     *         dimensions disagree, if an encoded class index is not within 1e-4 of a whole
     *         number, or if a decoded index falls outside [0, num_classes) -- all external
     *         boundaries, byte-for-byte `PolicyGradientLoss::forward`'s own classification.
     * @note Not yet backend-generic -- dereferences Tensor::data() directly in a raw host
     *       loop. PULSATRIX_REQUIRE_HOST on both inputs guards against
     *       silent UB on a CUDA-backed Tensor.
     */
    [[nodiscard]] float forward(const Tensor& probs, const Tensor& target_class);

    /**
     * @brief Gradient w.r.t. `probs`: `2(p[n,k] − y[n,k]) / N`.
     * @return Gradient tensor, shape (N, num_classes) -- the shape of `probs` passed to
     *         forward().
     * @throws std::logic_error if forward() has never been called.
     */
    [[nodiscard]] Tensor backward() const;

private:
    DeviceBackend* backend_;
    Tensor last_probs_;
    std::vector<int64_t> last_target_indices_;
    bool has_forwarded_ = false;
};

}  // namespace pulsatrix
