/** @file grad_clipping.hpp
 *  @brief Global gradient-norm clipping (roadmap TRN-3).
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/**
 * @brief Scales every trainable parameter's gradient so their global L2 norm is at most
 *        `max_norm`, as torch.nn.utils.clip_grad_norm_ does, and returns the norm before clipping.
 * @param module Its parameters with requires_grad() set are counted and scaled; frozen ones are
 *        left alone (FND-2).
 * @param max_norm Positive, finite limit.
 * @param error_if_nonfinite Throw instead of returning when the norm is NaN or infinite.
 * @return The global norm before clipping, which is also the usual quantity to log.
 * @note Call it after gradient accumulation and before the optimizer step. When the norm exceeds
 *       max_norm, every gradient is multiplied by the same factor max_norm / (norm + 1e-6), so the
 *       update's direction is kept.
 * @note One difference from PyTorch: a NaN or infinite norm leaves the gradients unchanged
 *       (PyTorch scales them anyway, spreading the NaN). Check the returned norm and skip the step.
 * @throws std::invalid_argument if max_norm isn't positive and finite.
 * @throws std::runtime_error if error_if_nonfinite is set and the norm is NaN or infinite.
 */
[[nodiscard]] float ClipGradNorm(Module& module, float max_norm, bool error_if_nonfinite = false);

}  // namespace pulsatrix
