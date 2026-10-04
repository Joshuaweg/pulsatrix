/** @file batch_norm_fold.hpp
 *  @brief Folds an eval-mode BatchNorm into the Conv2D before it for the duration of an LRP
 *         explanation -- Zennit's BatchNorm canonizer (roadmap FND-5, lrp_issues #8).
 *  @ingroup interpretability_lrp
 */
#pragma once

#include <vector>

#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/conv2d_module.hpp"

namespace pulsatrix {

/**
 * @brief While alive, merges `bn`'s affine map into `conv`'s weights and makes `bn` an exact
 *        identity; on destruction, restores both bit for bit.
 * @note With s_c = gamma_c / sqrt(running_var_c + eps), the merged convolution has kernel
 *       w'[c] = s_c * w[c] and bias b'[c] = s_c * (b[c] - running_mean_c) + beta_c. The pair
 *       computes the same function before and during the fold (up to float rounding), but LRP
 *       now distributes relevance through one affine layer with the convolution's own rule,
 *       instead of passing it unchanged through BatchNorm's identity rule.
 * @note Use it as a scope around explaining:
 *       `{ BatchNormFold fold(conv, bn); auto r = lrp.explain(...); }`. Don't train while a
 *       fold is active: gradients would update the merged weights, and the restore would
 *       overwrite those updates.
 */
class BatchNormFold {
public:
    /**
     * @throws std::invalid_argument if `bn` is in training mode (its statistics are not fixed,
     *         so there is no single affine map to fold), or if `bn`'s channel count differs from
     *         `conv`'s output channels.
     * @throws std::logic_error if `bn` is already folded.
     */
    BatchNormFold(Conv2DModule& conv, BatchNormModule& bn);
    ~BatchNormFold();

    BatchNormFold(const BatchNormFold&) = delete;
    BatchNormFold& operator=(const BatchNormFold&) = delete;
    BatchNormFold(BatchNormFold&&) = delete;
    BatchNormFold& operator=(BatchNormFold&&) = delete;

private:
    Conv2DModule& conv_;
    BatchNormModule& bn_;
    std::vector<float> original_kernel_;
    std::vector<float> original_bias_;
};

}  // namespace pulsatrix
