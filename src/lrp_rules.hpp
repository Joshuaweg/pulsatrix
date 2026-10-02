// Zennit-compatible LRP rules for affine layers (LinearModule, Conv2DModule), composed from
// DeviceBackend primitives so they run unchanged on CPU, CUDA and HIP. Private to src/.
// LRP-rules campaign Mission 2.
//
// An affine layer is described abstractly as out = forward(x, W) + b, with backward(g, W) its
// input-gradient product (g @ W^T for a dense layer). Notation as in Zennit 1.0.0 (rules.py):
// stab(z) = z + eps * sign(z) with sign(0) = +1; x+ = max(x, 0), x- = min(x, 0), likewise W, b.
//
//   Epsilon (bias in the denominator):
//     z = xW + b;  R_in = x * ((R / stab(z)) W^T)
//   AlphaBeta(alpha, beta), alpha, beta >= 0, alpha - beta == 1:
//     z+ = (x+ W+ + b+) + x- W- ;  z- = (x+ W- + b-) + x- W+
//     g+ = R / stab(z+);  g- = R / stab(z-)
//     R_in = alpha (x+ (g+ W+^T) + x- (g+ W-^T)) - beta (x+ (g- W-^T) + x- (g- W+^T))
//   Gamma(gamma): Wa = W + gamma W+, Wb = W + gamma W-, ba = b + gamma b+, bb = b + gamma b-
//     o0 = x+ Wa + ba;  o1 = x- Wb;  o2 = x+ Wb + bb;  o3 = x- Wa;  z = xW + b (original output)
//     gpos = [z > 0] R / stab(o0 + o1);  gneg = [z < 0] R / stab(o2 + o3)
//     R_in = x+ (gpos Wa^T) + x- (gpos Wb^T) + x+ (gneg Wb^T) + x- (gneg Wa^T)
//   ZBox(low, high): L, H = x-shaped fills
//     den = xW - L W+ - H W-   (Zennit's three biases cancel exactly: b - b+ - b- == 0)
//     g = R / stab(den);  R_in = x (g W^T) - L (g W+^T) - H (g W-^T)
#pragma once

#include <cstddef>
#include <functional>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace lrp_rules {

/** @brief An affine layer seen by the rules: buffers are raw device pointers of the given sizes. */
struct AffineOp {
    DeviceBackend* backend = nullptr;
    DeviceType device = DeviceType::Cpu;
    size_t input_numel = 0;
    size_t output_numel = 0;
    size_t weight_numel = 0;
    size_t bias_numel = 0;
    /// out (output-shaped) = x (input-shaped) through weights w, no bias; overwrites out.
    std::function<void(const float* x, const float* w, float* out)> forward;
    /// out (input-shaped) = the input-gradient product of g (output-shaped) through w; overwrites out.
    std::function<void(const float* g, const float* w, float* out)> backward;
    /// out = in + b broadcast over the output (out may alias in).
    std::function<void(const float* in, const float* b, float* out)> add_bias;
};

/**
 * @brief Throws std::invalid_argument (naming `module_name`) if config's rule parameters are
 *        invalid: AlphaBeta needs alpha >= 0, beta >= 0 and alpha - beta == 1 (Zennit's check).
 */
void validate(const LRPRuleConfig& config, const char* module_name);

/** @brief True when config is the module's original (pre-bias) epsilon rule. */
inline bool is_legacy_epsilon(const LRPRuleConfig& config) {
    return config.rule == LRPRule::Epsilon && !config.epsilon_bias_in_denominator;
}

/**
 * @brief Applies config.rule (anything but the legacy pre-bias epsilon rule) to one affine layer.
 * @param x Cached forward input; w, b the layer's weights and bias; pre_bias = forward(x, w).
 * @param r Relevance at the output; r_in (input-shaped) is overwritten with the input relevance.
 */
void apply(const AffineOp& op, const float* x, const float* w, const float* b, const float* pre_bias, const float* r,
           float* r_in, const LRPRuleConfig& config);

}  // namespace lrp_rules
}  // namespace pulsatrix
