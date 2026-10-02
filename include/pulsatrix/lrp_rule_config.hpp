/** @file lrp_rule_config.hpp
 *  @brief Selects which LRP rule variant a Module::propagate_relevance() call uses.
 *  @ingroup interpretability_dl
 *  @see Module::propagate_relevance(), and the per-layer LRP rule implementations in
 *       @ref dl_modules (e.g. LinearModule, Conv2DModule, TransformerBlock) -- LRP rules
 *       live alongside each layer's forward/backward math, not in a separate file.
 */
#pragma once

#include <string>

namespace pulsatrix {

/**
 * @brief The LRP rule family a module applies. Semantics follow Zennit 1.0.0 exactly
 *        (Anders et al. 2021); every rule other than Epsilon is defined only for affine layers
 *        (LinearModule, Conv2DModule) -- see Module::supports_lrp_rule().
 */
enum class LRPRule {
    /// R_in = x * ((R / stab(z)) @ W^T). The bias is left out of z unless
    /// LRPRuleConfig::epsilon_bias_in_denominator is set (Zennit's Epsilon includes it).
    Epsilon,
    /// Zennit's generalized Gamma rule: weights W + gamma W^+ / W + gamma W^-, split by the sign
    /// of the input and of the original output (Montavon et al. 2019, generalized to negative
    /// activations).
    Gamma,
    /// Alpha-beta rule (Bach et al. 2015) with alpha - beta == 1; alpha 1, beta 0 is ZPlus.
    AlphaBeta,
    /// Z^B rule (Montavon et al. 2017) for a box-bounded input [low, high]; meant for the
    /// input layer.
    ZBox
};

/** @brief Lower-case rule name ("epsilon", "gamma", "alpha_beta", "zbox") for messages/metadata. */
inline std::string lrp_rule_name(LRPRule rule) {
    switch (rule) {
        case LRPRule::Epsilon:
            return "epsilon";
        case LRPRule::Gamma:
            return "gamma";
        case LRPRule::AlphaBeta:
            return "alpha_beta";
        case LRPRule::ZBox:
            return "zbox";
    }
    return "unknown";
}

/**
 * @brief Configuration for LRP relevance propagation: which rule a module applies and its
 *        hyperparameters. A module that does not implement the requested rule throws (see
 *        Module::supports_lrp_rule()) -- the rule is never silently substituted.
 * @note `epsilon` stays the first member so aggregate initialization `LRPRuleConfig{0.01f}`
 *       keeps meaning "epsilon rule, epsilon 0.01". It is also the denominator stabilizer of
 *       Gamma / AlphaBeta / ZBox (Zennit's `stabilizer`, default 1e-6 there as well).
 */
struct LRPRuleConfig {
    /** @brief Stabilizer added to every rule's denominator: z + epsilon * sign(z), sign(0) = +1. */
    float epsilon = 1e-6f;
    /** @brief Which rule to apply. */
    LRPRule rule = LRPRule::Epsilon;
    /** @brief Gamma rule: weight of the positive (or negative) part added to the weights. */
    float gamma = 0.25f;
    /** @brief AlphaBeta rule: weight of the positive contributions. Needs alpha, beta >= 0, alpha - beta == 1. */
    float alpha = 1.0f;
    /** @brief AlphaBeta rule: weight of the negative contributions. */
    float beta = 0.0f;
    /** @brief ZBox rule: lowest admissible input value (scalar, applied to every input element). */
    float low = 0.0f;
    /** @brief ZBox rule: highest admissible input value. */
    float high = 1.0f;
    /**
     * @brief Epsilon rule only, honoured by LinearModule / Conv2DModule: include the bias in the
     *        denominator, z = xW + b (Zennit's Epsilon). The default (false) keeps this codebase's
     *        conservative pre-bias z = xW, under which the bias absorbs no relevance.
     */
    bool epsilon_bias_in_denominator = false;
};

}  // namespace pulsatrix
