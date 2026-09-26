/** @file lrp_rule_config.hpp
 *  @brief Selects which LRP rule variant a Module::propagate_relevance() call uses.
 *  @ingroup interpretability_dl
 *  @see Module::propagate_relevance(), and the per-layer LRP rule implementations in
 *       @ref dl_modules (e.g. LinearModule, Conv2DModule, TransformerBlock) -- LRP rules
 *       live alongside each layer's forward/backward math, not in a separate file.
 */
#pragma once

namespace pulsatrix {

/**
 * @brief Configuration for LRP relevance propagation. Selects *which* rule variant to
 *        apply (currently: epsilon rule only); it never changes *whether* a module
 *        supports LRP -- that's guaranteed at compile time by Module::propagate_relevance
 *        being pure-virtual. Gamma/alpha-beta fields are added when a module first needs
 *        them, not speculatively now.
 */
struct LRPRuleConfig {
    /** @brief Stabilizer added to the epsilon rule's denominator, signed to match z_j. */
    float epsilon = 1e-6f;
};

}  // namespace pulsatrix
