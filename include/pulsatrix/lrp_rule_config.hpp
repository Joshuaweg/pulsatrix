/** @file lrp_rule_config.hpp
 *  @brief Selects which LRP rule variant a Module::propagate_relevance() call uses.
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
