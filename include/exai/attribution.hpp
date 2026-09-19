/** @file attribution.hpp
 *  @brief First-class explanation result type -- values, method, and metadata together.
 */
#pragma once

#include <string>
#include <unordered_map>

#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief An explanation result: the raw attribution values, the method that produced
 *        them, and any relevant metadata (charter Part 2 SS4).
 * @note Deliberately not just a Tensor -- downstream code (audit tooling, a future viz
 *       layer) shouldn't have to guess what produced a number. metadata is a simple
 *       string-keyed map (e.g. baseline description for Integrated Gradients, kernel
 *       width for LIME) -- no richer variant type until a real method actually needs one.
 */
struct Attribution {
    /** @brief Which explanation method produced this result (e.g. "saliency", "grad_cam"). */
    std::string method;

    /** @brief The raw attribution values, same shape as the explained input (or a
     *         method-specific shape, e.g. Grad-CAM's per-channel map). */
    Tensor values;

    /** @brief Method-specific metadata (e.g. {"baseline", "zero"} for Integrated Gradients). */
    std::unordered_map<std::string, std::string> metadata;
};

}  // namespace exai
