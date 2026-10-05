/** @file attribution.hpp
 *  @brief First-class explanation result type -- values, method, and metadata together.
 *  @ingroup interpretability_dl
 */
#pragma once

#include <string>
#include <unordered_map>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief An explanation result: the raw attribution values, the method that produced
 *        them, and any relevant metadata (charter Part 2 SS4).
 * @note Deliberately not just a Tensor -- downstream code (audit tooling, the viz layer and
 *       its JSON documents) shouldn't have to guess what produced a number. metadata is a simple
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

/**
 * @brief Shared helpers for the post-hoc explainers' host boundary.
 * @note The explainers run the network's forward/backward on whatever device it lives on, but
 *       their own bookkeeping (seeds, interpolation, perturbation, CAM weighting) is small
 *       host arithmetic: tensors are read to the host once (Tensor::to_host_vector /
 *       read_element) and every tensor handed back is uploaded with the values constructor.
 */
namespace explainer_detail {

/**
 * @brief The backend to allocate a tensor through that must live beside `like`.
 * @return `backend` when it serves like's device (the caller's choice, as before), else like's
 *         own owning backend -- so a host-only backend argument never receives device data.
 */
inline DeviceBackend* backend_beside(const Tensor& like, DeviceBackend* backend) {
    return (backend != nullptr && backend->device() == like.device()) ? backend : like.backend();
}

}  // namespace explainer_detail

}  // namespace pulsatrix
