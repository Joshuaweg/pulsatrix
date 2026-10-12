/** @file concept_relevance.hpp
 *  @brief Concept Relevance Propagation (INT-6): which of a layer's channels an LRP explanation
 *         passes through, where each one looks in the input, and which inputs show it at work.
 *  @ingroup mech_interp
 *
 * An LRP heatmap says where in the input the evidence for a class is, but not what it is. CRP
 * (Achtibat et al., Nature Machine Intelligence 2023) takes the relevance at a hidden layer,
 * where each channel (a convolution's filter, a dense layer's unit) stands for a learned concept,
 * and asks three questions:
 *
 * - **How much does each concept matter for this prediction?** LayerRelevance() and
 *   ChannelRelevance(): the relevance at the layer, summed per channel.
 * - **Where in the input is each concept?** ConditionalHeatmaps(): LRP with the relevance at the
 *   layer zeroed outside the chosen channels, so only that concept's share reaches the input.
 *   Conditions can mask several layers at once, a concept and its sub-concepts below it.
 * - **What is the concept?** ConceptReferenceSamples(): the inputs in which the channel is most
 *   relevant for a class (relevance maximization), or most active (activation maximization).
 *   Relevance shows the concept as the model uses it for the class; activation shows everything
 *   that excites the channel, used or not.
 *
 * LRP's rules are linear in the relevance they pass down, so the conditional heatmaps of every
 * channel of a layer add up to the plain LRP heatmap. Everything uses the same LRP object, so the
 * rules are the ones LRP::explain() would apply.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief Keep the relevance of these channels at this layer, and zero the rest. */
struct ConceptMask {
    /** @brief The node: module i's output is node i + 1. Between 1 and the number of modules. */
    NodeId layer = 1;
    /** @brief Indices along the layer's second dimension (channels of an (N, C, H, W) layer, units
     *         of an (N, D) one). */
    std::vector<int64_t> channels;
};

/** @brief One conditional heatmap's masks, at most one per layer. */
using ConceptCondition = std::vector<ConceptMask>;

/**
 * @brief The relevance at @p layer for explaining @p target: the layer's shape, (N, C, ...).
 * @throws std::invalid_argument if layer isn't between 1 and the number of modules, or for what
 *         LRP::explain() rejects.
 */
[[nodiscard]] Tensor LayerRelevance(const LRP& lrp, ExplainerContext& ctx, const Tensor& input, const LRPTarget& target, NodeId layer,
                                    DeviceBackend* backend);

/**
 * @brief Each row's relevance per channel, summed over the remaining dimensions: (N, C) values,
 *        row-major. With a conserving rule they add up to the explained output.
 * @throws std::invalid_argument if @p layer_relevance has rank below 2.
 */
[[nodiscard]] std::vector<double> ChannelRelevance(const Tensor& layer_relevance);

/**
 * @brief One input heatmap per condition: LRP from @p target with the relevance zeroed, at each
 *        masked layer, outside the mask's channels.
 * @return Attribution{"crp", the input's shape, metadata: `condition` (`layer:c,c;...`),
 *         `relevance_in_sum`}.
 * @throws std::invalid_argument if a condition is empty, masks a layer twice, names a layer
 *         outside [1, number of modules] or a channel outside the layer, or for what
 *         LRP::explain() rejects.
 */
[[nodiscard]] std::vector<Attribution> ConditionalHeatmaps(const LRP& lrp, ExplainerContext& ctx, const Tensor& input, const LRPTarget& target,
                                                           const std::vector<ConceptCondition>& conditions, DeviceBackend* backend);

/** @brief How ConceptReferenceSamples() ranks inputs. */
enum class ReferenceMode {
    /** @brief By the channel's relevance for the target class, summed over positions. */
    Relevance,
    /** @brief By the channel's largest activation over positions. */
    Activation,
};

/**
 * @brief The @p k inputs of @p candidates that show a channel best, best first: the reference
 *        samples by which CRP shows what a concept is.
 * @param candidates A batch of inputs, the model's input shape with a leading batch dimension.
 * @param target The class the relevance explains; unused for ReferenceMode::Activation.
 * @note Ties keep the lower index first.
 * @throws std::invalid_argument if k is below 1, the channel is outside the layer, or for what
 *         LayerRelevance() rejects.
 */
[[nodiscard]] std::vector<int64_t> ConceptReferenceSamples(const LRP& lrp, ExplainerContext& ctx, const Tensor& candidates, int64_t target,
                                                           NodeId layer, int64_t channel, int64_t k, ReferenceMode mode,
                                                           DeviceBackend* backend);

}  // namespace pulsatrix
