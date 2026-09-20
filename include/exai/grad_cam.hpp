/** @file grad_cam.hpp
 *  @brief Grad-CAM -- gradient-weighted class activation mapping over the last conv layer
 *         (charter Part 1, Phase 2; theory: xai_context.aDNA's vision_gradcam.md).
 */
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "exai/assert.hpp"
#include "exai/attribution.hpp"
#include "exai/device_backend.hpp"
#include "exai/explainer_context.hpp"
#include "exai/tensor.hpp"

namespace exai {

/**
 * @brief L^c = ReLU(sum_k alpha^c_k * A^k), where alpha^c_k = mean_ij(d(y^c)/d(A^k_ij))
 *        and A is the last OpType::Conv node's activation.
 * @note Finds the target layer via ComputationGraph::nodes_by_op_type(OpType::Conv)'s
 *       last entry -- the theory file's stated default ("the last convolutional layer
 *       before the classifier head"), found by op-type tag per charter Part 2 SS3, never
 *       by layer name.
 * @note Reads the target node's own activation/gradient as ComputationGraph tagged it --
 *       in this codebase Conv and its following ReLU are separate graph nodes, so "the
 *       last conv layer" here means the pre-ReLU conv output specifically. This is a
 *       legitimate placement (Grad-CAM's formula doesn't require a post-ReLU activation),
 *       just worth noting for a reader used to "conv block" framing elsewhere.
 * @note A pure graph walker built entirely against ExplainerContext's public interface --
 *       no core (Tensor/ComputationGraph/Autograd/Module) changes needed.
 */
class GradCAM {
public:
    /**
     * @brief Computes the Grad-CAM map for one target class index.
     * @param ctx Context to run the forward/backward pass through. Its graph must contain
     *        at least one OpType::Conv node.
     * @param input Input to explain.
     * @param target_index Which output element (class score) to attribute (0-based, flat
     *        index into the network's final output).
     * @param backend Backend to allocate the one-hot seed and result tensors through.
     * @return An Attribution with method "grad_cam", values = the N x H x W CAM batch
     *         (same spatial size as the target conv layer's feature maps, one map per
     *         batch example), and metadata recording the target index and which node was
     *         used.
     * @note Assumes a rank-2 (N, num_classes) network output and a rank-4 (N, channels, H,
     *       W) target conv layer activation -- migrated by
     *       campaign_exai_dl_library_batch_dimension_support from the original rank-1/
     *       rank-3 assumptions. Same single-shared-target_index scope boundary as
     *       Saliency's identical migration.
     */
    [[nodiscard]] Attribution explain(ExplainerContext& ctx, const Tensor& input, int64_t target_index,
                                       DeviceBackend* backend) const {
        Tensor output = ctx.forward_pass(input);
        // External boundary (Mission 2, finding 15 systemic sweep; rank check added by
        // campaign_exai_dl_library_batch_dimension_support) -- escalated from
        // EXAI_ASSERT-only.
        if (output.rank() != 2) {
            throw std::invalid_argument("GradCAM::explain: network output must be rank-2 (N, num_classes)");
        }
        if (target_index < 0 || target_index >= output.shape().dim(1)) {
            throw std::invalid_argument("GradCAM::explain: target_index out of range");
        }
        int64_t N = output.shape().dim(0);

        Tensor seed(output.shape(), backend);
        seed.fill(0.0f);
        for (int64_t n = 0; n < N; ++n) {
            seed.at({n, target_index}) = 1.0f;
        }
        (void)ctx.backward_pass(seed);

        std::vector<NodeId> conv_nodes = ctx.graph().nodes_by_op_type(OpType::Conv);
        // External boundary -- whether this ExplainerContext was built with a Conv layer
        // is a caller-configuration fact, not an internal invariant this library controls.
        if (conv_nodes.empty()) {
            throw std::invalid_argument("GradCAM::explain: graph has no Conv layer");
        }
        NodeId target_node = conv_nodes.back();

        const Tensor& activation = ctx.activation(target_node);
        const Tensor& grad = ctx.gradient(target_node);

        int64_t channels = activation.shape().dim(1);
        int64_t height = activation.shape().dim(2);
        int64_t width = activation.shape().dim(3);

        Tensor cam(Shape({N, height, width}), backend);
        for (int64_t n = 0; n < N; ++n) {
            std::vector<float> alpha(static_cast<size_t>(channels), 0.0f);
            for (int64_t c = 0; c < channels; ++c) {
                float sum = 0.0f;
                for (int64_t h = 0; h < height; ++h) {
                    for (int64_t w = 0; w < width; ++w) {
                        sum += grad.at({n, c, h, w});
                    }
                }
                alpha[static_cast<size_t>(c)] = sum / static_cast<float>(height * width);
            }

            for (int64_t h = 0; h < height; ++h) {
                for (int64_t w = 0; w < width; ++w) {
                    float value = 0.0f;
                    for (int64_t c = 0; c < channels; ++c) {
                        value += alpha[static_cast<size_t>(c)] * activation.at({n, c, h, w});
                    }
                    cam.at({n, h, w}) = value > 0.0f ? value : 0.0f;
                }
            }
        }

        return Attribution{"grad_cam", std::move(cam),
                            {{"target_index", std::to_string(target_index)},
                             {"layer_node_id", std::to_string(target_node)}}};
    }
};

}  // namespace exai
