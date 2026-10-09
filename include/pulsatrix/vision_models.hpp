/** @file vision_models.hpp
 *  @brief torchvision's ResNet and VGG, built from pulsatrix layers with torchvision's parameter
 *         names, so their published ImageNet weights load once converted to safetensors (KS-9).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pulsatrix/adaptive_avg_pool2d_module.hpp"
#include "pulsatrix/batch_norm_fold.hpp"
#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/max_pool2d_module.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/residual_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {

/** @brief A torchvision ResNet with basic blocks (ResNet18 and ResNet34). */
struct ResNetConfig {
    /** @brief Blocks per stage: {2, 2, 2, 2} is ResNet18, {3, 4, 6, 3} ResNet34. */
    std::vector<int64_t> blocks = {2, 2, 2, 2};
    /** @brief The first stage's channels, doubled at each later stage (torchvision: 64). */
    int64_t width = 64;
    int64_t num_classes = 1000;
    int64_t in_channels = 3;
};

/**
 * @brief torchvision's `ResNet(BasicBlock, ...)`: a 7x7 stride-2 convolution, BatchNorm, ReLU
 *        and 3x3 stride-2 max pooling; four stages of basic blocks (two 3x3 convolutions with
 *        BatchNorm, and a 1x1 convolution with BatchNorm on the shortcut where the shape changes);
 *        global average pooling and a linear classifier. Input (N, 3, H, W), output (N, classes).
 *
 * - **Parameter names are torchvision's** (`conv1.weight`, `layer2.0.downsample.0.weight`,
 *   `bn1.running_mean`, `fc.weight`, ...), so LoadTorchvisionWeights() loads a converted
 *   state dict. torchvision's convolutions have no bias; theirs stay zero.
 * - **Explaining it:** put it in eval mode, fold its BatchNorms into their convolutions for as
 *   long as you explain (fold_batch_norms(), Zennit's canonizer), and explain through
 *   `ExplainerContext ctx(model.layers())` so a composite sees the stem, each block and the
 *   classifier as separate modules. Each block is a ResidualModule: relevance is split between
 *   its two branches in proportion to their contributions, as Zennit's ResNet canonizer does.
 */
class TorchvisionResNet : public Module {
public:
    /** @throws std::invalid_argument for an empty block list or a size < 1. */
    TorchvisionResNet(const ResNetConfig& config, DeviceBackend* backend);

    /** @brief ResNet18's config (11.7M parameters). */
    [[nodiscard]] static ResNetConfig ResNet18() { return {}; }

    /** @brief The top-level layers, in order: conv1, bn1, relu, maxpool, one per block, avgpool,
     *         flatten, fc. For ExplainerContext. */
    [[nodiscard]] std::vector<Module*> layers();
    /** @brief Folds every BatchNorm into the convolution before it, until the returned objects
     *         are destroyed. @throws std::invalid_argument unless in eval mode. */
    [[nodiscard]] std::vector<std::unique_ptr<BatchNormFold>> fold_batch_norms();

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override;
    [[nodiscard]] const ResNetConfig& config() const { return config_; }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    struct Block {
        std::string name;  ///< torchvision's, e.g. "layer2.0"
        std::unique_ptr<Conv2DModule> conv1, conv2, down_conv;
        std::unique_ptr<BatchNormModule> bn1, bn2, down_bn;
        std::unique_ptr<ReluModule> relu1, relu_out;
        std::unique_ptr<SequentialModule> main, shortcut;
        std::unique_ptr<ResidualModule> residual;
        std::unique_ptr<SequentialModule> block;  ///< residual, then ReLU
    };
    ResNetConfig config_;
    DeviceBackend* backend_;
    std::unique_ptr<Conv2DModule> conv1_;
    std::unique_ptr<BatchNormModule> bn1_;
    std::unique_ptr<ReluModule> relu_;
    std::unique_ptr<MaxPool2DModule> maxpool_;
    std::vector<std::unique_ptr<Block>> blocks_;
    std::unique_ptr<AdaptiveAvgPool2DModule> avgpool_;
    std::unique_ptr<FlattenModule> flatten_;
    std::unique_ptr<LinearModule> fc_;
    std::unique_ptr<SequentialModule> net_;
};

/** @brief A torchvision VGG without BatchNorm (VGG11 to VGG19). */
struct VGGConfig {
    /** @brief The convolution widths in order, 0 for a 2x2 max pool: torchvision's config "D"
     *         for VGG16. */
    std::vector<int64_t> features = {64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512, 512, 512, 0};
    /** @brief The adaptive average pool's output size (7 in torchvision). */
    int64_t pool_size = 7;
    /** @brief The two hidden classifier widths (4096 in torchvision). */
    int64_t hidden = 4096;
    int64_t num_classes = 1000;
    int64_t in_channels = 3;
};

/**
 * @brief torchvision's `VGG`: 3x3 convolutions with bias and ReLU, 2x2 max pooling, adaptive
 *        average pooling to (pool_size, pool_size), then Linear, ReLU, Dropout, Linear, ReLU,
 *        Dropout, Linear. Parameter names are torchvision's (`features.0.weight`,
 *        `classifier.6.bias`, ...). Every layer is top-level in layers(), so the first
 *        convolution can take ZBox in an EpsilonGammaBox composite.
 */
class TorchvisionVGG : public Module {
public:
    /** @throws std::invalid_argument for no convolutions or a size < 1. */
    TorchvisionVGG(const VGGConfig& config, DeviceBackend* backend);

    /** @brief VGG16's config (138M parameters). */
    [[nodiscard]] static VGGConfig VGG16() { return {}; }

    [[nodiscard]] std::vector<Module*> layers();

    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override;
    [[nodiscard]] const VGGConfig& config() const { return config_; }

protected:
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    VGGConfig config_;
    DeviceBackend* backend_;
    std::vector<std::unique_ptr<Module>> owned_;
    std::vector<Module*> layers_;
    std::vector<std::pair<std::string, Module*>> named_;  ///< torchvision's prefix of each layer with parameters
    std::unique_ptr<SequentialModule> net_;
};

/** @brief What LoadTorchvisionWeights() did. */
struct TorchvisionLoadReport {
    int64_t loaded = 0;
    /** @brief Model parameters the file doesn't have, left as they were (torchvision's absent
     *         convolution biases). */
    std::vector<std::string> left_at_default;
};

/**
 * @brief Loads a torchvision state dict converted to safetensors
 *        (tools/convert/pickle_to_safetensors.py) into @p model by name: every parameter and
 *        buffer of the model is looked up under the same name. Linear weights (every
 *        rank-2 parameter), stored (out, in) by PyTorch, are transposed; `num_batches_tracked` counters are ignored.
 * @throws std::invalid_argument for a tensor the model doesn't have, a shape that doesn't fit, or
 *         a missing parameter other than a convolution bias.
 */
TorchvisionLoadReport LoadTorchvisionWeights(Module& model, const std::string& safetensors_path);

}  // namespace pulsatrix
