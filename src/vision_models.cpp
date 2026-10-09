#include "pulsatrix/vision_models.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {

namespace {

void Prefix(std::vector<NamedParamRef>& out, const std::string& prefix, Module& m) {
    for (NamedParamRef& p : m.named_parameters()) out.push_back({prefix + "." + p.name, p.ref});
}

void PrefixBuffers(std::vector<NamedBufferRef>& out, const std::string& prefix, Module& m) {
    for (NamedBufferRef& b : m.named_buffers()) out.push_back({prefix + "." + b.name, b.value});
}

std::unique_ptr<Conv2DModule> Conv(int64_t in, int64_t out, int64_t k, int64_t stride, int64_t pad, DeviceBackend* backend) {
    auto c = std::make_unique<Conv2DModule>(in, out, k, k, backend, stride, pad);
    c->set_bias(std::vector<float>(static_cast<size_t>(out), 0.0f));  // torchvision's ResNet convolutions have none
    return c;
}

std::unique_ptr<BatchNormModule> Bn(int64_t channels, DeviceBackend* backend) {
    return std::make_unique<BatchNormModule>(channels, backend, backend->device(), /*eps=*/1e-5f);  // torchvision's eps
}

}  // namespace

// ---- ResNet ------------------------------------------------------------------------------------

TorchvisionResNet::TorchvisionResNet(const ResNetConfig& config, DeviceBackend* backend) : config_(config), backend_(backend) {
    if (config.blocks.empty() || config.width < 1 || config.num_classes < 1 || config.in_channels < 1 ||
        std::any_of(config.blocks.begin(), config.blocks.end(), [](int64_t b) { return b < 1; })) {
        throw std::invalid_argument("TorchvisionResNet: needs one or more stages of one or more blocks, and positive sizes");
    }
    const int64_t w = config.width;
    conv1_ = Conv(config.in_channels, w, 7, 2, 3, backend);
    bn1_ = Bn(w, backend);
    relu_ = std::make_unique<ReluModule>(backend);
    maxpool_ = std::make_unique<MaxPool2DModule>(3, 3, 2, 2, 1, 1, backend);
    std::vector<Module*> seq = {conv1_.get(), bn1_.get(), relu_.get(), maxpool_.get()};
    int64_t in = w;
    for (size_t stage = 0; stage < config.blocks.size(); ++stage) {
        const int64_t out = w << stage;
        for (int64_t i = 0; i < config.blocks[stage]; ++i) {
            const int64_t stride = stage > 0 && i == 0 ? 2 : 1;
            auto b = std::make_unique<Block>();
            b->name = "layer" + std::to_string(stage + 1) + "." + std::to_string(i);
            b->conv1 = Conv(in, out, 3, stride, 1, backend);
            b->bn1 = Bn(out, backend);
            b->relu1 = std::make_unique<ReluModule>(backend);
            b->conv2 = Conv(out, out, 3, 1, 1, backend);
            b->bn2 = Bn(out, backend);
            b->main = std::make_unique<SequentialModule>(
                std::vector<Module*>{b->conv1.get(), b->bn1.get(), b->relu1.get(), b->conv2.get(), b->bn2.get()});
            if (stride != 1 || in != out) {
                b->down_conv = Conv(in, out, 1, stride, 0, backend);
                b->down_bn = Bn(out, backend);
                b->shortcut = std::make_unique<SequentialModule>(std::vector<Module*>{b->down_conv.get(), b->down_bn.get()});
                b->residual = std::make_unique<ResidualModule>(b->main.get(), b->shortcut.get(), backend);
            } else {
                b->residual = std::make_unique<ResidualModule>(b->main.get(), backend);
            }
            b->relu_out = std::make_unique<ReluModule>(backend);
            b->block = std::make_unique<SequentialModule>(std::vector<Module*>{b->residual.get(), b->relu_out.get()});
            seq.push_back(b->block.get());
            blocks_.push_back(std::move(b));
            in = out;
        }
    }
    avgpool_ = std::make_unique<AdaptiveAvgPool2DModule>(1, 1, backend);
    flatten_ = std::make_unique<FlattenModule>(backend);
    fc_ = std::make_unique<LinearModule>(in, config.num_classes, backend);
    seq.push_back(avgpool_.get());
    seq.push_back(flatten_.get());
    seq.push_back(fc_.get());
    net_ = std::make_unique<SequentialModule>(seq);
}

std::vector<Module*> TorchvisionResNet::layers() { return net_->layers(); }

std::vector<std::unique_ptr<BatchNormFold>> TorchvisionResNet::fold_batch_norms() {
    if (is_training()) throw std::invalid_argument("TorchvisionResNet::fold_batch_norms: put the model in eval mode first");
    std::vector<std::unique_ptr<BatchNormFold>> folds;
    folds.push_back(std::make_unique<BatchNormFold>(*conv1_, *bn1_));
    for (auto& b : blocks_) {
        folds.push_back(std::make_unique<BatchNormFold>(*b->conv1, *b->bn1));
        folds.push_back(std::make_unique<BatchNormFold>(*b->conv2, *b->bn2));
        if (b->down_conv) folds.push_back(std::make_unique<BatchNormFold>(*b->down_conv, *b->down_bn));
    }
    return folds;
}

Tensor TorchvisionResNet::forward_impl(const Tensor& input) { return net_->forward(input); }
Tensor TorchvisionResNet::backward(const Tensor& grad_output) { return net_->backward(grad_output); }
Tensor TorchvisionResNet::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    return net_->propagate_relevance(relevance_out, config);
}

std::vector<NamedParamRef> TorchvisionResNet::named_parameters() {
    std::vector<NamedParamRef> out;
    Prefix(out, "conv1", *conv1_);
    Prefix(out, "bn1", *bn1_);
    for (auto& b : blocks_) {
        Prefix(out, b->name + ".conv1", *b->conv1);
        Prefix(out, b->name + ".bn1", *b->bn1);
        Prefix(out, b->name + ".conv2", *b->conv2);
        Prefix(out, b->name + ".bn2", *b->bn2);
        if (b->down_conv) {
            Prefix(out, b->name + ".downsample.0", *b->down_conv);
            Prefix(out, b->name + ".downsample.1", *b->down_bn);
        }
    }
    Prefix(out, "fc", *fc_);
    return out;
}

std::vector<NamedBufferRef> TorchvisionResNet::named_buffers() {
    std::vector<NamedBufferRef> out;
    PrefixBuffers(out, "bn1", *bn1_);
    for (auto& b : blocks_) {
        PrefixBuffers(out, b->name + ".bn1", *b->bn1);
        PrefixBuffers(out, b->name + ".bn2", *b->bn2);
        if (b->down_bn) PrefixBuffers(out, b->name + ".downsample.1", *b->down_bn);
    }
    return out;
}

void TorchvisionResNet::set_training(bool training) {
    Module::set_training(training);
    net_->set_training(training);
}

// ---- VGG ---------------------------------------------------------------------------------------

TorchvisionVGG::TorchvisionVGG(const VGGConfig& config, DeviceBackend* backend) : config_(config), backend_(backend) {
    if (std::none_of(config.features.begin(), config.features.end(), [](int64_t c) { return c > 0; }) || config.pool_size < 1 ||
        config.hidden < 1 || config.num_classes < 1 || config.in_channels < 1 ||
        std::any_of(config.features.begin(), config.features.end(), [](int64_t c) { return c < 0; })) {
        throw std::invalid_argument("TorchvisionVGG: needs one or more convolutions, and positive sizes");
    }
    auto own = [&](std::unique_ptr<Module> m) {
        layers_.push_back(m.get());
        owned_.push_back(std::move(m));
        return layers_.back();
    };
    int64_t in = config.in_channels, index = 0;
    for (int64_t c : config.features) {
        if (c == 0) {
            own(std::make_unique<MaxPool2DModule>(2, 2, backend));
            ++index;
        } else {
            named_.emplace_back("features." + std::to_string(index), own(std::make_unique<Conv2DModule>(in, c, 3, 3, backend, 1, 1)));
            own(std::make_unique<ReluModule>(backend));
            index += 2;
            in = c;
        }
    }
    own(std::make_unique<AdaptiveAvgPool2DModule>(config.pool_size, config.pool_size, backend));
    own(std::make_unique<FlattenModule>(backend));
    const int64_t flat = in * config.pool_size * config.pool_size;
    named_.emplace_back("classifier.0", own(std::make_unique<LinearModule>(flat, config.hidden, backend)));
    own(std::make_unique<ReluModule>(backend));
    own(std::make_unique<DropoutModule>(0.5f, backend));
    named_.emplace_back("classifier.3", own(std::make_unique<LinearModule>(config.hidden, config.hidden, backend)));
    own(std::make_unique<ReluModule>(backend));
    own(std::make_unique<DropoutModule>(0.5f, backend));
    named_.emplace_back("classifier.6", own(std::make_unique<LinearModule>(config.hidden, config.num_classes, backend)));
    net_ = std::make_unique<SequentialModule>(layers_);
}

std::vector<Module*> TorchvisionVGG::layers() { return layers_; }
Tensor TorchvisionVGG::forward_impl(const Tensor& input) { return net_->forward(input); }
Tensor TorchvisionVGG::backward(const Tensor& grad_output) { return net_->backward(grad_output); }
Tensor TorchvisionVGG::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    return net_->propagate_relevance(relevance_out, config);
}

std::vector<NamedParamRef> TorchvisionVGG::named_parameters() {
    std::vector<NamedParamRef> out;
    for (auto& [prefix, m] : named_) Prefix(out, prefix, *m);
    return out;
}

void TorchvisionVGG::set_training(bool training) {
    Module::set_training(training);
    net_->set_training(training);
}

// ---- Loading -----------------------------------------------------------------------------------

TorchvisionLoadReport LoadTorchvisionWeights(Module& model, const std::string& path) {
    const SafetensorsFile file = SafetensorsFile::Map(path);
    CPUBackend cpu;
    TorchvisionLoadReport report;
    std::map<std::string, bool> used;
    for (const std::string& name : file.names()) {
        if (name.size() >= 20 && name.compare(name.size() - 20, 20, ".num_batches_tracked") == 0) continue;
        used[name] = false;
    }
    auto load = [&](const std::string& name, Tensor* target, bool is_param) {
        auto it = used.find(name);
        if (it == used.end()) {
            const bool conv_bias = is_param && name.size() > 5 && name.compare(name.size() - 5, 5, ".bias") == 0 && target->rank() == 1 &&
                                   file.contains(name.substr(0, name.size() - 5) + ".weight") &&
                                   file.info(name.substr(0, name.size() - 5) + ".weight").shape.size() == 4;
            if (!conv_bias) throw std::invalid_argument("LoadTorchvisionWeights: " + path + " has no " + name);
            report.left_at_default.push_back(name);
            return;
        }
        const std::vector<int64_t> shape = file.info(name).shape;
        std::vector<float> v = file.tensor(name, &cpu).to_host_vector();
        std::vector<int64_t> want;
        for (int64_t i = 0; i < target->rank(); ++i) want.push_back(target->shape().dim(static_cast<int>(i)));
        if (want.size() == 2) {
            // A Linear weight, square ones included: PyTorch stores (out, in), pulsatrix (in, out).
            if (shape.size() != 2 || shape[0] != want[1] || shape[1] != want[0]) {
                throw std::invalid_argument("LoadTorchvisionWeights: " + name + " doesn't fit the model's shape");
            }
            std::vector<float> t(v.size());
            for (int64_t r = 0; r < shape[0]; ++r) {
                for (int64_t c = 0; c < shape[1]; ++c) t[static_cast<size_t>(c * shape[0] + r)] = v[static_cast<size_t>(r * shape[1] + c)];
            }
            v = std::move(t);
        } else if (shape != want) {
            throw std::invalid_argument("LoadTorchvisionWeights: " + name + " doesn't fit the model's shape");
        }
        *target = Tensor(target->shape(), target->backend(), v, target->device());
        it->second = true;
        ++report.loaded;
    };
    for (const NamedParamRef& p : model.named_parameters()) load(p.name, p.ref.value, true);
    for (const NamedBufferRef& b : model.named_buffers()) load(b.name, b.value, false);
    for (const auto& [name, done] : used) {
        if (!done) throw std::invalid_argument("LoadTorchvisionWeights: the model has nothing named " + name);
    }
    return report;
}

}  // namespace pulsatrix
