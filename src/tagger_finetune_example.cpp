#include "pulsatrix/tagger_finetune_example.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/grad_clipping.hpp"
#include "pulsatrix/lr_scheduler.hpp"
#include "pulsatrix/param_groups.hpp"
#include "pulsatrix/safetensors.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"

namespace pulsatrix {
namespace {

// A portable generator (std:: distributions differ between standard libraries).
struct Rng {
    uint64_t state;
    uint64_t next() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    float uniform() { return static_cast<float>(next() >> 40) / static_cast<float>(1ULL << 24); }  // [0, 1)
};

Tensor reshaped(const Tensor& t, Shape shape) {
    Tensor out = t;
    out.reshape(std::move(shape));
    return out;
}

int64_t tag_of(TaggingRule rule, int64_t token, int64_t previous) {
    const int64_t c = TinyTagger::kClasses;
    return rule == TaggingRule::SumWithPrevious ? (token + previous) % c : ((token - previous) % c + c) % c;
}

}  // namespace

TinyTagger::TinyTagger(DeviceBackend* backend)
    : embed_(kVocab, kDim, backend), block_(kDim, kHeads, kFeedForward, backend), head_(kDim, kClasses, backend) {}

Tensor TinyTagger::forward_impl(const Tensor& input) {
    if (input.rank() != 2) {
        throw std::invalid_argument("TinyTagger::forward: input must be (N, L) token ids");
    }
    last_n_ = input.shape().dim(0);
    const int64_t l = input.shape().dim(1);
    Tensor h = block_.forward(embed_.forward(input));
    return reshaped(head_.forward(reshaped(h, Shape({last_n_ * l, kDim}))), Shape({last_n_, l, kClasses}));
}

Tensor TinyTagger::backward(const Tensor& grad_output) {
    const int64_t l = grad_output.numel() / (last_n_ * kClasses);
    Tensor g = head_.backward(reshaped(grad_output, Shape({last_n_ * l, kClasses})));
    return embed_.backward(block_.backward(reshaped(g, Shape({last_n_, l, kDim}))));
}

Tensor TinyTagger::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    const int64_t l = relevance_out.numel() / (last_n_ * kClasses);
    Tensor r = head_.propagate_relevance(reshaped(relevance_out, Shape({last_n_ * l, kClasses})), config);
    return embed_.propagate_relevance(block_.propagate_relevance(reshaped(r, Shape({last_n_, l, kDim})), config),
                                      config);
}

std::vector<NamedParamRef> TinyTagger::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "embed", embed_);
    append_named_parameters(out, "block", block_);
    append_named_parameters(out, "head", head_);
    return out;
}

std::vector<NamedBufferRef> TinyTagger::named_buffers() {
    std::vector<NamedBufferRef> out;
    append_named_buffers(out, "embed", embed_);
    append_named_buffers(out, "block", block_);
    append_named_buffers(out, "head", head_);
    return out;
}

void TinyTagger::set_training(bool training) {
    Module::set_training(training);
    embed_.set_training(training);
    block_.set_training(training);
    head_.set_training(training);
}

void InitTagger(TinyTagger& tagger, uint64_t seed) {
    Rng rng{seed};
    for (const NamedParamRef& p : tagger.named_parameters()) {
        Tensor& v = *p.ref.value;
        std::vector<float> values(static_cast<size_t>(v.numel()));
        if (v.rank() >= 2) {
            const float a = 1.0f / std::sqrt(static_cast<float>(v.shape().dim(0)));
            for (float& x : values) x = (2.0f * rng.uniform() - 1.0f) * a;
        } else {
            const float fill = p.name.find("norm") != std::string::npos ? 1.0f : 0.0f;
            for (float& x : values) x = fill;
        }
        v = Tensor(v.shape(), v.backend(), values, v.device());
    }
}

TaggingBatch MakeTaggingBatch(TaggingRule rule, int64_t batch_size, uint64_t seed, DeviceBackend* backend) {
    static CPUBackend default_backend;
    DeviceBackend* b = backend != nullptr ? backend : &default_backend;
    Rng rng{seed};
    const int64_t l = TinyTagger::kSeqLen;
    std::vector<float> inputs(static_cast<size_t>(batch_size * l), 0.0f);
    std::vector<float> targets(static_cast<size_t>(batch_size * l), static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex));
    for (int64_t n = 0; n < batch_size; ++n) {
        const int64_t length = 5 + static_cast<int64_t>(rng.next() % static_cast<uint64_t>(l - 4));  // 5..L
        int64_t previous = 0;
        for (int64_t t = 0; t < length; ++t) {
            const auto token = 1 + static_cast<int64_t>(rng.next() % static_cast<uint64_t>(TinyTagger::kVocab - 1));
            inputs[static_cast<size_t>(n * l + t)] = static_cast<float>(token);
            targets[static_cast<size_t>(n * l + t)] = static_cast<float>(tag_of(rule, token, previous));
            previous = token;
        }
    }
    return {Tensor(Shape({batch_size, l}), b, inputs), Tensor(Shape({batch_size, l}), b, targets)};
}

std::vector<float> TrainTagger(TinyTagger& tagger, TaggingRule rule, const FineTuneConfig& config,
                               DeviceBackend* backend) {
    AdamWOptimizer optimizer(config.learning_rate, backend, config.weight_decay);
    optimizer.set_param_groups({{"no decay", param_select::one_dimensional(), config.learning_rate, 0.0f}});
    int64_t start = 0;
    if (!config.resume_from.empty()) {
        LoadCheckpoint(config.resume_from, tagger, optimizer);
        start = std::stoll(SafetensorsFile::Read(config.resume_from).metadata().at("step"));
    }
    LRScheduler scheduler(optimizer, LRSchedule::Cosine(config.warmup, config.steps));
    scheduler.set_last_step(start);

    std::vector<float> losses;
    for (int64_t step = start; step < config.steps; ++step) {
        if (config.stop_after >= 0 && step == start + config.stop_after) {
            SaveCheckpoint(config.checkpoint_path, tagger, optimizer, {{"step", std::to_string(step)}});
            return losses;
        }
        std::vector<TaggingBatch> window;
        int64_t tokens = 0;
        for (int64_t k = 0; k < config.micro_batches; ++k) {
            window.push_back(MakeTaggingBatch(rule, config.micro_batch_size,
                                              config.data_seed + static_cast<uint64_t>(step * config.micro_batches + k),
                                              backend));
            tokens += CountTargetTokens(window.back().targets);
        }
        optimizer.zero_grad(tagger);
        float loss_value = 0.0f;
        for (const TaggingBatch& batch : window) {
            TokenCrossEntropyLoss loss(backend);
            loss_value += loss.forward(tagger.forward(batch.inputs), batch.targets, static_cast<float>(tokens));
            (void)tagger.backward(loss.backward());
        }
        (void)ClipGradNorm(tagger, config.max_grad_norm);
        optimizer.step(tagger);
        scheduler.step();
        losses.push_back(loss_value);
    }
    return losses;
}

float TaggingAccuracy(TinyTagger& tagger, TaggingRule rule, int64_t num_sequences) {
    TaggingBatch batch = MakeTaggingBatch(rule, num_sequences, 987654321ULL, tagger.named_parameters()[0].ref.value->backend());
    const std::vector<float> logits = tagger.forward(batch.inputs).to_host_vector();
    const std::vector<float> targets = batch.targets.to_host_vector();
    int64_t correct = 0, total = 0;
    for (size_t r = 0; r < targets.size(); ++r) {
        if (targets[r] < 0.0f) {
            continue;
        }
        int64_t best = 0;
        for (int64_t c = 1; c < TinyTagger::kClasses; ++c) {
            if (logits[r * TinyTagger::kClasses + static_cast<size_t>(c)] >
                logits[r * TinyTagger::kClasses + static_cast<size_t>(best)]) {
                best = c;
            }
        }
        correct += best == static_cast<int64_t>(targets[r]) ? 1 : 0;
        ++total;
    }
    return total == 0 ? 0.0f : static_cast<float>(correct) / static_cast<float>(total);
}

}  // namespace pulsatrix
