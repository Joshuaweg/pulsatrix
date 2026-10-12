/** @file lora.hpp
 *  @brief LoRA (Hu et al., 2021): fine-tune a frozen model through a low-rank update on each of
 *         its Linear layers, merge it into the weights or take it out again, and save and load it
 *         as a PEFT adapter.
 *  @ingroup dl_modules
 *
 * A LoRA layer computes y = x W + b + (alpha / r) x A B, with the base weight W frozen and only A
 * `(in, r)` and B `(r, out)` trained. B starts at zero, so the adapted model starts out as the base
 * model. The adapter lives inside a LinearModule (LinearModule::attach_lora()), because the
 * projections of every model in pulsatrix are LinearModule members: ApplyLora() finds them by name
 * and attaches one to each, and the model trains, generates and explains as before.
 *
 * Use it as "LoRA Without Regret" (Schulman and Thinking Machines Lab, 2025) recommends: on every
 * Linear layer, MLP included (ApplyLora()'s default), with a learning rate about 10 times the one
 * full fine-tuning would use.
 */
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

class LinearModule;

/** @brief LoRA's settings. The defaults are PEFT's `LoraConfig`'s. */
struct LoRAConfig {
    /** @brief The rank r of the update. */
    int64_t rank = 8;
    /** @brief The update is scaled by alpha / rank. */
    float alpha = 8.0f;
    /** @brief Dropout on the adapter's input, in training mode only. */
    float dropout = 0.0f;
    /** @brief Seeds A's initialization (and the dropout masks), the same on every platform. */
    uint64_t seed = 0;
    /**
     * @brief The layers ApplyLora() adapts, PEFT style: a layer matches a name if its own name is
     *        that name or ends with "." and that name (`q_proj` matches `layers.0.mha.q_proj`).
     *        Empty means every LinearModule but an output head named `lm_head`, as PEFT's
     *        `"all-linear"`. (PEFT's own default for Llama is q_proj and v_proj only.)
     */
    std::vector<std::string> target_modules;
};

/**
 * @brief One layer's low-rank update: A `(in, r)` and B `(r, out)`, stored like LinearModule's
 *        weight, with x A B as the update. PEFT's lora_A and lora_B are the transposes,
 *        `(r, in)` and `(out, r)`; SaveLoraAdapter() and LoadLoraAdapter() convert.
 * @note LinearModule owns and drives it; use LinearModule::lora() to read it.
 */
class LoRAAdapter {
public:
    /**
     * @brief A as PEFT initializes it (Kaiming-uniform with a = sqrt(5), so uniform on
     *        ±1 / sqrt(in)), and B zero.
     * @throws std::invalid_argument if rank < 1, alpha isn't finite, or dropout isn't in [0, 1).
     */
    LoRAAdapter(int64_t in_features, int64_t out_features, const LoRAConfig& config, DeviceBackend* backend,
                DeviceType device);

    [[nodiscard]] int64_t rank() const { return rank_; }
    [[nodiscard]] float alpha() const { return alpha_; }
    /** @brief alpha / rank. */
    [[nodiscard]] float scale() const { return alpha_ / static_cast<float>(rank_); }
    [[nodiscard]] float dropout() const { return dropout_p_; }
    [[nodiscard]] const Tensor& a() const { return a_; }
    [[nodiscard]] const Tensor& b() const { return b_; }
    [[nodiscard]] const Tensor& a_grad() const { return a_grad_; }
    [[nodiscard]] const Tensor& b_grad() const { return b_grad_; }
    /** @brief Replaces A, `(in, r)` row-major. @throws std::invalid_argument on a wrong size. */
    void set_a(const std::vector<float>& values);
    /** @brief Replaces B, `(r, out)` row-major. @throws std::invalid_argument on a wrong size. */
    void set_b(const std::vector<float>& values);

    /** @brief True while the update is added into the base weight (LinearModule::merge_lora()). */
    [[nodiscard]] bool merged() const { return merged_; }
    /** @brief False while the layer runs as its base layer (LinearModule::set_lora_enabled()). */
    [[nodiscard]] bool enabled() const { return enabled_; }

    /** @brief The adapted weight W + scale() A B, as a new tensor. */
    [[nodiscard]] Tensor effective_weight(const Tensor& weight) const;
    /** @brief The update scale() A B alone, `(in, out)`. */
    [[nodiscard]] Tensor update() const;

    /**
     * @brief True after a relevance pass that split this layer (LoRARelevance::Split, adapter
     *        enabled and unmerged), until the next forward().
     */
    [[nodiscard]] bool has_relevance() const { return last_adapter_relevance_.numel() > 0; }
    /** @brief The last split's input relevance through the base weight W, `(N, in)`.
     *  @throws std::logic_error unless has_relevance(). */
    [[nodiscard]] const Tensor& last_base_relevance() const;
    /** @brief The last split's input relevance through the adapter, `(N, in)`: what the
     *         fine-tuning contributes to this layer's explanation. The two parts sum to the layer's
     *         input relevance. @throws std::logic_error unless has_relevance(). */
    [[nodiscard]] const Tensor& last_adapter_relevance() const;

private:
    friend class LinearModule;
    /** @brief Adds scale() x A B to @p pre_bias, caching what backward() needs. */
    void add_forward(const Tensor& input, Tensor& pre_bias, bool training);
    /** @brief Accumulates A's and B's gradients and adds the adapter's part to @p grad_input. */
    void backward(const Tensor& grad_output, Tensor& grad_input);
    /** @brief W += sign * scale() A B. */
    void add_update(Tensor& weight, float sign) const;
    void release_activations();
    void clear_relevance();

    int64_t in_features_;
    int64_t out_features_;
    int64_t rank_;
    float alpha_;
    float dropout_p_;
    DeviceBackend* backend_;
    Tensor a_;
    Tensor b_;
    Tensor a_grad_;
    Tensor b_grad_;
    DropoutModule dropout_;
    Tensor last_input_;   // the adapter's input, after dropout
    Tensor last_hidden_;  // scale() x A, (N, r)
    Tensor last_base_relevance_;
    Tensor last_adapter_relevance_;
    bool used_dropout_ = false;
    bool merged_ = false;
    bool enabled_ = true;
};

/**
 * @brief Attaches a LoRA adapter to each LinearModule that config.target_modules selects, then
 *        freezes everything else: only the adapters train, as with PEFT's get_peft_model().
 * @return The adapted layers' names, in the model's order.
 * @note Each layer's A is drawn from config.seed and the layer's position in that order.
 * @throws std::invalid_argument if a target matches no module or a module that isn't a
 *         LinearModule, nothing is selected, or a selected layer already has an adapter. Nothing is
 *         changed then.
 */
std::vector<std::string> ApplyLora(Module& model, const LoRAConfig& config = {});

/** @brief The model's LinearModules that have an adapter, by name, in the model's order. */
[[nodiscard]] std::vector<std::pair<std::string, LinearModule*>> LoraLayers(Module& model);

/** @brief One adapted layer's share of a relevance pass (TRN-8). */
struct LoraLayerRelevance {
    /** @brief The layer's name, as LoraLayers() gives it. */
    std::string name;
    /** @brief Input relevance through the base weight, `(N, in)`. */
    Tensor base;
    /** @brief Input relevance through the adapter, `(N, in)`. */
    Tensor adapter;
    /** @brief Sums over everything: the layer's relevance carried by each path. */
    double base_total = 0;
    double adapter_total = 0;
};

/**
 * @brief How much of each adapted layer's relevance went through its adapter in the model's last
 *        relevance pass (LoRARelevance::Split, the default), in the model's order: where, and for
 *        which inputs, the fine-tuning changed the explanation.
 * @note Layers the pass didn't split (disabled or merged adapters, LoRARelevance::Merged) are left
 *       out.
 * @throws std::logic_error if no layer was split.
 */
[[nodiscard]] std::vector<LoraLayerRelevance> LoraRelevanceSplit(Module& model);

/** @brief Merges every adapter into its base weight (LinearModule::merge_lora()). */
void MergeLora(Module& model);
/** @brief Takes every merged adapter out of its base weight again. */
void UnmergeLora(Module& model);
/** @brief Turns every adapter on or off, so one model runs as the base or as the fine-tune. */
void SetLoraEnabled(Module& model, bool enabled);
/**
 * @brief Removes every adapter. Merged ones stay in the weights (PEFT's merge_and_unload() is
 *        MergeLora() then this); unmerged ones are dropped (PEFT's unload()).
 */
void RemoveLora(Module& model);

/**
 * @brief Writes the model's adapters as a PEFT adapter directory: `adapter_model.safetensors` and
 *        `adapter_config.json`, which PEFT's PeftModel.from_pretrained() loads onto the Hugging
 *        Face model.
 * @param mapping How the model's parameter names become checkpoint names, the manifest used to load
 *        the base model (HuggingFaceMapping() for a CausalLM, EsmMapping() for an EncoderLM). A
 *        layer whose `.weight` it maps whole names the adapter after the source. Empty keeps
 *        pulsatrix's names.
 * @note Keys are `base_model.model.<module>.lora_A.weight` `(r, in)` and `...lora_B.weight`
 *       `(out, r)`, float32.
 * @throws std::invalid_argument if the model has no adapters, the adapters' rank, alpha or dropout
 *         differ, a merged adapter is saved (its update is in the base weight), or a layer has no
 *         whole-weight mapping; std::runtime_error if the files can't be written.
 */
void SaveLoraAdapter(Module& model, const std::string& directory, const std::vector<WeightMapping>& mapping = {});

/**
 * @brief Reads a PEFT LoRA adapter directory onto the model: attaches an adapter to each layer it
 *        has weights for, with r from their shapes and lora_alpha and lora_dropout from
 *        `adapter_config.json`, and freezes the rest as ApplyLora() does.
 * @param mapping As for SaveLoraAdapter().
 * @throws std::invalid_argument for a config that isn't plain LoRA (DoRA, rsLoRA, fan_in_fan_out,
 *         trained biases, modules_to_save, rank or alpha patterns), a key that isn't a LoRA weight,
 *         a layer that doesn't exist or already has an adapter, or a shape that doesn't match.
 *         Everything is checked before the model is changed. std::runtime_error if a file can't be
 *         read.
 */
void LoadLoraAdapter(Module& model, const std::string& directory, const std::vector<WeightMapping>& mapping = {});

}  // namespace pulsatrix
