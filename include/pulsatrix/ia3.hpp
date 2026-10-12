/** @file ia3.hpp
 *  @brief IA³ (Liu et al., 2022): fine-tune a frozen model by learning one rescaling vector per
 *         adapted Linear layer, on its output (keys, values) or its input (the MLP's down projection),
 *         and save and load it as a PEFT adapter.
 *  @ingroup dl_modules
 *
 * An IA³ layer computes y = (x W + b) ⊙ l, or, for a feedforward layer, y = (x ⊙ l) W + b. Each
 * vector starts at 1, so the adapted model starts as the base model, and it has one value per
 * channel: about 58 thousand values for SmolLM2-135M on the default layers. Either form is the base layer with
 * rescaled weights, W diag(l) or diag(l) W, so the fine-tune can be read off directly (which
 * channels it amplified, which it switched off), and LRP runs on the rescaled layer exactly.
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/causal_lm.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

class LinearModule;

/** @brief Which layers IA³ rescales, PEFT style (see LoRAConfig::target_modules for the matching). */
struct IA3Config {
    /** @brief The layers to rescale. Empty: the paper's choice, keys, values and the MLP's inner
     *         activations: every `k_proj`, `v_proj`, and `down_proj` or `fc2`. (PEFT's default for
     *         Qwen2 is q_proj, v_proj and down_proj; for Llama it is the paper's.) */
    std::vector<std::string> target_modules;
    /** @brief Those whose input is rescaled rather than their output. Each must be among the
     *         targets. Empty with empty targets: `down_proj` and `fc2`. */
    std::vector<std::string> feedforward_modules;
};

/**
 * @brief One layer's rescaling vector l: `(out)` for an output-scaling layer, `(in)` for a
 *        feedforward one. LinearModule owns and drives it; read it through LinearModule::ia3().
 */
class IA3Adapter {
public:
    /** @brief l = 1: the layer is unchanged. */
    IA3Adapter(int64_t size, bool feedforward, DeviceBackend* backend, DeviceType device);

    [[nodiscard]] bool feedforward() const { return feedforward_; }
    [[nodiscard]] const Tensor& scale() const { return scale_; }
    [[nodiscard]] const Tensor& scale_grad() const { return grad_; }
    /** @brief Replaces l. @throws std::invalid_argument on a wrong size. */
    void set_scale(const std::vector<float>& values);
    [[nodiscard]] bool merged() const { return merged_; }
    [[nodiscard]] bool enabled() const { return enabled_; }

private:
    friend class LinearModule;
    bool feedforward_;
    DeviceBackend* backend_;
    Tensor scale_;
    Tensor grad_;
    Tensor last_input_;   // feedforward: x before the rescaling
    Tensor last_output_;  // otherwise: x W + b before the rescaling
    bool merged_ = false;
    bool enabled_ = true;
};

/**
 * @brief Attaches an IA³ vector to each LinearModule config selects, then freezes everything
 *        else, as PEFT's get_peft_model() does.
 * @return The adapted layers' names, in the model's order.
 * @throws std::invalid_argument if a target or feedforward name matches no module or a module
 *         that isn't a LinearModule, a feedforward name isn't a target, nothing is selected, or a
 *         selected layer already has an IA³ vector or a LoRA adapter. Nothing changes then.
 */
std::vector<std::string> ApplyIA3(Module& model, const IA3Config& config = {});

/** @brief The model's LinearModules with an IA³ vector, by name, in the model's order. */
[[nodiscard]] std::vector<std::pair<std::string, LinearModule*>> IA3Layers(Module& model);
/** @brief Folds every vector into its layer's weights (and an output-scaling layer's bias). */
void MergeIA3(Module& model);
/** @brief Divides it back out, as PEFT does. Entries of l within 1e-8 of 0 can't be undone exactly. */
void UnmergeIA3(Module& model);
/** @brief Turns every vector on or off. @throws std::logic_error while one is merged. */
void SetIA3Enabled(Module& model, bool enabled);
/** @brief Removes every vector; merged ones stay in the weights. */
void RemoveIA3(Module& model);

/**
 * @brief Writes the model's IA³ vectors as a PEFT adapter directory (`adapter_model.safetensors` with
 *        `base_model.model.<module>.ia3_l`, `(out, 1)` or `(1, in)`, and `adapter_config.json`).
 * @param mapping As for SaveLoraAdapter(): the base model's weight manifest, or empty for pulsatrix's
 *        names.
 * @throws std::invalid_argument if there are none, one is merged, or a layer has no whole-weight
 *         mapping.
 */
void SaveIA3Adapter(Module& model, const std::string& directory, const std::vector<WeightMapping>& mapping = {});

/**
 * @brief Reads a PEFT IA³ adapter directory onto the model, telling feedforward layers by the
 *        vectors' shapes, and freezes the rest of the model.
 * @throws std::invalid_argument for a config that isn't plain IA³ (fan_in_fan_out, modules_to_save),
 *         a key that isn't an IA³ vector, a shape that fits neither side of its layer or disagrees with
 *         feedforward_modules, or a layer that doesn't exist or is already adapted. Nothing changes then.
 */
void LoadIA3Adapter(Module& model, const std::string& directory, const std::vector<WeightMapping>& mapping = {});

}  // namespace pulsatrix
