/** @file param_groups.hpp
 *  @brief Optimizer parameter groups: a learning rate and weight decay per set of parameters,
 *         chosen by name (roadmap TRN-1).
 *  @ingroup dl_modules
 */
#pragma once

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pulsatrix/module.hpp"

namespace pulsatrix {

/** @brief Picks parameters by their named_parameters() name and value. */
using ParamSelector = std::function<bool(const std::string& name, const Tensor& value)>;

/**
 * @brief Optimizer settings for the parameters `selects` picks.
 * @note An optimizer puts each parameter in the first group that selects it, and every
 *       parameter no group selects in its default group (the optimizer's own learning rate and
 *       weight decay). learning_rate and weight_decay may be changed between steps, which is how
 *       a learning-rate schedule adjusts a group.
 */
struct ParamGroup {
    /** @brief For error messages and logs. */
    std::string name;
    ParamSelector selects;
    float learning_rate;
    /** @brief L2 penalty: the step uses grad + weight_decay * value, as PyTorch's SGD and Adam do. */
    float weight_decay = 0.0f;
};

/** @brief Common selectors. */
namespace param_select {

/** @brief The parameter named exactly `prefix`, and every parameter under it (dot boundaries). */
inline ParamSelector name_prefix(std::string prefix) {
    return [prefix = std::move(prefix)](const std::string& name, const Tensor&) {
        return name == prefix || name.rfind(prefix + ".", 0) == 0;
    };
}

/**
 * @brief Parameters of rank 0 or 1: biases and normalization scales. The usual recipe gives
 *        these no weight decay.
 */
inline ParamSelector one_dimensional() {
    return [](const std::string&, const Tensor& value) { return value.rank() <= 1; };
}

}  // namespace param_select

/** @brief Throws std::invalid_argument unless `value` is finite and non-negative. */
inline void check_optimizer_setting(float value, const std::string& what) {
    if (!std::isfinite(value) || value < 0.0f) {
        throw std::invalid_argument(what + " must be finite and non-negative");
    }
}

/** @brief The group machinery SGDOptimizer and AdamOptimizer share. */
class ParamGroupSet {
public:
    /** @brief One parameter with the settings that apply to it this step. */
    struct Assignment {
        ParamRef ref;
        float learning_rate;
        float weight_decay;
    };

    /** @throws std::invalid_argument if a group has no selector or an invalid setting. */
    void set(std::vector<ParamGroup> groups) {
        for (const ParamGroup& g : groups) {
            check(g);
        }
        groups_ = std::move(groups);
    }

    [[nodiscard]] std::vector<ParamGroup>& groups() { return groups_; }
    [[nodiscard]] const std::vector<ParamGroup>& groups() const { return groups_; }

    /**
     * @brief Every parameter of `module`, in parameters() order, with its group's settings.
     * @note A parameter a legacy module reports without a name gets the default settings.
     * @throws std::invalid_argument if a group's settings were changed to invalid values.
     */
    [[nodiscard]] std::vector<Assignment> resolve(Module& module, float default_learning_rate,
                                                  float default_weight_decay) const {
        for (const ParamGroup& g : groups_) {
            check(g);
        }
        std::unordered_map<const Tensor*, const ParamGroup*> chosen;
        if (!groups_.empty()) {
            for (const NamedParamRef& p : module.named_parameters()) {
                for (const ParamGroup& g : groups_) {
                    if (g.selects(p.name, *p.ref.value)) {
                        chosen.emplace(p.ref.value, &g);
                        break;
                    }
                }
            }
        }
        std::vector<Assignment> out;
        for (ParamRef p : module.parameters()) {
            auto it = chosen.find(p.value);
            if (it == chosen.end()) {
                out.push_back({p, default_learning_rate, default_weight_decay});
            } else {
                out.push_back({p, it->second->learning_rate, it->second->weight_decay});
            }
        }
        return out;
    }

private:
    static void check(const ParamGroup& g) {
        if (!g.selects) {
            throw std::invalid_argument("parameter group \"" + g.name + "\" has no selector");
        }
        check_optimizer_setting(g.learning_rate, "parameter group \"" + g.name + "\" learning_rate");
        check_optimizer_setting(g.weight_decay, "parameter group \"" + g.name + "\" weight_decay");
    }

    std::vector<ParamGroup> groups_;
};

}  // namespace pulsatrix
