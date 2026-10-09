/** @file lrp.hpp
 *  @brief LRP -- whole-model Layer-wise Relevance Propagation explainer.
 *  @ingroup interpretability_lrp
 */
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp_rule_config.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief How LRP seeds relevance at the target output. */
enum class LRPSeed {
    /// R[n, t] = y[n, t]: the total relevance is the explained logit itself (the classic
    /// Bach et al. 2015 setup, where conservation reads "attributions sum to the output").
    OutputValue,
    /// R[n, t] = 1: unit relevance at the target, the convention Zennit / LXT attributors use
    /// when handed a one-hot output gradient. Use this to compare against those libraries.
    OneHot
};

/**
 * @brief What LRP explains: one target class (and optionally one contrast class) per row of
 *        the network's (N, num_classes) output.
 * @note `targets` / `contrasts` hold either a single index (applied to every row) or one index
 *       per row. With contrasts, the seed is seed(target) - seed(contrast): LRP propagation is
 *       linear in the output relevance for fixed activations, so the result is exactly
 *       explanation(target) - explanation(contrast) -- "why t rather than c".
 */
struct LRPTarget {
    std::vector<int64_t> targets;
    std::vector<int64_t> contrasts = {};
    LRPSeed seed = LRPSeed::OutputValue;
};

/**
 * @brief Per-layer LRP rule choice: maps (top-level module index in forward order, module) to
 *        the LRPRuleConfig that module applies.
 * @note LRP calls a composite exactly once per module, in ascending index order starting at 0,
 *       on every explain() -- presets that depend on position (epsilon_gamma_box's "first
 *       Conv2D") rely on that order.
 * @note Applied to the ExplainerContext's top-level modules: a SequentialModule receives one
 *       config, which it forwards to all of its layers (it supports a rule only if they all do).
 */
using LRPComposite = std::function<LRPRuleConfig(size_t layer_index, const Module& module)>;

/**
 * @brief Zennit 1.0.0's composite presets (zennit.composites), mapped onto pulsatrix modules by
 *        type: LinearModule = torch Linear, Conv2DModule = torch Conv2d; every other module gets
 *        the epsilon rule (the pass-through modules -- ReLU, Flatten, Dropout, MaxPool -- ignore
 *        it, as Zennit's Pass rule / plain gradient does).
 * @note Every Epsilon entry sets epsilon_bias_in_denominator = true: Zennit's Epsilon rule puts
 *       the bias in z, so matching Zennit numerically requires it (this codebase's own default
 *       epsilon rule leaves the bias out to be conservative).
 */
namespace lrp_composite {

namespace detail {
inline bool is_conv(const Module& module) { return dynamic_cast<const Conv2DModule*>(&module) != nullptr; }
inline LRPRuleConfig zennit_epsilon(float epsilon) {
    LRPRuleConfig config{epsilon};
    config.epsilon_bias_in_denominator = true;
    return config;
}
inline LRPRuleConfig alpha_beta(float alpha, float beta, float epsilon) {
    LRPRuleConfig config{epsilon};
    config.rule = LRPRule::AlphaBeta;
    config.alpha = alpha;
    config.beta = beta;
    return config;
}
}  // namespace detail

/** @brief Epsilon for this module, and @p conv for every Conv2D inside it (residual blocks,
 *         sequences): Zennit's by-type mapping carried into nested modules. */
inline LRPRuleConfig epsilon_with_conv(float epsilon, const LRPRuleConfig& conv) {
    LRPRuleConfig config = detail::zennit_epsilon(epsilon);
    config.conv_rule = std::make_shared<const LRPRuleConfig>(conv);
    return config;
}

/** @brief Zennit EpsilonPlus: Epsilon for Linear, ZPlus (AlphaBeta 1, 0) for Conv2D, including
 *         convolutions inside residual blocks and other containers. */
inline LRPComposite epsilon_plus(float epsilon = 1e-6f) {
    return [epsilon](size_t, const Module& module) {
        const LRPRuleConfig zplus = detail::alpha_beta(1.0f, 0.0f, epsilon);
        return detail::is_conv(module) ? zplus : epsilon_with_conv(epsilon, zplus);
    };
}

/** @brief Zennit EpsilonAlpha2Beta1: Epsilon for Linear, AlphaBeta(2, 1) for Conv2D, nested
 *         ones included. */
inline LRPComposite epsilon_alpha2_beta1(float epsilon = 1e-6f) {
    return [epsilon](size_t, const Module& module) {
        const LRPRuleConfig ab = detail::alpha_beta(2.0f, 1.0f, epsilon);
        return detail::is_conv(module) ? ab : epsilon_with_conv(epsilon, ab);
    };
}

/**
 * @brief Zennit EpsilonGammaBox: ZBox(low, high) for the first Conv2D layer (lowest index),
 *        Gamma(gamma) for every other Conv2D, nested ones included, Epsilon for every Linear.
 * @note The first Conv2D must be a top-level module, as it is in ResNet and VGG: one nested
 *       in a container before any top-level Conv2D gets Gamma.
 * @note As in Zennit 1.0.0, whose first_map holds only Convolution: a Linear is never ZBox'd,
 *       even when it is the first layer, so on a Conv2D-free network this preset is Epsilon on
 *       every layer (checked against Zennit in tests/lrp_reference_test.cpp).
 * @note Remembers the first Conv2D it has seen since the last index-0 call (see LRPComposite's
 *       calling order).
 */
inline LRPComposite epsilon_gamma_box(float low, float high, float gamma = 0.25f, float epsilon = 1e-6f) {
    auto first_conv_seen = std::make_shared<bool>(false);
    return [=](size_t layer_index, const Module& module) {
        if (layer_index == 0) {
            *first_conv_seen = false;
        }
        if (!detail::is_conv(module)) {
            LRPRuleConfig gamma_rule{epsilon};
            gamma_rule.rule = LRPRule::Gamma;
            gamma_rule.gamma = gamma;
            return epsilon_with_conv(epsilon, gamma_rule);
        }
        LRPRuleConfig config{epsilon};
        if (!*first_conv_seen) {
            *first_conv_seen = true;
            config.rule = LRPRule::ZBox;
            config.low = low;
            config.high = high;
        } else {
            config.rule = LRPRule::Gamma;
            config.gamma = gamma;
        }
        return config;
    };
}

}  // namespace lrp_composite

/**
 * @brief Whole-model LRP: runs the forward pass, seeds relevance at the chosen output(s), and
 *        propagates it to the input through every module's own propagate_relevance() rule.
 * @note The per-layer rules are the ones each Module implements (see each module's
 *       propagate_relevance() doc); either one `config` is passed to all of them, or an
 *       LRPComposite chooses one per module. A module asked for a rule it does not implement
 *       throws (ExplainerContext::relevance_pass) -- there is no silent fallback to epsilon.
 *       With LRPRuleConfig::epsilon_bias_in_denominator set, bias terms absorb relevance, and
 *       the AttnLRP softmax / attention rules do not conserve exactly, so `sum(values)` matches
 *       the seeded total only for conservative stacks. The Attribution's metadata reports both
 *       sums.
 * @note Device-generic: the seed is assembled on the host (one device->host copy of the
 *       network output when seeding with output values) and uploaded through `backend`, which
 *       must be the backend the network's output lives on.
 */
class LRP {
public:
    /** @brief Uniform rule: every module applies `config`. */
    explicit LRP(LRPRuleConfig config = LRPRuleConfig{}) : config_(config) {}

    /**
     * @brief Per-layer rules from a composite; `name` is reported as "composite:<name>".
     * @throws std::invalid_argument if composite is empty.
     */
    explicit LRP(LRPComposite composite, std::string name = "custom")
        : config_(), composite_(std::move(composite)), composite_name_(std::move(name)) {
        if (!composite_) {
            throw std::invalid_argument("LRP: composite must not be empty");
        }
    }

    /** @brief Zennit EpsilonPlus preset (lrp_composite::epsilon_plus). */
    [[nodiscard]] static LRP epsilon_plus(float epsilon = 1e-6f) {
        return LRP(lrp_composite::epsilon_plus(epsilon), "epsilon_plus");
    }
    /** @brief Zennit EpsilonGammaBox preset (lrp_composite::epsilon_gamma_box). */
    [[nodiscard]] static LRP epsilon_gamma_box(float low, float high, float gamma = 0.25f, float epsilon = 1e-6f) {
        return LRP(lrp_composite::epsilon_gamma_box(low, high, gamma, epsilon), "epsilon_gamma_box");
    }
    /** @brief Zennit EpsilonAlpha2Beta1 preset (lrp_composite::epsilon_alpha2_beta1). */
    [[nodiscard]] static LRP epsilon_alpha2_beta1(float epsilon = 1e-6f) {
        return LRP(lrp_composite::epsilon_alpha2_beta1(epsilon), "epsilon_alpha2_beta1");
    }

    /** @brief Explains `target_index` for every row, OutputValue seed. */
    [[nodiscard]] Attribution explain(ExplainerContext& ctx, const Tensor& input, int64_t target_index,
                                      DeviceBackend* backend) const {
        return explain(ctx, input, LRPTarget{{target_index}}, backend);
    }

    /**
     * @brief Explains the given target(s) / contrast(s).
     * @return Attribution{"lrp", relevance (input's shape), metadata}: rule (the uniform rule's
     *         name, or "composite:<name>"), rules (each module's rule, comma-separated, forward
     *         order), epsilon (the uniform config's), seed, targets, contrasts,
     *         relevance_out_sum, relevance_in_sum.
     * @throws std::invalid_argument if the network output isn't rank-2, a target/contrast list
     *         has neither 1 nor N entries, an index is out of range, or a row's contrast equals
     *         its target (that seed is all zeros and explains nothing), a module does not
     *         implement its rule, or a rule's parameters are invalid.
     * @throws std::logic_error if the context's last forward pass was patched (relevance_pass()).
     */
    [[nodiscard]] Attribution explain(ExplainerContext& ctx, const Tensor& input, const LRPTarget& target,
                                      DeviceBackend* backend) const {
        Tensor output = ctx.forward_pass(input);
        if (output.rank() != 2) {
            throw std::invalid_argument("LRP::explain: network output must be rank-2 (N, num_classes)");
        }
        const int64_t N = output.shape().dim(0);
        const int64_t C = output.shape().dim(1);
        const std::vector<int64_t> targets = per_row(target.targets, N, C, "targets");
        const std::vector<int64_t> contrasts =
            target.contrasts.empty() ? std::vector<int64_t>{} : per_row(target.contrasts, N, C, "contrasts");
        for (size_t n = 0; n < contrasts.size(); ++n) {
            if (contrasts[n] == targets[n]) {
                throw std::invalid_argument(
                    "LRP::explain: a row's contrast equals its target (the seed would be zero)");
            }
        }

        // Seed on the host, then upload: valid whatever device the output lives on.
        std::vector<float> y;
        if (target.seed == LRPSeed::OutputValue) {
            y.resize(static_cast<size_t>(N * C));
            output.backend()->copy(y.data(), output.data(), y.size() * sizeof(float),
                                   output.device() == DeviceType::Cpu ? CopyDirection::HostToHost
                                                                      : CopyDirection::DeviceToHost);
        }
        auto seed_value = [&](int64_t n, int64_t c) {
            return target.seed == LRPSeed::OutputValue ? y[static_cast<size_t>(n * C + c)] : 1.0f;
        };
        std::vector<float> seed(static_cast<size_t>(N * C), 0.0f);
        float relevance_out_sum = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            const int64_t t = targets[static_cast<size_t>(n)];
            seed[static_cast<size_t>(n * C + t)] = seed_value(n, t);
            relevance_out_sum += seed_value(n, t);
            if (!contrasts.empty()) {
                const int64_t c = contrasts[static_cast<size_t>(n)];
                seed[static_cast<size_t>(n * C + c)] = -seed_value(n, c);
                relevance_out_sum -= seed_value(n, c);
            }
        }
        Tensor seed_tensor(output.shape(), backend, seed, output.device());

        std::vector<LRPRuleConfig> configs;
        configs.reserve(ctx.modules().size());
        for (size_t i = 0; i < ctx.modules().size(); ++i) {
            configs.push_back(composite_ ? composite_(i, *ctx.modules()[i]) : config_);
        }
        std::string rules;
        for (size_t i = 0; i < configs.size(); ++i) {
            rules += (i == 0 ? "" : ",") + lrp_rule_name(configs[i].rule);
        }
        Tensor relevance = ctx.relevance_pass(seed_tensor, configs);
        const float relevance_in_sum =
            relevance.backend()->sum(relevance.data(), static_cast<size_t>(relevance.numel()));

        return Attribution{"lrp",
                           std::move(relevance),
                           {{"rule", composite_ ? "composite:" + composite_name_ : lrp_rule_name(config_.rule)},
                            {"rules", rules},
                            {"epsilon", std::to_string(config_.epsilon)},
                            {"seed", target.seed == LRPSeed::OutputValue ? "output_value" : "one_hot"},
                            {"targets", join(target.targets)},
                            {"contrasts", join(target.contrasts)},
                            {"relevance_out_sum", std::to_string(relevance_out_sum)},
                            {"relevance_in_sum", std::to_string(relevance_in_sum)}}};
    }

    /** @brief The uniform config (default-constructed when this LRP uses a composite). */
    [[nodiscard]] const LRPRuleConfig& config() const { return config_; }

    /** @brief The composite, or an empty function for a uniform-rule LRP. */
    [[nodiscard]] const LRPComposite& composite() const { return composite_; }

private:
    // Expands a 1-entry list to N rows and validates every index against C classes.
    static std::vector<int64_t> per_row(const std::vector<int64_t>& indices, int64_t N, int64_t C,
                                        const char* what) {
        if (indices.size() != 1 && static_cast<int64_t>(indices.size()) != N) {
            throw std::invalid_argument(std::string("LRP::explain: ") + what +
                                        " must have 1 entry or one per row of the output");
        }
        std::vector<int64_t> rows(static_cast<size_t>(N));
        for (int64_t n = 0; n < N; ++n) {
            const int64_t index = indices.size() == 1 ? indices[0] : indices[static_cast<size_t>(n)];
            if (index < 0 || index >= C) {
                throw std::invalid_argument(std::string("LRP::explain: ") + what + " index out of range");
            }
            rows[static_cast<size_t>(n)] = index;
        }
        return rows;
    }

    static std::string join(const std::vector<int64_t>& v) {
        std::string out;
        for (size_t i = 0; i < v.size(); ++i) {
            out += (i == 0 ? "" : ",") + std::to_string(v[i]);
        }
        return out;
    }

    LRPRuleConfig config_;
    LRPComposite composite_;
    std::string composite_name_;
};

}  // namespace pulsatrix
