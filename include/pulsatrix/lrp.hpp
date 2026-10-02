/** @file lrp.hpp
 *  @brief LRP -- whole-model Layer-wise Relevance Propagation explainer.
 *  @ingroup dl_explainers
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/attribution.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
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
 * @brief Whole-model LRP: runs the forward pass, seeds relevance at the chosen output(s), and
 *        propagates it to the input through every module's own propagate_relevance() rule.
 * @note The per-layer rules are the ones each Module implements (currently the epsilon-rule
 *       family -- see each module's propagate_relevance() doc); `config` is passed to all of
 *       them. Bias terms absorb relevance under the epsilon rule, and the AttnLRP softmax /
 *       attention rules do not conserve exactly, so `sum(values)` matches the seeded total only
 *       for bias-free, conservative stacks. The Attribution's metadata reports both sums.
 * @note Device-generic: the seed is assembled on the host (one device->host copy of the
 *       network output when seeding with output values) and uploaded through `backend`, which
 *       must be the backend the network's output lives on.
 */
class LRP {
public:
    explicit LRP(LRPRuleConfig config = LRPRuleConfig{}) : config_(config) {}

    /** @brief Explains `target_index` for every row, OutputValue seed. */
    [[nodiscard]] Attribution explain(ExplainerContext& ctx, const Tensor& input, int64_t target_index,
                                      DeviceBackend* backend) const {
        return explain(ctx, input, LRPTarget{{target_index}}, backend);
    }

    /**
     * @brief Explains the given target(s) / contrast(s).
     * @return Attribution{"lrp", relevance (input's shape), metadata}: rule, epsilon, seed,
     *         targets, contrasts, relevance_out_sum, relevance_in_sum.
     * @throws std::invalid_argument if the network output isn't rank-2, a target/contrast list
     *         has neither 1 nor N entries, an index is out of range, or a row's contrast equals
     *         its target (that seed is all zeros and explains nothing).
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

        Tensor relevance = ctx.relevance_pass(seed_tensor, config_);
        const float relevance_in_sum =
            relevance.backend()->sum(relevance.data(), static_cast<size_t>(relevance.numel()));

        return Attribution{"lrp",
                           std::move(relevance),
                           {{"rule", "epsilon"},
                            {"epsilon", std::to_string(config_.epsilon)},
                            {"seed", target.seed == LRPSeed::OutputValue ? "output_value" : "one_hot"},
                            {"targets", join(target.targets)},
                            {"contrasts", join(target.contrasts)},
                            {"relevance_out_sum", std::to_string(relevance_out_sum)},
                            {"relevance_in_sum", std::to_string(relevance_in_sum)}}};
    }

    [[nodiscard]] const LRPRuleConfig& config() const { return config_; }

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
};

}  // namespace pulsatrix
