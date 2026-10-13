/** @file contrastive.hpp
 *  @brief The contrastive explanation method (CEM; Dhurandhar et al., "Explanations based on the
 *         Missing", NeurIPS 2018): pertinent negatives, what is absent from an input and would change
 *         its class if present, and pertinent positives, the least of the input that keeps its class
 *         on its own (XAI-4).
 *  @ingroup interpretability_dl
 *
 * Both are found by FISTA on an elastic-net objective, with a binary search over the weight of the
 * prediction loss, as in the paper. With x the input, t0 its class, z the model's outputs and b the
 * absence value of each feature (the baseline, 0 by default):
 *  - a pertinent negative is the sparsest δ >= 0 on features absent from x (x_j == b_j) such that
 *    x + δ is no longer class t0. It minimizes
 *    c · max(z_t0(x + δ) - max_{i != t0} z_i(x + δ), -κ) + β |δ|_1 + |δ|_2^2;
 *  - a pertinent positive is the sparsest part δ of x - b, each feature between b_j and x_j, such that
 *    b + δ alone is still class t0. It minimizes
 *    c · max(max_{i != t0} z_i(b + δ) - z_t0(b + δ), -κ) + β |δ|_1 + |δ|_2^2.
 *
 * The paper also adds an autoencoder's reconstruction error to keep δ on the data manifold; that
 * term isn't here, so on images a pertinent negative can look like noise to a person.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief CEM's settings. The defaults are the paper's MNIST settings, scaled to its 1,000 steps. */
struct ContrastiveOptions {
    /** @brief The margin κ the class change (or keep) must reach in the outputs. */
    float kappa = 0.0f;
    /** @brief The L1 weight β: larger gives sparser explanations. */
    float beta = 0.1f;
    /** @brief The first weight c of the prediction loss; the binary search moves it. */
    float c_init = 10.0f;
    /** @brief Rounds of the binary search over c. */
    int search_steps = 9;
    /** @brief FISTA iterations in each round. */
    int steps = 1000;
    float learning_rate = 0.01f;
    /** @brief Each feature's absence value; empty for 0. */
    std::vector<float> baseline;
    /** @brief The most a pertinent negative may raise each feature to; empty for no limit. */
    std::vector<float> upper;
};

/** @brief A pertinent negative or positive. */
struct ContrastiveResult {
    /** @brief Whether one was found. When not, the other fields describe the input unchanged. */
    bool found = false;
    /** @brief The model's class for the input. */
    int64_t original_class = 0;
    /** @brief The class of the explanation: another class for a pertinent negative, the same one
     *         for a pertinent positive. */
    int64_t explanation_class = 0;
    /** @brief δ, one value per feature. */
    std::vector<float> delta;
    /** @brief What the model sees: x + δ for a pertinent negative, b + δ for a pertinent positive. */
    std::vector<float> explanation;
    /** @brief |δ|_1, and the features δ changes. */
    double l1 = 0.0;
    std::vector<int64_t> features;
    /** @brief The weight c of the explanation returned. */
    float c = 0.0f;
};

/**
 * @brief The pertinent negative of @p input: what would have to be added, where the input has
 *        nothing, to change its class.
 * @param ctx The model, run through forward_pass and backward_pass; its class is the index of its
 *        largest output. Parameter gradients accumulate in it, as with Saliency.
 * @param input One input with its batch dimension.
 * @throws std::invalid_argument if baseline or upper has the wrong length, or an option is out of range.
 */
[[nodiscard]] ContrastiveResult FindPertinentNegative(ExplainerContext& ctx, const Tensor& input,
                                                      const ContrastiveOptions& options = {});

/**
 * @brief The pertinent positive of @p input: the least of it that the model still gives its class.
 * @throws std::invalid_argument as FindPertinentNegative().
 */
[[nodiscard]] ContrastiveResult FindPertinentPositive(ExplainerContext& ctx, const Tensor& input,
                                                      const ContrastiveOptions& options = {});

/** @brief The result as a `pulsatrix.contrastive.v1` document; @p kind is "pertinent_negative" or
 *         "pertinent_positive". */
[[nodiscard]] JsonValue ToJson(const ContrastiveResult& result, const std::string& kind);

}  // namespace pulsatrix
