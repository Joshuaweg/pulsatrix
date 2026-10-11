/** @file representation_similarity.hpp
 *  @brief Linear CKA: how similar two layers' representations of the same inputs are.
 *  @ingroup mech_interp
 *
 * Centered kernel alignment (Kornblith et al., ICML 2019) compares two representations of the
 * same n inputs, for example one layer of two models, two layers of one model, a model before and
 * after fine-tuning, or the same layer computed on two backends. It is 1 when one representation
 * is a rotation and uniform scaling of the other, and near 0 when they are unrelated. Unlike
 * comparing activations directly, it doesn't need the layers to have the same width or their
 * neurons to line up.
 *
 * The debiased estimator (Song et al., JMLR 2012; Nguyen et al., ICLR 2021) is the default. The
 * original, biased one is pulled towards 1 when there are few inputs for the width: two random,
 * unrelated representations of 50 inputs with 1,000 features each score 0.95 with it and
 * about 0 debiased.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/** @brief Which estimator of CKA to use. */
enum class CkaEstimator {
    /** @brief Unbiased HSIC in place of HSIC: near 0 for unrelated representations at any
     *         number of inputs. Needs at least 4 inputs, and can be slightly negative. */
    Debiased,
    /** @brief Kornblith et al.'s original estimator. Between 0 and 1, but biased upwards when the
     *         inputs are few compared with the width. */
    Biased,
};

/**
 * @brief Linear CKA between two representations of the same inputs.
 * @param x one representation, shape (n, ...): a row per input. Dimensions after the first are
 *        flattened, so a (n, channels, height, width) activation is n vectors.
 * @param y another representation of the same n inputs, in the same order, of any width.
 * @return The similarity, 1 for representations equal up to rotation, reflection, uniform scale
 *         and a constant offset. NaN if either representation is constant over the inputs
 *         (or, debiased, its unbiased variance estimate isn't positive), since CKA is undefined
 *         there.
 * @note Computed in double precision on the host, whatever device the tensors are on. It costs
 *       about n × min(n (p + q), p q + p² + q²) multiply-adds for widths p and q.
 * @throws std::invalid_argument if a tensor has rank below 2, the two have different numbers of
 *         rows, or there are fewer than 2 inputs (4 for the debiased estimator).
 */
[[nodiscard]] double LinearCka(const Tensor& x, const Tensor& y, CkaEstimator estimator = CkaEstimator::Debiased);

/** @brief CKA between every pair from two lists of representations, such as the layers of two
 *         models: `values[i * cols + j]` compares `a[i]` with `b[j]`. */
struct CkaMatrix {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<double> values;

    [[nodiscard]] double at(int64_t row, int64_t col) const;
};

/**
 * @brief LinearCka() between every representation in @p a and every one in @p b.
 * @note Each representation's n × n Gram matrix is computed once and reused for every pair, so a
 *       12 × 12 layer comparison costs about 24 Gram matrices rather than 144 CKAs. When those
 *       wouldn't fit in @p max_gram_bytes, each pair is computed on its own instead.
 * @throws std::invalid_argument as LinearCka() does, for any representation, or if a list is
 *         empty.
 */
[[nodiscard]] CkaMatrix LinearCkaMatrix(const std::vector<Tensor>& a, const std::vector<Tensor>& b,
                                        CkaEstimator estimator = CkaEstimator::Debiased,
                                        std::size_t max_gram_bytes = std::size_t{2} << 30);

}  // namespace pulsatrix
