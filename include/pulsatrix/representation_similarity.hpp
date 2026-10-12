/** @file representation_similarity.hpp
 *  @brief How similar two layers' representations of the same inputs are: linear CKA, RSA and
 *         SVCCA.
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
 *
 * Two older measures are here too, for comparison with published results that use them:
 *
 * - **RSA** (representational similarity analysis; Kriegeskorte et al., Frontiers in Systems
 *   Neuroscience 2008) describes each representation by how far apart it puts every pair of
 *   inputs, its representational dissimilarity matrix (RDM), and correlates the two RDMs. It
 *   compares a model with anything that orders the same inputs, a brain recording or a
 *   hand-made model of the categories, and RsaPermutationTest() says whether the agreement could
 *   be chance.
 * - **SVCCA** (Raghu et al., NeurIPS 2017) keeps each representation's top singular directions
 *   (99% of the variance by default) and finds the pairs of directions, one in each, that
 *   correlate most. It is invariant to any invertible linear map, which CKA isn't. That makes it
 *   blind to how a layer weights its directions, and Kornblith et al. found it worse than CKA at
 *   matching corresponding layers. It is also biased upwards: canonical correlations are fitted
 *   to the inputs, so unrelated representations score well above 0 unless there are many more
 *   inputs than kept directions.
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

// ---- RSA ----

/** @brief How RepresentationalDissimilarity() measures the distance between two inputs. */
enum class RdmMetric {
    /** @brief 1 minus the Pearson correlation of the two rows, between 0 and 2: the classic RSA
     *         choice, blind to each row's mean and scale. */
    Correlation,
    /** @brief The Euclidean distance between the rows. */
    Euclidean,
    /** @brief 1 minus the cosine of the angle between the rows, between 0 and 2. */
    Cosine,
};

/** @brief A representational dissimilarity matrix: the distance between every pair of n inputs.
 *  @note Stored condensed, as scipy's `pdist` stores it: pairs (0, 1), (0, 2), ..., (0, n-1),
 *        (1, 2), ..., n (n - 1) / 2 values. */
struct Rdm {
    int64_t n = 0;
    std::vector<double> values;

    /** @brief The distance between inputs i and j; 0 when they are the same. */
    [[nodiscard]] double at(int64_t i, int64_t j) const;
};

/**
 * @brief The RDM of a representation, shape (n, ...): a row per input, later dimensions
 *        flattened.
 * @note Computed in double precision on the host. Correlation and cosine distances are clipped to
 *       [0, 2], as scipy clips them, so rounding can't make them negative.
 * @throws std::invalid_argument if x has rank below 2, fewer than 3 rows, or a NaN or infinite
 *         value, or if a row is constant (for Correlation) or zero (for Cosine), where the
 *         distance is undefined.
 */
[[nodiscard]] Rdm RepresentationalDissimilarity(const Tensor& x, RdmMetric metric = RdmMetric::Correlation);

/** @brief How CompareRdms() correlates two RDMs. */
enum class RdmComparison {
    /** @brief Spearman's rank correlation, ties given their average rank (scipy's `spearmanr`).
     *         Only the order of the distances counts, so it suits RDMs on different scales. */
    Spearman,
    /** @brief Pearson's correlation of the distances themselves. */
    Pearson,
};

/**
 * @brief The correlation between two RDMs of the same inputs, in the same order.
 * @return Between -1 and 1; NaN if either RDM is constant.
 * @throws std::invalid_argument if either describes fewer than 3 inputs, holds the wrong number of
 *         values or a NaN or infinite one, or the two describe different numbers of inputs.
 */
[[nodiscard]] double CompareRdms(const Rdm& a, const Rdm& b, RdmComparison comparison = RdmComparison::Spearman);

/** @brief RSA between two representations of the same inputs:
 *         CompareRdms(RepresentationalDissimilarity(x), RepresentationalDissimilarity(y)).
 *  @throws std::invalid_argument as those two do, or if x and y have different numbers of rows. */
[[nodiscard]] double Rsa(const Tensor& x, const Tensor& y, RdmMetric metric = RdmMetric::Correlation,
                         RdmComparison comparison = RdmComparison::Spearman);

/** @brief RsaPermutationTest()'s result. */
struct RsaTestResult {
    /** @brief CompareRdms() of the two RDMs as given. */
    double similarity = 0;
    /** @brief One-sided: the share of relabelings, counting the real one, that score at least
     *         @ref similarity. Never below 1 / (permutations + 1). */
    double p_value = 1;
    int64_t permutations = 0;
};

/**
 * @brief Whether two RDMs agree more than chance: the Mantel test RSA uses.
 *
 * The inputs of @p b are relabeled at random (rows and columns together, so each relabeling is
 * an RDM the same representation could have given in another order) and compared with @p a again.
 * Distances within an RDM aren't independent, which is why a correlation's usual p-value doesn't
 * apply and relabeling does.
 * @param permutations How many relabelings; 999 resolves p-values down to 0.001.
 * @param seed The relabelings come from it: the same seed, the same p-value on every platform.
 * @throws std::invalid_argument as CompareRdms() does, or if permutations is below 1.
 */
[[nodiscard]] RsaTestResult RsaPermutationTest(const Rdm& a, const Rdm& b, RdmComparison comparison = RdmComparison::Spearman,
                                               int64_t permutations = 999, uint64_t seed = 0);

// ---- SVCCA ----

/**
 * @brief The canonical correlations between two representations of the same inputs, largest
 *        first: min(rank x, rank y) values between 0 and 1.
 *
 * The first is the largest correlation between any direction in x's (centered) space and any in
 * y's; each later one the largest among directions uncorrelated with the earlier ones. They are
 * invariant to invertible linear maps of either representation.
 * @note Directions whose variance is rounding error (below the largest × max(n, width) × machine
 *       epsilon) don't count towards a rank.
 * @throws std::invalid_argument if a tensor has rank below 2 or a NaN or infinite value, the two
 *         have different numbers of rows, either is constant, or rank x + rank y isn't below the
 *         number of inputs: then at least one correlation is 1 whatever the data, and the rest
 *         are inflated.
 */
[[nodiscard]] std::vector<double> CanonicalCorrelations(const Tensor& x, const Tensor& y);

/** @brief Svcca()'s settings. */
struct SvccaOptions {
    /** @brief Keep each representation's fewest top singular directions whose share of the
     *         variance exceeds this, as scikit-learn's `PCA(n_components=0.99)` chooses. In (0, 1]. */
    double variance_kept = 0.99;
    /** @brief If above 0, keep exactly this many directions of each instead (capped at its rank),
     *         as Raghu et al. do to compare layers of different widths on equal terms. */
    int64_t directions = 0;
};

/** @brief Svcca()'s result. */
struct SvccaResult {
    /** @brief The mean canonical correlation, Raghu et al.'s similarity. */
    double similarity = 0;
    /** @brief The canonical correlations between the kept directions, largest first. */
    std::vector<double> correlations;
    /** @brief How many singular directions of x and of y were kept. */
    int64_t x_directions = 0;
    int64_t y_directions = 0;
};

/**
 * @brief SVCCA: CanonicalCorrelations() between each representation's top singular directions.
 * @throws std::invalid_argument as CanonicalCorrelations() does for the kept directions, or if an
 *         option is out of range.
 */
[[nodiscard]] SvccaResult Svcca(const Tensor& x, const Tensor& y, const SvccaOptions& options = {});

}  // namespace pulsatrix
