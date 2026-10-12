/** @file rtd.hpp
 *  @brief Representation Topology Divergence (RTD): how differently two representations of the same
 *         inputs are shaped, cluster by cluster and loop by loop, and CompareRepresentations(), one
 *         report with CKA, mutual-kNN alignment and RTD side by side.
 *  @ingroup tda
 *
 * CKA compares two representations' whole geometry at once, and can score two layers as similar
 * when one has clusters or loops the other lacks. RTD (Barannikov, Trofimov, Balabin and Burnaev,
 * ICML 2022) compares their topology at every scale. Build the Vietoris-Rips filtration of a graph
 * on two copies of the inputs, whose distances mix both representations; its barcode, the
 * R-Cross-Barcode, records the features one representation has at a scale where the other has
 * already lost them. RTD is the total length of its H1 bars. It is 0 for identical representations,
 * and grows with every cluster that merges at a different scale.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "pulsatrix/tda/distance_matrix.hpp"
#include "pulsatrix/tda/persistence_diagram.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief The R-Cross-Barcode in one dimension: the Rips diagram of the 2n-vertex graph with
 *        distances, for points A_1..A_n and copies A'_1..A'_n (the paper's practical form, its
 *        Equation 1 with the extra vertex O folded in):
 *        - A_i to A_j: 0;
 *        - A'_i to A'_j: min(w_ij, w~_ij);
 *        - A_i to A'_j: w_ij if i < j, 0 if i == j, and never connected if i > j.
 * @param w, w_tilde The two representations' distances, for the same n points in the same order,
 *        used as given (Rtd() normalizes them first).
 * @param dimension The homology dimension; RTD uses 1.
 * @note By Theorem 2.1 of the paper, the barcode is empty when w == w~, and when w~ is all zeros the
 *       barcode in dimension k + 1 is the ordinary Rips barcode of w in dimension k.
 * @throws std::invalid_argument if the sizes differ, there are fewer than 2 points, or dimension is
 *         outside [1, 7].
 */
[[nodiscard]] PersistenceDiagram CrossBarcode(const DistanceMatrix& w, const DistanceMatrix& w_tilde, int dimension = 1);

/** @brief Rtd()'s settings: the paper's Algorithm 2. */
struct RtdOptions {
    /** @brief Points per random batch; with fewer points than this, all of them, once. */
    int64_t sample_size = 500;
    /** @brief How many batches to average. */
    int64_t trials = 10;
    /** @brief The random seed for the batches; the same on every platform. */
    uint32_t seed = 0;
};

/** @brief Rtd()'s result. */
struct RtdResult {
    /** @brief The symmetric score, (RTD(a, b) + RTD(b, a)) / 2, averaged over the batches. */
    double rtd = 0;
    /** @brief RTD(a, b) alone: the total H1 length of CrossBarcode(w_a, w_b), averaged. */
    double rtd_ab = 0;
    /** @brief RTD(b, a). */
    double rtd_ba = 0;
    /** @brief The batches used. */
    int64_t trials = 0;
};

/**
 * @brief Representation Topology Divergence between two representations of the same inputs (the
 *        paper's Algorithm 2): in each batch, Euclidean distances within a and within b, each
 *        divided by its 0.9 quantile so that scale doesn't count, then the total H1 length of both
 *        cross-barcodes.
 * @param a, b shape (n, ...) each, rows for the same n inputs in the same order; widths may differ.
 * @note Each batch is a Vietoris-Rips computation on 2 x sample_size points.
 * @throws std::invalid_argument if the row counts differ, there are fewer than 4 rows, sample_size
 *         is below 4 or trials below 1, or a representation's distances are all 0.
 */
[[nodiscard]] RtdResult Rtd(const Tensor& a, const Tensor& b, const RtdOptions& options = {});

/** @brief CompareRepresentations()'s settings. */
struct RepresentationComparisonOptions {
    /** @brief Neighbors for the mutual-kNN alignment. */
    int64_t k = 10;
    /** @brief RTD's batches. */
    RtdOptions rtd;
};

/** @brief Three views of how alike two representations are. */
struct RepresentationComparison {
    /** @brief Debiased linear CKA: the whole geometry, up to rotation and scale; 1 is identical. */
    double cka = 0;
    /** @brief Mutual-kNN alignment: the share of each input's k nearest neighbors the two agree on;
     *         chance is about k / (n - 1). */
    double mutual_knn = 0;
    /** @brief RTD: differences in clusters and loops across scales; 0 is identical. */
    RtdResult rtd;
};

/**
 * @brief CKA, mutual-kNN alignment and RTD of the same two representations, for a report: they
 *        disagree informatively, as when CKA is high but RTD isn't 0 because one layer splits a
 *        cluster the other keeps whole.
 * @throws std::invalid_argument as the three measures do.
 */
[[nodiscard]] RepresentationComparison CompareRepresentations(const Tensor& a, const Tensor& b,
                                                              const RepresentationComparisonOptions& options = {});

}  // namespace pulsatrix
