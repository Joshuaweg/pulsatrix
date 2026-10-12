/** @file diagram_distance.hpp
 *  @brief Distances between persistence diagrams, bottleneck and Wasserstein, and Certified<T>,
 *         a value with the error bounds behind it.
 *  @ingroup tda
 *
 * Two diagrams are compared by matching their points: each point either to a point of the other
 * diagram or to the diagonal (where a feature of zero persistence would sit). The bottleneck
 * distance is the smallest possible largest move; the q-Wasserstein distance the smallest q-th
 * root of the sum of q-th powers of the moves. Moves are measured in the L-infinity norm, as in
 * the stability theorem, so a point (b, d) is (d - b) / 2 from the diagonal.
 *
 * The bottleneck distance is the one stability is stated in: two point clouds whose distance
 * matrices differ by at most e have diagrams within e (Chazal, de Silva and Oudot 2014). It lets
 * layers, models and seeds be compared with an error bar, which is why it returns a Certified
 * value.
 */
#pragma once

#include <numeric>
#include <string>
#include <vector>

#include "pulsatrix/tda/persistence_diagram.hpp"

namespace pulsatrix {

/** @brief One source of error in a Certified value. */
struct ErrorTerm {
    /** @brief Where it comes from, such as "first diagram: float32 distances". */
    std::string source;
    /** @brief Its contribution: the value can be off by at most this much because of it. */
    double bound = 0;
};

/**
 * @brief A computed value and a certified bound on its error, itemized by source: the exact value
 *        is within bound() of value.
 */
template <typename T>
struct Certified {
    /** @brief The value as computed. */
    T value{};
    /** @brief The error sources and what each contributes. */
    std::vector<ErrorTerm> terms;

    /** @brief The total bound: the sum of the terms. */
    [[nodiscard]] double bound() const {
        return std::accumulate(terms.begin(), terms.end(), 0.0,
                               [](double s, const ErrorTerm& t) { return s + t.bound; });
    }
};

/**
 * @brief The bottleneck distance between two diagrams of the same dimension.
 * @return The distance between the diagrams as given, certified to be within
 *         a.error_bound + b.error_bound of the distance between the exact diagrams (by the
 *         triangle inequality). Infinite if the diagrams have different numbers of essential
 *         features, which can't be matched to the diagonal.
 * @note Exact: a binary search over the candidate values, each tested by a maximum bipartite
 *       matching (Hopcroft-Karp) on the points and their diagonal projections. Comfortable up to a
 *       few hundred points per diagram; the cost grows as (n1 + n2)^2.5 times a logarithm.
 * @throws std::invalid_argument if the diagrams' dimensions differ.
 */
[[nodiscard]] Certified<double> BottleneckDistance(const PersistenceDiagram& a, const PersistenceDiagram& b);

/**
 * @brief The q-Wasserstein distance between two diagrams of the same dimension, with the
 *        L-infinity norm between points.
 * @param order q, at least 1. Infinity gives the bottleneck distance's value.
 * @return Infinite if the numbers of essential features differ.
 * @note Exact, by the Hungarian algorithm on n1 + n2 rows: (n1 + n2)^3 time, so up to about a
 *       thousand points in all. Unlike the bottleneck distance it isn't certified: an error of e in
 *       every distance can move it by up to e times the q-th root of the number of points.
 * @throws std::invalid_argument if the dimensions differ or order is below 1 or NaN.
 */
[[nodiscard]] double WassersteinDistance(const PersistenceDiagram& a, const PersistenceDiagram& b, double order = 2);

}  // namespace pulsatrix
