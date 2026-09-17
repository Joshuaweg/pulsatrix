/** @file shape.hpp
 *  @brief Tensor dimension arithmetic -- rank, element count, per-dimension access.
 */
#pragma once

#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace exai {

/**
 * @brief An N-dimensional shape. A plain aggregate of dimensions with no invariant beyond
 *        "non-negative dimensions" -- see oop_design/context_oop_design_fundamentals.md's
 *        struct-vs-class discussion for why this is still a class (numel()/is_reshape_compatible()
 *        are derived queries, not raw public fields the caller could desync from dims_).
 */
class Shape {
public:
    /** @brief Constructs a shape from a dimension list. An empty list is a rank-0 scalar (numel() == 1). */
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}

    /** @brief Number of dimensions. 0 for a scalar. */
    [[nodiscard]] int64_t rank() const { return static_cast<int64_t>(dims_.size()); }

    /**
     * @brief Total element count -- the product of all dimensions.
     * @note A rank-0 shape has numel() == 1 (the empty-product convention), matching scalar
     *       semantics. Any dimension of 0 makes numel() == 0.
     */
    [[nodiscard]] int64_t numel() const {
        return std::accumulate(dims_.begin(), dims_.end(), int64_t{1}, std::multiplies<>());
    }

    /**
     * @brief Size of a single dimension.
     * @param index Dimension index, must be in [0, rank()).
     * @return The dimension's size.
     */
    [[nodiscard]] int64_t dim(size_t index) const { return dims_[index]; }

    /**
     * @brief Whether this shape and other have the same numel() -- the precondition for a
     *        valid reshape between them (same underlying buffer, different dimension grouping).
     * @param other Candidate target shape.
     * @return true iff numel() == other.numel().
     */
    [[nodiscard]] bool is_reshape_compatible(const Shape& other) const { return numel() == other.numel(); }

    [[nodiscard]] bool operator==(const Shape& other) const { return dims_ == other.dims_; }
    [[nodiscard]] bool operator!=(const Shape& other) const { return !(*this == other); }

private:
    std::vector<int64_t> dims_;
};

}  // namespace exai
