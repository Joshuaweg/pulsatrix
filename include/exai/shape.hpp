/** @file shape.hpp
 *  @brief Tensor dimension arithmetic -- rank, element count, per-dimension access.
 */
#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "exai/assert.hpp"

namespace exai {

/**
 * @brief An N-dimensional shape. A plain aggregate of dimensions with no invariant beyond
 *        "non-negative dimensions" -- see oop_design/context_oop_design_fundamentals.md's
 *        struct-vs-class discussion for why this is still a class (numel()/is_reshape_compatible()
 *        are derived queries, not raw public fields the caller could desync from dims_).
 */
class Shape {
public:
    /**
     * @brief Constructs a shape from a dimension list. An empty list is a rank-0 scalar (numel() == 1).
     * @throws std::invalid_argument if any dimension is negative.
     * @note Shape is an external boundary -- Shape objects are built directly from
     *       externally-supplied dimension lists (e.g. bindings/exai_py.cpp's
     *       shape_from_list, from a Python caller). A negative dimension is exactly the
     *       kind of malformed input a well-formed *internal* caller would never produce
     *       but an external one can -- see
     *       campaign_exai_dl_library_adversarial_hardening.md's Mission 0 classification.
     */
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {
        for (int64_t d : dims_) {
            if (d < 0) {
                throw std::invalid_argument("Shape: dimensions must be non-negative");
            }
        }
    }

    /** @brief Number of dimensions. 0 for a scalar. */
    [[nodiscard]] int64_t rank() const { return static_cast<int64_t>(dims_.size()); }

    /**
     * @brief Total element count -- the product of all dimensions.
     * @throws std::overflow_error if the product overflows int64_t.
     * @note A rank-0 shape has numel() == 1 (the empty-product convention), matching scalar
     *       semantics. Any dimension of 0 makes numel() == 0.
     * @note Same external-boundary reasoning as the constructor -- externally-supplied
     *       dimensions can be chosen specifically to overflow this product, which would
     *       otherwise silently wrap into a small or negative value that gets cast to
     *       size_t for an allocation size downstream (a huge- or mis-sized allocation).
     */
    [[nodiscard]] int64_t numel() const {
        int64_t result = 1;
        for (int64_t d : dims_) {
            if (d != 0 && result > std::numeric_limits<int64_t>::max() / (d == 0 ? 1 : d)) {
                throw std::overflow_error("Shape::numel: dimension product overflows int64_t");
            }
            result *= d;
        }
        return result;
    }

    /**
     * @brief Size of a single dimension.
     * @param index Dimension index, must be in [0, rank()).
     * @return The dimension's size.
     * @note EXAI_ASSERT, not throw -- internal invariant per Mission 0's classification
     *       table: every call site in this codebase computes index from an
     *       already-known-valid rank, never from unvalidated external input directly.
     */
    [[nodiscard]] int64_t dim(size_t index) const {
        EXAI_ASSERT(index < dims_.size());
        return dims_[index];
    }

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
