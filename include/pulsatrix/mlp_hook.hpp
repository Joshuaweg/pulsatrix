/** @file mlp_hook.hpp
 *  @brief Reads or replaces a transformer block's MLP output during forward() (FEAT-5), for
 *         collecting a transcoder's training pairs and splicing it in.
 *  @ingroup mech_interp
 */
#pragma once

#include <functional>
#include <stdexcept>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Called with an MLP's input (after the block's norm, where it has one) and its output
 *        (the branch added to the residual stream), both shaped like the block's input. What it
 *        returns, the same shape, is added to the residual stream instead of the output: return
 *        the output unchanged to only read it.
 * @note backward() and the relevance passes after a forward() that changed the output run as if
 *       the MLP had produced it: no gradient flows through the hook.
 */
using MlpHook = std::function<Tensor(const Tensor& mlp_input, const Tensor& mlp_output)>;

namespace detail {
/** @brief Runs @p hook and checks its result keeps the shape. @throws std::invalid_argument if not. */
inline Tensor ApplyMlpHook(const MlpHook& hook, const Tensor& input, const Tensor& output) {
    Tensor out = hook(input, output);
    if (out.shape() != output.shape()) throw std::invalid_argument("MLP hook: the replacement must keep the MLP output's shape");
    return out;
}
}  // namespace detail

}  // namespace pulsatrix
