/** @file sinusoidal_timestep_embedding.hpp
 *  @brief Fixed sinusoidal encoding of a diffusion timestep, for conditioning a denoiser.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Standard Transformer-style sinusoidal encoding of a diffusion timestep `t`:
 *        `emb[2i] = sin(t / base^(2i/embedding_dim))`,
 *        `emb[2i+1] = cos(t / base^(2i/embedding_dim))` for `i in [0, embedding_dim/2)`.
 *
 * This is how a DDPM denoiser is told *which* noise level it is looking at. It is a plain
 * deterministic function of `t` -- nothing is learned, nothing is cached, and no gradient
 * flows into it. The same fixed-positional-quantity disposition RoPEModule already
 * established for its rotation angles, one step further: RoPE is at least a Module because
 * it transforms a tensor, whereas this produces one from an integer and so is a free
 * function, not a class.
 *
 * @param t The timestep to encode. Any value is legal, including 0 and negatives -- sin/cos
 *        are defined everywhere and a schedule's valid range is NoiseSchedule's contract to
 *        enforce, not this function's.
 * @param embedding_dim Size of the produced embedding. Must be positive and even (the
 *        encoding fills sin/cos *pairs*, so an odd size has no valid pairing -- the same
 *        even-dimension requirement, for the same structural reason, as RoPEModule's
 *        head_dim).
 * @param backend Backend to allocate the result through. Not owned; must outlive the
 *        returned Tensor.
 * @param base Frequency base of the geometric wavelength schedule; 10000.0 is the standard
 *        convention shared by the Transformer and DDPM papers.
 * @return The embedding, shape `(1, embedding_dim)`.
 * @throws std::invalid_argument if embedding_dim <= 0 or embedding_dim is odd -- external
 *         boundary, same classification as RoPEModule's head_dim check.
 *
 * @note **Shape is `(1, embedding_dim)`, not rank-1 `(embedding_dim,)`.** This codebase is
 *       always-batched (a single example is N = 1, not a structurally different case --
 *       campaign_exai_dl_library_batch_dimension_support), and the one thing callers do with
 *       this result is concatenate it onto each row of an `(N, D)` batch of noisy samples.
 *       A `(1, embedding_dim)` row is directly that row; a rank-1 result would make every
 *       caller reshape first.
 * @note Writes through Tensor::data() on a tensor it allocates itself, which is therefore
 *       always Cpu-resident -- unlike NoiseSchedule's methods there is no caller-supplied
 *       tensor whose device could be wrong, so there is nothing here to device-guard.
 */
[[nodiscard]] Tensor SinusoidalTimestepEmbedding(int64_t t, int64_t embedding_dim, DeviceBackend* backend,
                                                 float base = 10000.0f);

}  // namespace pulsatrix
