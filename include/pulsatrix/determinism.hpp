/** @file determinism.hpp
 *  @brief One global seed for everything that isn't given its own, and a deterministic mode
 *         that forbids nondeterministic computation (roadmap FND-7).
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>

namespace pulsatrix {

/**
 * @brief Sets the global seed and restarts the seed stream next_seed() draws from.
 * @note Like `torch.manual_seed`: after set_seed(s), building the same objects in the same
 *       order gives the same initial weights, dropout masks and shuffles. Components that take
 *       an explicit seed ignore the global one. The default global seed is 0, so a program that
 *       never calls set_seed() is reproducible too.
 * @note Process-wide state. Call it before building models, not while other threads are
 *       constructing seeded components.
 */
void set_seed(uint64_t seed);

/** @brief The seed most recently passed to set_seed() (0 by default). */
[[nodiscard]] uint64_t global_seed();

/**
 * @brief The next seed in the global stream: a distinct, well-mixed 64-bit value per call,
 *        reproducible for a given global seed. Components built without an explicit seed take
 *        theirs from here, so two of them never share a random stream by accident.
 */
[[nodiscard]] uint64_t next_seed();

/**
 * @brief Turns deterministic mode on (the default) or off.
 * @note On: the GPU backends forbid atomics in their BLAS libraries (hipBLAS / cuBLAS), and any
 *       code path that would give run-to-run different results refuses to run
 *       (check_deterministic_allowed()). Off: those paths may run, possibly faster. pulsatrix's
 *       own kernels use no atomics and are deterministic either way.
 */
void set_deterministic(bool enabled);

/** @brief Whether deterministic mode is on. */
[[nodiscard]] bool deterministic();

/**
 * @brief Guard for a nondeterministic code path: call it before running one.
 * @param operation Names the operation in the error message.
 * @throws std::logic_error if deterministic mode is on.
 */
void check_deterministic_allowed(const char* operation);

}  // namespace pulsatrix
