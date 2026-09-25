/** @file assert.hpp
 *  @brief PULSATRIX_ASSERT -- debug-only invariant check for programmer errors, distinct from
 *         throw (used for caller-facing contract violations). See
 *         cpp_style_guide/context_style_project_conventions.md's assert-vs-throw table.
 */
#pragma once

#include <cstdio>
#include <cstdlib>

#ifdef NDEBUG
/** @brief No-op in release builds (NDEBUG defined). */
#define PULSATRIX_ASSERT(cond) ((void)0)
#else
/**
 * @brief Aborts with a diagnostic message if cond is false. Debug-only -- use for
 *        conditions that indicate a bug in this library's own code (a caller violating
 *        a documented precondition), never for conditions a well-formed external caller
 *        can legitimately trigger (use throw for those instead).
 */
#define PULSATRIX_ASSERT(cond)                                                                 \
    do {                                                                                   \
        if (!(cond)) {                                                                      \
            std::fprintf(stderr, "PULSATRIX_ASSERT failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::abort();                                                                    \
        }                                                                                     \
    } while (0)
#endif
