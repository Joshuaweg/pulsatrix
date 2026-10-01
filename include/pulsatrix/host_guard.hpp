/** @file host_guard.hpp
 *  @brief PULSATRIX_REQUIRE_HOST -- always-on guard for code paths that dereference
 *         Tensor::data() on the host.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdio>
#include <cstdlib>

#include "pulsatrix/device_backend.hpp"

/**
 * @brief Aborts if tensor is not Cpu-resident. Unlike PULSATRIX_ASSERT this is NOT compiled
 *        out under NDEBUG.
 * @note A host loop over a Cuda/Hip tensor's buffer is not a recoverable contract violation
 *       but undefined behaviour (a device pointer dereferenced on the host), so the check has
 *       to survive Release builds -- PULSATRIX_ASSERT, which these guards used before the
 *       GPU-native-kernels campaign, silently vanished there. The cost is one enum compare
 *       per guarded call.
 * @note The message keeps the "PULSATRIX_ASSERT failed" prefix so the existing guard death
 *       tests, which match on it, keep working unchanged.
 * @note Each remaining use marks a path still awaiting a device kernel (or a deliberate host
 *       boundary such as an RL environment); the campaign's exit gate is zero uses outside
 *       documented host-boundary code.
 */
#define PULSATRIX_REQUIRE_HOST(tensor)                                                              \
    do {                                                                                            \
        if ((tensor).device() != ::pulsatrix::DeviceType::Cpu) {                                    \
            std::fprintf(stderr,                                                                    \
                         "PULSATRIX_ASSERT failed: PULSATRIX_REQUIRE_HOST(%s): host-only code path "  \
                         "reached with a non-Cpu tensor at %s:%d\n",                                \
                         #tensor, __FILE__, __LINE__);                                              \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)
