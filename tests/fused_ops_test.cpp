// HIP-6: the fused training ops on the CPU (fused_ops_cases.hpp).
#include <gtest/gtest.h>

#include "fused_ops_cases.hpp"

namespace pulsatrix {
namespace {

TEST(FusedOps, AdamMultiMatchesPerTensor) {
    CPUBackend cpu;
    fused_cases::AdamMultiMatchesPerTensor(&cpu);
}

TEST(FusedOps, DotIntoMatchesDot) {
    CPUBackend cpu;
    fused_cases::DotIntoMatchesDot(&cpu);
}

TEST(FusedOps, TokenCrossEntropyOnDevice) {
    CPUBackend cpu;
    fused_cases::TokenCrossEntropyOnDevice(&cpu);
}

TEST(FusedOps, FusedLinearReluMatchesLayerByLayer) {
    CPUBackend cpu;
    fused_cases::FusedLinearReluMatchesLayerByLayer(&cpu);
}

}  // namespace
}  // namespace pulsatrix
