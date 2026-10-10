// HIP-6: the fused training ops on the Hip backend (fused_ops_cases.hpp).
#include <gtest/gtest.h>

#include "fused_ops_cases.hpp"
#include "pulsatrix/hip_backend.hpp"

namespace pulsatrix {
namespace {

TEST(FusedOpsHip, AdamMultiMatchesPerTensor) {
    HIPBackend gpu;
    fused_cases::AdamMultiMatchesPerTensor(&gpu);
}

TEST(FusedOpsHip, DotIntoMatchesDot) {
    HIPBackend gpu;
    fused_cases::DotIntoMatchesDot(&gpu);
}

TEST(FusedOpsHip, TokenCrossEntropyOnDevice) {
    HIPBackend gpu;
    fused_cases::TokenCrossEntropyOnDevice(&gpu);
}

TEST(FusedOpsHip, FusedLinearReluMatchesLayerByLayer) {
    HIPBackend gpu;
    fused_cases::FusedLinearReluMatchesLayerByLayer(&gpu);
}

}  // namespace
}  // namespace pulsatrix
