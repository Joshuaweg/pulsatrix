// HIP-6: the fused training ops on the Cuda backend (fused_ops_cases.hpp).
#include <gtest/gtest.h>

#include "fused_ops_cases.hpp"
#include "pulsatrix/cuda_backend.hpp"

namespace pulsatrix {
namespace {

TEST(FusedOpsCuda, AdamMultiMatchesPerTensor) {
    CUDABackend gpu;
    fused_cases::AdamMultiMatchesPerTensor(&gpu);
}

TEST(FusedOpsCuda, DotIntoMatchesDot) {
    CUDABackend gpu;
    fused_cases::DotIntoMatchesDot(&gpu);
}

TEST(FusedOpsCuda, TokenCrossEntropyOnDevice) {
    CUDABackend gpu;
    fused_cases::TokenCrossEntropyOnDevice(&gpu);
}

TEST(FusedOpsCuda, FusedLinearReluMatchesLayerByLayer) {
    CUDABackend gpu;
    fused_cases::FusedLinearReluMatchesLayerByLayer(&gpu);
}

}  // namespace
}  // namespace pulsatrix
