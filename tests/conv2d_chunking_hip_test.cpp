// HIP-7: Conv2DModule in batch chunks, on the Hip backend (conv2d_chunking_cases.hpp).
#include <gtest/gtest.h>

#include "conv2d_chunking_cases.hpp"
#include "pulsatrix/hip_backend.hpp"

namespace pulsatrix {
namespace {

TEST(Conv2DChunkingHip, ChunkedMatchesWholeBatch) {
    HIPBackend gpu;
    conv_chunking_cases::ChunkedMatchesWholeBatch(&gpu);
}

TEST(Conv2DChunkingHip, FrozenKernelInChunks) {
    HIPBackend gpu;
    conv_chunking_cases::FrozenKernelInChunks(&gpu);
}

}  // namespace
}  // namespace pulsatrix
