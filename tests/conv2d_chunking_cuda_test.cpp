// HIP-7: Conv2DModule in batch chunks, on the Cuda backend (conv2d_chunking_cases.hpp).
#include <gtest/gtest.h>

#include "conv2d_chunking_cases.hpp"
#include "pulsatrix/cuda_backend.hpp"

namespace pulsatrix {
namespace {

TEST(Conv2DChunkingCuda, ChunkedMatchesWholeBatch) {
    CUDABackend gpu;
    conv_chunking_cases::ChunkedMatchesWholeBatch(&gpu);
}

TEST(Conv2DChunkingCuda, FrozenKernelInChunks) {
    CUDABackend gpu;
    conv_chunking_cases::FrozenKernelInChunks(&gpu);
}

}  // namespace
}  // namespace pulsatrix
