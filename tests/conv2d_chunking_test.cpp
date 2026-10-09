// HIP-7: Conv2DModule in batch chunks, on the CPU (conv2d_chunking_cases.hpp).
#include <gtest/gtest.h>

#include "conv2d_chunking_cases.hpp"

namespace pulsatrix {
namespace {

TEST(Conv2DChunking, ChunkedMatchesWholeBatch) {
    CPUBackend cpu;
    conv_chunking_cases::ChunkedMatchesWholeBatch(&cpu);
}

TEST(Conv2DChunking, FrozenKernelInChunks) {
    CPUBackend cpu;
    conv_chunking_cases::FrozenKernelInChunks(&cpu);
}

TEST(Conv2DChunking, DefaultBudget) {
    CPUBackend cpu;
    Conv2DModule conv(1, 1, 1, 1, &cpu);
    EXPECT_EQ(conv.max_workspace_bytes(), Conv2DModule::kDefaultMaxWorkspaceBytes);
}

}  // namespace
}  // namespace pulsatrix
