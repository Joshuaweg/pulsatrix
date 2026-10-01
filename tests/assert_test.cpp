#include <gtest/gtest.h>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/host_guard.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace {

TEST(PulsatrixAssertTest, TrueConditionDoesNotAbort) {
    EXPECT_NO_THROW(PULSATRIX_ASSERT(1 + 1 == 2));
}

TEST(PulsatrixAssertDeathTest, FalseConditionAborts) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    EXPECT_DEATH({ PULSATRIX_ASSERT(1 + 1 == 3); }, "PULSATRIX_ASSERT failed");
}

// PULSATRIX_REQUIRE_HOST is deliberately NOT NDEBUG-gated: no GTEST_SKIP here, so this runs
// (and must pass) in Release too. A host loop over a device buffer is UB, not a
// debug-only concern.
TEST(PulsatrixRequireHostTest, CpuTensorPasses) {
    CPUBackend backend;
    Tensor t(Shape({2}), &backend);
    PULSATRIX_REQUIRE_HOST(t);
}

TEST(PulsatrixRequireHostDeathTest, NonCpuTensorAbortsInEveryBuildType) {
    CPUBackend backend;
    Tensor mislabelled(Shape({2}), &backend, DeviceType::Hip);  // host memory, device tag
    EXPECT_DEATH({ PULSATRIX_REQUIRE_HOST(mislabelled); }, "PULSATRIX_REQUIRE_HOST\\(mislabelled\\)");
}

}  // namespace
}  // namespace pulsatrix
