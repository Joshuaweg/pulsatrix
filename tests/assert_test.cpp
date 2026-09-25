#include <gtest/gtest.h>

#include "pulsatrix/assert.hpp"

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

}  // namespace
}  // namespace pulsatrix
