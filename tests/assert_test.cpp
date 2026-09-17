#include <gtest/gtest.h>

#include "exai/assert.hpp"

namespace exai {
namespace {

TEST(ExaiAssertTest, TrueConditionDoesNotAbort) {
    EXPECT_NO_THROW(EXAI_ASSERT(1 + 1 == 2));
}

TEST(ExaiAssertDeathTest, FalseConditionAborts) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    EXPECT_DEATH({ EXAI_ASSERT(1 + 1 == 3); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
