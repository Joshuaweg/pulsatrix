#include <gtest/gtest.h>

#include "exai/assert.hpp"

namespace exai {
namespace {

TEST(ExaiAssertTest, TrueConditionDoesNotAbort) {
    EXPECT_NO_THROW(EXAI_ASSERT(1 + 1 == 2));
}

TEST(ExaiAssertDeathTest, FalseConditionAborts) {
    EXPECT_DEATH({ EXAI_ASSERT(1 + 1 == 3); }, "EXAI_ASSERT failed");
}

}  // namespace
}  // namespace exai
