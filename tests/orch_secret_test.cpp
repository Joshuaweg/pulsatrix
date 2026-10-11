// AGT-1: orch::Secret, the holder for API keys.
#include "pulsatrix/orch/secret.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace pulsatrix::orch {
namespace {

void SetEnv(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);  // an empty value removes the variable on Windows
#else
    if (*value == '\0') {
        ::unsetenv(name);
    } else {
        ::setenv(name, value, 1);
    }
#endif
}

// Detects `std::ostream << T`: a Secret must not have one.
template <class T, class = void>
struct Streamable : std::false_type {};
template <class T>
struct Streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>> : std::true_type {};

TEST(OrchSecretTest, CannotBePrintedOrConvertedToAString) {
    static_assert(!Streamable<Secret>::value, "a Secret must not be printable");
    static_assert(!std::is_convertible_v<Secret, std::string>, "a Secret must not convert to a string");
    static_assert(!std::is_convertible_v<std::string, Secret>, "making a Secret must be explicit");
}

TEST(OrchSecretTest, RevealsTheValueAndRedactsItElsewhere) {
    Secret s("sk-ant-0123456789");
    EXPECT_EQ(s.reveal(), "sk-ant-0123456789");
    EXPECT_EQ(s.size(), 17u);
    EXPECT_FALSE(s.empty());
    EXPECT_EQ(s.redacted(), "<secret, 17 characters>");
    EXPECT_EQ(s.redacted().find("sk-"), std::string::npos);
    EXPECT_EQ(Secret().redacted(), "<empty secret>");
}

TEST(OrchSecretTest, CopiesAndMovesKeepTheValueAndEmptyTheSource) {
    Secret a("short");  // in the small-string buffer, which a move copies rather than steals
    Secret b(a);
    EXPECT_EQ(b.reveal(), "short");
    Secret c(std::move(a));
    EXPECT_EQ(c.reveal(), "short");
    EXPECT_TRUE(a.empty());  // NOLINT(bugprone-use-after-move): the moved-from state is specified

    Secret d("a much longer value that lives on the heap, not in the small-string buffer");
    d = b;
    EXPECT_EQ(d.reveal(), "short");
    Secret e;
    e = std::move(c);
    EXPECT_EQ(e.reveal(), "short");
    EXPECT_TRUE(c.empty());  // NOLINT(bugprone-use-after-move)
    const Secret& same = e;
    e = same;  // self-assignment keeps the value
    EXPECT_EQ(e.reveal(), "short");
}

TEST(OrchSecretTest, ReadsTheEnvironment) {
    SetEnv("PULSATRIX_TEST_SECRET", "from-the-environment");
    std::optional<Secret> s = Secret::FromEnv("PULSATRIX_TEST_SECRET");
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(s->reveal(), "from-the-environment");

    SetEnv("PULSATRIX_TEST_SECRET", "");
    EXPECT_FALSE(Secret::FromEnv("PULSATRIX_TEST_SECRET").has_value());
    EXPECT_FALSE(Secret::FromEnv("PULSATRIX_TEST_SECRET_THAT_IS_NEVER_SET").has_value());
}

}  // namespace
}  // namespace pulsatrix::orch
