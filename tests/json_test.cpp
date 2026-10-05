#include "pulsatrix/json.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

void ExpectRejected(const std::string& text) {
    EXPECT_THROW((void)ParseJson(text), std::invalid_argument) << "accepted: " << text;
}

TEST(JsonParseTest, ParsesEveryValueType) {
    JsonValue v = ParseJson(R"( {"a": [1, -2.5, true, false, null], "b": {"c": "d"}, "e": ""} )");
    ASSERT_EQ(v.type(), JsonValue::Type::Object);
    const JsonValue::Array& a = v.find("a")->as_array();
    ASSERT_EQ(a.size(), 5u);
    EXPECT_EQ(a[0].as_int64(), 1);
    EXPECT_EQ(a[1].as_double(), -2.5);
    EXPECT_TRUE(a[2].as_bool());
    EXPECT_FALSE(a[3].as_bool());
    EXPECT_TRUE(a[4].is_null());
    EXPECT_EQ(v.find("b")->find("c")->as_string(), "d");
    EXPECT_EQ(v.find("e")->as_string(), "");
    EXPECT_EQ(v.find("missing"), nullptr);
}

TEST(JsonParseTest, KeepsObjectMemberOrder) {
    JsonValue v = ParseJson(R"({"z": 1, "a": 2, "m": 3})");
    const JsonValue::Object& o = v.as_object();
    ASSERT_EQ(o.size(), 3u);
    EXPECT_EQ(o[0].first, "z");
    EXPECT_EQ(o[1].first, "a");
    EXPECT_EQ(o[2].first, "m");
}

TEST(JsonParseTest, NumbersConvertExactlyToEachType) {
    // 0.1 read as float directly is the float nearest 0.1, not the double nearest 0.1 narrowed.
    EXPECT_EQ(ParseJson("0.1").as_float(), 0.1f);
    EXPECT_EQ(ParseJson("0.1").as_double(), 0.1);
    EXPECT_EQ(ParseJson("9223372036854775807").as_int64(), std::numeric_limits<int64_t>::max());
    EXPECT_EQ(ParseJson("-9223372036854775808").as_int64(), std::numeric_limits<int64_t>::min());
    EXPECT_EQ(ParseJson("1E2").as_double(), 100.0);
    EXPECT_EQ(ParseJson("-0").as_int64(), 0);
    EXPECT_THROW((void)ParseJson("3.5").as_int64(), std::invalid_argument);
    EXPECT_THROW((void)ParseJson("1e3").as_int64(), std::invalid_argument);
    EXPECT_THROW((void)ParseJson("9223372036854775808").as_int64(), std::invalid_argument);
    // Out of float range: an error, not a silent infinity.
    EXPECT_THROW((void)ParseJson("1e39").as_float(), std::invalid_argument);
    EXPECT_THROW((void)ParseJson("1e309").as_double(), std::invalid_argument);
}

TEST(JsonParseTest, DecodesStringEscapesToUtf8) {
    EXPECT_EQ(ParseJson(R"("a\"b\\c\/d\b\f\n\r\t")").as_string(), "a\"b\\c/d\b\f\n\r\t");
    EXPECT_EQ(ParseJson(R"("\u00e9")").as_string(), "\xC3\xA9");
    EXPECT_EQ(ParseJson(R"("\u20AC")").as_string(), "\xE2\x82\xAC");
    EXPECT_EQ(ParseJson(R"("\ud83d\ude00")").as_string(), "\xF0\x9F\x98\x80");
    EXPECT_EQ(ParseJson(R"("\u0000")").as_string(), std::string(1, '\0'));
    EXPECT_EQ(ParseJson("\"caf\xC3\xA9\"").as_string(), "caf\xC3\xA9");
}

TEST(JsonParseTest, RejectsMalformedInput) {
    for (const char* text : {
             "", " ", "{", "[1,]", "{\"a\":1,}", "[1 2]", "{\"a\" 1}", "{a: 1}", "'a'",
             "// c\n1", "/* c */ 1", "NaN", "Infinity", "-Infinity", "nan", "01", "+1", ".5", "1.",
             "1e", "-", "0x10", "tru", "nul", "[1] [2]", "1 x", "\"abc", "\"\\x\"", "\"\\u12\"",
             "{\"a\": 1, \"a\": 2}",
         }) {
        ExpectRejected(text);
    }
}

TEST(JsonParseTest, RejectsInvalidStrings) {
    ExpectRejected("\"\\ud83d\"");          // unpaired high surrogate
    ExpectRejected("\"\\ude00\"");          // unpaired low surrogate
    ExpectRejected("\"\\ud83d\\u0041\"");   // high surrogate followed by a non-surrogate
    ExpectRejected(std::string("\"a\nb\""));  // raw control character
    ExpectRejected("\"\x80\"");             // lone continuation byte
    ExpectRejected("\"\xC0\x80\"");         // overlong encoding of U+0000
    ExpectRejected("\"\xED\xA0\x80\"");     // UTF-8-encoded surrogate
    ExpectRejected("\"\xF4\x90\x80\x80\"");  // above U+10FFFF
    ExpectRejected("\"\xE2\x82\"");         // truncated sequence
}

TEST(JsonParseTest, LimitsNestingDepth) {
    auto nested = [](int depth) { return std::string(static_cast<size_t>(depth), '[') + std::string(static_cast<size_t>(depth), ']'); };
    EXPECT_NO_THROW((void)ParseJson(nested(256)));
    ExpectRejected(nested(257));
    ExpectRejected(std::string(100000, '['));
}

TEST(JsonParseTest, ErrorNamesTheByteOffset) {
    try {
        (void)ParseJson("[1, 2, x]");
        FAIL() << "accepted";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("offset 7"), std::string::npos) << e.what();
    }
}

TEST(JsonWriteTest, LayoutIsFixed) {
    JsonValue v{JsonValue::Object{}};
    v.add("name", "x");
    v.add("numbers", JsonValue::Array{1, 2.5, nullptr, true});
    v.add("empty_array", JsonValue::Array{});
    v.add("empty_object", JsonValue::Object{});
    v.add("strings", JsonValue::Array{"a", "b"});
    JsonValue inner{JsonValue::Object{}};
    inner.add("k", false);
    v.add("objects", JsonValue::Array{inner});
    EXPECT_EQ(WriteJson(v),
              "{\n"
              "  \"name\": \"x\",\n"
              "  \"numbers\": [1, 2.5, null, true],\n"
              "  \"empty_array\": [],\n"
              "  \"empty_object\": {},\n"
              "  \"strings\": [\n"
              "    \"a\",\n"
              "    \"b\"\n"
              "  ],\n"
              "  \"objects\": [\n"
              "    {\n"
              "      \"k\": false\n"
              "    }\n"
              "  ]\n"
              "}\n");
    EXPECT_EQ(WriteJson(JsonValue(3)), "3\n");
}

TEST(JsonWriteTest, NumbersAreShortestAndJavaScriptStyled) {
    auto f = [](float x) { std::string s = WriteJson(JsonValue::Float(x)); return s.substr(0, s.size() - 1); };
    auto d = [](double x) { std::string s = WriteJson(JsonValue(x)); return s.substr(0, s.size() - 1); };
    EXPECT_EQ(f(0.1f), "0.1");
    EXPECT_EQ(f(-1.25f), "-1.25");
    EXPECT_EQ(f(0.0f), "0");
    EXPECT_EQ(f(-0.0f), "-0");
    EXPECT_EQ(f(3.0f), "3");
    EXPECT_EQ(f(1e-7f), "1e-7");
    EXPECT_EQ(f(1.5e-6f), "0.0000015");
    EXPECT_EQ(f(123456789.0f), "123456790");
    EXPECT_EQ(f(1e21f), "1e+21");  // exponent form from 1e21 up, as in JavaScript
    EXPECT_EQ(f(3.4028235e38f), "3.4028235e+38");
    EXPECT_EQ(d(0.1), "0.1");
    EXPECT_EQ(d(0.0005), "0.0005");
    EXPECT_EQ(d(1e21), "1e+21");
    EXPECT_EQ(d(1e20), "100000000000000000000");
    EXPECT_EQ(d(5e-324), "5e-324");
    EXPECT_EQ(WriteJson(JsonValue(int64_t{-9007199254740993})), "-9007199254740993\n");
}

TEST(JsonWriteTest, FloatsAndDoublesRoundTripBitExactly) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<uint32_t> bits32;
    std::uniform_int_distribution<uint64_t> bits64;
    int checked = 0;
    while (checked < 20000) {
        uint32_t b = bits32(rng);
        float x;
        std::memcpy(&x, &b, sizeof x);
        if (!std::isfinite(x)) {
            continue;
        }
        float back = ParseJson(WriteJson(JsonValue::Float(x))).as_float();
        uint32_t back_bits;
        std::memcpy(&back_bits, &back, sizeof back);
        ASSERT_EQ(back_bits, b) << x;
        uint64_t b64 = bits64(rng);
        double y;
        std::memcpy(&y, &b64, sizeof y);
        if (std::isfinite(y)) {
            double back64 = ParseJson(WriteJson(JsonValue(y))).as_double();
            uint64_t back64_bits;
            std::memcpy(&back64_bits, &back64, sizeof back64);
            ASSERT_EQ(back64_bits, b64) << y;
        }
        ++checked;
    }
    for (float x : {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
                    std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()}) {
        EXPECT_EQ(ParseJson(WriteJson(JsonValue::Float(x))).as_float(), x);
    }
}

TEST(JsonWriteTest, EscapesStrings) {
    EXPECT_EQ(WriteJson(JsonValue(std::string("a\"b\\c\n\x01/\xC3\xA9", 10))), "\"a\\\"b\\\\c\\n\\u0001/\xC3\xA9\"\n");
    EXPECT_EQ(WriteJson(JsonValue(std::string(1, '\0'))), "\"\\u0000\"\n");
}

TEST(JsonWriteTest, RejectsWhatJsonCannotHold) {
    EXPECT_THROW(JsonValue(std::nan("")), std::invalid_argument);
    EXPECT_THROW(JsonValue(std::numeric_limits<double>::infinity()), std::invalid_argument);
    EXPECT_THROW(JsonValue::Float(-std::numeric_limits<float>::infinity()), std::invalid_argument);
    EXPECT_THROW((void)WriteJson(JsonValue(std::string("\xFF"))), std::invalid_argument);
}

TEST(JsonWriteTest, ParseOfWriteIsTheSameValue) {
    std::string text = "{\n  \"a\": [1, -0.5, null],\n  \"b\": {\n    \"c\": \"\xE2\x82\xAC\"\n  }\n}\n";
    EXPECT_EQ(WriteJson(ParseJson(text)), text);
}

TEST(JsonValueTest, BuildersCheckTheirType) {
    JsonValue o{JsonValue::Object{}};
    o.add("a", 1);
    EXPECT_THROW(o.add("a", 2), std::invalid_argument);
    EXPECT_THROW(o.push_back(1), std::invalid_argument);
    JsonValue a{JsonValue::Array{}};
    a.push_back(1);
    EXPECT_THROW(a.add("x", 1), std::invalid_argument);
    EXPECT_THROW((void)a.find("x"), std::invalid_argument);
    EXPECT_THROW((void)JsonValue("s").as_double(), std::invalid_argument);
    EXPECT_THROW((void)JsonValue(1).as_string(), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
