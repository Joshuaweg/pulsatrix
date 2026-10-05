/** @file json.hpp
 *  @brief A small strict JSON value, parser and deterministic writer (RFC 8259).
 *  @ingroup visualization
 */
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pulsatrix {

/**
 * @brief One JSON value: null, a boolean, a number, a string, an array or an object.
 * @note Objects keep their keys in insertion order, and the writer emits them in that order, so
 *       the same value always serializes to the same bytes. A number keeps the exact text it was
 *       parsed from, so as_float() and as_double() each round the decimal text once, directly to
 *       their own type, instead of going through double first.
 */
class JsonValue {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };
    using Array = std::vector<JsonValue>;
    using Object = std::vector<std::pair<std::string, JsonValue>>;

    /** @brief A null. */
    JsonValue() = default;
    /** @brief A null. */
    JsonValue(std::nullptr_t) {}
    JsonValue(bool b);
    /** @brief A number written with the shortest text that reads back as the same double.
     *  @throws std::invalid_argument if @p d is NaN or infinite, which JSON can't represent. */
    JsonValue(double d);
    JsonValue(int64_t i);
    JsonValue(int i) : JsonValue(static_cast<int64_t>(i)) {}
    JsonValue(std::string s);
    JsonValue(const char* s) : JsonValue(std::string(s)) {}
    JsonValue(Array a);
    JsonValue(Object o);

    /** @brief A number written with the shortest text that reads back as the same float.
     *  @throws std::invalid_argument if @p f is NaN or infinite. */
    static JsonValue Float(float f);

    [[nodiscard]] Type type() const { return type_; }
    [[nodiscard]] bool is_null() const { return type_ == Type::Null; }

    /** @name Typed access. Each throws std::invalid_argument if the value is another type. */
    ///@{
    [[nodiscard]] bool as_bool() const;
    [[nodiscard]] double as_double() const;
    [[nodiscard]] float as_float() const;
    /** @throws std::invalid_argument also if the number isn't an integer in int64_t's range. */
    [[nodiscard]] int64_t as_int64() const;
    [[nodiscard]] const std::string& as_string() const;
    [[nodiscard]] const Array& as_array() const;
    [[nodiscard]] const Object& as_object() const;
    /** @brief A number's text exactly as written or parsed, for example `"0.1"`. */
    [[nodiscard]] const std::string& number_text() const;
    ///@}

    /** @brief The member named @p key of an object, or nullptr if there is none.
     *  @throws std::invalid_argument if this value isn't an object. */
    [[nodiscard]] const JsonValue* find(std::string_view key) const;

    /** @brief Appends a member to an object. Keys must be unique.
     *  @throws std::invalid_argument if this value isn't an object or already has @p key. */
    void add(std::string key, JsonValue value);

    /** @brief Appends an element to an array.
     *  @throws std::invalid_argument if this value isn't an array. */
    void push_back(JsonValue value);

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    std::string text_;  // a number's text, or a string's contents
    Array array_;
    Object object_;

    friend JsonValue ParseJson(std::string_view text);
    friend class JsonParser;
};

/**
 * @brief Parses one JSON document.
 * @note Strict: no comments, trailing commas, NaN or Infinity literals, duplicate object keys,
 *       invalid UTF-8, unpaired surrogate escapes or trailing content. Nesting deeper than 256
 *       levels is rejected, so a hostile file can't exhaust the stack.
 * @throws std::invalid_argument naming the byte offset of the first error.
 */
[[nodiscard]] JsonValue ParseJson(std::string_view text);

/**
 * @brief Writes @p value as JSON, indented by two spaces and ending with a newline.
 * @note Deterministic: the same value always gives the same bytes on every platform. Arrays that
 *       hold only numbers, booleans and nulls go on one line, so long value arrays stay compact.
 *       Strings are written as UTF-8, with `"`, `\` and control characters escaped.
 * @throws std::invalid_argument if a string isn't valid UTF-8.
 */
[[nodiscard]] std::string WriteJson(const JsonValue& value);

}  // namespace pulsatrix
