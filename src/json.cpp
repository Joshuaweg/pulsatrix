#include "pulsatrix/json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <system_error>

namespace pulsatrix {

namespace {

constexpr int kMaxDepth = 256;

[[noreturn]] void TypeError(const char* wanted) {
    throw std::invalid_argument(std::string("JSON value is not ") + wanted);
}

// The length of the valid UTF-8 sequence starting at s[i], or 0 if it isn't one. Rejects
// overlong encodings, surrogates (U+D800..U+DFFF) and code points above U+10FFFF.
size_t Utf8SequenceLength(std::string_view s, size_t i) {
    auto byte = [&](size_t k) { return static_cast<unsigned char>(s[k]); };
    auto continuation = [&](size_t k) { return k < s.size() && (byte(k) & 0xC0) == 0x80; };
    unsigned char c = byte(i);
    if (c < 0x80) {
        return 1;
    }
    if (c >= 0xC2 && c <= 0xDF) {
        return continuation(i + 1) ? 2 : 0;
    }
    if (c >= 0xE0 && c <= 0xEF) {
        if (!continuation(i + 1) || !continuation(i + 2)) {
            return 0;
        }
        unsigned char c1 = byte(i + 1);
        if ((c == 0xE0 && c1 < 0xA0) || (c == 0xED && c1 >= 0xA0)) {
            return 0;  // overlong, or a surrogate
        }
        return 3;
    }
    if (c >= 0xF0 && c <= 0xF4) {
        if (!continuation(i + 1) || !continuation(i + 2) || !continuation(i + 3)) {
            return 0;
        }
        unsigned char c1 = byte(i + 1);
        if ((c == 0xF0 && c1 < 0x90) || (c == 0xF4 && c1 >= 0x90)) {
            return 0;  // overlong, or above U+10FFFF
        }
        return 4;
    }
    return 0;
}

void AppendUtf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Formats the shortest round-tripping digits the way JavaScript's Number.prototype.toString
// does: plain decimals from 1e-6 up to 1e21, exponent notation outside that. `scientific` is
// std::to_chars's shortest scientific form, for example "-1.5e-06".
std::string JavaScriptStyle(std::string_view scientific) {
    std::string sign;
    if (!scientific.empty() && scientific.front() == '-') {
        sign = "-";
        scientific.remove_prefix(1);
    }
    size_t e_pos = scientific.find('e');
    std::string digits;
    for (char ch : scientific.substr(0, e_pos)) {
        if (ch != '.') {
            digits += ch;
        }
    }
    int exponent = std::stoi(std::string(scientific.substr(e_pos + 1)));
    const int k = static_cast<int>(digits.size());
    const int n = exponent + 1;  // where the decimal point falls, counted from the first digit
    if (k <= n && n <= 21) {
        return sign + digits + std::string(static_cast<size_t>(n - k), '0');
    }
    if (0 < n && n <= 21) {
        return sign + digits.substr(0, static_cast<size_t>(n)) + "." + digits.substr(static_cast<size_t>(n));
    }
    if (-6 < n && n <= 0) {
        return sign + "0." + std::string(static_cast<size_t>(-n), '0') + digits;
    }
    std::string out = sign + digits.substr(0, 1);
    if (k > 1) {
        out += "." + digits.substr(1);
    }
    out += (n - 1 < 0) ? "e-" : "e+";
    out += std::to_string(std::abs(n - 1));
    return out;
}

template <typename T>
std::string ShortestText(T value) {
    char buf[64];
    auto result = std::to_chars(buf, buf + sizeof buf, value, std::chars_format::scientific);
    return JavaScriptStyle(std::string_view(buf, static_cast<size_t>(result.ptr - buf)));
}

// Whether a valid JSON number's magnitude is below 1 (so an out-of-range conversion is an
// underflow, not an overflow): the decimal exponent of its first nonzero digit is negative.
bool MagnitudeBelowOne(std::string_view text) {
    size_t i = (text.front() == '-') ? 1 : 0;
    long position = 0;  // decimal exponent of the first nonzero digit, before the 'e' part
    bool seen_point = false;
    bool found = false;
    long int_digits = 0;
    long leading_fraction_zeros = 0;
    for (; i < text.size() && text[i] != 'e' && text[i] != 'E'; ++i) {
        char ch = text[i];
        if (ch == '.') {
            seen_point = true;
        } else if (!found) {
            if (ch != '0') {
                found = true;
                position = seen_point ? -(leading_fraction_zeros + 1) : 0;
                if (!seen_point) {
                    int_digits = 1;
                }
            } else if (seen_point) {
                ++leading_fraction_zeros;
            }
        } else if (!seen_point) {
            ++int_digits;
        }
    }
    if (!found) {
        return true;  // zero
    }
    if (int_digits > 0) {
        position = int_digits - 1;
    }
    long exponent = 0;
    if (i < text.size()) {
        // The exponent can be huge ("1e-99999999"); clamp instead of overflowing.
        std::string_view e = text.substr(i + 1);
        bool negative = !e.empty() && e.front() == '-';
        if (!e.empty() && (e.front() == '-' || e.front() == '+')) {
            e.remove_prefix(1);
        }
        for (char ch : e) {
            exponent = std::min(exponent * 10 + (ch - '0'), 100000L);
        }
        if (negative) {
            exponent = -exponent;
        }
    }
    return position + exponent < 0;
}

template <typename T>
T ParseFloating(const std::string& text, const char* type_name) {
    T value{};
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec == std::errc::result_out_of_range) {
        if (MagnitudeBelowOne(text)) {
            return text.front() == '-' ? -T(0) : T(0);  // underflow rounds to a signed zero
        }
        throw std::invalid_argument("JSON number " + text + " is out of " + type_name + " range");
    }
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        throw std::invalid_argument("JSON number " + text + " can't be read as " + type_name);
    }
    return value;
}

}  // namespace

JsonValue::JsonValue(bool b) : type_(Type::Bool), bool_(b) {}

JsonValue::JsonValue(double d) : type_(Type::Number) {
    if (!std::isfinite(d)) {
        throw std::invalid_argument("JSON can't hold NaN or infinity");
    }
    text_ = ShortestText(d);
}

JsonValue::JsonValue(int64_t i) : type_(Type::Number), text_(std::to_string(i)) {}

JsonValue::JsonValue(std::string s) : type_(Type::String), text_(std::move(s)) {}

JsonValue::JsonValue(Array a) : type_(Type::Array), array_(std::move(a)) {}

JsonValue::JsonValue(Object o) : type_(Type::Object), object_(std::move(o)) {}

JsonValue JsonValue::Float(float f) {
    if (!std::isfinite(f)) {
        throw std::invalid_argument("JSON can't hold NaN or infinity");
    }
    JsonValue v;
    v.type_ = Type::Number;
    v.text_ = ShortestText(f);
    return v;
}

bool JsonValue::as_bool() const {
    if (type_ != Type::Bool) {
        TypeError("a boolean");
    }
    return bool_;
}

double JsonValue::as_double() const {
    if (type_ != Type::Number) {
        TypeError("a number");
    }
    return ParseFloating<double>(text_, "double");
}

float JsonValue::as_float() const {
    if (type_ != Type::Number) {
        TypeError("a number");
    }
    return ParseFloating<float>(text_, "float");
}

int64_t JsonValue::as_int64() const {
    if (type_ != Type::Number) {
        TypeError("a number");
    }
    std::string_view t = text_;
    if (t == "-0") {
        return 0;
    }
    int64_t value = 0;
    auto result = std::from_chars(t.data(), t.data() + t.size(), value);
    if (result.ec != std::errc() || result.ptr != t.data() + t.size()) {
        throw std::invalid_argument("JSON number " + text_ + " is not an integer in int64 range");
    }
    return value;
}

const std::string& JsonValue::as_string() const {
    if (type_ != Type::String) {
        TypeError("a string");
    }
    return text_;
}

const JsonValue::Array& JsonValue::as_array() const {
    if (type_ != Type::Array) {
        TypeError("an array");
    }
    return array_;
}

const JsonValue::Object& JsonValue::as_object() const {
    if (type_ != Type::Object) {
        TypeError("an object");
    }
    return object_;
}

const std::string& JsonValue::number_text() const {
    if (type_ != Type::Number) {
        TypeError("a number");
    }
    return text_;
}

const JsonValue* JsonValue::find(std::string_view key) const {
    for (const auto& [k, v] : as_object()) {
        if (k == key) {
            return &v;
        }
    }
    return nullptr;
}

void JsonValue::add(std::string key, JsonValue value) {
    if (find(key) != nullptr) {
        throw std::invalid_argument("JSON object already has key \"" + key + "\"");
    }
    object_.emplace_back(std::move(key), std::move(value));
}

void JsonValue::push_back(JsonValue value) {
    if (type_ != Type::Array) {
        TypeError("an array");
    }
    array_.push_back(std::move(value));
}

// ---- parser --------------------------------------------------------------------------------

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : s_(text) {}

    JsonValue ParseDocument() {
        SkipWhitespace();
        JsonValue v = ParseValue(0);
        SkipWhitespace();
        if (pos_ != s_.size()) {
            Fail("unexpected content after the document");
        }
        return v;
    }

private:
    std::string_view s_;
    size_t pos_ = 0;

    [[noreturn]] void Fail(const std::string& what) const {
        throw std::invalid_argument("JSON parse error at offset " + std::to_string(pos_) + ": " + what);
    }

    void SkipWhitespace() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    void Expect(char c) {
        if (pos_ >= s_.size() || s_[pos_] != c) {
            Fail(std::string("expected '") + c + "'");
        }
        ++pos_;
    }

    void ExpectLiteral(std::string_view word) {
        if (s_.substr(pos_, word.size()) != word) {
            Fail("invalid literal");
        }
        pos_ += word.size();
    }

    JsonValue ParseValue(int depth) {
        if (pos_ >= s_.size()) {
            Fail("unexpected end of input");
        }
        switch (s_[pos_]) {
            case '{':
                return ParseObject(depth + 1);
            case '[':
                return ParseArray(depth + 1);
            case '"':
                return JsonValue(ParseString());
            case 't':
                ExpectLiteral("true");
                return JsonValue(true);
            case 'f':
                ExpectLiteral("false");
                return JsonValue(false);
            case 'n':
                ExpectLiteral("null");
                return JsonValue();
            default:
                return ParseNumber();
        }
    }

    JsonValue ParseObject(int depth) {
        if (depth > kMaxDepth) {
            Fail("nesting deeper than " + std::to_string(kMaxDepth));
        }
        Expect('{');
        JsonValue obj{JsonValue::Object{}};
        SkipWhitespace();
        if (pos_ < s_.size() && s_[pos_] == '}') {
            ++pos_;
            return obj;
        }
        while (true) {
            SkipWhitespace();
            if (pos_ >= s_.size() || s_[pos_] != '"') {
                Fail("expected a string key");
            }
            size_t key_pos = pos_;
            std::string key = ParseString();
            if (obj.find(key) != nullptr) {
                pos_ = key_pos;
                Fail("duplicate key \"" + key + "\"");
            }
            SkipWhitespace();
            Expect(':');
            SkipWhitespace();
            JsonValue value = ParseValue(depth);
            obj.object_.emplace_back(std::move(key), std::move(value));
            SkipWhitespace();
            if (pos_ < s_.size() && s_[pos_] == ',') {
                ++pos_;
                continue;
            }
            Expect('}');
            return obj;
        }
    }

    JsonValue ParseArray(int depth) {
        if (depth > kMaxDepth) {
            Fail("nesting deeper than " + std::to_string(kMaxDepth));
        }
        Expect('[');
        JsonValue arr{JsonValue::Array{}};
        SkipWhitespace();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            ++pos_;
            return arr;
        }
        while (true) {
            SkipWhitespace();
            arr.array_.push_back(ParseValue(depth));
            SkipWhitespace();
            if (pos_ < s_.size() && s_[pos_] == ',') {
                ++pos_;
                continue;
            }
            Expect(']');
            return arr;
        }
    }

    uint32_t ParseHex4() {
        if (pos_ + 4 > s_.size()) {
            Fail("truncated \\u escape");
        }
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') {
                v |= static_cast<uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                v |= static_cast<uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                v |= static_cast<uint32_t>(c - 'A' + 10);
            } else {
                --pos_;
                Fail("invalid hex digit in \\u escape");
            }
        }
        return v;
    }

    std::string ParseString() {
        Expect('"');
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) {
                Fail("unterminated string");
            }
            unsigned char c = static_cast<unsigned char>(s_[pos_]);
            if (c == '"') {
                ++pos_;
                return out;
            }
            if (c < 0x20) {
                Fail("unescaped control character in string");
            }
            if (c == '\\') {
                ++pos_;
                if (pos_ >= s_.size()) {
                    Fail("unterminated escape");
                }
                char e = s_[pos_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'u': {
                        uint32_t cp = ParseHex4();
                        if (cp >= 0xDC00 && cp <= 0xDFFF) {
                            Fail("unpaired low surrogate");
                        }
                        if (cp >= 0xD800 && cp <= 0xDBFF) {
                            if (s_.substr(pos_, 2) != "\\u") {
                                Fail("unpaired high surrogate");
                            }
                            pos_ += 2;
                            uint32_t low = ParseHex4();
                            if (low < 0xDC00 || low > 0xDFFF) {
                                Fail("unpaired high surrogate");
                            }
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        }
                        AppendUtf8(out, cp);
                        break;
                    }
                    default:
                        --pos_;
                        Fail("invalid escape");
                }
                continue;
            }
            size_t n = Utf8SequenceLength(s_, pos_);
            if (n == 0) {
                Fail("invalid UTF-8");
            }
            out.append(s_.substr(pos_, n));
            pos_ += n;
        }
    }

    JsonValue ParseNumber() {
        size_t start = pos_;
        auto digit = [&](size_t i) { return i < s_.size() && s_[i] >= '0' && s_[i] <= '9'; };
        if (pos_ < s_.size() && s_[pos_] == '-') {
            ++pos_;
        }
        if (!digit(pos_)) {
            Fail("invalid value");
        }
        if (s_[pos_] == '0') {
            ++pos_;
            if (digit(pos_)) {
                Fail("leading zero in number");
            }
        } else {
            while (digit(pos_)) {
                ++pos_;
            }
        }
        if (pos_ < s_.size() && s_[pos_] == '.') {
            ++pos_;
            if (!digit(pos_)) {
                Fail("expected a digit after the decimal point");
            }
            while (digit(pos_)) {
                ++pos_;
            }
        }
        if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) {
                ++pos_;
            }
            if (!digit(pos_)) {
                Fail("expected a digit in the exponent");
            }
            while (digit(pos_)) {
                ++pos_;
            }
        }
        JsonValue v;
        v.type_ = JsonValue::Type::Number;
        v.text_ = std::string(s_.substr(start, pos_ - start));
        return v;
    }
};

JsonValue ParseJson(std::string_view text) { return JsonParser(text).ParseDocument(); }

// ---- writer --------------------------------------------------------------------------------

namespace {

void WriteString(std::string& out, const std::string& s) {
    out += '"';
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"': out += "\\\""; ++i; continue;
            case '\\': out += "\\\\"; ++i; continue;
            case '\b': out += "\\b"; ++i; continue;
            case '\f': out += "\\f"; ++i; continue;
            case '\n': out += "\\n"; ++i; continue;
            case '\r': out += "\\r"; ++i; continue;
            case '\t': out += "\\t"; ++i; continue;
            default: break;
        }
        if (c < 0x20) {
            static const char* hex = "0123456789abcdef";
            out += "\\u00";
            out += hex[c >> 4];
            out += hex[c & 0xF];
            ++i;
            continue;
        }
        size_t n = Utf8SequenceLength(s, i);
        if (n == 0) {
            throw std::invalid_argument("JSON string is not valid UTF-8 (byte " + std::to_string(i) + ")");
        }
        out.append(s, i, n);
        i += n;
    }
    out += '"';
}

bool IsScalar(const JsonValue& v) {
    return v.type() == JsonValue::Type::Null || v.type() == JsonValue::Type::Bool || v.type() == JsonValue::Type::Number;
}

void WriteValue(std::string& out, const JsonValue& v, int indent) {
    const std::string pad(static_cast<size_t>(indent) * 2, ' ');
    const std::string inner(static_cast<size_t>(indent + 1) * 2, ' ');
    switch (v.type()) {
        case JsonValue::Type::Null:
            out += "null";
            return;
        case JsonValue::Type::Bool:
            out += v.as_bool() ? "true" : "false";
            return;
        case JsonValue::Type::Number:
            // A number's stored text is already canonical: either written by this library or
            // validated JSON from the parser.
            out += v.number_text();
            return;
        case JsonValue::Type::String:
            WriteString(out, v.as_string());
            return;
        case JsonValue::Type::Array: {
            const JsonValue::Array& a = v.as_array();
            if (a.empty()) {
                out += "[]";
                return;
            }
            bool flat = true;
            for (const JsonValue& e : a) {
                flat = flat && IsScalar(e);
            }
            out += '[';
            for (size_t i = 0; i < a.size(); ++i) {
                if (flat) {
                    out += (i == 0) ? "" : ", ";
                } else {
                    out += (i == 0) ? "\n" : ",\n";
                    out += inner;
                }
                WriteValue(out, a[i], indent + 1);
            }
            if (!flat) {
                out += '\n';
                out += pad;
            }
            out += ']';
            return;
        }
        case JsonValue::Type::Object: {
            const JsonValue::Object& o = v.as_object();
            if (o.empty()) {
                out += "{}";
                return;
            }
            out += '{';
            for (size_t i = 0; i < o.size(); ++i) {
                out += (i == 0) ? "\n" : ",\n";
                out += inner;
                WriteString(out, o[i].first);
                out += ": ";
                WriteValue(out, o[i].second, indent + 1);
            }
            out += '\n';
            out += pad;
            out += '}';
            return;
        }
    }
}

}  // namespace

std::string WriteJson(const JsonValue& value) {
    std::string out;
    WriteValue(out, value, 0);
    out += '\n';
    return out;
}

}  // namespace pulsatrix
