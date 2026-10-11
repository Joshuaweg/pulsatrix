#include "pulsatrix/orch/json_schema.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pulsatrix::orch {

namespace {

// Deeper schemas than this are refused, so a schema can't exhaust the stack.
constexpr int kMaxSchemaDepth = 64;

using Type = JsonValue::Type;

bool IsTypeName(std::string_view s) {
    return s == "null" || s == "boolean" || s == "object" || s == "array" || s == "number" || s == "string" ||
           s == "integer";
}

bool IsSchemaUri(std::string_view s) {
    if (!s.empty() && s.back() == '#') {
        s.remove_suffix(1);
    }
    return s == "http://json-schema.org/draft-07/schema" || s == "https://json-schema.org/draft/2020-12/schema";
}

// A JSON Pointer reference token: `~` and `/` escaped.
std::string Escape(std::string_view key) {
    std::string out;
    for (char c : key) {
        if (c == '~') {
            out += "~0";
        } else if (c == '/') {
            out += "~1";
        } else {
            out += c;
        }
    }
    return out;
}

// A property name the model wrote, safe to put in a message: short and plain, or a placeholder.
std::string ModelKey(std::string_view key) {
    if (key.empty() || key.size() > 64) {
        return "*";
    }
    for (char c : key) {
        const bool plain = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                           c == '-' || c == '.';
        if (!plain) {
            return "*";
        }
    }
    return std::string(key);
}

bool IsInteger(const JsonValue& v) {
    if (v.type() != Type::Number) {
        return false;
    }
    const double d = v.as_double();
    return std::isfinite(d) && d == std::floor(d);
}

bool IsCount(const JsonValue& v) { return IsInteger(v) && v.as_double() >= 0; }

const char* TypeName(const JsonValue& v) {
    switch (v.type()) {
        case Type::Null: return "null";
        case Type::Bool: return "a boolean";
        case Type::Number: return IsInteger(v) ? "an integer" : "a number";
        case Type::String: return "a string";
        case Type::Array: return "an array";
        case Type::Object: return "an object";
    }
    return "a value";
}

std::string Article(std::string_view type) {
    if (type == "null") {
        return "null";
    }
    const bool vowel = type == "object" || type == "array" || type == "integer";
    return std::string(vowel ? "an " : "a ") + std::string(type);
}

bool HasType(const JsonValue& v, std::string_view type) {
    switch (v.type()) {
        case Type::Null: return type == "null";
        case Type::Bool: return type == "boolean";
        case Type::Number: return type == "number" || (type == "integer" && IsInteger(v));
        case Type::String: return type == "string";
        case Type::Array: return type == "array";
        case Type::Object: return type == "object";
    }
    return false;
}

std::size_t CodePoints(const std::string& s) {
    std::size_t n = 0;
    for (unsigned char c : s) {
        n += (c & 0xC0) != 0x80 ? 1 : 0;
    }
    return n;
}

// ---- checking a schema ---------------------------------------------------------------------

[[noreturn]] void Bad(const std::string& at, const std::string& what) {
    throw std::invalid_argument("JSON Schema" + (at.empty() ? std::string() : " at " + at) + ": " + what);
}

const char* UnsupportedHint(std::string_view key) {
    if (key == "$ref" || key == "$defs" || key == "definitions" || key == "$id" || key == "$anchor" ||
        key == "$dynamicRef") {
        return "; references aren't supported, since draft-7 and 2020-12 spell them differently, so write the "
               "schema out";
    }
    if (key == "pattern" || key == "patternProperties") {
        return "; regular expressions aren't supported, so check the value in the tool";
    }
    if (key == "dependencies" || key == "dependentRequired" || key == "dependentSchemas" || key == "prefixItems" ||
        key == "additionalItems" || key == "unevaluatedProperties" || key == "unevaluatedItems") {
        return "; it differs between draft-7 and 2020-12";
    }
    return "";
}

void CheckSubschemas(const JsonValue& list, const std::string& at, int depth);

void Check(const JsonValue& s, const std::string& at, int depth) {
    if (depth > kMaxSchemaDepth) {
        Bad(at, "nested more than " + std::to_string(kMaxSchemaDepth) + " levels deep");
    }
    if (s.type() == Type::Bool) {
        return;
    }
    if (s.type() != Type::Object) {
        Bad(at, "a schema must be an object or a boolean");
    }
    for (const auto& [key, v] : s.as_object()) {
        const std::string here = at + "/" + Escape(key);
        if (key == "type") {
            if (v.type() == Type::String) {
                if (!IsTypeName(v.as_string())) {
                    Bad(here, "unknown type \"" + v.as_string() + "\"");
                }
            } else if (v.type() == Type::Array && !v.as_array().empty()) {
                for (std::size_t i = 0; i < v.as_array().size(); ++i) {
                    const JsonValue& t = v.as_array()[i];
                    if (t.type() != Type::String || !IsTypeName(t.as_string())) {
                        Bad(here, "every element must be a type name");
                    }
                    for (std::size_t j = 0; j < i; ++j) {
                        if (v.as_array()[j].as_string() == t.as_string()) {
                            Bad(here, "the type names must be unique");
                        }
                    }
                }
            } else {
                Bad(here, "must be a type name or a non-empty array of them");
            }
        } else if (key == "enum") {
            if (v.type() != Type::Array || v.as_array().empty()) {
                Bad(here, "must be a non-empty array");
            }
        } else if (key == "properties") {
            if (v.type() != Type::Object) {
                Bad(here, "must be an object");
            }
            for (const auto& [name, sub] : v.as_object()) {
                Check(sub, here + "/" + Escape(name), depth + 1);
            }
        } else if (key == "required") {
            if (v.type() != Type::Array) {
                Bad(here, "must be an array of property names");
            }
            for (std::size_t i = 0; i < v.as_array().size(); ++i) {
                const JsonValue& name = v.as_array()[i];
                if (name.type() != Type::String) {
                    Bad(here, "must be an array of property names");
                }
                for (std::size_t j = 0; j < i; ++j) {
                    if (v.as_array()[j].as_string() == name.as_string()) {
                        Bad(here, "the names must be unique");
                    }
                }
            }
        } else if (key == "items") {
            if (v.type() == Type::Array) {
                Bad(here, "an array of schemas means different things in draft-7 and 2020-12 (where it's "
                          "prefixItems), so give one schema for every element");
            }
            Check(v, here, depth + 1);
        } else if (key == "additionalProperties" || key == "not") {
            Check(v, here, depth + 1);
        } else if (key == "allOf" || key == "anyOf" || key == "oneOf") {
            CheckSubschemas(v, here, depth);
        } else if (key == "minItems" || key == "maxItems" || key == "minLength" || key == "maxLength" ||
                   key == "minProperties" || key == "maxProperties") {
            if (!IsCount(v)) {
                Bad(here, "must be a non-negative integer");
            }
        } else if (key == "uniqueItems") {
            if (v.type() != Type::Bool) {
                Bad(here, "must be a boolean");
            }
        } else if (key == "minimum" || key == "maximum" || key == "exclusiveMinimum" || key == "exclusiveMaximum") {
            if (v.type() != Type::Number) {
                Bad(here, v.type() == Type::Bool ? "must be a number (the draft-4 boolean form isn't supported)"
                                                 : "must be a number");
            }
        } else if (key == "multipleOf") {
            if (v.type() != Type::Number || !(v.as_double() > 0)) {
                Bad(here, "must be a number greater than 0");
            }
        } else if (key == "$schema") {
            if (v.type() != Type::String || !IsSchemaUri(v.as_string())) {
                Bad(here, "only draft-07 and 2020-12 are supported");
            }
        } else if (key == "format") {
            if (v.type() != Type::String) {
                Bad(here, "must be a string");
            }
        } else if (key == "const" || key == "title" || key == "description" || key == "default" ||
                   key == "examples" || key == "$comment") {
            // const takes any value; the rest are annotations.
        } else {
            Bad(here, "the keyword \"" + key + "\" isn't supported" + UnsupportedHint(key));
        }
    }
}

void CheckSubschemas(const JsonValue& list, const std::string& at, int depth) {
    if (list.type() != Type::Array || list.as_array().empty()) {
        Bad(at, "must be a non-empty array of schemas");
    }
    for (std::size_t i = 0; i < list.as_array().size(); ++i) {
        Check(list.as_array()[i], at + "/" + std::to_string(i), depth + 1);
    }
}

// ---- validating a value --------------------------------------------------------------------

class Validator {
public:
    explicit Validator(std::size_t max_errors) : max_errors_(max_errors) {}

    // Whether @p v matches @p s. With @p record, the reasons are added to errors() (up to the
    // limit); without, it returns at the first failure.
    bool validate(const JsonValue& s, const JsonValue& v, const std::string& path, bool record) {
        bool ok = true;
        // Records a failure; returns whether to keep looking for more.
        auto fail = [&](const std::string& where, std::string message) {
            ok = false;
            if (record && errors_.size() < max_errors_) {
                errors_.push_back({where, std::move(message)});
            }
            return record && errors_.size() < max_errors_;
        };

        if (s.type() == Type::Bool) {
            if (!s.as_bool()) {
                fail(path, "no value is allowed here");
            }
            return ok;
        }

        if (const JsonValue* type = s.find("type")) {
            bool matches = false;
            std::string expected;
            if (type->type() == Type::String) {
                matches = HasType(v, type->as_string());
                expected = Article(type->as_string());
            } else {
                for (const JsonValue& t : type->as_array()) {
                    matches = matches || HasType(v, t.as_string());
                    expected += (expected.empty() ? "" : " or ") + Article(t.as_string());
                }
            }
            if (!matches) {
                // The other keywords would only repeat the problem.
                fail(path, "must be " + expected + ", not " + TypeName(v));
                return false;
            }
        }
        if (const JsonValue* allowed = s.find("enum")) {
            bool found = false;
            for (const JsonValue& a : allowed->as_array()) {
                found = found || JsonEquals(a, v);
            }
            if (!found && !fail(path, "must be one of the values the schema's enum lists")) {
                return false;
            }
        }
        if (const JsonValue* c = s.find("const"); c != nullptr && !JsonEquals(*c, v)) {
            if (!fail(path, "must equal the schema's const value")) {
                return false;
            }
        }

        if (v.type() == Type::Object && !object_keywords(s, v, path, record, ok, fail)) {
            return false;
        }
        if (v.type() == Type::Array && !array_keywords(s, v, path, record, ok, fail)) {
            return false;
        }
        if (v.type() == Type::String) {
            const std::size_t n = CodePoints(v.as_string());
            if (const JsonValue* min = s.find("minLength"); min != nullptr && static_cast<double>(n) < min->as_double()) {
                if (!fail(path, "must be at least " + min->number_text() + " characters long")) {
                    return false;
                }
            }
            if (const JsonValue* max = s.find("maxLength"); max != nullptr && static_cast<double>(n) > max->as_double()) {
                if (!fail(path, "must be at most " + max->number_text() + " characters long")) {
                    return false;
                }
            }
        }
        if (v.type() == Type::Number && !number_keywords(s, v, path, fail)) {
            return false;
        }

        if (const JsonValue* all = s.find("allOf")) {
            for (const JsonValue& sub : all->as_array()) {
                if (!validate(sub, v, path, record)) {
                    ok = false;
                    if (!record || errors_.size() >= max_errors_) {
                        return false;
                    }
                }
            }
        }
        if (const JsonValue* any = s.find("anyOf")) {
            bool matched = false;
            for (const JsonValue& sub : any->as_array()) {
                if (validate(sub, v, path, false)) {
                    matched = true;
                    break;
                }
            }
            if (!matched && !fail(path, "must match at least one of the schema's anyOf alternatives")) {
                return false;
            }
        }
        if (const JsonValue* one = s.find("oneOf")) {
            std::size_t matched = 0;
            for (const JsonValue& sub : one->as_array()) {
                matched += validate(sub, v, path, false) ? 1 : 0;
            }
            if (matched != 1 && !fail(path, matched == 0 ? "must match one of the schema's oneOf alternatives"
                                                         : "must match exactly one of the schema's oneOf "
                                                           "alternatives, not " + std::to_string(matched))) {
                return false;
            }
        }
        if (const JsonValue* n = s.find("not"); n != nullptr && validate(*n, v, path, false)) {
            fail(path, "must not match the schema under not");
        }
        return ok;
    }

    std::vector<SchemaError> take() { return std::move(errors_); }

private:
    template <class Fail>
    bool object_keywords(const JsonValue& s, const JsonValue& v, const std::string& path, bool record, bool& ok,
                         Fail& fail) {
        const JsonValue* properties = s.find("properties");
        if (const JsonValue* required = s.find("required")) {
            for (const JsonValue& name : required->as_array()) {
                if (v.find(name.as_string()) == nullptr &&
                    !fail(path, "is missing the required property \"" + name.as_string() + "\"")) {
                    return false;
                }
            }
        }
        const JsonValue* additional = s.find("additionalProperties");
        for (const auto& [key, member] : v.as_object()) {
            const JsonValue* sub = properties != nullptr ? properties->find(key) : nullptr;
            if (sub != nullptr) {
                if (!validate(*sub, member, path + "/" + Escape(key), record)) {
                    ok = false;
                    if (!keep_going(record)) {
                        return false;
                    }
                }
            } else if (additional != nullptr) {
                const std::string where = path + "/" + Escape(ModelKey(key));
                if (additional->type() == Type::Bool && !additional->as_bool()) {
                    if (!fail(where, "is a property the schema doesn't allow")) {
                        return false;
                    }
                } else if (!validate(*additional, member, where, record)) {
                    ok = false;
                    if (!keep_going(record)) {
                        return false;
                    }
                }
            }
        }
        const std::size_t n = v.as_object().size();
        if (const JsonValue* min = s.find("minProperties"); min != nullptr && static_cast<double>(n) < min->as_double()) {
            if (!fail(path, "must have at least " + min->number_text() + " properties")) {
                return false;
            }
        }
        if (const JsonValue* max = s.find("maxProperties"); max != nullptr && static_cast<double>(n) > max->as_double()) {
            if (!fail(path, "must have at most " + max->number_text() + " properties")) {
                return false;
            }
        }
        return true;
    }

    template <class Fail>
    bool array_keywords(const JsonValue& s, const JsonValue& v, const std::string& path, bool record, bool& ok,
                        Fail& fail) {
        const JsonValue::Array& elements = v.as_array();
        if (const JsonValue* items = s.find("items")) {
            for (std::size_t i = 0; i < elements.size(); ++i) {
                if (!validate(*items, elements[i], path + "/" + std::to_string(i), record)) {
                    ok = false;
                    if (!keep_going(record)) {
                        return false;
                    }
                }
            }
        }
        if (const JsonValue* min = s.find("minItems"); min != nullptr && static_cast<double>(elements.size()) < min->as_double()) {
            if (!fail(path, "must have at least " + min->number_text() + " elements")) {
                return false;
            }
        }
        if (const JsonValue* max = s.find("maxItems"); max != nullptr && static_cast<double>(elements.size()) > max->as_double()) {
            if (!fail(path, "must have at most " + max->number_text() + " elements")) {
                return false;
            }
        }
        if (const JsonValue* unique = s.find("uniqueItems"); unique != nullptr && unique->as_bool()) {
            for (std::size_t i = 1; i < elements.size(); ++i) {
                for (std::size_t j = 0; j < i; ++j) {
                    if (JsonEquals(elements[i], elements[j])) {
                        if (!fail(path + "/" + std::to_string(i),
                                  "repeats element " + std::to_string(j) + ", and the elements must be unique")) {
                            return false;
                        }
                        j = i;  // one report per repeated element
                    }
                }
            }
        }
        return true;
    }

    template <class Fail>
    static bool number_keywords(const JsonValue& s, const JsonValue& v, const std::string& path, Fail& fail) {
        const double x = v.as_double();
        struct Bound {
            const char* keyword;
            bool (*passes)(double x, double bound);
            const char* words;
        };
        static const Bound kBounds[] = {
            {"minimum", [](double a, double b) { return a >= b; }, "at least "},
            {"maximum", [](double a, double b) { return a <= b; }, "at most "},
            {"exclusiveMinimum", [](double a, double b) { return a > b; }, "greater than "},
            {"exclusiveMaximum", [](double a, double b) { return a < b; }, "less than "},
        };
        for (const Bound& bound : kBounds) {
            const JsonValue* b = s.find(bound.keyword);
            if (b != nullptr && !bound.passes(x, b->as_double()) &&
                !fail(path, std::string("must be ") + bound.words + b->number_text())) {
                return false;
            }
        }
        if (const JsonValue* m = s.find("multipleOf")) {
            // Divides in doubles with a small tolerance, so 0.3 counts as a multiple of 0.1.
            const double q = x / m->as_double();
            const bool multiple = std::isfinite(q) && std::fabs(q - std::round(q)) <= 1e-9 * std::fmax(1.0, std::fabs(q));
            if (!multiple && !fail(path, "must be a multiple of " + m->number_text())) {
                return false;
            }
        }
        return true;
    }

    // After a nested failure: whether to keep looking for more.
    bool keep_going(bool record) const { return record && errors_.size() < max_errors_; }

    std::size_t max_errors_;
    std::vector<SchemaError> errors_;
};

// ---- strict mode ---------------------------------------------------------------------------

bool DescribesObject(const JsonValue& s) {
    if (s.find("properties") != nullptr || s.find("additionalProperties") != nullptr) {
        return true;
    }
    const JsonValue* type = s.find("type");
    if (type == nullptr) {
        return false;
    }
    if (type->type() == Type::String) {
        return type->as_string() == "object";
    }
    for (const JsonValue& t : type->as_array()) {
        if (t.as_string() == "object") {
            return true;
        }
    }
    return false;
}

std::string StrictProblem(const JsonValue& s, const std::string& at) {
    const std::string where = at.empty() ? "the schema" : at;
    if (s.type() == Type::Bool) {
        return where + " is a boolean schema, which strict mode doesn't support";
    }
    for (const auto& [key, v] : s.as_object()) {
        if (key == "minimum" || key == "maximum" || key == "exclusiveMinimum" || key == "exclusiveMaximum" ||
            key == "multipleOf" || key == "minLength" || key == "maxLength" || key == "maxItems" ||
            key == "uniqueItems" || key == "minProperties" || key == "maxProperties" || key == "oneOf" || key == "not") {
            return where + " uses " + key + ", which strict mode doesn't support";
        }
        if (key == "minItems" && v.as_double() > 1) {
            return where + " sets minItems above 1, which strict mode doesn't support";
        }
        if (key == "enum") {
            for (const JsonValue& e : v.as_array()) {
                if (e.type() == Type::Array || e.type() == Type::Object) {
                    return where + " has an enum value that is an array or an object, which strict mode doesn't "
                                   "support";
                }
            }
        }
    }
    if (DescribesObject(s)) {
        const JsonValue* additional = s.find("additionalProperties");
        if (additional == nullptr || additional->type() != Type::Bool || additional->as_bool()) {
            return where + " describes an object without \"additionalProperties\": false, which strict mode needs";
        }
    }
    if (const JsonValue* properties = s.find("properties")) {
        for (const auto& [name, sub] : properties->as_object()) {
            std::string problem = StrictProblem(sub, at + "/properties/" + Escape(name));
            if (!problem.empty()) {
                return problem;
            }
        }
    }
    if (const JsonValue* items = s.find("items")) {
        std::string problem = StrictProblem(*items, at + "/items");
        if (!problem.empty()) {
            return problem;
        }
    }
    for (const char* key : {"allOf", "anyOf"}) {
        if (const JsonValue* list = s.find(key)) {
            for (std::size_t i = 0; i < list->as_array().size(); ++i) {
                std::string problem =
                    StrictProblem(list->as_array()[i], at + "/" + key + "/" + std::to_string(i));
                if (!problem.empty()) {
                    return problem;
                }
            }
        }
    }
    return "";
}

}  // namespace

void CheckSchema(const JsonValue& schema) { Check(schema, "", 0); }

std::vector<SchemaError> ValidateJson(const JsonValue& schema, const JsonValue& value, std::size_t max_errors) {
    CheckSchema(schema);
    Validator validator(max_errors == 0 ? 1 : max_errors);
    validator.validate(schema, value, "", true);
    return validator.take();
}

std::string StrictModeProblem(const JsonValue& schema) {
    CheckSchema(schema);
    return StrictProblem(schema, "");
}

bool JsonEquals(const JsonValue& a, const JsonValue& b) {
    if (a.type() != b.type()) {
        return false;
    }
    switch (a.type()) {
        case Type::Null: return true;
        case Type::Bool: return a.as_bool() == b.as_bool();
        case Type::Number: return a.number_text() == b.number_text() || a.as_double() == b.as_double();
        case Type::String: return a.as_string() == b.as_string();
        case Type::Array: {
            const JsonValue::Array& x = a.as_array();
            const JsonValue::Array& y = b.as_array();
            if (x.size() != y.size()) {
                return false;
            }
            for (std::size_t i = 0; i < x.size(); ++i) {
                if (!JsonEquals(x[i], y[i])) {
                    return false;
                }
            }
            return true;
        }
        case Type::Object: {
            if (a.as_object().size() != b.as_object().size()) {
                return false;
            }
            for (const auto& [key, member] : a.as_object()) {
                const JsonValue* other = b.find(key);
                if (other == nullptr || !JsonEquals(member, *other)) {
                    return false;
                }
            }
            return true;
        }
    }
    return false;
}

}  // namespace pulsatrix::orch
