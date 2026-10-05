// Shared reading and writing machinery for pulsatrix.<kind>.v<N> documents: schema checks, the
// "nonfinite" JSON Pointer convention, and readers whose errors name the problem's pointer.
// Private to the library (src/), used by viz/document.cpp and benchmark.cpp.
#pragma once

#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/viz/document.hpp"

namespace pulsatrix::document_io {


inline std::string SchemaFor(std::string_view kind) {
    return "pulsatrix." + std::string(kind) + ".v" + std::to_string(kVizDocumentVersion);
}

// Splits "pulsatrix.<kind>.v<N>" and checks N; returns the kind.
inline std::string KindFromSchema(const std::string& schema) {
    const std::string prefix = "pulsatrix.";
    size_t v = schema.rfind(".v");
    bool well_formed = schema.compare(0, prefix.size(), prefix) == 0 && v != std::string::npos && v > prefix.size() &&
                       v + 2 < schema.size();
    if (well_formed) {
        for (size_t i = prefix.size(); i < v; ++i) {
            char c = schema[i];
            well_formed = well_formed && ((c >= 'a' && c <= 'z') || c == '_');
        }
        for (size_t i = v + 2; i < schema.size(); ++i) {
            well_formed = well_formed && schema[i] >= '0' && schema[i] <= '9';
        }
    }
    if (!well_formed) {
        throw std::invalid_argument("\"" + schema + "\" is not a pulsatrix document schema (pulsatrix.<kind>.v<N>)");
    }
    if (schema.substr(v + 2) != std::to_string(kVizDocumentVersion)) {
        throw std::invalid_argument("unsupported document version: " + schema + " (this build reads v" +
                                    std::to_string(kVizDocumentVersion) + ")");
    }
    return schema.substr(prefix.size(), v - prefix.size());
}

inline std::string KindOf(const JsonValue& root) {
    if (root.type() != JsonValue::Type::Object) {
        throw std::invalid_argument("a pulsatrix document must be a JSON object");
    }
    const JsonValue* schema = root.find("schema");
    if (schema == nullptr || schema->type() != JsonValue::Type::String) {
        throw std::invalid_argument("a pulsatrix document needs a string \"schema\" member");
    }
    return KindFromSchema(schema->as_string());
}

inline std::string Index(const std::string& pointer, size_t i) { return pointer + "/" + std::to_string(i); }

inline const char* NonFiniteName(double v) {
    if (std::isnan(v)) {
        return "nan";
    }
    return v > 0 ? "inf" : "-inf";
}

// Builds one document. Numbers go through num()/num_double(), which write a non-finite value
// as null and remember its JSON Pointer for the "nonfinite" member.
class DocWriter {
public:
    explicit DocWriter(std::string_view kind) : root_(JsonValue::Object{}) { root_.add("schema", SchemaFor(kind)); }

    JsonValue& root() { return root_; }

    JsonValue num(float v, const std::string& pointer) {
        if (std::isfinite(v)) {
            return JsonValue::Float(v);
        }
        nonfinite_.add(pointer, NonFiniteName(v));
        return JsonValue();
    }

    JsonValue num_double(double v, const std::string& pointer) {
        if (std::isfinite(v)) {
            return JsonValue(v);
        }
        nonfinite_.add(pointer, NonFiniteName(v));
        return JsonValue();
    }

    JsonValue floats(const std::vector<float>& values, const std::string& pointer) {
        JsonValue a{JsonValue::Array{}};
        for (size_t i = 0; i < values.size(); ++i) {
            a.push_back(num(values[i], Index(pointer, i)));
        }
        return a;
    }

    std::string finish() {
        if (!nonfinite_.as_object().empty()) {
            root_.add("nonfinite", std::move(nonfinite_));
        }
        return WriteJson(root_);
    }

private:
    JsonValue root_;
    JsonValue nonfinite_{JsonValue::Object{}};
};

inline JsonValue Ints(const std::vector<int64_t>& values) {
    JsonValue a{JsonValue::Array{}};
    for (int64_t v : values) {
        a.push_back(JsonValue(v));
    }
    return a;
}

inline JsonValue Strings(const std::vector<std::string>& values) {
    JsonValue a{JsonValue::Array{}};
    for (const std::string& v : values) {
        a.push_back(JsonValue(v));
    }
    return a;
}

// Reads one document of a known kind. Every error names the JSON Pointer of the problem.
class DocReader {
public:
    DocReader(std::string_view json, std::string_view kind) : schema_(SchemaFor(kind)) {
        try {
            root_ = ParseJson(json);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(schema_ + " document: " + e.what());
        }
        std::string found;
        try {
            found = KindOf(root_);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(schema_ + " document: " + e.what());
        }
        if (found != kind) {
            throw std::invalid_argument("expected a " + schema_ + " document, got " + SchemaFor(found));
        }
        if (const JsonValue* nf = root_.find("nonfinite")) {
            if (nf->type() != JsonValue::Type::Object) {
                fail("/nonfinite", "expected an object");
            }
            for (const auto& [pointer, value] : nf->as_object()) {
                if (value.type() != JsonValue::Type::String ||
                    (value.as_string() != "nan" && value.as_string() != "inf" && value.as_string() != "-inf")) {
                    fail("/nonfinite", "the entry for " + pointer + " must be \"nan\", \"inf\" or \"-inf\"");
                }
                nonfinite_.emplace(pointer, value.as_string());
            }
        }
    }

    const JsonValue& root() const { return root_; }

    [[noreturn]] void fail(const std::string& pointer, const std::string& what) const {
        throw std::invalid_argument(schema_ + " document: " + (pointer.empty() ? "/" : pointer) + ": " + what);
    }

    const JsonValue& member(const JsonValue& object, const std::string& object_pointer, const char* key) const {
        const JsonValue* v = object.find(key);
        if (v == nullptr) {
            fail(object_pointer + "/" + key, "missing");
        }
        return *v;
    }

    const JsonValue::Object& object(const JsonValue& v, const std::string& pointer) const {
        if (v.type() != JsonValue::Type::Object) {
            fail(pointer, "expected an object");
        }
        return v.as_object();
    }

    const JsonValue::Array& array(const JsonValue& v, const std::string& pointer) const {
        if (v.type() != JsonValue::Type::Array) {
            fail(pointer, "expected an array");
        }
        return v.as_array();
    }

    bool boolean(const JsonValue& v, const std::string& pointer) const {
        if (v.type() != JsonValue::Type::Bool) {
            fail(pointer, "expected true or false");
        }
        return v.as_bool();
    }

    const std::string& string(const JsonValue& v, const std::string& pointer) const {
        if (v.type() != JsonValue::Type::String) {
            fail(pointer, "expected a string");
        }
        return v.as_string();
    }

    int64_t integer(const JsonValue& v, const std::string& pointer, int64_t min = std::numeric_limits<int64_t>::min(),
                    int64_t max = std::numeric_limits<int64_t>::max()) const {
        if (v.type() != JsonValue::Type::Number) {
            fail(pointer, "expected an integer");
        }
        int64_t i = 0;
        try {
            i = v.as_int64();
        } catch (const std::invalid_argument&) {
            fail(pointer, "expected an integer, got " + v.number_text());
        }
        if (i < min || i > max) {
            fail(pointer, std::to_string(i) + " is out of range [" + std::to_string(min) + ", " + std::to_string(max) + "]");
        }
        return i;
    }

    float number(const JsonValue& v, const std::string& pointer) {
        if (v.is_null()) {
            return static_cast<float>(non_finite(pointer));
        }
        if (v.type() != JsonValue::Type::Number) {
            fail(pointer, "expected a number");
        }
        try {
            return v.as_float();
        } catch (const std::invalid_argument& e) {
            fail(pointer, e.what());
        }
    }

    double number_double(const JsonValue& v, const std::string& pointer) {
        if (v.is_null()) {
            return non_finite(pointer);
        }
        if (v.type() != JsonValue::Type::Number) {
            fail(pointer, "expected a number");
        }
        try {
            return v.as_double();
        } catch (const std::invalid_argument& e) {
            fail(pointer, e.what());
        }
    }

    std::vector<float> floats(const JsonValue& v, const std::string& pointer) {
        const JsonValue::Array& a = array(v, pointer);
        std::vector<float> out;
        out.reserve(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            out.push_back(number(a[i], Index(pointer, i)));
        }
        return out;
    }

    std::vector<int64_t> ints(const JsonValue& v, const std::string& pointer, int64_t min, int64_t max) const {
        const JsonValue::Array& a = array(v, pointer);
        std::vector<int64_t> out;
        out.reserve(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            out.push_back(integer(a[i], Index(pointer, i), min, max));
        }
        return out;
    }

    std::vector<std::string> strings(const JsonValue& v, const std::string& pointer) const {
        const JsonValue::Array& a = array(v, pointer);
        std::vector<std::string> out;
        out.reserve(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            out.push_back(string(a[i], Index(pointer, i)));
        }
        return out;
    }

    // Every "nonfinite" entry must have explained a null number the reader read; one that
    // didn't points at nothing, or at a number that isn't null, so the document is corrupt.
    void finish() const {
        for (const auto& [pointer, value] : nonfinite_) {
            if (used_.count(pointer) == 0) {
                fail(pointer, "has a \"nonfinite\" entry but is not a null number");
            }
        }
    }

private:
    std::string schema_;
    JsonValue root_;
    std::map<std::string, std::string> nonfinite_;
    std::set<std::string> used_;

    double non_finite(const std::string& pointer) {
        auto it = nonfinite_.find(pointer);
        if (it == nonfinite_.end()) {
            fail(pointer, "null number with no \"nonfinite\" entry");
        }
        used_.insert(pointer);
        if (it->second == "nan") {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return it->second == "inf" ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }
};

// rows * cols or a shape's product, refusing overflow; returns false if it exceeds @p limit.
inline bool ProductAtMost(const std::vector<int64_t>& dims, size_t limit, size_t& product) {
    product = 1;
    for (int64_t d : dims) {
        if (d < 0) {
            return false;
        }
        if (d != 0 && product > limit / static_cast<size_t>(d)) {
            return false;
        }
        product *= static_cast<size_t>(d);
    }
    return true;
}

// Write-side checks share their wording with the reader's: "<field>: <problem>".
[[noreturn]] inline void Invalid(std::string_view kind, const std::string& what) {
    throw std::invalid_argument("invalid " + SchemaFor(kind) + " document: " + what);
}

inline constexpr int64_t kMaxIndex = std::numeric_limits<int64_t>::max();


}  // namespace pulsatrix::document_io
