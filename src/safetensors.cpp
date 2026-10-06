#include "pulsatrix/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

// The format's data section is little-endian, and this file copies it as-is. GCC and Clang say
// which byte order they target; MSVC doesn't, but every Windows target is little-endian.
#if defined(__BYTE_ORDER__)
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "safetensors.cpp assumes a little-endian host"
#endif
#elif !defined(_WIN32)
#error "safetensors.cpp can't determine the host byte order; it assumes little-endian"
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pulsatrix {
namespace {

// The reference implementation's cap on the JSON header.
constexpr uint64_t kMaxHeaderBytes = 100000000;

[[noreturn]] void reject(const std::string& why) { throw std::invalid_argument("safetensors: " + why); }

struct DtypeName {
    const char* name;
    SafetensorsDtype dtype;
    uint64_t size;
};

constexpr DtypeName kDtypes[] = {
    {"BOOL", SafetensorsDtype::Bool, 1},       {"U8", SafetensorsDtype::U8, 1},
    {"I8", SafetensorsDtype::I8, 1},           {"I16", SafetensorsDtype::I16, 2},
    {"U16", SafetensorsDtype::U16, 2},         {"I32", SafetensorsDtype::I32, 4},
    {"U32", SafetensorsDtype::U32, 4},         {"I64", SafetensorsDtype::I64, 8},
    {"U64", SafetensorsDtype::U64, 8},         {"F8_E4M3", SafetensorsDtype::F8_E4M3, 1},
    {"F8_E5M2", SafetensorsDtype::F8_E5M2, 1}, {"F16", SafetensorsDtype::F16, 2},
    {"BF16", SafetensorsDtype::BF16, 2},       {"F32", SafetensorsDtype::F32, 4},
    {"F64", SafetensorsDtype::F64, 8},
};

uint64_t dtype_size(SafetensorsDtype dtype) {
    for (const DtypeName& d : kDtypes) {
        if (d.dtype == dtype) {
            return d.size;
        }
    }
    return 0;
}

// Overflow-checked a * b.
bool mul_overflows(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return true;
    }
    out = a * b;
    return false;
}

bool valid_utf8(const std::string& s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const auto c = static_cast<unsigned char>(s[i]);
        size_t len;
        uint32_t cp;
        if (c < 0x80) {
            ++i;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
            cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
            cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
            cp = c & 0x07;
        } else {
            return false;
        }
        if (i + len > n) {
            return false;
        }
        for (size_t k = 1; k < len; ++k) {
            const auto cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) {
                return false;
            }
            cp = (cp << 6) | (cc & 0x3F);
        }
        // Overlong encodings, surrogates and code points past U+10FFFF are invalid.
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) ||
            (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
            return false;
        }
        i += len;
    }
    return true;
}

void append_utf8(std::string& out, uint32_t cp) {
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

// A strict reader for the JSON subset a safetensors header uses: objects, arrays, strings and
// non-negative integers. Booleans, null, floats, exponents and leading zeros are rejected, as
// are duplicate keys. Every read is bounds-checked against the header's own length.
class HeaderParser {
public:
    explicit HeaderParser(const std::string& text) : s_(text) {}

    void parse(std::map<std::string, SafetensorsTensorInfo>& infos, std::map<std::string, std::string>& metadata) {
        skip_ws();
        if (pos_ >= s_.size() || s_[pos_] != '{') {
            reject("header must be a JSON object");
        }
        expect('{');
        std::set<std::string> seen;
        skip_ws();
        if (peek() != '}') {
            while (true) {
                skip_ws();
                std::string key = string_value();
                if (!seen.insert(key).second) {
                    reject("duplicate key \"" + key + "\"");
                }
                skip_ws();
                expect(':');
                skip_ws();
                if (key == "__metadata__") {
                    metadata = string_map();
                } else {
                    infos.emplace(key, tensor_entry());
                }
                skip_ws();
                if (peek() == ',') {
                    ++pos_;
                    continue;
                }
                break;
            }
        }
        expect('}');
        // Trailing whitespace is padding; anything else is not.
        skip_ws();
        if (pos_ != s_.size()) {
            reject("unexpected data after the header object");
        }
    }

private:
    char peek() const {
        if (pos_ >= s_.size()) {
            reject("header ends early");
        }
        return s_[pos_];
    }

    void expect(char c) {
        if (peek() != c) {
            reject(std::string("expected '") + c + "' in header");
        }
        ++pos_;
    }

    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) {
            ++pos_;
        }
    }

    uint32_t hex4() {
        if (pos_ + 4 > s_.size()) {
            reject("truncated \\u escape");
        }
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s_[pos_++];
            v <<= 4;
            if (c >= '0' && c <= '9') {
                v |= static_cast<uint32_t>(c - '0');
            } else if (c >= 'a' && c <= 'f') {
                v |= static_cast<uint32_t>(c - 'a' + 10);
            } else if (c >= 'A' && c <= 'F') {
                v |= static_cast<uint32_t>(c - 'A' + 10);
            } else {
                reject("bad \\u escape");
            }
        }
        return v;
    }

    std::string string_value() {
        expect('"');
        std::string out;
        while (true) {
            const char c = peek();
            ++pos_;
            if (c == '"') {
                break;
            }
            if (static_cast<unsigned char>(c) < 0x20) {
                reject("unescaped control character in a string");
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            const char e = peek();
            ++pos_;
            switch (e) {
                case '"':
                case '\\':
                case '/':
                    out += e;
                    break;
                case 'b':
                    out += '\b';
                    break;
                case 'f':
                    out += '\f';
                    break;
                case 'n':
                    out += '\n';
                    break;
                case 'r':
                    out += '\r';
                    break;
                case 't':
                    out += '\t';
                    break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        reject("unpaired low surrogate");
                    }
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (pos_ + 2 > s_.size() || s_[pos_] != '\\' || s_[pos_ + 1] != 'u') {
                            reject("unpaired high surrogate");
                        }
                        pos_ += 2;
                        const uint32_t low = hex4();
                        if (low < 0xDC00 || low > 0xDFFF) {
                            reject("unpaired high surrogate");
                        }
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    append_utf8(out, cp);
                    break;
                }
                default:
                    reject("bad escape in a string");
            }
        }
        if (!valid_utf8(out)) {
            reject("string is not valid UTF-8");
        }
        return out;
    }

    // A non-negative integer without sign, fraction, exponent or leading zero, that fits uint64.
    uint64_t integer_value() {
        const char first = peek();
        if (first < '0' || first > '9') {
            reject("expected a non-negative integer");
        }
        uint64_t v = 0;
        const size_t start = pos_;
        while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') {
            const auto digit = static_cast<uint64_t>(s_[pos_] - '0');
            if (v > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
                reject("integer overflows 64 bits");
            }
            v = v * 10 + digit;
            ++pos_;
        }
        if (s_[start] == '0' && pos_ - start > 1) {
            reject("integer with a leading zero");
        }
        if (pos_ < s_.size() && (s_[pos_] == '.' || s_[pos_] == 'e' || s_[pos_] == 'E')) {
            reject("expected an integer, got a number with a fraction or exponent");
        }
        return v;
    }

    std::vector<uint64_t> integer_array() {
        expect('[');
        std::vector<uint64_t> out;
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return out;
        }
        while (true) {
            skip_ws();
            out.push_back(integer_value());
            skip_ws();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect(']');
            return out;
        }
    }

    std::map<std::string, std::string> string_map() {
        expect('{');
        std::map<std::string, std::string> out;
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return out;
        }
        while (true) {
            skip_ws();
            std::string key = string_value();
            skip_ws();
            expect(':');
            skip_ws();
            if (peek() != '"') {
                reject("__metadata__ values must be strings");
            }
            std::string value = string_value();
            if (!out.emplace(std::move(key), std::move(value)).second) {
                reject("duplicate __metadata__ key");
            }
            skip_ws();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            return out;
        }
    }

    SafetensorsTensorInfo tensor_entry() {
        if (peek() != '{') {
            reject("a tensor entry must be an object");
        }
        expect('{');
        bool has_dtype = false, has_shape = false, has_offsets = false;
        SafetensorsTensorInfo info{SafetensorsDtype::F32, {}, 0, 0};
        while (true) {
            skip_ws();
            const std::string field = string_value();
            skip_ws();
            expect(':');
            skip_ws();
            if (field == "dtype") {
                if (has_dtype) {
                    reject("duplicate field \"dtype\"");
                }
                has_dtype = true;
                const std::string name = string_value();
                const DtypeName* found = nullptr;
                for (const DtypeName& d : kDtypes) {
                    if (name == d.name) {
                        found = &d;
                    }
                }
                if (found == nullptr) {
                    reject("unsupported dtype \"" + name + "\"");
                }
                info.dtype = found->dtype;
            } else if (field == "shape") {
                if (has_shape) {
                    reject("duplicate field \"shape\"");
                }
                has_shape = true;
                if (peek() != '[') {
                    reject("shape must be an array");
                }
                for (uint64_t d : integer_array()) {
                    if (d > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                        reject("dimension too large");
                    }
                    info.shape.push_back(static_cast<int64_t>(d));
                }
            } else if (field == "data_offsets") {
                if (has_offsets) {
                    reject("duplicate field \"data_offsets\"");
                }
                has_offsets = true;
                if (peek() != '[') {
                    reject("data_offsets must be an array");
                }
                const std::vector<uint64_t> offsets = integer_array();
                if (offsets.size() != 2) {
                    reject("data_offsets must have exactly two entries");
                }
                info.data_begin = offsets[0];
                info.data_end = offsets[1];
            } else {
                reject("unknown tensor field \"" + field + "\"");
            }
            skip_ws();
            if (peek() == ',') {
                ++pos_;
                continue;
            }
            expect('}');
            break;
        }
        if (!has_dtype || !has_shape || !has_offsets) {
            reject("a tensor entry needs dtype, shape and data_offsets");
        }
        return info;
    }

    const std::string& s_;
    size_t pos_ = 0;
};

std::string json_string(const std::string& s) {
    static const char* kHex = "0123456789abcdef";
    std::string out = "\"";
    for (const char c : s) {
        const auto u = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (u < 0x20) {
            out += "\\u00";
            out += kHex[u >> 4];
            out += kHex[u & 0xF];
        } else {
            out += c;
        }
    }
    return out + "\"";
}

std::string shape_json(const Shape& shape) {
    std::string out = "[";
    for (int64_t d = 0; d < shape.rank(); ++d) {
        out += (d ? "," : "") + std::to_string(shape.dim(d));
    }
    return out + "]";
}

}  // namespace

SafetensorsFile SafetensorsFile::FromStorage(std::shared_ptr<const Storage> storage) {
    const uint8_t* const raw = storage->data;
    const size_t total = storage->size;
    struct View {
        const uint8_t* p;
        size_t n;
        size_t size() const { return n; }
        const uint8_t* data() const { return p; }
        uint8_t operator[](size_t i) const { return p[i]; }
    } bytes{raw, total};
    if (bytes.size() < 8) {
        reject("file shorter than its 8-byte header length");
    }
    uint64_t header_len = 0;
    for (int i = 0; i < 8; ++i) {
        header_len |= static_cast<uint64_t>(bytes[static_cast<size_t>(i)]) << (8 * i);
    }
    if (header_len == 0) {
        reject("empty header");
    }
    if (header_len > kMaxHeaderBytes) {
        reject("header longer than 100 MB");
    }
    if (header_len > bytes.size() - 8) {
        reject("header length runs past the end of the file");
    }

    SafetensorsFile f;
    const std::string header(reinterpret_cast<const char*>(bytes.data()) + 8, static_cast<size_t>(header_len));
    if (!valid_utf8(header)) {
        reject("header is not valid UTF-8");
    }
    HeaderParser(header).parse(f.infos_, f.metadata_);

    f.data_start_ = 8 + static_cast<size_t>(header_len);
    const uint64_t data_size = bytes.size() - f.data_start_;

    // Each range must be exactly numel * element size, and inside the data section.
    std::vector<std::pair<const std::string*, const SafetensorsTensorInfo*>> ordered;
    for (const auto& [name, info] : f.infos_) {
        if (info.data_begin > info.data_end) {
            reject("tensor \"" + name + "\" has data_offsets begin > end");
        }
        if (info.data_end > data_size) {
            reject("tensor \"" + name + "\" runs past the end of the data");
        }
        uint64_t numel = 1;
        for (int64_t d : info.shape) {
            if (mul_overflows(numel, static_cast<uint64_t>(d), numel)) {
                reject("tensor \"" + name + "\" has an element count that overflows");
            }
        }
        uint64_t nbytes = 0;
        if (mul_overflows(numel, dtype_size(info.dtype), nbytes)) {
            reject("tensor \"" + name + "\" has a byte size that overflows");
        }
        if (info.data_end - info.data_begin != nbytes) {
            reject("tensor \"" + name + "\" byte range doesn't match its shape and dtype");
        }
        ordered.emplace_back(&name, &info);
    }

    // The ranges must tile the data section exactly: no overlaps, no holes, nothing left over.
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
        if (a.second->data_begin != b.second->data_begin) {
            return a.second->data_begin < b.second->data_begin;
        }
        return a.second->data_end < b.second->data_end;
    });
    uint64_t cursor = 0;
    for (const auto& [name, info] : ordered) {
        if (info->data_begin != cursor) {
            reject(info->data_begin < cursor ? "tensor \"" + *name + "\" overlaps another"
                                             : "hole in the data section before tensor \"" + *name + "\"");
        }
        cursor = info->data_end;
        f.names_.push_back(*name);
    }
    if (cursor != data_size) {
        reject("data section has bytes no tensor indexes");
    }

    f.storage_ = std::move(storage);
    return f;
}

SafetensorsFile SafetensorsFile::Parse(std::vector<uint8_t> bytes) {
    struct Owned : Storage {
        std::vector<uint8_t> bytes;
    };
    auto owned = std::make_shared<Owned>();
    owned->bytes = std::move(bytes);
    owned->data = owned->bytes.data();
    owned->size = owned->bytes.size();
    return FromStorage(std::move(owned));
}

SafetensorsFile SafetensorsFile::Read(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("safetensors: cannot open " + path);
    }
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) {
        throw std::runtime_error("safetensors: cannot read " + path);
    }
    return Parse(std::move(bytes));
}

SafetensorsFile SafetensorsFile::Map(const std::string& path) {
#ifdef _WIN32
    struct Mapped : Storage {
        HANDLE file = INVALID_HANDLE_VALUE;
        HANDLE mapping = nullptr;
        ~Mapped() override {
            if (data != nullptr) UnmapViewOfFile(data);
            if (mapping != nullptr) CloseHandle(mapping);
            if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        }
    };
    auto m = std::make_shared<Mapped>();
    m->file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                          nullptr);
    if (m->file == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("safetensors: cannot open " + path);
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(m->file, &size)) {
        throw std::runtime_error("safetensors: cannot read the size of " + path);
    }
    if (size.QuadPart == 0) {
        return Parse({});  // an empty file can't be mapped; Parse rejects it
    }
    m->mapping = CreateFileMappingA(m->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (m->mapping == nullptr) {
        throw std::runtime_error("safetensors: cannot map " + path);
    }
    m->data = static_cast<const uint8_t*>(MapViewOfFile(m->mapping, FILE_MAP_READ, 0, 0, 0));
    if (m->data == nullptr) {
        throw std::runtime_error("safetensors: cannot map " + path);
    }
    m->size = static_cast<size_t>(size.QuadPart);
    return FromStorage(std::move(m));
#else
    struct Mapped : Storage {
        ~Mapped() override {
            if (data != nullptr) munmap(const_cast<uint8_t*>(data), size);
        }
    };
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("safetensors: cannot open " + path);
    }
    struct stat st {};
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("safetensors: cannot read the size of " + path);
    }
    if (st.st_size == 0) {
        close(fd);
        return Parse({});  // an empty file can't be mapped; Parse rejects it
    }
    void* p = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);  // the mapping keeps the file open
    if (p == MAP_FAILED) {
        throw std::runtime_error("safetensors: cannot map " + path);
    }
    auto m = std::make_shared<Mapped>();
    m->data = static_cast<const uint8_t*>(p);
    m->size = static_cast<size_t>(st.st_size);
    return FromStorage(std::move(m));
#endif
}

const SafetensorsTensorInfo& SafetensorsFile::info(const std::string& name) const {
    auto it = infos_.find(name);
    if (it == infos_.end()) {
        throw std::invalid_argument("safetensors: no tensor named \"" + name + "\"");
    }
    return it->second;
}

std::pair<const uint8_t*, size_t> SafetensorsFile::bytes(const std::string& name) const {
    const SafetensorsTensorInfo& i = info(name);
    return {storage_->data + data_start_ + i.data_begin, static_cast<size_t>(i.data_end - i.data_begin)};
}

namespace {

float FromBits(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// An IEEE-style binary float with the given exponent and mantissa widths, widened exactly to
// float. ieee_specials: an all-ones exponent means infinity or NaN (F16, F8_E5M2); otherwise only
// the all-ones pattern is NaN and the rest are ordinary numbers (F8_E4M3, the "fn" variant).
float WidenFloat(uint32_t v, int exp_bits, int man_bits, bool ieee_specials) {
    const uint32_t sign = (v >> (exp_bits + man_bits)) & 1u;
    const uint32_t exp = (v >> man_bits) & ((1u << exp_bits) - 1u);
    const uint32_t man = v & ((1u << man_bits) - 1u);
    const int bias = (1 << (exp_bits - 1)) - 1;
    const float s = sign ? -1.0f : 1.0f;
    const uint32_t exp_max = (1u << exp_bits) - 1u;
    if (ieee_specials && exp == exp_max) {
        return man == 0 ? s * std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN();
    }
    if (!ieee_specials && exp == exp_max && man == (1u << man_bits) - 1u) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    if (exp == 0) {  // zero or subnormal: man * 2^(1 - bias - man_bits), exact in float
        return s * std::ldexp(static_cast<float>(man), 1 - bias - man_bits);
    }
    return s * std::ldexp(static_cast<float>((1u << man_bits) | man), static_cast<int>(exp) - bias - man_bits);
}

}  // namespace

Tensor SafetensorsFile::tensor(const std::string& name, DeviceBackend* backend) const {
    const SafetensorsTensorInfo& i = info(name);
    auto [ptr, size] = bytes(name);
    std::vector<float> values;
    auto read = [&](size_t k, size_t width) {
        uint64_t v = 0;
        for (size_t b = 0; b < width; ++b) v |= static_cast<uint64_t>(ptr[k * width + b]) << (8 * b);  // little-endian
        return v;
    };
    switch (i.dtype) {
        case SafetensorsDtype::F32:
            values.resize(size / 4);
            for (size_t k = 0; k < values.size(); ++k) values[k] = FromBits(static_cast<uint32_t>(read(k, 4)));
            break;
        case SafetensorsDtype::BF16:  // the top half of a float32: exact
            values.resize(size / 2);
            for (size_t k = 0; k < values.size(); ++k) values[k] = FromBits(static_cast<uint32_t>(read(k, 2)) << 16);
            break;
        case SafetensorsDtype::F16:
            values.resize(size / 2);
            for (size_t k = 0; k < values.size(); ++k) values[k] = WidenFloat(static_cast<uint32_t>(read(k, 2)), 5, 10, true);
            break;
        case SafetensorsDtype::F8_E4M3:
            values.resize(size);
            for (size_t k = 0; k < values.size(); ++k) values[k] = WidenFloat(ptr[k], 4, 3, false);
            break;
        case SafetensorsDtype::F8_E5M2:
            values.resize(size);
            for (size_t k = 0; k < values.size(); ++k) values[k] = WidenFloat(ptr[k], 5, 2, true);
            break;
        case SafetensorsDtype::F64:  // rounded to nearest; out-of-range values become infinities
            values.resize(size / 8);
            for (size_t k = 0; k < values.size(); ++k) {
                const uint64_t bits = read(k, 8);
                double d;
                std::memcpy(&d, &bits, sizeof d);
                values[k] = static_cast<float>(d);
            }
            break;
        default:
            throw std::invalid_argument("safetensors: tensor \"" + name +
                                        "\" is an integer or bool tensor; only floating-point dtypes convert to a Tensor");
    }
    return Tensor(Shape(i.shape), backend, values);
}

std::vector<uint8_t> SerializeSafetensors(const std::vector<std::pair<std::string, const Tensor*>>& tensors,
                                          const std::map<std::string, std::string>& metadata) {
    std::set<std::string> seen;
    std::string header = "{";
    if (!metadata.empty()) {
        header += "\"__metadata__\":{";
        bool first = true;
        for (const auto& [k, v] : metadata) {
            if (!valid_utf8(k) || !valid_utf8(v)) {
                throw std::invalid_argument("safetensors: metadata must be valid UTF-8");
            }
            header += (first ? "" : ",") + json_string(k) + ":" + json_string(v);
            first = false;
        }
        header += "}";
    }
    std::vector<uint8_t> data;
    for (const auto& [name, t] : tensors) {
        if (name == "__metadata__") {
            throw std::invalid_argument("safetensors: \"__metadata__\" is reserved and can't name a tensor");
        }
        if (!seen.insert(name).second) {
            throw std::invalid_argument("safetensors: duplicate tensor name \"" + name + "\"");
        }
        if (!valid_utf8(name)) {
            throw std::invalid_argument("safetensors: tensor names must be valid UTF-8");
        }
        const size_t begin = data.size();
        const auto nbytes = static_cast<size_t>(t->numel()) * sizeof(float);
        data.resize(begin + nbytes);
        if (nbytes > 0) {
            t->backend()->copy(data.data() + begin, t->data(), nbytes,
                               t->device() == DeviceType::Cpu ? CopyDirection::HostToHost : CopyDirection::DeviceToHost);
        }
        header += (header.size() > 1 ? "," : "") + json_string(name) + ":{\"dtype\":\"F32\",\"shape\":" +
                  shape_json(t->shape()) + ",\"data_offsets\":[" + std::to_string(begin) + "," +
                  std::to_string(data.size()) + "]}";
    }
    header += "}";
    // Pad with spaces so the data section starts on an 8-byte boundary.
    while ((8 + header.size()) % 8 != 0) {
        header += ' ';
    }

    std::vector<uint8_t> out(8);
    const uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) {
        out[static_cast<size_t>(i)] = static_cast<uint8_t>(n >> (8 * i));
    }
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

void WriteSafetensors(const std::string& path, const std::vector<std::pair<std::string, const Tensor*>>& tensors,
                      const std::map<std::string, std::string>& metadata) {
    const std::vector<uint8_t> bytes = SerializeSafetensors(tensors, metadata);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("safetensors: cannot open " + path + " for writing");
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!out) {
        throw std::runtime_error("safetensors: cannot write " + path);
    }
}

}  // namespace pulsatrix
