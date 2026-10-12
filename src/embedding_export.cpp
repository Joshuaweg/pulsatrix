#include "pulsatrix/embedding_export.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pulsatrix {
namespace {

constexpr char kMagic[] = "\x93NUMPY";
constexpr std::size_t kMagicSize = 6;
constexpr std::size_t kMaxRank = 8;

[[noreturn]] void Bad(const std::string& why) { throw std::invalid_argument("ParseNpy: " + why); }

// The header's Python dict literal, as numpy writes it: {'descr': '<f4', 'fortran_order': False,
// 'shape': (3, 4), }. Only strings, booleans and tuples of non-negative integers are accepted.
struct Header {
    std::string descr;
    bool fortran_order = false;
    std::vector<int64_t> shape;
    bool has_descr = false, has_order = false, has_shape = false;
};

class HeaderParser {
public:
    explicit HeaderParser(const std::string& text) : s_(text) {}

    Header Parse() {
        Header h;
        Expect('{');
        while (true) {
            Skip();
            if (Peek() == '}') {
                ++i_;
                break;
            }
            const std::string key = String();
            Expect(':');
            Skip();
            if (key == "descr") {
                if (h.has_descr) Bad("the header repeats 'descr'");
                h.descr = String();
                h.has_descr = true;
            } else if (key == "fortran_order") {
                if (h.has_order) Bad("the header repeats 'fortran_order'");
                h.fortran_order = Bool();
                h.has_order = true;
            } else if (key == "shape") {
                if (h.has_shape) Bad("the header repeats 'shape'");
                h.shape = Tuple();
                h.has_shape = true;
            } else {
                Bad("the header has an unknown key '" + key + "'");
            }
            Skip();
            if (Peek() == ',') {
                ++i_;
            } else if (Peek() != '}') {
                Bad("expected ',' or '}' in the header");
            }
        }
        Skip();
        if (i_ != s_.size()) Bad("the header has text after its dict");
        if (!h.has_descr || !h.has_order || !h.has_shape) Bad("the header needs 'descr', 'fortran_order' and 'shape'");
        return h;
    }

private:
    char Peek() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    void Skip() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
    }
    void Expect(char c) {
        Skip();
        if (Peek() != c) Bad(std::string("expected '") + c + "' in the header");
        ++i_;
    }
    std::string String() {
        Skip();
        const char quote = Peek();
        if (quote != '\'' && quote != '"') Bad("expected a quoted string in the header");
        ++i_;
        const std::size_t start = i_;
        while (i_ < s_.size() && s_[i_] != quote) {
            if (s_[i_] == '\\') Bad("escapes in header strings aren't supported");
            ++i_;
        }
        if (i_ >= s_.size()) Bad("an unterminated string in the header");
        return s_.substr(start, i_++ - start);
    }
    bool Bool() {
        if (s_.compare(i_, 4, "True") == 0) {
            i_ += 4;
            return true;
        }
        if (s_.compare(i_, 5, "False") == 0) {
            i_ += 5;
            return false;
        }
        Bad("'fortran_order' must be True or False");
    }
    std::vector<int64_t> Tuple() {
        if (Peek() != '(') Bad("'shape' must be a tuple");
        ++i_;
        std::vector<int64_t> dims;
        while (true) {
            Skip();
            if (Peek() == ')') {
                ++i_;
                return dims;
            }
            if (Peek() < '0' || Peek() > '9') Bad("'shape' must hold non-negative integers");
            int64_t v = 0;
            while (Peek() >= '0' && Peek() <= '9') {
                if (v > (std::numeric_limits<int64_t>::max() - 9) / 10) Bad("a dimension in 'shape' is too large");
                v = v * 10 + (Peek() - '0');
                ++i_;
            }
            // Python writes 3 as 3 and may write 3L in old files.
            if (Peek() == 'L') ++i_;
            dims.push_back(v);
            if (dims.size() > kMaxRank) Bad("more than " + std::to_string(kMaxRank) + " dimensions");
            Skip();
            if (Peek() == ',') {
                ++i_;
            } else if (Peek() != ')') {
                Bad("expected ',' or ')' in 'shape'");
            }
        }
    }

    const std::string& s_;
    std::size_t i_ = 0;
};

uint32_t ReadLe(const std::vector<uint8_t>& b, std::size_t at, int bytes) {
    uint32_t v = 0;
    for (int k = bytes - 1; k >= 0; --k) v = (v << 8) | b[at + static_cast<std::size_t>(k)];
    return v;
}

std::string Clean(std::string s) {
    for (char& c : s) {
        if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    }
    return s;
}

void WriteFile(const std::string& path, const char* data, std::size_t size, const char* function) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error(std::string(function) + ": cannot open " + path + " for writing");
    out.write(data, static_cast<std::streamsize>(size));
    if (!out) throw std::runtime_error(std::string(function) + ": cannot write " + path);
}

}  // namespace

std::vector<uint8_t> SerializeNpy(const Tensor& x) {
    // Python's tuple syntax: (), (5,), (3, 4).
    std::string shape = "(";
    for (int64_t d = 0; d < x.rank(); ++d) shape += (d ? ", " : "") + std::to_string(x.shape().dim(d));
    shape += x.rank() == 1 ? ",)" : ")";
    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }";
    // numpy pads with spaces and ends with a newline so the data starts on a 64-byte boundary.
    const std::size_t unpadded = kMagicSize + 4 + header.size() + 1;
    header.append((64 - unpadded % 64) % 64, ' ');
    header += '\n';
    std::vector<uint8_t> out(kMagic, kMagic + kMagicSize);
    out.push_back(1);
    out.push_back(0);
    out.push_back(static_cast<uint8_t>(header.size() & 0xff));
    out.push_back(static_cast<uint8_t>(header.size() >> 8));
    out.insert(out.end(), header.begin(), header.end());
    const std::vector<float> values = x.to_host_vector();
    for (float v : values) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof bits);
        for (int k = 0; k < 4; ++k) out.push_back(static_cast<uint8_t>(bits >> (8 * k)));
    }
    return out;
}

void WriteNpy(const std::string& path, const Tensor& x) {
    const std::vector<uint8_t> bytes = SerializeNpy(x);
    WriteFile(path, reinterpret_cast<const char*>(bytes.data()), bytes.size(), "WriteNpy");
}

Tensor ParseNpy(const std::vector<uint8_t>& bytes, DeviceBackend* backend) {
    if (bytes.size() < kMagicSize + 4 || std::memcmp(bytes.data(), kMagic, kMagicSize) != 0) Bad("not a .npy file");
    const uint8_t major = bytes[kMagicSize];
    if (major < 1 || major > 3) Bad("format version " + std::to_string(major) + " isn't 1, 2 or 3");
    const int length_bytes = major == 1 ? 2 : 4;
    const std::size_t start = kMagicSize + 2;
    if (bytes.size() < start + static_cast<std::size_t>(length_bytes)) Bad("the file ends inside its header");
    const std::size_t header_size = ReadLe(bytes, start, length_bytes);
    const std::size_t data_at = start + static_cast<std::size_t>(length_bytes) + header_size;
    if (header_size > bytes.size() || data_at > bytes.size()) Bad("the file ends inside its header");
    const std::string text(bytes.begin() + static_cast<std::ptrdiff_t>(start + static_cast<std::size_t>(length_bytes)),
                           bytes.begin() + static_cast<std::ptrdiff_t>(data_at));
    const Header h = HeaderParser(text).Parse();

    std::size_t item = 0;
    if (h.descr == "<f4") {
        item = 4;
    } else if (h.descr == "<f8") {
        item = 8;
    } else {
        Bad("dtype '" + h.descr + "' isn't little-endian float32 or float64 ('<f4' or '<f8')");
    }
    const std::size_t available = (bytes.size() - data_at) / item;
    std::size_t count = 1;
    bool empty = false;
    for (int64_t d : h.shape) empty = empty || d == 0;
    if (empty) {
        // No data, but the other dimensions must still multiply within int64_t, as Shape needs.
        int64_t product = 1;
        for (int64_t d : h.shape) {
            if (d == 0) continue;
            if (product > std::numeric_limits<int64_t>::max() / d) Bad("the shape's dimensions multiply past int64_t");
            product *= d;
        }
        count = 0;
    } else {
        // Checked before multiplying, so a huge shape can't overflow into a small count.
        for (int64_t d : h.shape) {
            if (count > available / static_cast<std::size_t>(d)) Bad("the shape holds more values than the file");
            count *= static_cast<std::size_t>(d);
        }
    }
    if (bytes.size() - data_at != count * item) {
        Bad("the shape needs " + std::to_string(count * item) + " bytes of data; the file has " + std::to_string(bytes.size() - data_at));
    }
    std::vector<float> raw(count);
    for (std::size_t k = 0; k < count; ++k) {
        const std::size_t at = data_at + k * item;
        if (item == 4) {
            const uint32_t bits = ReadLe(bytes, at, 4);
            std::memcpy(&raw[k], &bits, 4);
        } else {
            uint64_t bits = 0;
            for (int b = 7; b >= 0; --b) bits = (bits << 8) | bytes[at + static_cast<std::size_t>(b)];
            double v = 0;
            std::memcpy(&v, &bits, 8);
            if (std::isfinite(v) && std::fabs(v) > static_cast<double>(std::numeric_limits<float>::max())) {
                Bad("a float64 value is too large for float32");
            }
            raw[k] = static_cast<float>(v);
        }
    }
    const std::vector<int64_t>& dims = h.shape;
    std::vector<float> values = raw;
    if (h.fortran_order && dims.size() > 1) {
        // Column-major on disk: the first index varies fastest.
        for (std::size_t k = 0; k < count; ++k) {
            std::size_t rest = k, offset = 0, stride = 1;
            std::vector<std::size_t> index(dims.size());
            for (std::size_t d = dims.size(); d-- > 0;) {
                index[d] = rest % static_cast<std::size_t>(dims[d]);
                rest /= static_cast<std::size_t>(dims[d]);
            }
            for (std::size_t d = 0; d < dims.size(); ++d) {
                offset += index[d] * stride;
                stride *= static_cast<std::size_t>(dims[d]);
            }
            values[k] = raw[offset];
        }
    }
    return Tensor(Shape(dims), backend, values);
}

Tensor ReadNpy(const std::string& path, DeviceBackend* backend) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("ReadNpy: cannot open " + path);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) throw std::runtime_error("ReadNpy: cannot read " + path);
    return ParseNpy(bytes, backend);
}

std::string ProjectorVectorsTsv(const Tensor& x) {
    if (x.rank() < 2) throw std::invalid_argument("ProjectorVectorsTsv: x must have shape (n, ...), a row per point");
    const int64_t n = x.shape().dim(0);
    const int64_t p = n == 0 ? 0 : static_cast<int64_t>(x.numel()) / n;
    const std::vector<float> v = x.to_host_vector();
    std::string out;
    char buf[32];
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t a = 0; a < p; ++a) {
            const float t = v[static_cast<std::size_t>(i * p + a)];
            if (!std::isfinite(t)) {
                throw std::invalid_argument("ProjectorVectorsTsv: row " + std::to_string(i) + " has a NaN or infinite value");
            }
            std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(t));
            if (a > 0) out += '\t';
            out += buf;
        }
        out += '\n';
    }
    return out;
}

std::string ProjectorMetadataTsv(const std::vector<MetadataColumn>& columns) {
    if (columns.empty()) throw std::invalid_argument("ProjectorMetadataTsv: no columns");
    const std::size_t n = columns.front().values.size();
    for (const MetadataColumn& c : columns) {
        if (c.values.size() != n) {
            throw std::invalid_argument("ProjectorMetadataTsv: column '" + c.name + "' has " + std::to_string(c.values.size()) +
                                        " values, column '" + columns.front().name + "' " + std::to_string(n));
        }
    }
    std::string out;
    if (columns.size() > 1) {
        for (std::size_t c = 0; c < columns.size(); ++c) out += (c ? "\t" : "") + Clean(columns[c].name);
        out += '\n';
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < columns.size(); ++c) out += (c ? "\t" : "") + Clean(columns[c].values[i]);
        out += '\n';
    }
    return out;
}

void ExportEmbeddings(const std::string& directory, const Tensor& x, const std::vector<MetadataColumn>& columns) {
    const std::string vectors = ProjectorVectorsTsv(x);
    std::string metadata;
    if (!columns.empty()) {
        const auto n = static_cast<std::size_t>(x.shape().dim(0));
        for (const MetadataColumn& c : columns) {
            if (c.values.size() != n) {
                throw std::invalid_argument("ExportEmbeddings: column '" + c.name + "' has " + std::to_string(c.values.size()) +
                                            " values for " + std::to_string(n) + " rows");
            }
        }
        metadata = ProjectorMetadataTsv(columns);
    }
    const std::filesystem::path dir(directory);
    WriteNpy((dir / "vectors.npy").string(), x);
    WriteFile((dir / "vectors.tsv").string(), vectors.data(), vectors.size(), "ExportEmbeddings");
    if (!columns.empty()) WriteFile((dir / "metadata.tsv").string(), metadata.data(), metadata.size(), "ExportEmbeddings");
}

}  // namespace pulsatrix
