#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

// Written by the reference implementation (Python safetensors 0.8.0, numpy 2.5.3):
//   save({"b.weight": f32 [[1.5,-2,0.25],[3,-0.5,7.75]], "a.bias": f32 [0.1,0.2,0.3],
//         "scalar": f32 42 (shape []), "empty": f32 zeros (0, 4), "half": f16 [1,-2.5],
//         "ids": i64 [7,-1,123456789012]}, metadata={"format": "pt", "note": "café"})
const std::vector<uint8_t> kReference = {
    0x98, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7b, 0x22, 0x5f, 0x5f, 0x6d, 0x65, 0x74, 0x61,
    0x64, 0x61, 0x74, 0x61, 0x5f, 0x5f, 0x22, 0x3a, 0x7b, 0x22, 0x66, 0x6f, 0x72, 0x6d, 0x61, 0x74,
    0x22, 0x3a, 0x22, 0x70, 0x74, 0x22, 0x2c, 0x22, 0x6e, 0x6f, 0x74, 0x65, 0x22, 0x3a, 0x22, 0x63,
    0x61, 0x66, 0xc3, 0xa9, 0x22, 0x7d, 0x2c, 0x22, 0x69, 0x64, 0x73, 0x22, 0x3a, 0x7b, 0x22, 0x64,
    0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x49, 0x36, 0x34, 0x22, 0x2c, 0x22, 0x73, 0x68, 0x61,
    0x70, 0x65, 0x22, 0x3a, 0x5b, 0x33, 0x5d, 0x2c, 0x22, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x6f, 0x66,
    0x66, 0x73, 0x65, 0x74, 0x73, 0x22, 0x3a, 0x5b, 0x30, 0x2c, 0x32, 0x34, 0x5d, 0x7d, 0x2c, 0x22,
    0x61, 0x2e, 0x62, 0x69, 0x61, 0x73, 0x22, 0x3a, 0x7b, 0x22, 0x64, 0x74, 0x79, 0x70, 0x65, 0x22,
    0x3a, 0x22, 0x46, 0x33, 0x32, 0x22, 0x2c, 0x22, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x3a, 0x5b,
    0x33, 0x5d, 0x2c, 0x22, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x73,
    0x22, 0x3a, 0x5b, 0x32, 0x34, 0x2c, 0x33, 0x36, 0x5d, 0x7d, 0x2c, 0x22, 0x62, 0x2e, 0x77, 0x65,
    0x69, 0x67, 0x68, 0x74, 0x22, 0x3a, 0x7b, 0x22, 0x64, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22,
    0x46, 0x33, 0x32, 0x22, 0x2c, 0x22, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x3a, 0x5b, 0x32, 0x2c,
    0x33, 0x5d, 0x2c, 0x22, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x73,
    0x22, 0x3a, 0x5b, 0x33, 0x36, 0x2c, 0x36, 0x30, 0x5d, 0x7d, 0x2c, 0x22, 0x65, 0x6d, 0x70, 0x74,
    0x79, 0x22, 0x3a, 0x7b, 0x22, 0x64, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x46, 0x33, 0x32,
    0x22, 0x2c, 0x22, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x3a, 0x5b, 0x30, 0x2c, 0x34, 0x5d, 0x2c,
    0x22, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x73, 0x22, 0x3a, 0x5b,
    0x36, 0x30, 0x2c, 0x36, 0x30, 0x5d, 0x7d, 0x2c, 0x22, 0x73, 0x63, 0x61, 0x6c, 0x61, 0x72, 0x22,
    0x3a, 0x7b, 0x22, 0x64, 0x74, 0x79, 0x70, 0x65, 0x22, 0x3a, 0x22, 0x46, 0x33, 0x32, 0x22, 0x2c,
    0x22, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x3a, 0x5b, 0x5d, 0x2c, 0x22, 0x64, 0x61, 0x74, 0x61,
    0x5f, 0x6f, 0x66, 0x66, 0x73, 0x65, 0x74, 0x73, 0x22, 0x3a, 0x5b, 0x36, 0x30, 0x2c, 0x36, 0x34,
    0x5d, 0x7d, 0x2c, 0x22, 0x68, 0x61, 0x6c, 0x66, 0x22, 0x3a, 0x7b, 0x22, 0x64, 0x74, 0x79, 0x70,
    0x65, 0x22, 0x3a, 0x22, 0x46, 0x31, 0x36, 0x22, 0x2c, 0x22, 0x73, 0x68, 0x61, 0x70, 0x65, 0x22,
    0x3a, 0x5b, 0x32, 0x5d, 0x2c, 0x22, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x6f, 0x66, 0x66, 0x73, 0x65,
    0x74, 0x73, 0x22, 0x3a, 0x5b, 0x36, 0x34, 0x2c, 0x36, 0x38, 0x5d, 0x7d, 0x7d, 0x20, 0x20, 0x20,
    0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x14, 0x1a, 0x99, 0xbe, 0x1c, 0x00, 0x00, 0x00, 0xcd, 0xcc, 0xcc, 0x3d, 0xcd, 0xcc, 0x4c, 0x3e,
    0x9a, 0x99, 0x99, 0x3e, 0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x00, 0x80, 0x3e,
    0x00, 0x00, 0x40, 0x40, 0x00, 0x00, 0x00, 0xbf, 0x00, 0x00, 0xf8, 0x40, 0x00, 0x00, 0x28, 0x42,
    0x00, 0x3c, 0x00, 0xc1,
};

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

// A file from a header and a data section, with no padding: the format doesn't require any.
std::vector<uint8_t> make_file(const std::string& header, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> out(8);
    uint64_t n = header.size();
    for (int i = 0; i < 8; ++i) out[static_cast<size_t>(i)] = static_cast<uint8_t>(n >> (8 * i));
    out.insert(out.end(), header.begin(), header.end());
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

std::vector<uint8_t> zeros(size_t n) { return std::vector<uint8_t>(n, 0); }

// One F32 tensor of 2 elements: the smallest valid file, for the rejection tests to bend.
std::string one_tensor(const std::string& fields) { return "{\"w\":{" + fields + "}}"; }
const std::string kValidFields = "\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]";

void expect_rejected(const std::vector<uint8_t>& bytes, const std::string& why) {
    EXPECT_THROW((void)SafetensorsFile::Parse(bytes), std::invalid_argument) << why;
}

class SafetensorsTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

// --- Reading the reference implementation's output ------------------------------------

TEST_F(SafetensorsTest, ReadsTheReferenceImplementationsFile) {
    SafetensorsFile f = SafetensorsFile::Parse(kReference);
    EXPECT_EQ(f.names(), (std::vector<std::string>{"ids", "a.bias", "b.weight", "empty", "scalar", "half"}));
    EXPECT_EQ(f.metadata(), (std::map<std::string, std::string>{{"format", "pt"}, {"note", "caf\xc3\xa9"}}));

    Tensor w = f.tensor("b.weight", &backend);
    EXPECT_EQ(w.shape(), Shape({2, 3}));
    EXPECT_EQ(values_of(w), (std::vector<float>{1.5f, -2.0f, 0.25f, 3.0f, -0.5f, 7.75f}));
    EXPECT_EQ(values_of(f.tensor("a.bias", &backend)), (std::vector<float>{0.1f, 0.2f, 0.3f}));
    Tensor scalar = f.tensor("scalar", &backend);
    EXPECT_EQ(scalar.rank(), 0);
    EXPECT_EQ(scalar.data()[0], 42.0f);
    EXPECT_EQ(f.tensor("empty", &backend).shape(), Shape({0, 4}));

    const SafetensorsTensorInfo& ids = f.info("ids");
    EXPECT_EQ(ids.dtype, SafetensorsDtype::I64);
    EXPECT_EQ(ids.shape, (std::vector<int64_t>{3}));
    auto [ptr, size] = f.bytes("ids");
    ASSERT_EQ(size, 24u);
    int64_t third = 0;
    std::memcpy(&third, ptr + 16, 8);
    EXPECT_EQ(third, 123456789012LL);
    EXPECT_EQ(f.info("half").dtype, SafetensorsDtype::F16);
}

TEST_F(SafetensorsTest, TensorRequiresF32AndAKnownName) {
    SafetensorsFile f = SafetensorsFile::Parse(kReference);
    EXPECT_THROW((void)f.tensor("half", &backend), std::invalid_argument);  // conversion is IO-6
    EXPECT_THROW((void)f.tensor("missing", &backend), std::invalid_argument);
    EXPECT_THROW((void)f.info("missing"), std::invalid_argument);
    EXPECT_FALSE(f.contains("missing"));
    EXPECT_TRUE(f.contains("ids"));
}

// --- Writing ---------------------------------------------------------------------------

TEST_F(SafetensorsTest, WriteThenReadRoundTripsBitForBit) {
    Tensor a(Shape({2, 2}), &backend, {1.0f, -0.0f, 3.4e38f, 1e-45f});
    Tensor b(Shape({3}), &backend, {0.5f, 0.25f, 0.125f});
    Tensor scalar(Shape({}), &backend, {7.0f});
    Tensor empty(Shape({0, 5}), &backend);
    std::vector<uint8_t> bytes = SerializeSafetensors(
        {{"layer.weight", &a}, {"layer.bias", &b}, {"s", &scalar}, {"e", &empty}}, {{"format_version", "1"}});
    SafetensorsFile f = SafetensorsFile::Parse(bytes);
    EXPECT_EQ(f.metadata().at("format_version"), "1");
    for (const auto& [name, t] : std::vector<std::pair<std::string, const Tensor*>>{
             {"layer.weight", &a}, {"layer.bias", &b}, {"s", &scalar}, {"e", &empty}}) {
        Tensor back = f.tensor(name, &backend);
        EXPECT_EQ(back.shape(), t->shape()) << name;
        ASSERT_EQ(back.numel(), t->numel()) << name;
        if (t->numel() > 0) EXPECT_EQ(std::memcmp(back.data(), t->data(), static_cast<size_t>(t->numel()) * 4), 0);
    }
}

TEST_F(SafetensorsTest, WriterAlignsTheDataSectionAndEscapesNames) {
    Tensor t(Shape({1}), &backend, {1.0f});
    std::vector<uint8_t> bytes = SerializeSafetensors({{"quote\"back\\slash\nnewline\x01", &t}});
    uint64_t n = 0;
    for (int i = 0; i < 8; ++i) n |= static_cast<uint64_t>(bytes[static_cast<size_t>(i)]) << (8 * i);
    EXPECT_EQ((8 + n) % 8, 0u);
    SafetensorsFile f = SafetensorsFile::Parse(bytes);
    EXPECT_EQ(f.names(), (std::vector<std::string>{"quote\"back\\slash\nnewline\x01"}));
}

TEST_F(SafetensorsTest, WriterRejectsDuplicateAndReservedNames) {
    Tensor t(Shape({1}), &backend, {1.0f});
    EXPECT_THROW((void)SerializeSafetensors({{"x", &t}, {"x", &t}}), std::invalid_argument);
    EXPECT_THROW((void)SerializeSafetensors({{"__metadata__", &t}}), std::invalid_argument);
    EXPECT_THROW((void)SerializeSafetensors({{"bad\xff", &t}}), std::invalid_argument);  // not UTF-8
}

TEST_F(SafetensorsTest, WriteAndReadAFile) {
    Tensor t(Shape({2}), &backend, {4.0f, 5.0f});
    const std::string path = ::testing::TempDir() + "pulsatrix_safetensors_roundtrip.safetensors";
    WriteSafetensors(path, {{"t", &t}});
    EXPECT_EQ(values_of(SafetensorsFile::Read(path).tensor("t", &backend)), (std::vector<float>{4.0f, 5.0f}));
    EXPECT_THROW((void)SafetensorsFile::Read(path + ".missing"), std::runtime_error);
}

// --- Rejections: every rule in the format, plus the roadmap's (offsets, overlaps, holes, overflow)

TEST_F(SafetensorsTest, AcceptsTheBaselineUsedByTheRejectionTests) {
    EXPECT_NO_THROW((void)SafetensorsFile::Parse(make_file(one_tensor(kValidFields), zeros(8))));
    EXPECT_NO_THROW((void)SafetensorsFile::Parse(make_file(one_tensor(kValidFields) + "   ", zeros(8))));
    EXPECT_NO_THROW((void)SafetensorsFile::Parse(make_file("{}", {})));
}

TEST_F(SafetensorsTest, RejectsABrokenLengthPrefix) {
    expect_rejected({}, "empty file");
    expect_rejected({1, 0, 0}, "shorter than the length prefix");
    std::vector<uint8_t> f = make_file(one_tensor(kValidFields), zeros(8));
    f[0] = 0xff;  // header length past the end of the file
    expect_rejected(f, "header length past the end");
    std::vector<uint8_t> huge(8, 0);
    huge[7] = 0x80;  // 2^63
    expect_rejected(huge, "header length overflows");
    expect_rejected(make_file("", {}), "empty header");
    std::vector<uint8_t> over_cap = make_file("{}", {});
    const uint64_t cap_plus_one = 100000001ULL;
    for (int i = 0; i < 8; ++i) over_cap[static_cast<size_t>(i)] = static_cast<uint8_t>(cap_plus_one >> (8 * i));
    over_cap.resize(8 + cap_plus_one, ' ');
    expect_rejected(over_cap, "header over the 100 MB cap");
}

TEST_F(SafetensorsTest, RejectsMalformedJson) {
    for (const std::string& header : std::vector<std::string>{
             "[]", "{", "{\"w\":}", "{\"w\":{" + kValidFields + "}} x", "{\"w\":{" + kValidFields + "},}",
             "{'w':{}}", "{\"w\":{" + kValidFields + "}}{}", " {}", "{\"a\":1}", "null", "{\"w\":{" + kValidFields + ",}}",
             "{\"\\x\":{}}", "{\"\\ud800\":{}}", "{\"\x01\":{}}", "{\"w\xff\":{}}", "{\"\\u00\":{}}"}) {
        expect_rejected(make_file(header, zeros(8)), header);
    }
}

TEST_F(SafetensorsTest, RejectsDuplicateKeys) {
    expect_rejected(make_file("{\"w\":{" + kValidFields + "},\"w\":{" + kValidFields + "}}", zeros(8)), "tensor twice");
    expect_rejected(make_file(one_tensor(kValidFields + ",\"dtype\":\"F32\""), zeros(8)), "field twice");
    expect_rejected(make_file("{\"__metadata__\":{\"a\":\"1\",\"a\":\"2\"}}", {}), "metadata key twice");
}

TEST_F(SafetensorsTest, RejectsBadTensorEntries) {
    for (const std::string& fields : std::vector<std::string>{
             "\"shape\":[2],\"data_offsets\":[0,8]",                                 // no dtype
             "\"dtype\":\"F32\",\"data_offsets\":[0,8]",                             // no shape
             "\"dtype\":\"F32\",\"shape\":[2]",                                      // no offsets
             kValidFields + ",\"extra\":1",                                           // unknown field
             "\"dtype\":\"F31\",\"shape\":[2],\"data_offsets\":[0,8]",               // unknown dtype
             "\"dtype\":\"F32\",\"shape\":[-2],\"data_offsets\":[0,8]",              // negative dim
             "\"dtype\":\"F32\",\"shape\":[2.0],\"data_offsets\":[0,8]",             // not an integer
             "\"dtype\":\"F32\",\"shape\":[2e0],\"data_offsets\":[0,8]",             // exponent
             "\"dtype\":\"F32\",\"shape\":[02],\"data_offsets\":[0,8]",              // leading zero
             "\"dtype\":\"F32\",\"shape\":[99999999999999999999],\"data_offsets\":[0,8]",  // overflows
             "\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0]",                 // one offset
             "\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,0]",               // begin > end
             "\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,8]",               // size != numel * 4
             "\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,16]",              // past the end
             "\"dtype\":\"F32\",\"shape\":\"2\",\"data_offsets\":[0,8]",             // shape not array
         }) {
        expect_rejected(make_file(one_tensor(fields), zeros(8)), fields);
    }
}

TEST_F(SafetensorsTest, RejectsElementCountOverflow) {
    // 2^32 * 2^32 elements wraps to 0 in 64 bits, which would "match" an empty range.
    expect_rejected(make_file(one_tensor("\"dtype\":\"F32\",\"shape\":[4294967296,4294967296],\"data_offsets\":[0,0]"), {}),
                    "numel wraps to zero");
    // numel fits, numel * 4 does not.
    expect_rejected(
        make_file(one_tensor("\"dtype\":\"F32\",\"shape\":[4611686018427387904],\"data_offsets\":[0,0]"), {}),
        "byte size wraps");
}

TEST_F(SafetensorsTest, RejectsOverlapsHolesAndUnindexedBytes) {
    const std::string a = "\"a\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}";
    expect_rejected(make_file("{" + a + ",\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}", zeros(12)),
                    "overlap");
    expect_rejected(make_file("{" + a + ",\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[12,16]}}", zeros(16)),
                    "hole");
    expect_rejected(make_file("{" + a + "}", zeros(12)), "trailing unindexed bytes");
    expect_rejected(make_file("{" + a + ",\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", zeros(8)),
                    "two tensors on the same bytes");
    // Zero-size tensors occupy no bytes, so they never overlap.
    EXPECT_NO_THROW((void)SafetensorsFile::Parse(
        make_file("{" + a + ",\"z\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[8,8]}}", zeros(8))));
}

TEST_F(SafetensorsTest, RejectsNonStringMetadata) {
    expect_rejected(make_file("{\"__metadata__\":{\"a\":1}}", {}), "number value");
    expect_rejected(make_file("{\"__metadata__\":[]}", {}), "array");
    expect_rejected(make_file("{\"__metadata__\":{\"a\":{}}}", {}), "object value");
}

// Deterministic mutation sweep: every corruption either parses or throws invalid_argument --
// never another exception, a crash or an out-of-bounds read (run under ASan to see the last).
TEST_F(SafetensorsTest, CorruptedFilesThrowInvalidArgumentOrParse) {
    uint64_t state = 12345;
    auto rnd = [&state](uint64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (state >> 33) % bound;
    };
    int parsed = 0, rejected = 0;
    for (int trial = 0; trial < 20000; ++trial) {
        std::vector<uint8_t> f = kReference;
        switch (rnd(4)) {
            case 0:  // flip bits
                for (uint64_t k = 0, n = 1 + rnd(4); k < n; ++k) f[rnd(f.size())] ^= static_cast<uint8_t>(1u << rnd(8));
                break;
            case 1:  // overwrite a header byte with a JSON-significant character
                f[8 + rnd(408)] = static_cast<uint8_t>("{}[]\",:0123456789-.eE\\u \x00\xff"[rnd(28)]);
                break;
            case 2:  // truncate
                f.resize(rnd(f.size()));
                break;
            default:  // corrupt the length prefix
                f[rnd(8)] = static_cast<uint8_t>(rnd(256));
                break;
        }
        try {
            (void)SafetensorsFile::Parse(f);
            ++parsed;
        } catch (const std::invalid_argument&) {
            ++rejected;
        }
    }
    EXPECT_GT(rejected, 10000);
    EXPECT_GT(parsed, 0);  // e.g. bit flips inside the data section
}

}  // namespace
}  // namespace pulsatrix
