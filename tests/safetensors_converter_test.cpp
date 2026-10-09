// IO-3: pulsatrix reads what the legacy-pickle converter (tools/convert/pickle_to_safetensors.py)
// writes. The fixture comes from tools/convert/make_converter_fixture.py: a torch.save
// checkpoint with known values, converted with --strip-prefix module.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

SafetensorsFile Converted() {
    return SafetensorsFile::Map(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/convert/converted.safetensors");
}

TEST(SafetensorsConverter, NamesShapesAndMetadataComeThrough) {
    const SafetensorsFile f = Converted();
    // The state dict was found under "state_dict" and "module." was stripped; "epoch" is gone.
    EXPECT_EQ(f.names().size(), 5u);
    for (const char* name : {"linear.weight", "linear.bias", "bn.num_batches_tracked", "embed.weight", "head.weight"}) {
        EXPECT_TRUE(f.contains(name)) << name;
    }
    EXPECT_EQ(f.info("linear.weight").shape, (std::vector<int64_t>{3, 2}));
    EXPECT_EQ(f.info("linear.weight").dtype, SafetensorsDtype::F32);
    EXPECT_EQ(f.info("linear.bias").dtype, SafetensorsDtype::BF16);
    EXPECT_EQ(f.info("bn.num_batches_tracked").dtype, SafetensorsDtype::I64);
    EXPECT_TRUE(f.info("bn.num_batches_tracked").shape.empty());
    EXPECT_EQ(f.metadata().at("format"), "pt");
    EXPECT_EQ(f.metadata().at("key"), "state_dict");
    EXPECT_EQ(f.metadata().at("source_sha256").size(), 64u);
}

TEST(SafetensorsConverter, ValuesComeThroughContiguousAndUpcast) {
    CPUBackend cpu;
    const SafetensorsFile f = Converted();
    // arange(6).reshape(2, 3) transposed, written contiguous in row-major order.
    EXPECT_EQ(f.tensor("linear.weight", &cpu).to_host_vector(), (std::vector<float>{0, 3, 1, 4, 2, 5}));
    // bf16, upcast exactly.
    EXPECT_EQ(f.tensor("linear.bias", &cpu).to_host_vector(), (std::vector<float>{0.5f, -1.5f}));
    // Two names that shared one tensor in PyTorch each have their own bytes.
    EXPECT_EQ(f.tensor("embed.weight", &cpu).to_host_vector(), (std::vector<float>{1, 2, 3, 4}));
    EXPECT_EQ(f.tensor("head.weight", &cpu).to_host_vector(), (std::vector<float>{1, 2, 3, 4}));
    EXPECT_NE(f.info("embed.weight").data_begin, f.info("head.weight").data_begin);
    // Integer buffers are kept as integers.
    auto [ptr, size] = f.bytes("bn.num_batches_tracked");
    ASSERT_EQ(size, sizeof(int64_t));
    int64_t count = 0;
    std::memcpy(&count, ptr, sizeof count);
    EXPECT_EQ(count, 7);
}

}  // namespace
}  // namespace pulsatrix
