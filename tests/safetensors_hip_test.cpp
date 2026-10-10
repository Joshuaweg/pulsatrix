// KS-8: the safetensors writer and checkpoints with tensors that live on the GPU (the IO follow-up
// "running the safetensors writer's GPU path on hardware"): what's written from device memory
// reads back bit for bit, on the host and on the device.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "pulsatrix/checkpoint.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/hip_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/safetensors.hpp"

namespace pulsatrix {
namespace {

std::vector<float> Values(size_t n, float scale) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = scale * (static_cast<float>(i % 13) - 6.0f) / 7.0f + 1e-7f * static_cast<float>(i);
    return v;
}

std::vector<float> ToHost(const Tensor& t) {
    CPUBackend cpu;
    Tensor h = t;
    h.to(DeviceType::Cpu, &cpu);
    return h.to_host_vector();
}

TEST(SafetensorsHip, DeviceTensorsWriteAndReadBackBitForBit) {
    HIPBackend hip;
    CPUBackend cpu;
    const std::vector<float> a = Values(1001, 3.5f), b = Values(6, -0.25f);
    const Tensor ta(Shape({7, 11, 13}), &hip, a, DeviceType::Hip), tb(Shape({2, 3}), &hip, b, DeviceType::Hip);
    const std::string path = ::testing::TempDir() + "safetensors_hip_test.safetensors";
    WriteSafetensors(path, {{"a", &ta}, {"b", &tb}}, {{"written_from", "hip"}});

    const SafetensorsFile f = SafetensorsFile::Map(path);
    EXPECT_EQ(f.metadata().at("written_from"), "hip");
    EXPECT_EQ(f.info("a").shape, (std::vector<int64_t>{7, 11, 13}));
    const std::vector<float> host = f.tensor("a", &cpu).to_host_vector();
    ASSERT_EQ(host.size(), a.size());
    EXPECT_EQ(std::memcmp(host.data(), a.data(), a.size() * sizeof(float)), 0);
    EXPECT_EQ(f.tensor("b", &cpu).to_host_vector(), b);

    // And onto the device again.
    const Tensor back = f.tensor("a", &hip);
    EXPECT_EQ(ToHost(back), a);
}

TEST(SafetensorsHip, CheckpointOfADeviceModuleRoundTrips) {
    HIPBackend hip;
    LinearModule saved(5, 3, &hip, DeviceType::Hip), loaded(5, 3, &hip, DeviceType::Hip);
    saved.set_weight(Values(15, 2.0f));
    saved.set_bias(Values(3, 0.5f));
    const std::string path = ::testing::TempDir() + "safetensors_hip_checkpoint.safetensors";
    SaveCheckpoint(path, saved);
    LoadCheckpoint(path, loaded);
    EXPECT_EQ(loaded.weight().device(), DeviceType::Hip);
    EXPECT_EQ(ToHost(loaded.weight()), ToHost(saved.weight()));
    EXPECT_EQ(ToHost(loaded.bias()), ToHost(saved.bias()));
}

}  // namespace
}  // namespace pulsatrix
