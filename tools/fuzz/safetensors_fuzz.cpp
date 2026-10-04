// Fuzz target for the safetensors reader (roadmap IO-1). The property: any input either parses
// into a file whose every tensor can be read back, or throws std::invalid_argument. Anything else
// -- another exception type, a crash, or a sanitizer report -- is a bug.
//
// With libFuzzer (clang):
//   cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DPULSATRIX_BUILD_FUZZERS=ON \
//         -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
//   cmake --build build-fuzz --target safetensors_fuzz && ./build-fuzz/safetensors_fuzz
//
// Without libFuzzer, define PULSATRIX_FUZZ_STANDALONE (the CMake target does when
// PULSATRIX_FUZZ_STANDALONE=ON): main() then mutates a few valid files with a seeded generator.
//   ./safetensors_fuzz [iterations] [seed]
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/safetensors.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static pulsatrix::CPUBackend backend;
    try {
        pulsatrix::SafetensorsFile f = pulsatrix::SafetensorsFile::Parse(std::vector<uint8_t>(data, data + size));
        for (const std::string& name : f.names()) {
            auto [ptr, n] = f.bytes(name);
            volatile uint8_t sink = 0;
            for (size_t i = 0; i < n; ++i) {
                sink = sink ^ ptr[i];  // touch every byte the parser vouched for
            }
            if (f.info(name).dtype == pulsatrix::SafetensorsDtype::F32) {
                (void)f.tensor(name, &backend);
            }
        }
    } catch (const std::invalid_argument&) {
        // Rejected: the expected outcome for malformed input.
    }
    return 0;
}

#ifdef PULSATRIX_FUZZ_STANDALONE
namespace {

std::vector<std::vector<uint8_t>> seeds() {
    pulsatrix::CPUBackend b;
    pulsatrix::Tensor w(pulsatrix::Shape({2, 3}), &b, {1, 2, 3, 4, 5, 6});
    pulsatrix::Tensor s(pulsatrix::Shape({}), &b, {7});
    pulsatrix::Tensor e(pulsatrix::Shape({0, 2}), &b);
    return {
        pulsatrix::SerializeSafetensors({{"w", &w}, {"s", &s}, {"e", &e}}, {{"format", "pt"}}),
        pulsatrix::SerializeSafetensors({{"a\\u00e9\"", &w}}),
        pulsatrix::SerializeSafetensors({}),
    };
}

}  // namespace

int main(int argc, char** argv) {
    const long iterations = argc > 1 ? std::atol(argv[1]) : 1000000;
    uint64_t state = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
    auto rnd = [&state](uint64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return bound == 0 ? 0 : (state >> 33) % bound;
    };
    static const char kTokens[] = "{}[]\",:0123456789-.eE\\u \t\n";
    const std::vector<std::vector<uint8_t>> corpus = seeds();
    for (long it = 0; it < iterations; ++it) {
        std::vector<uint8_t> f = corpus[rnd(corpus.size())];
        for (uint64_t m = 0, n = 1 + rnd(6); m < n; ++m) {
            switch (rnd(6)) {
                case 0:
                    if (!f.empty()) f[rnd(f.size())] ^= static_cast<uint8_t>(1u << rnd(8));
                    break;
                case 1:
                    if (!f.empty()) f[rnd(f.size())] = static_cast<uint8_t>(kTokens[rnd(sizeof(kTokens) - 1)]);
                    break;
                case 2:
                    f.resize(rnd(f.size() + 1));
                    break;
                case 3:
                    f.insert(f.begin() + static_cast<long>(rnd(f.size() + 1)), static_cast<uint8_t>(rnd(256)));
                    break;
                case 4:
                    if (f.size() >= 8) f[rnd(8)] = static_cast<uint8_t>(rnd(256));
                    break;
                default:
                    if (!f.empty()) f.erase(f.begin() + static_cast<long>(rnd(f.size())));
                    break;
            }
        }
        LLVMFuzzerTestOneInput(f.data(), f.size());
    }
    std::printf("safetensors_fuzz: %ld inputs, no failures\n", iterations);
    return 0;
}
#endif
