// Fuzz target for the .npy reader (roadmap INT-8). The property: any input either parses into a
// tensor whose every value can be read, or throws std::invalid_argument. Anything else -- another
// exception type, a crash, or a sanitizer report -- is a bug.
//
// With libFuzzer (clang):
//   cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DPULSATRIX_BUILD_FUZZERS=ON \
//         -DCMAKE_CXX_FLAGS="-fsanitize=fuzzer-no-link,address,undefined"
//   cmake --build build-fuzz --target npy_fuzz && ./build-fuzz/npy_fuzz
//
// Without libFuzzer, define PULSATRIX_FUZZ_STANDALONE (the CMake target does when
// PULSATRIX_FUZZ_STANDALONE=ON): main() then mutates a few valid files with a seeded generator.
//   ./npy_fuzz [iterations] [seed]
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/embedding_export.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    static pulsatrix::CPUBackend backend;
    try {
        const pulsatrix::Tensor t = pulsatrix::ParseNpy(std::vector<uint8_t>(data, data + size), &backend);
        const std::vector<float> v = t.to_host_vector();
        if (static_cast<int64_t>(v.size()) != t.shape().numel()) std::abort();
        volatile float sink = 0;
        for (float x : v) sink = sink + x;  // touch every value the parser produced
    } catch (const std::invalid_argument&) {
        // Rejected: the expected outcome for malformed input.
    }
    return 0;
}

#ifdef PULSATRIX_FUZZ_STANDALONE
namespace {

// A .npy file with this header dict and these data bytes, laid out as numpy lays it out.
std::vector<uint8_t> File(const std::string& dict, int major, size_t data_bytes) {
    std::string h = dict;
    const size_t prefix = major == 1 ? 10 : 12;
    h.append((64 - (prefix + h.size() + 1) % 64) % 64, ' ');
    h += '\n';
    std::vector<uint8_t> b = {0x93, 'N', 'U', 'M', 'P', 'Y', static_cast<uint8_t>(major), 0};
    for (int k = 0; k < (major == 1 ? 2 : 4); ++k) b.push_back(static_cast<uint8_t>(h.size() >> (8 * k)));
    b.insert(b.end(), h.begin(), h.end());
    for (size_t k = 0; k < data_bytes; ++k) b.push_back(static_cast<uint8_t>(k * 37));
    return b;
}

std::vector<std::vector<uint8_t>> seeds() {
    pulsatrix::CPUBackend b;
    pulsatrix::Tensor w(pulsatrix::Shape({2, 3}), &b, {1, 2, 3, 4, 5, 6});
    pulsatrix::Tensor s(pulsatrix::Shape({}), &b, {7});
    pulsatrix::Tensor e(pulsatrix::Shape({0, 2}), &b);
    return {
        pulsatrix::SerializeNpy(w),
        pulsatrix::SerializeNpy(s),
        pulsatrix::SerializeNpy(e),
        File("{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3, 2), }", 1, 96),
        File("{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }", 2, 16),
        File("{\"descr\": \"<f4\", \"fortran_order\": True, \"shape\": (3, 2)}", 3, 24),
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
    static const char kTokens[] = "{}()'\",:0123456789L<>fi48TrueFalse \n";
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
                    if (f.size() >= 12) f[6 + rnd(6)] = static_cast<uint8_t>(rnd(256));  // version and header length
                    break;
                default:
                    if (!f.empty()) f.erase(f.begin() + static_cast<long>(rnd(f.size())));
                    break;
            }
        }
        LLVMFuzzerTestOneInput(f.data(), f.size());
    }
    std::printf("npy_fuzz: %ld inputs, no failures\n", iterations);
    return 0;
}
#endif
