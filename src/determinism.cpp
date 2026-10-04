#include "pulsatrix/determinism.hpp"

#include <atomic>
#include <stdexcept>
#include <string>

namespace pulsatrix {
namespace {

std::atomic<uint64_t> g_seed{0};
std::atomic<uint64_t> g_counter{0};
std::atomic<bool> g_deterministic{true};

// SplitMix64's finalizer (Steele, Lea & Flood 2014): a bijection that spreads adjacent inputs
// across all 64 bits, so seeds drawn in sequence give unrelated random streams.
uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

}  // namespace

void set_seed(uint64_t seed) {
    g_seed.store(seed);
    g_counter.store(0);
}

uint64_t global_seed() { return g_seed.load(); }

uint64_t next_seed() {
    const uint64_t index = g_counter.fetch_add(1);
    return mix(g_seed.load() + 0x9E3779B97F4A7C15ULL * (index + 1));
}

void set_deterministic(bool enabled) { g_deterministic.store(enabled); }

bool deterministic() { return g_deterministic.load(); }

void check_deterministic_allowed(const char* operation) {
    if (deterministic()) {
        throw std::logic_error(std::string(operation) +
                               " is nondeterministic; call set_deterministic(false) to allow it");
    }
}

}  // namespace pulsatrix
