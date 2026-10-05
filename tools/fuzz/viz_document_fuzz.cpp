// Fuzz target for the JSON parser and the viz document readers (roadmap VIZ-1). The property:
// any input either reads as a document or throws std::invalid_argument, and a document that
// reads writes back to JSON that reads to the same bytes again. Anything else -- another
// exception type, a crash, a sanitizer report or a failed check -- is a bug.
//
// Built and run like tools/fuzz/safetensors_fuzz.cpp (see its header): with libFuzzer, or with
// PULSATRIX_FUZZ_STANDALONE=ON, where main() mutates a few valid documents with a seeded
// generator.
//   ./viz_document_fuzz [iterations] [seed]
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "pulsatrix/json.hpp"
#include "pulsatrix/viz/document.hpp"

namespace {

// Reads with Parse, then checks that ToJson of the result is a fixed point.
template <typename Parse>
void ReadAndRewrite(std::string_view text, Parse parse) {
    try {
        auto doc = parse(text);
        std::string once = pulsatrix::ToJson(doc);
        std::string twice = pulsatrix::ToJson(parse(once));
        if (once != twice) {
            std::fprintf(stderr, "rewrite is not stable:\n%s\n---\n%s\n", once.c_str(), twice.c_str());
            std::abort();
        }
    } catch (const std::invalid_argument&) {
        // Rejected: the expected outcome for malformed input.
    }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    std::string_view text(reinterpret_cast<const char*>(data), size);
    try {
        pulsatrix::JsonValue v = pulsatrix::ParseJson(text);
        std::string once = pulsatrix::WriteJson(v);
        if (pulsatrix::WriteJson(pulsatrix::ParseJson(once)) != once) {
            std::fprintf(stderr, "JSON rewrite is not stable:\n%s\n", once.c_str());
            std::abort();
        }
    } catch (const std::invalid_argument&) {
    }
    try {
        (void)pulsatrix::VizDocumentKind(text);
    } catch (const std::invalid_argument&) {
    }
    ReadAndRewrite(text, pulsatrix::ParseAttributionDocument);
    ReadAndRewrite(text, pulsatrix::ParseHeatmapDocument);
    ReadAndRewrite(text, pulsatrix::ParseTokenRelevanceDocument);
    ReadAndRewrite(text, pulsatrix::ParseCircuitGraphDocument);
    ReadAndRewrite(text, pulsatrix::ParseTrainingLogDocument);
    ReadAndRewrite(text, pulsatrix::ParseFeatureDashboardDocument);
    return 0;
}

#ifdef PULSATRIX_FUZZ_STANDALONE
namespace {

std::vector<std::string> seeds() {
    using namespace pulsatrix;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    AttributionDocument a{"ig", {2, 2}, {0.5f, nan, -inf, 1e-30f}, {{"k", "v"}}};
    HeatmapDocument h{"t", 2, 1, {1.0f, -2.0f}, {"a", "b"}, {}};
    TokenRelevanceDocument t{"lrp", {"caf\xC3\xA9", "\"x\""}, {0.25f, inf}, "y"};
    CircuitGraphDocument c;
    c.nodes = {{0, "Linear", "fc", 1.0f}, {3, "Attention", std::nullopt, nan}};
    c.edges = {{0, 3, 0.5f}};
    TrainingLogDocument l;
    l.scalars = {{"loss", {0, 1}, {0.5, std::numeric_limits<double>::infinity()}}};
    l.histograms = {{"w", {1.0f, 2.0f}}};
    FeatureDashboardDocument f;
    f.source = "sae";
    f.histogram_edges = {0.0f, 1.0f};
    f.histogram_counts = {3};
    f.top_examples = {{"e", {"a"}, {2.0f}}};
    return {ToJson(a), ToJson(h), ToJson(t), ToJson(c), ToJson(l), ToJson(f)};
}

}  // namespace

int main(int argc, char** argv) {
    const long iterations = argc > 1 ? std::atol(argv[1]) : 200000;
    uint64_t state = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
    auto rnd = [&state](uint64_t bound) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return bound == 0 ? 0 : (state >> 33) % bound;
    };
    static const char kTokens[] = "{}[]\",:0123456789-.eE\\u \t\nnulltruefalse/";
    const std::vector<std::string> corpus = seeds();
    for (long it = 0; it < iterations; ++it) {
        std::string f = corpus[rnd(corpus.size())];
        for (uint64_t m = 0, n = 1 + rnd(6); m < n; ++m) {
            switch (rnd(5)) {
                case 0:
                    if (!f.empty()) f[rnd(f.size())] ^= static_cast<char>(1u << rnd(8));
                    break;
                case 1:
                    if (!f.empty()) f[rnd(f.size())] = kTokens[rnd(sizeof(kTokens) - 1)];
                    break;
                case 2:
                    f.resize(rnd(f.size() + 1));
                    break;
                case 3:
                    f.insert(f.begin() + static_cast<long>(rnd(f.size() + 1)), kTokens[rnd(sizeof(kTokens) - 1)]);
                    break;
                default:
                    if (!f.empty()) f.erase(f.begin() + static_cast<long>(rnd(f.size())));
                    break;
            }
        }
        LLVMFuzzerTestOneInput(reinterpret_cast<const uint8_t*>(f.data()), f.size());
    }
    std::printf("viz_document_fuzz: %ld inputs, no failures\n", iterations);
    return 0;
}
#endif
