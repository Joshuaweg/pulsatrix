// pulsatrix_attnlrp: compares pulsatrix's AttnLRP relevance on a Hugging Face model with LXT's own
// (roadmap LLM-7).
//
//   docker run ... pulsatrix-lrpref:latest python3 tools/golden/make_attnlrp_reference.py golden/SmolLM2-135M
//   pulsatrix_attnlrp golden/SmolLM2-135M                # reads config, weights and attnlrp.safetensors
//   pulsatrix_attnlrp golden/SmolLM2-135M --device hip --threshold 0.99
//
// Exit status: 0 if every correlation is above the threshold, 1 if not or on an error, 2 on a usage error.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "pulsatrix/attnlrp_parity.hpp"
#include "pulsatrix/cpu_backend.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_attnlrp: " << problem << "\n"
              << "usage: pulsatrix_attnlrp MODEL_DIR [--reference FILE] [--threshold T] [--device cpu|hip]\n";
    std::exit(2);
}

int Run(pulsatrix::DeviceBackend* backend, const std::string& dir, const std::string& reference_path, float threshold) {
    using namespace pulsatrix;
    std::unique_ptr<CausalLM> model = LoadCausalLM(dir, backend);
    AttnLrpReport r = CompareToAttnLrp(*model, SafetensorsFile::Map(reference_path), LxtAttnLrpConfig(), threshold);
    auto meta = [&](const char* k) { auto it = r.metadata.find(k); return it == r.metadata.end() ? std::string("?") : it->second; };
    std::printf("%s @ %s (lxt %s, transformers %s)\n", meta("model").c_str(), meta("revision").c_str(), meta("lxt").c_str(),
                meta("transformers").c_str());
    for (size_t i = 0; i < r.sequences.size(); ++i) {
        const auto& s = r.sequences[i];
        std::printf("  text %zu: %3lld tokens  token r %.6f  K-head r %.6f  V-head r %.6f  rel diff %.2g  sum %.4g (LXT %.4g)\n",
                    i, static_cast<long long>(s.num_tokens), s.token_correlation, s.key_correlation, s.value_correlation,
                    s.max_relative_diff, s.relevance_sum, s.reference_sum);
    }
    std::printf("%s: lowest token correlation %.6f, lowest K/V-head correlation %.6f, threshold %.3g (largest rel diff %.2g)\n",
                r.passed ? "PASS" : "FAIL", r.min_token_correlation, r.min_kv_correlation, r.threshold, r.max_relative_diff);
    return r.passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) Usage("no model directory");
    const std::string dir = argv[1];
    std::string reference = (std::filesystem::path(dir) / "attnlrp.safetensors").string();
    float threshold = 0.99f;
    std::string device = "cpu";
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--reference") reference = value();
        else if (a == "--threshold") threshold = std::strtof(value().c_str(), nullptr);
        else if (a == "--device") device = value();
        else Usage("unknown option " + a);
    }
    try {
        if (device == "cpu") {
            pulsatrix::CPUBackend cpu;
            return Run(&cpu, dir, reference, threshold);
        }
#ifdef PULSATRIX_GOLDEN_WITH_HIP
        if (device == "hip") {
            pulsatrix::HIPBackend hip;
            return Run(&hip, dir, reference, threshold);
        }
#endif
        Usage("device \"" + device + "\" is not in this build");
    } catch (const std::exception& e) {
        std::cerr << "pulsatrix_attnlrp: " << e.what() << "\n";
        return 1;
    }
}
