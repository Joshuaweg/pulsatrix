// pulsatrix_golden: compares a Hugging Face model loaded into pulsatrix with transformers' own logits
// on real text (roadmap LLM-6).
//
//   python3 tools/golden/make_golden.py --model HuggingFaceTB/SmolLM2-135M --out golden/SmolLM2-135M
//   pulsatrix_golden golden/SmolLM2-135M                 # reads config, weights and golden.safetensors
//   pulsatrix_golden golden/SmolLM2-135M --device hip --threshold 1e-3
//
// Exit status: 0 if every logit is within the threshold, 1 if not or on an error, 2 on a usage error.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/golden_logits.hpp"
#ifdef PULSATRIX_GOLDEN_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

namespace {

[[noreturn]] void Usage(const std::string& problem) {
    std::cerr << "pulsatrix_golden: " << problem << "\n"
              << "usage: pulsatrix_golden MODEL_DIR [--golden FILE] [--threshold T] [--device cpu|hip]\n";
    std::exit(2);
}

int Run(pulsatrix::DeviceBackend* backend, const std::string& dir, const std::string& golden_path, float threshold) {
    using namespace pulsatrix;
    std::unique_ptr<CausalLM> model = LoadCausalLM(dir, backend);
    GoldenReport r = CompareToGolden(*model, SafetensorsFile::Map(golden_path), threshold);
    auto meta = [&](const char* k) { auto it = r.metadata.find(k); return it == r.metadata.end() ? std::string("?") : it->second; };
    std::printf("%s @ %s (transformers %s)\n", meta("model").c_str(), meta("revision").c_str(), meta("transformers").c_str());
    for (size_t i = 0; i < r.sequences.size(); ++i) {
        const auto& s = r.sequences[i];
        std::printf("  text %zu: %3lld tokens  max |diff| %.3g  mean %.3g  argmax disagreements %lld\n", i,
                    static_cast<long long>(s.num_tokens), s.max_abs_diff, s.mean_abs_diff,
                    static_cast<long long>(s.argmax_disagreements));
    }
    std::printf("%s: max |diff| %.3g against threshold %.3g\n", r.passed ? "PASS" : "FAIL", r.max_abs_diff, r.threshold);
    return r.passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) Usage("no model directory");
    const std::string dir = argv[1];
    std::string golden = (std::filesystem::path(dir) / "golden.safetensors").string();
    float threshold = 1e-3f;
    std::string device = "cpu";
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) Usage(a + " needs a value");
            return argv[++i];
        };
        if (a == "--golden") golden = value();
        else if (a == "--threshold") threshold = std::strtof(value().c_str(), nullptr);
        else if (a == "--device") device = value();
        else Usage("unknown option " + a);
    }
    try {
        if (device == "cpu") {
            pulsatrix::CPUBackend cpu;
            return Run(&cpu, dir, golden, threshold);
        }
#ifdef PULSATRIX_GOLDEN_WITH_HIP
        if (device == "hip") {
            pulsatrix::HIPBackend hip;
            return Run(&hip, dir, golden, threshold);
        }
#endif
        Usage("device \"" + device + "\" is not in this build");
    } catch (const std::exception& e) {
        std::cerr << "pulsatrix_golden: " << e.what() << "\n";
        return 1;
    }
}
