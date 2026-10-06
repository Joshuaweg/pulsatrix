// pulsatrix_tokenizer_parity: compares a tokenizer.json loaded into pulsatrix with Hugging Face
// `tokenizers` on a reference corpus (roadmap TOK-2).
//
//   python3 tools/tokenizers/make_tokenizer_reference.py reference tokenizer.json corpus.jsonl ref.jsonl
//   pulsatrix_tokenizer_parity tokenizer.json ref.jsonl [--examples N]
//
// Exit status: 0 if every line matches (ids, offsets, decoded text), 1 if not or on an error,
// 2 on a usage error.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "pulsatrix/tokenizer_json.hpp"
#include "pulsatrix/tokenizer_parity.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: pulsatrix_tokenizer_parity TOKENIZER_JSON REFERENCE_JSONL [--examples N]\n";
        return 2;
    }
    size_t examples = 5;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--examples") examples = static_cast<size_t>(std::strtoul(argv[i + 1], nullptr, 10));
    }
    try {
        const auto t0 = std::chrono::steady_clock::now();
        const pulsatrix::TextTokenizer tok = pulsatrix::LoadTokenizerJson(argv[1]);
        const auto t1 = std::chrono::steady_clock::now();
        const pulsatrix::TokenizerParityReport r = pulsatrix::CompareToTokenizerReference(tok, argv[2], examples);
        const auto t2 = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::printf("%s: %zu lines, %zu id mismatches, %zu offset mismatches, %zu decode mismatches, %zu offset differences on lines that don't round-trip "
                    "(load %.0f ms, compare %.0f ms)\n",
                    r.passed() ? "PASS" : "FAIL", r.lines, r.id_mismatches, r.offset_mismatches, r.decode_mismatches,
                    r.inexact_line_offset_differences, ms(t0, t1), ms(t1, t2));
        for (const auto& m : r.examples) {
            std::printf("  line %zu, %s: %s\n    text: %s\n", m.line, m.field.c_str(), m.detail.c_str(), m.text.c_str());
        }
        return r.passed() ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "pulsatrix_tokenizer_parity: " << e.what() << "\n";
        return 1;
    }
}
