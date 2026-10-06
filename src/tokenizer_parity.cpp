#include "pulsatrix/tokenizer_parity.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "pulsatrix/json.hpp"

namespace pulsatrix {

namespace {

bool IsLeadByte(char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }

std::vector<int64_t> Ids(const JsonValue& row, std::string_view key) {
    const JsonValue* v = row.find(key);
    if (v == nullptr) throw std::invalid_argument("CompareToTokenizerReference: a line has no \"" + std::string(key) + "\"");
    std::vector<int64_t> ids;
    for (const JsonValue& id : v->as_array()) ids.push_back(id.as_int64());
    return ids;
}

std::string Join(const std::vector<int64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) s += (i ? " " : "") + std::to_string(v[i]);
    return s;
}

}  // namespace

Offset ByteToCharOffset(std::string_view text, Offset bytes) {
    Offset chars;
    for (size_t i = 0; i < bytes.begin && i < text.size(); ++i) chars.begin += IsLeadByte(text[i]) ? 1 : 0;
    // A begin inside a character belongs to that character.
    if (bytes.begin < text.size() && !IsLeadByte(text[bytes.begin])) --chars.begin;
    for (size_t i = 0; i < bytes.end && i < text.size(); ++i) chars.end += IsLeadByte(text[i]) ? 1 : 0;
    return chars;
}

TokenizerParityReport CompareToTokenizerReference(const TextTokenizer& tokenizer, const std::string& path, size_t max_examples) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("CompareToTokenizerReference: can't read " + path);
    TokenizerParityReport report;
    auto note = [&](size_t line, const char* field, const std::string& text, std::string detail) {
        if (report.examples.size() < max_examples) report.examples.push_back({line, field, text, std::move(detail)});
    };
    std::string raw;
    while (std::getline(in, raw)) {
        if (raw.empty()) continue;
        const JsonValue row = ParseJson(raw);
        const size_t line = report.lines++;
        const std::string& text = row.find("text")->as_string();
        const Encoding e = tokenizer.encode(text);
        const std::vector<int64_t> expected = Ids(row, "ids");
        const std::vector<int64_t> plain = tokenizer.encode(text, /*add_special_tokens=*/false).ids;
        const std::vector<int64_t> expected_plain = Ids(row, "plain_ids");
        if (e.ids != expected || plain != expected_plain) {
            ++report.id_mismatches;
            note(line, e.ids != expected ? "ids" : "plain_ids", text,
                 "got [" + Join(e.ids != expected ? e.ids : plain) + "], want [" + Join(e.ids != expected ? expected : expected_plain) + "]");
            continue;
        }
        const JsonValue::Array& offsets = row.find("offsets")->as_array();
        const JsonValue* normalized_flag = row.find("normalized");
        const bool exact = !(normalized_flag != nullptr && normalized_flag->as_bool()) && tokenizer.decode(plain) == text;
        for (size_t i = 0; i < e.size(); ++i) {
            const Offset got = ByteToCharOffset(text, e.offsets[i]);
            const JsonValue::Array& want = offsets[i].as_array();
            const auto want_begin = static_cast<size_t>(want[0].as_int64());
            const auto want_end = static_cast<size_t>(want[1].as_int64());
            if (got.begin != want_begin || got.end != want_end) {
                if (!exact) {
                    ++report.inexact_line_offset_differences;  // see the header
                    break;
                }
                ++report.offset_mismatches;
                note(line, "offsets", text,
                     "token " + std::to_string(i) + " (" + e.tokens[i] + "): got [" + std::to_string(got.begin) + ", " +
                         std::to_string(got.end) + "), want [" + want[0].number_text() + ", " + want[1].number_text() + ")");
                break;
            }
        }
        const std::string decoded = tokenizer.decode(e.ids);
        const std::string decoded_skip = tokenizer.decode(e.ids, /*skip_special_tokens=*/true);
        if (decoded != row.find("decoded")->as_string() || decoded_skip != row.find("decoded_skip")->as_string()) {
            ++report.decode_mismatches;
            const bool plain_differs = decoded != row.find("decoded")->as_string();
            note(line, plain_differs ? "decoded" : "decoded_skip", text,
                 "got \"" + (plain_differs ? decoded : decoded_skip) + "\", want \"" +
                     row.find(plain_differs ? "decoded" : "decoded_skip")->as_string() + "\"");
        }
    }
    return report;
}

}  // namespace pulsatrix
