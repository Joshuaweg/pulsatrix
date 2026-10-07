#include "pulsatrix/protein_sequences.hpp"

#include <algorithm>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "pulsatrix/byte_level_bpe.hpp"
#include "pulsatrix/tokenizer_components.hpp"

namespace pulsatrix {
namespace {

std::string ReadAll(const std::string& path, const char* who) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string(who) + ": can't read " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool IsSpace(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\v' || c == '\f'; }

std::string Trim(std::string_view s) {
    size_t b = 0, e = s.size();
    while (b < e && IsSpace(s[b])) ++b;
    while (e > b && IsSpace(s[e - 1])) --e;
    return std::string(s.substr(b, e - b));
}

}  // namespace

TextTokenizer MakeEsmTokenizer(const std::vector<std::string>& vocab) {
    // EsmTokenizer splits on every vocabulary token first (they're all "no split" tokens), then on
    // whitespace, and looks each remaining piece up whole. Added tokens do the first step; a
    // word-level model the last.
    auto model = std::make_shared<WordLevelModel>(vocab, "<unk>");
    TextTokenizer tok(model);
    for (size_t id = 0; id < vocab.size(); ++id) {
        const std::string& t = vocab[id];
        const bool special = t.size() > 1 && t.front() == '<' && t.back() == '>' && t != "<null_1>";
        tok.add_token({t, static_cast<int64_t>(id), special});
    }
    tok.set_pre_tokenizer(std::make_shared<SplitPreTokenizer>("\\s+", SplitBehavior::Removed));
    const auto cls = model->token_to_id("<cls>"), eos = model->token_to_id("<eos>");
    if (!cls || !eos) throw std::invalid_argument("MakeEsmTokenizer: the vocabulary needs <cls> and <eos>");
    tok.set_post_processor(std::make_shared<TemplatePostProcessor>(std::vector<TemplatePostProcessor::Item>{
        {{*cls}, {"<cls>"}}, {}, {{*eos}, {"<eos>"}}}));
    return tok;
}

TextTokenizer LoadEsmTokenizer(const std::string& vocab_path) {
    std::vector<std::string> vocab;
    std::istringstream lines(ReadAll(vocab_path, "LoadEsmTokenizer"));
    for (std::string line; std::getline(lines, line);) {
        line = Trim(line);
        if (!line.empty()) vocab.push_back(line);
    }
    return MakeEsmTokenizer(vocab);
}

std::vector<FastaRecord> ParseFasta(std::string_view text) {
    std::vector<FastaRecord> out;
    size_t line_no = 0;
    for (size_t pos = 0; pos <= text.size();) {
        size_t end = text.find('\n', pos);
        if (end == std::string_view::npos) end = text.size();
        std::string_view line = text.substr(pos, end - pos);
        ++line_no;
        pos = end + 1;
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        const std::string trimmed = Trim(line);
        if (trimmed.empty() || trimmed[0] == ';') {
            if (end == text.size()) break;
            continue;
        }
        if (trimmed[0] == '>') {
            const std::string header = Trim(std::string_view(trimmed).substr(1));
            if (header.empty()) throw std::invalid_argument("ParseFasta: empty header on line " + std::to_string(line_no));
            const size_t space = std::find_if(header.begin(), header.end(), IsSpace) - header.begin();
            out.push_back({header.substr(0, space), space < header.size() ? Trim(std::string_view(header).substr(space)) : "", ""});
        } else {
            if (out.empty()) throw std::invalid_argument("ParseFasta: sequence before the first header, on line " + std::to_string(line_no));
            for (char c : trimmed) {
                if (!IsSpace(c)) out.back().sequence += c;
            }
        }
        if (end == text.size()) break;
    }
    return out;
}

std::vector<FastaRecord> ReadFasta(const std::string& path) { return ParseFasta(ReadAll(path, "ReadFasta")); }

std::string WriteFasta(const std::vector<FastaRecord>& records, size_t width) {
    std::string out;
    for (const auto& r : records) {
        out += ">" + r.id + (r.description.empty() ? "" : " " + r.description) + "\n";
        const size_t step = width == 0 ? std::max<size_t>(r.sequence.size(), 1) : width;
        for (size_t i = 0; i < r.sequence.size(); i += step) out += r.sequence.substr(i, step) + "\n";
    }
    return out;
}

}  // namespace pulsatrix
