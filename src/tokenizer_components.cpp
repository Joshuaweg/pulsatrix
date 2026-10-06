#include "pulsatrix/tokenizer_components.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>

#include "utf8.hpp"

namespace pulsatrix {

namespace {

bool IsAsciiSpace(char32_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r'; }
bool IsAsciiAlnum(char32_t c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

std::unordered_map<std::string, int64_t> IndexTokens(const std::vector<std::string>& tokens, const char* who) {
    std::unordered_map<std::string, int64_t> ids;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (!ids.emplace(tokens[i], static_cast<int64_t>(i)).second) {
            throw std::invalid_argument(std::string(who) + ": token \"" + tokens[i] + "\" repeats");
        }
    }
    return ids;
}

}  // namespace

void AsciiLowercase::normalize(NormalizedString& text) const {
    std::vector<NormalizedString::Piece> pieces;
    pieces.reserve(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text.text()[i];
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        pieces.push_back({std::string(1, c), i, i + 1});
    }
    text.rebuild(pieces);
}

void WordPunctuationSplit::pre_tokenize(std::vector<NormalizedString>& splits) const {
    std::vector<NormalizedString> out;
    for (const NormalizedString& split : splits) {
        const std::string& s = split.text();
        size_t word_start = 0;
        bool in_word = false;
        auto flush = [&](size_t end) {
            if (in_word) out.push_back(split.slice(word_start, end));
            in_word = false;
        };
        for (size_t i = 0; i < s.size();) {
            const utf8::CodePoint c = utf8::Decode(s, i);
            const size_t length = c.length == 0 ? 1 : c.length;
            if (IsAsciiAlnum(c.value) || c.value == '\'') {
                if (!in_word) word_start = i;
                in_word = true;
            } else if (IsAsciiSpace(c.value)) {
                flush(i);
            } else {
                flush(i);
                out.push_back(split.slice(i, i + length));
            }
            i += length;
        }
        flush(s.size());
    }
    splits = std::move(out);
}

// ---- WordLevelModel ------------------------------------------------------------------------

WordLevelModel::WordLevelModel(std::vector<std::string> vocab, const std::string& unk_token)
    : vocab_(std::move(vocab)), ids_(IndexTokens(vocab_, "WordLevelModel")) {
    const auto unk = ids_.find(unk_token);
    if (unk == ids_.end()) throw std::invalid_argument("WordLevelModel: the unknown token \"" + unk_token + "\" isn't in the vocabulary");
    unk_id_ = unk->second;
}

std::vector<ModelToken> WordLevelModel::tokenize(std::string_view piece) const {
    const auto it = ids_.find(std::string(piece));
    const int64_t id = it == ids_.end() ? unk_id_ : it->second;
    return {ModelToken{id, vocab_[static_cast<size_t>(id)], 0, piece.size()}};
}

std::optional<int64_t> WordLevelModel::token_to_id(std::string_view token) const {
    const auto it = ids_.find(std::string(token));
    if (it == ids_.end()) return std::nullopt;
    return it->second;
}

std::optional<std::string> WordLevelModel::id_to_token(int64_t id) const {
    if (id < 0 || id >= vocab_size()) return std::nullopt;
    return vocab_[static_cast<size_t>(id)];
}

// ---- ByteModel -----------------------------------------------------------------------------

std::vector<ModelToken> ByteModel::tokenize(std::string_view piece) const {
    std::vector<ModelToken> out;
    out.reserve(piece.size());
    for (size_t i = 0; i < piece.size(); ++i) {
        out.push_back({static_cast<unsigned char>(piece[i]), std::string(1, piece[i]), i, i + 1});
    }
    return out;
}

std::optional<int64_t> ByteModel::token_to_id(std::string_view token) const {
    if (token.size() != 1) return std::nullopt;
    return static_cast<unsigned char>(token[0]);
}

std::optional<std::string> ByteModel::id_to_token(int64_t id) const {
    if (id < 0 || id > 255) return std::nullopt;
    return std::string(1, static_cast<char>(id));
}

// ---- CharModel -----------------------------------------------------------------------------

CharModel::CharModel(std::vector<std::string> chars, const std::string& unk_token)
    : chars_(std::move(chars)), ids_(IndexTokens(chars_, "CharModel")) {
    for (const std::string& c : chars_) {
        if (c == unk_token) continue;
        if (c.empty() || utf8::Decode(c, 0).length != c.size()) {
            throw std::invalid_argument("CharModel: \"" + c + "\" isn't one UTF-8 character");
        }
    }
    const auto unk = ids_.find(unk_token);
    if (unk == ids_.end()) throw std::invalid_argument("CharModel: the unknown token \"" + unk_token + "\" isn't in the list");
    unk_id_ = unk->second;
}

std::vector<ModelToken> CharModel::tokenize(std::string_view piece) const {
    std::vector<ModelToken> out;
    for (size_t i = 0; i < piece.size();) {
        const size_t length = std::max<size_t>(1, utf8::Decode(piece, i).length);
        const auto it = ids_.find(std::string(piece.substr(i, length)));
        const int64_t id = it == ids_.end() ? unk_id_ : it->second;
        out.push_back({id, chars_[static_cast<size_t>(id)], i, i + length});
        i += length;
    }
    return out;
}

std::optional<int64_t> CharModel::token_to_id(std::string_view token) const {
    const auto it = ids_.find(std::string(token));
    if (it == ids_.end()) return std::nullopt;
    return it->second;
}

std::optional<std::string> CharModel::id_to_token(int64_t id) const {
    if (id < 0 || id >= vocab_size()) return std::nullopt;
    return chars_[static_cast<size_t>(id)];
}

std::string FuseDecoder::decode(const std::vector<std::string>& tokens) const {
    std::string text;
    for (const std::string& t : tokens) text += t;
    return text;
}

// ---- Ready-made tokenizers -----------------------------------------------------------------

TextTokenizer MakeWordLevelTokenizer(const Vocabulary& vocab) {
    std::vector<std::string> tokens;
    tokens.reserve(static_cast<size_t>(vocab.size()));
    for (int64_t i = 0; i < vocab.size(); ++i) tokens.push_back(vocab.TokenAt(i));
    TextTokenizer t(std::make_shared<WordLevelModel>(std::move(tokens), Vocabulary::kUnkToken));
    t.set_normalizer(std::make_shared<AsciiLowercase>());
    t.set_pre_tokenizer(std::make_shared<WordPunctuationSplit>());
    return t;
}

TextTokenizer MakeByteTokenizer(const std::vector<std::string>& special_tokens) {
    TextTokenizer t(std::make_shared<ByteModel>());
    t.set_decoder(std::make_shared<FuseDecoder>());
    for (size_t i = 0; i < special_tokens.size(); ++i) t.add_token({special_tokens[i], 256 + static_cast<int64_t>(i), true});
    return t;
}

TextTokenizer MakeCharTokenizer(std::vector<std::string> chars, const std::vector<std::string>& special_tokens) {
    bool has_unk = false;
    for (const std::string& c : chars) has_unk = has_unk || c == Vocabulary::kUnkToken;
    if (!has_unk) chars.emplace_back(Vocabulary::kUnkToken);
    const auto first_special = static_cast<int64_t>(chars.size());
    TextTokenizer t(std::make_shared<CharModel>(std::move(chars), Vocabulary::kUnkToken));
    t.set_decoder(std::make_shared<FuseDecoder>());
    for (size_t i = 0; i < special_tokens.size(); ++i) t.add_token({special_tokens[i], first_special + static_cast<int64_t>(i), true});
    return t;
}

}  // namespace pulsatrix
