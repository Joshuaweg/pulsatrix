#include "pulsatrix/sentencepiece_bpe.hpp"

#include <algorithm>
#include <stdexcept>

#include "utf8.hpp"

namespace pulsatrix {

namespace {

/** @brief The byte ranges of every match of a literal or a regex, left to right, not overlapping. */
std::vector<UnicodeRegex::Match> Find(const std::optional<UnicodeRegex>& regex, const std::string& literal,
                                      const std::string& text) {
    if (regex) return regex->find_all(text);
    std::vector<UnicodeRegex::Match> out;
    for (size_t at = text.find(literal); at != std::string::npos; at = text.find(literal, at + literal.size())) {
        out.push_back({at, at + literal.size()});
    }
    return out;
}

std::string ReplaceAll(const std::optional<UnicodeRegex>& regex, const std::string& literal, const std::string& content,
                       const std::string& text) {
    std::string out;
    size_t prev = 0;
    for (const UnicodeRegex::Match& m : Find(regex, literal, text)) {
        out.append(text, prev, m.begin - prev);
        out += content;
        prev = m.end;
    }
    out.append(text, prev, std::string::npos);
    return out;
}

/** @brief A normalized string with every match replaced; each replacement comes from its match. */
void ReplaceIn(NormalizedString& text, const std::optional<UnicodeRegex>& regex, const std::string& literal,
               const std::string& content) {
    const std::vector<UnicodeRegex::Match> matches = Find(regex, literal, text.text());
    if (matches.empty()) return;
    std::vector<NormalizedString::Piece> pieces;
    size_t prev = 0;
    for (const UnicodeRegex::Match& m : matches) {
        for (size_t i = prev; i < m.begin; ++i) pieces.push_back({text.text().substr(i, 1), i, i + 1});
        pieces.push_back({content, m.begin, m.end});
        prev = m.end;
    }
    for (size_t i = prev; i < text.size(); ++i) pieces.push_back({text.text().substr(i, 1), i, i + 1});
    text.rebuild(pieces);
}

/** @brief Puts @p prefix in front of non-empty @p text. The prefix comes from the first
 *         character's source, as in Hugging Face's NormalizedString::prepend, so a `▁` that ends
 *         up a token of its own still points at the word it starts. */
void Prepend(NormalizedString& text, const std::string& prefix) {
    if (text.empty()) return;
    const size_t first = std::max<size_t>(1, utf8::Decode(text.text(), 0).length);
    std::vector<NormalizedString::Piece> pieces = {{prefix, 0, first}};
    for (size_t i = 0; i < text.size(); ++i) pieces.push_back({text.text().substr(i, 1), i, i + 1});
    text.rebuild(pieces);
}

}  // namespace

// ---- Normalizers ---------------------------------------------------------------------------

ReplaceNormalizer::ReplaceNormalizer(std::optional<UnicodeRegex> regex, std::string literal, std::string content)
    : regex_(std::move(regex)), literal_(std::move(literal)), content_(std::move(content)) {}

ReplaceNormalizer::ReplaceNormalizer(std::string pattern, std::string content)
    : ReplaceNormalizer(std::nullopt, std::move(pattern), std::move(content)) {
    if (literal_.empty()) throw std::invalid_argument("ReplaceNormalizer: empty pattern");
}

ReplaceNormalizer ReplaceNormalizer::Regex(const std::string& pattern, std::string content) {
    return ReplaceNormalizer(UnicodeRegex(pattern), "", std::move(content));
}

void ReplaceNormalizer::normalize(NormalizedString& text) const { ReplaceIn(text, regex_, literal_, content_); }

PrependNormalizer::PrependNormalizer(std::string prepend) : prepend_(std::move(prepend)) {}

void PrependNormalizer::normalize(NormalizedString& text) const { Prepend(text, prepend_); }

// ---- Metaspace -----------------------------------------------------------------------------

MetaspacePreTokenizer::MetaspacePreTokenizer(std::string replacement, PrependScheme scheme, bool split)
    : replacement_(std::move(replacement)), scheme_(scheme), split_(split) {
    if (replacement_.empty() || utf8::Decode(replacement_, 0).length != replacement_.size()) {
        throw std::invalid_argument("MetaspacePreTokenizer: the replacement must be one character");
    }
}

void MetaspacePreTokenizer::pre_tokenize(std::vector<NormalizedString>& splits) const {
    std::vector<NormalizedString> out;
    for (NormalizedString& split : splits) {
        ReplaceIn(split, std::nullopt, " ", replacement_);
        const bool starts_with = split.text().compare(0, replacement_.size(), replacement_) == 0;
        // First: only the piece that starts the original text.
        const bool first = !split.empty() && split.original(0, split.size()).begin == 0;
        if (!starts_with && (scheme_ == PrependScheme::Always || (scheme_ == PrependScheme::First && first))) {
            Prepend(split, replacement_);
        }
        if (!split_) {
            out.push_back(std::move(split));
            continue;
        }
        // Split before each replacement character (MergedWithNext).
        size_t start = 0;
        for (size_t at = split.text().find(replacement_, replacement_.size()); at != std::string::npos;
             at = split.text().find(replacement_, at + replacement_.size())) {
            out.push_back(split.slice(start, at));
            start = at;
        }
        if (start < split.size()) out.push_back(split.slice(start, split.size()));
    }
    splits = std::move(out);
}

// ---- Decoders ------------------------------------------------------------------------------

ReplaceDecoder::ReplaceDecoder(std::optional<UnicodeRegex> regex, std::string literal, std::string content)
    : regex_(std::move(regex)), literal_(std::move(literal)), content_(std::move(content)) {}

ReplaceDecoder::ReplaceDecoder(std::string pattern, std::string content)
    : ReplaceDecoder(std::nullopt, std::move(pattern), std::move(content)) {
    if (literal_.empty()) throw std::invalid_argument("ReplaceDecoder: empty pattern");
}

ReplaceDecoder ReplaceDecoder::Regex(const std::string& pattern, std::string content) {
    return ReplaceDecoder(UnicodeRegex(pattern), "", std::move(content));
}

std::vector<std::string> ReplaceDecoder::decode_chain(std::vector<std::string> tokens) const {
    for (std::string& t : tokens) t = ReplaceAll(regex_, literal_, content_, t);
    return tokens;
}

std::vector<std::string> ByteFallbackDecoder::decode_chain(std::vector<std::string> tokens) const {
    std::vector<std::string> out;
    std::string bytes;
    auto flush = [&] {
        if (bytes.empty()) return;
        if (utf8::IsValid(bytes)) {
            out.push_back(bytes);
        } else {
            for (size_t i = 0; i < bytes.size(); ++i) out.emplace_back("\xEF\xBF\xBD");
        }
        bytes.clear();
    };
    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    for (std::string& t : tokens) {
        const bool is_byte = t.size() == 6 && t.compare(0, 3, "<0x") == 0 && t[5] == '>' && hex(t[3]) >= 0 && hex(t[4]) >= 0;
        if (is_byte) {
            bytes.push_back(static_cast<char>(hex(t[3]) * 16 + hex(t[4])));
            continue;
        }
        flush();
        out.push_back(std::move(t));
    }
    flush();
    return out;
}

StripDecoder::StripDecoder(std::string content, size_t start, size_t stop)
    : content_(std::move(content)), start_(start), stop_(stop) {
    if (content_.empty() || utf8::Decode(content_, 0).length != content_.size()) {
        throw std::invalid_argument("StripDecoder: the content must be one character");
    }
}

std::vector<std::string> StripDecoder::decode_chain(std::vector<std::string> tokens) const {
    const size_t n = content_.size();
    for (std::string& t : tokens) {
        size_t begin = 0;
        for (size_t k = 0; k < start_ && t.compare(begin, n, content_) == 0 && begin + n <= t.size(); ++k) begin += n;
        size_t end = t.size();
        for (size_t k = 0; k < stop_ && end >= begin + n && t.compare(end - n, n, content_) == 0; ++k) end -= n;
        t = t.substr(begin, end - begin);
    }
    return tokens;
}

MetaspaceDecoder::MetaspaceDecoder(std::string replacement, PrependScheme scheme)
    : replacement_(std::move(replacement)), scheme_(scheme) {}

std::vector<std::string> MetaspaceDecoder::decode_chain(std::vector<std::string> tokens) const {
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::string out;
        const std::string& t = tokens[i];
        for (size_t p = 0; p < t.size();) {
            if (t.compare(p, replacement_.size(), replacement_) == 0) {
                // Hugging Face drops every replacement in the first token, not only the one the
                // pre-tokenizer added, so "▁▁▁▁word" as one first token decodes to "word".
                if (!(i == 0 && scheme_ != PrependScheme::Never)) out += ' ';
                p += replacement_.size();
            } else {
                out += t[p++];
            }
        }
        tokens[i] = std::move(out);
    }
    return tokens;
}

SequenceDecoder::SequenceDecoder(std::vector<std::shared_ptr<const Decoder>> decoders) : decoders_(std::move(decoders)) {}

std::vector<std::string> SequenceDecoder::decode_chain(std::vector<std::string> tokens) const {
    for (const auto& d : decoders_) tokens = d->decode_chain(std::move(tokens));
    return tokens;
}

}  // namespace pulsatrix
