#include "pulsatrix/byte_level_bpe.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>

#include "unicode.hpp"
#include "utf8.hpp"

namespace pulsatrix {

namespace {

/** @brief A piece of a split's text, and whether it matched the split pattern. */
struct Piece {
    size_t begin;
    size_t end;
    bool is_match;
};

/** @brief Matches -> the alternating match / gap pieces covering [0, size). */
std::vector<Piece> Cover(const std::vector<UnicodeRegex::Match>& matches, size_t size) {
    std::vector<Piece> pieces;
    size_t prev = 0;
    for (const UnicodeRegex::Match& m : matches) {
        if (m.begin > prev) pieces.push_back({prev, m.begin, false});
        pieces.push_back({m.begin, m.end, true});
        prev = m.end;
    }
    if (prev < size) pieces.push_back({prev, size, false});
    return pieces;
}

/** @brief Hugging Face's NormalizedString::split: applies a behavior to the pieces. */
std::vector<Piece> ApplyBehavior(std::vector<Piece> pieces, SplitBehavior behavior, bool invert) {
    if (invert) {
        for (Piece& p : pieces) p.is_match = !p.is_match;
    }
    std::vector<Piece> out;
    bool previous_match = false;
    switch (behavior) {
        case SplitBehavior::Isolated:
            return pieces;
        case SplitBehavior::Removed:
            for (const Piece& p : pieces) {
                if (!p.is_match) out.push_back(p);
            }
            return out;
        case SplitBehavior::MergedWithPrevious:
            for (const Piece& p : pieces) {
                if (p.is_match && !previous_match && !out.empty()) {
                    out.back().end = p.end;
                } else {
                    out.push_back(p);
                }
                previous_match = p.is_match;
            }
            return out;
        case SplitBehavior::MergedWithNext:
            for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) {
                if (it->is_match && !previous_match && !out.empty()) {
                    out.back().begin = it->begin;
                } else {
                    out.push_back(*it);
                }
                previous_match = it->is_match;
            }
            std::reverse(out.begin(), out.end());
            return out;
        case SplitBehavior::Contiguous:
            for (size_t i = 0; i < pieces.size(); ++i) {
                if (i > 0 && pieces[i].is_match == previous_match) {
                    out.back().end = pieces[i].end;
                } else {
                    out.push_back(pieces[i]);
                }
                previous_match = pieces[i].is_match;
            }
            return out;
    }
    return pieces;
}

void AppendSlices(const NormalizedString& split, const std::vector<Piece>& pieces, std::vector<NormalizedString>& out) {
    for (const Piece& p : pieces) {
        if (p.end > p.begin) out.push_back(split.slice(p.begin, p.end));
    }
}

const char* kGpt2Pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

struct ByteTables {
    std::vector<std::string> byte_to_char;              // 256 entries
    std::unordered_map<char32_t, unsigned char> char_to_byte;
};

const ByteTables& Tables() {
    static const ByteTables tables = [] {
        ByteTables t;
        t.byte_to_char.resize(256);
        std::vector<bool> printable(256, false);
        for (int b = '!'; b <= '~'; ++b) printable[static_cast<size_t>(b)] = true;
        for (int b = 0xA1; b <= 0xAC; ++b) printable[static_cast<size_t>(b)] = true;
        for (int b = 0xAE; b <= 0xFF; ++b) printable[static_cast<size_t>(b)] = true;
        char32_t next = 256;
        for (int b = 0; b < 256; ++b) {
            const char32_t c = printable[static_cast<size_t>(b)] ? static_cast<char32_t>(b) : next++;
            t.byte_to_char[static_cast<size_t>(b)] = utf8::Encode(c);
            t.char_to_byte[c] = static_cast<unsigned char>(b);
        }
        return t;
    }();
    return tables;
}

}  // namespace

const std::vector<std::string>& ByteLevelAlphabet() { return Tables().byte_to_char; }

// ---- Normalizers ---------------------------------------------------------------------------

void NfcNormalizer::normalize(NormalizedString& text) const {
    const std::vector<unicode::NfcChar> chars = unicode::Nfc(text.text());
    if (chars.empty()) return;  // already NFC
    std::vector<NormalizedString::Piece> pieces;
    pieces.reserve(chars.size());
    for (const unicode::NfcChar& c : chars) pieces.push_back({utf8::Encode(c.cp), c.source_begin, c.source_end});
    text.rebuild(pieces);
}

SequenceNormalizer::SequenceNormalizer(std::vector<std::shared_ptr<const Normalizer>> normalizers)
    : normalizers_(std::move(normalizers)) {}

void SequenceNormalizer::normalize(NormalizedString& text) const {
    for (const auto& n : normalizers_) n->normalize(text);
}

// ---- Pre-tokenizers ------------------------------------------------------------------------

SplitPreTokenizer::SplitPreTokenizer(std::optional<UnicodeRegex> regex, std::string literal, SplitBehavior behavior,
                                     bool invert)
    : regex_(std::move(regex)), literal_(std::move(literal)), behavior_(behavior), invert_(invert) {}

SplitPreTokenizer::SplitPreTokenizer(const std::string& regex, SplitBehavior behavior, bool invert)
    : SplitPreTokenizer(UnicodeRegex(regex), "", behavior, invert) {}

SplitPreTokenizer SplitPreTokenizer::Literal(const std::string& text, SplitBehavior behavior, bool invert) {
    if (text.empty()) throw std::invalid_argument("SplitPreTokenizer::Literal: empty pattern");
    return SplitPreTokenizer(std::nullopt, text, behavior, invert);
}

void SplitPreTokenizer::pre_tokenize(std::vector<NormalizedString>& splits) const {
    std::vector<NormalizedString> out;
    for (const NormalizedString& split : splits) {
        std::vector<UnicodeRegex::Match> matches;
        if (regex_) {
            matches = regex_->find_all(split.text());
        } else {
            for (size_t at = split.text().find(literal_); at != std::string::npos;
                 at = split.text().find(literal_, at + literal_.size())) {
                matches.push_back({at, at + literal_.size()});
            }
        }
        AppendSlices(split, ApplyBehavior(Cover(matches, split.size()), behavior_, invert_), out);
    }
    splits = std::move(out);
}

DigitsPreTokenizer::DigitsPreTokenizer(bool individual_digits) : individual_(individual_digits) {}

void DigitsPreTokenizer::pre_tokenize(std::vector<NormalizedString>& splits) const {
    std::vector<NormalizedString> out;
    for (const NormalizedString& split : splits) {
        std::vector<UnicodeRegex::Match> matches;  // each numeric character
        const std::string& s = split.text();
        for (size_t i = 0; i < s.size();) {
            const utf8::CodePoint c = utf8::Decode(s, i);
            if (unicode::IsNumber(c.value)) matches.push_back({i, i + c.length});
            i += std::max<size_t>(1, c.length);
        }
        const SplitBehavior behavior = individual_ ? SplitBehavior::Isolated : SplitBehavior::Contiguous;
        AppendSlices(split, ApplyBehavior(Cover(matches, s.size()), behavior, false), out);
    }
    splits = std::move(out);
}

ByteLevelPreTokenizer::ByteLevelPreTokenizer(bool add_prefix_space, bool use_regex)
    : add_prefix_space_(add_prefix_space) {
    if (use_regex) regex_.emplace(kGpt2Pattern);
}

void ByteLevelPreTokenizer::pre_tokenize(std::vector<NormalizedString>& splits) const {
    std::vector<NormalizedString> pieces;
    for (NormalizedString& split : splits) {
        if (add_prefix_space_ && split.text().rfind(' ', 0) != 0) {
            // An inserted space (empty origin), then every byte with its own origin.
            std::vector<NormalizedString::Piece> rebuilt = {{" ", 0, 0}};
            for (size_t i = 0; i < split.size(); ++i) rebuilt.push_back({split.text().substr(i, 1), i, i + 1});
            split.rebuild(rebuilt);
        }
        if (regex_) {
            AppendSlices(split, Cover(regex_->find_all(split.text()), split.size()), pieces);
        } else {
            pieces.push_back(std::move(split));
        }
    }
    const std::vector<std::string>& alphabet = Tables().byte_to_char;
    for (NormalizedString& piece : pieces) {
        std::vector<NormalizedString::Piece> mapped;
        mapped.reserve(piece.size());
        for (size_t i = 0; i < piece.size(); ++i) {
            mapped.push_back({alphabet[static_cast<unsigned char>(piece.text()[i])], i, i + 1});
        }
        piece.rebuild(mapped);
    }
    splits = std::move(pieces);
}

SequencePreTokenizer::SequencePreTokenizer(std::vector<std::shared_ptr<const PreTokenizer>> pre_tokenizers)
    : pre_tokenizers_(std::move(pre_tokenizers)) {}

void SequencePreTokenizer::pre_tokenize(std::vector<NormalizedString>& splits) const {
    for (const auto& p : pre_tokenizers_) p->pre_tokenize(splits);
}

// ---- BPE -----------------------------------------------------------------------------------

BpeModel::BpeModel(std::unordered_map<std::string, int64_t> vocab,
                   const std::vector<std::pair<std::string, std::string>>& merges, BpeOptions options)
    : vocab_(std::move(vocab)), options_(std::move(options)) {
    for (const auto& [token, id] : vocab_) {
        tokens_[id] = token;
        vocab_size_ = std::max(vocab_size_, id + 1);
    }
    const size_t prefix = options_.continuing_subword_prefix.size();
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const auto& [a, b] = merges[rank];
        const auto ia = vocab_.find(a);
        const auto ib = vocab_.find(b);
        // The merged token drops b's continuing-subword prefix (Hugging Face's BPE builder).
        const std::string merged = a + (prefix > 0 && b.compare(0, prefix, options_.continuing_subword_prefix) == 0 ? b.substr(prefix) : b);
        const auto im = vocab_.find(merged);
        if (ia == vocab_.end() || ib == vocab_.end() || im == vocab_.end()) {
            throw std::invalid_argument("BpeModel: merge " + std::to_string(rank) + " (\"" + a + "\" \"" + b +
                                        "\") uses a token that isn't in the vocabulary");
        }
        // A repeated pair takes its last rank, as Hugging Face's HashMap collect does.
        merges_.insert_or_assign(std::make_pair(ia->second, ib->second), std::make_pair(static_cast<int64_t>(rank), im->second));
    }
    if (options_.unk_token) {
        const auto unk = vocab_.find(*options_.unk_token);
        if (unk == vocab_.end()) throw std::invalid_argument("BpeModel: the unknown token isn't in the vocabulary");
        unk_id_ = unk->second;
    }
}

std::vector<ModelToken> BpeModel::tokenize(std::string_view piece) const {
    if (piece.empty()) return {};
    if (options_.ignore_merges) {
        const auto whole = vocab_.find(std::string(piece));
        if (whole != vocab_.end()) return {ModelToken{whole->second, whole->first, 0, piece.size()}};
    }
    // Symbols: one per character to start, as a doubly linked list over this vector.
    struct Symbol {
        int64_t id;
        size_t begin, end;
        ptrdiff_t prev, next;
    };
    std::vector<Symbol> symbols;
    for (size_t i = 0; i < piece.size();) {
        const size_t length = std::max<size_t>(1, utf8::Decode(piece, i).length);
        std::string ch(piece.substr(i, length));
        if (i > 0) ch = options_.continuing_subword_prefix + ch;
        if (i + length == piece.size()) ch += options_.end_of_word_suffix;
        const auto it = vocab_.find(ch);
        if (it != vocab_.end()) {
            symbols.push_back({it->second, i, i + length, 0, 0});
        } else if (unk_id_) {
            symbols.push_back({*unk_id_, i, i + length, 0, 0});
        }  // else dropped, as in Hugging Face
        i += length;
    }
    for (size_t i = 0; i < symbols.size(); ++i) {
        symbols[i].prev = static_cast<ptrdiff_t>(i) - 1;
        symbols[i].next = i + 1 < symbols.size() ? static_cast<ptrdiff_t>(i) + 1 : -1;
    }
    // Hugging Face's Word::merge_all: lowest rank first, then leftmost; stale entries are skipped
    // by re-checking the pair at pop time.
    struct Merge {
        int64_t rank;
        size_t pos;
        int64_t new_id;
        bool operator>(const Merge& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
    };
    std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>> queue;
    auto push = [&](size_t left) {
        const ptrdiff_t right = symbols[left].next;
        if (right < 0) return;
        const auto m = merges_.find({symbols[left].id, symbols[static_cast<size_t>(right)].id});
        if (m != merges_.end()) queue.push({m->second.first, left, m->second.second});
    };
    for (size_t i = 0; i + 1 < symbols.size(); ++i) push(i);
    std::vector<bool> removed(symbols.size(), false);
    while (!queue.empty()) {
        const Merge top = queue.top();
        queue.pop();
        if (removed[top.pos] || symbols[top.pos].next < 0) continue;
        const auto right = static_cast<size_t>(symbols[top.pos].next);
        const auto m = merges_.find({symbols[top.pos].id, symbols[right].id});
        if (m == merges_.end() || m->second.second != top.new_id) continue;
        Symbol& left = symbols[top.pos];
        left.id = top.new_id;
        left.end = symbols[right].end;
        left.next = symbols[right].next;
        removed[right] = true;
        if (left.next >= 0) symbols[static_cast<size_t>(left.next)].prev = static_cast<ptrdiff_t>(top.pos);
        if (left.prev >= 0) push(static_cast<size_t>(left.prev));
        push(top.pos);
    }
    std::vector<ModelToken> out;
    for (size_t i = 0; i < symbols.size(); ++i) {
        if (removed[i]) continue;
        out.push_back({symbols[i].id, tokens_.at(symbols[i].id), symbols[i].begin, symbols[i].end});
    }
    return out;
}

std::optional<int64_t> BpeModel::token_to_id(std::string_view token) const {
    const auto it = vocab_.find(std::string(token));
    if (it == vocab_.end()) return std::nullopt;
    return it->second;
}

std::optional<std::string> BpeModel::id_to_token(int64_t id) const {
    const auto it = tokens_.find(id);
    if (it == tokens_.end()) return std::nullopt;
    return it->second;
}

// ---- Post-processors -----------------------------------------------------------------------

ByteLevelPostProcessor::ByteLevelPostProcessor(bool add_prefix_space, bool trim_offsets)
    : add_prefix_space_(add_prefix_space), trim_offsets_(trim_offsets) {}

Encoding ByteLevelPostProcessor::process(Encoding encoding, bool /*add_special_tokens*/) const {
    if (!trim_offsets_) return encoding;
    // Hugging Face's byte_level::process_offsets. 'Ġ' (U+0120) stands for a space byte, one byte
    // of the input, so counting characters counts bytes.
    const std::string space = Tables().byte_to_char[' '];
    for (size_t i = 0; i < encoding.size(); ++i) {
        const std::string& token = encoding.tokens[i];
        Offset& o = encoding.offsets[i];
        size_t leading = 0, trailing = 0;
        while ((leading + 1) * space.size() <= token.size() && token.compare(leading * space.size(), space.size(), space) == 0) {
            ++leading;
        }
        while ((trailing + 1) * space.size() <= token.size() &&
               token.compare(token.size() - (trailing + 1) * space.size(), space.size(), space) == 0) {
            ++trailing;
        }
        if (leading > 0) {
            const bool is_first = i == 0 || o.begin == 0;
            if (is_first && add_prefix_space_ && leading == 1) leading = 0;
            o.begin = std::min(o.begin + leading, o.end);
        }
        if (trailing > 0 && o.end >= trailing) o.end = std::max(o.end - trailing, o.begin);
    }
    return encoding;
}

TemplatePostProcessor::TemplatePostProcessor(std::vector<Item> single) : single_(std::move(single)) {
    for (const Item& item : single_) {
        if (item.special_ids.size() != item.special_tokens.size()) {
            throw std::invalid_argument("TemplatePostProcessor: a special token's ids and tokens differ in length");
        }
    }
}

Encoding TemplatePostProcessor::process(Encoding encoding, bool add_special_tokens) const {
    if (!add_special_tokens) return encoding;
    Encoding out;
    for (const Item& item : single_) {
        if (item.is_sequence()) {
            for (size_t i = 0; i < encoding.size(); ++i) {
                out.push_back(encoding.ids[i], encoding.tokens[i], encoding.offsets[i], encoding.special_tokens_mask[i] != 0);
            }
        } else {
            for (size_t i = 0; i < item.special_ids.size(); ++i) out.push_back(item.special_ids[i], item.special_tokens[i], {0, 0}, true);
        }
    }
    return out;
}

SequencePostProcessor::SequencePostProcessor(std::vector<std::shared_ptr<const PostProcessor>> processors)
    : processors_(std::move(processors)) {}

Encoding SequencePostProcessor::process(Encoding encoding, bool add_special_tokens) const {
    for (const auto& p : processors_) encoding = p->process(std::move(encoding), add_special_tokens);
    return encoding;
}

// ---- Decoder -------------------------------------------------------------------------------

namespace {

/** @brief @p bytes as UTF-8, each maximal invalid subsequence replaced by U+FFFD (Rust's
 *         String::from_utf8_lossy, which Hugging Face uses). */
std::string Utf8Lossy(const std::string& bytes) {
    std::string out;
    const auto at = [&](size_t i) { return static_cast<unsigned char>(bytes[i]); };
    for (size_t i = 0; i < bytes.size();) {
        const utf8::CodePoint c = utf8::Decode(bytes, i);
        if (c.length != 0) {
            out.append(bytes, i, c.length);
            i += c.length;
            continue;
        }
        // The maximal prefix of a valid sequence (Unicode 15.0, Table 3-7).
        const unsigned char b0 = at(i);
        size_t need = 0;
        unsigned char lo = 0x80, hi = 0xBF;
        if (b0 >= 0xC2 && b0 <= 0xDF) {
            need = 1;
        } else if (b0 >= 0xE0 && b0 <= 0xEF) {
            need = 2;
            if (b0 == 0xE0) lo = 0xA0;
            if (b0 == 0xED) hi = 0x9F;
        } else if (b0 >= 0xF0 && b0 <= 0xF4) {
            need = 3;
            if (b0 == 0xF0) lo = 0x90;
            if (b0 == 0xF4) hi = 0x8F;
        }
        size_t j = i + 1;
        for (size_t k = 0; k < need && j < bytes.size(); ++k, ++j) {
            const unsigned char b = at(j);
            if (k == 0 ? (b < lo || b > hi) : (b < 0x80 || b > 0xBF)) break;
        }
        out += "\xEF\xBF\xBD";
        i = j;
    }
    return out;
}

}  // namespace

std::string ByteLevelDecoder::decode(const std::vector<std::string>& tokens) const {
    const auto& char_to_byte = Tables().char_to_byte;
    std::string bytes;
    for (const std::string& t : tokens) {
        for (size_t i = 0; i < t.size();) {
            const utf8::CodePoint c = utf8::Decode(t, i);
            const size_t length = std::max<size_t>(1, c.length);
            const auto b = char_to_byte.find(c.value);
            if (c.length != 0 && b != char_to_byte.end()) {
                bytes.push_back(static_cast<char>(b->second));
            } else {
                bytes.append(t, i, length);
            }
            i += length;
        }
    }
    return Utf8Lossy(bytes);
}

}  // namespace pulsatrix
