#include "pulsatrix/unicode_regex.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "unicode.hpp"
#include "utf8.hpp"

namespace pulsatrix {

namespace {

using unicode::Category;

/** @brief A set of code points: explicit ranges plus category and shorthand-class tests. */
struct CharSet {
    struct Range {
        char32_t first;
        char32_t last;
    };
    std::vector<Range> ranges;
    /** @brief Bit i set: category i is in the set. */
    uint32_t categories = 0;
    bool space = false, not_space = false, word = false, not_word = false;
    std::vector<uint32_t> not_categories;  ///< each entry: a \P{...} mask (in the set if outside it)
    bool negated = false;

    [[nodiscard]] bool contains_raw(char32_t cp) const {
        for (const Range& r : ranges) {
            if (cp >= r.first && cp <= r.last) return true;
        }
        if (categories != 0 || !not_categories.empty() || word || not_word) {
            const uint32_t bit = 1u << static_cast<unsigned>(unicode::GetCategory(cp));
            if ((categories & bit) != 0) return true;
            for (uint32_t mask : not_categories) {
                if ((mask & bit) == 0) return true;
            }
            const bool is_word = unicode::IsLetter(cp) || unicode::IsMark(cp) ||
                                 unicode::GetCategory(cp) == Category::Nd || unicode::GetCategory(cp) == Category::Pc;
            if ((word && is_word) || (not_word && !is_word)) return true;
        }
        if (space || not_space) {
            const bool ws = unicode::IsWhiteSpace(cp);
            if ((space && ws) || (not_space && !ws)) return true;
        }
        return false;
    }

    [[nodiscard]] bool contains(char32_t cp, bool ignore_case) const {
        bool in = contains_raw(cp);
        if (!in && ignore_case) {
            const char32_t folded = unicode::FoldCase(cp);
            in = (folded != cp && contains_raw(folded)) || (cp >= 'a' && cp <= 'z' && contains_raw(cp - 'a' + 'A'));
        }
        return in != negated;
    }
};

uint32_t CategoryMask(std::string_view name) {
    static const char* kNames[] = {"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe",
                                   "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"};
    uint32_t mask = 0;
    if (name == "L&" || name == "LC") return 0b111;  // Lu, Ll, Lt
    for (unsigned i = 0; i < 30; ++i) {
        const std::string_view n = kNames[i];
        if (n == name || (name.size() == 1 && n[0] == name[0])) mask |= 1u << i;
    }
    if (name == "Any") return ~0u;
    return mask;
}

enum class NodeKind { Set, Literal, Group, Repeat, Lookahead };

struct Node;
using Sequence = std::vector<std::unique_ptr<Node>>;

struct Node {
    NodeKind kind = NodeKind::Literal;
    bool ignore_case = false;
    char32_t literal = 0;            // Literal
    CharSet set;                     // Set
    std::vector<Sequence> branches;  // Group, Lookahead: alternatives
    std::unique_ptr<Node> child;     // Repeat
    size_t min = 0, max = 0;         // Repeat
    bool greedy = true;              // Repeat
    bool negate = false;             // Lookahead

    [[nodiscard]] bool single_char() const { return kind == NodeKind::Set || kind == NodeKind::Literal; }
    [[nodiscard]] bool matches_char(char32_t cp) const {
        if (kind == NodeKind::Literal) {
            return cp == literal || (ignore_case && unicode::FoldCase(cp) == unicode::FoldCase(literal));
        }
        return set.contains(cp, ignore_case);
    }
};

constexpr size_t kUnbounded = std::numeric_limits<size_t>::max();

// ---- Parser --------------------------------------------------------------------------------

class Parser {
public:
    explicit Parser(std::string_view pattern) {
        for (size_t i = 0; i < pattern.size();) {
            const utf8::CodePoint c = utf8::Decode(pattern, i);
            if (c.length == 0) throw std::invalid_argument("UnicodeRegex: the pattern isn't valid UTF-8");
            cps_.push_back(c.value);
            i += c.length;
        }
    }

    std::vector<Sequence> parse() {
        std::vector<Sequence> alternatives = parse_alternatives(false);
        if (pos_ != cps_.size()) fail("unbalanced ')'");
        return alternatives;
    }

private:
    [[noreturn]] void fail(const std::string& what) const {
        throw std::invalid_argument("UnicodeRegex: " + what + " at character " + std::to_string(pos_));
    }
    bool done() const { return pos_ >= cps_.size(); }
    char32_t peek() const { return cps_[pos_]; }
    bool accept(char32_t c) {
        if (!done() && peek() == c) {
            ++pos_;
            return true;
        }
        return false;
    }

    std::vector<Sequence> parse_alternatives(bool ignore_case) {
        std::vector<Sequence> alternatives;
        alternatives.push_back(parse_sequence(ignore_case));
        while (accept('|')) alternatives.push_back(parse_sequence(ignore_case));
        return alternatives;
    }

    Sequence parse_sequence(bool ignore_case) {
        Sequence seq;
        while (!done() && peek() != '|' && peek() != ')') {
            std::unique_ptr<Node> atom = parse_atom(ignore_case);
            seq.push_back(parse_quantifier(std::move(atom)));
        }
        return seq;
    }

    std::unique_ptr<Node> parse_quantifier(std::unique_ptr<Node> atom) {
        if (done()) return atom;
        size_t min = 0, max = 0;
        const size_t start = pos_;
        if (accept('?')) {
            max = 1;
        } else if (accept('*')) {
            max = kUnbounded;
        } else if (accept('+')) {
            min = 1;
            max = kUnbounded;
        } else if (peek() == '{' && parse_braces(min, max)) {
        } else {
            return atom;
        }
        if (atom->kind == NodeKind::Lookahead) {
            pos_ = start;
            fail("a quantifier on a lookahead");
        }
        auto repeat = std::make_unique<Node>();
        repeat->kind = NodeKind::Repeat;
        repeat->min = min;
        repeat->max = max;
        repeat->greedy = !accept('?');
        if (!done() && (peek() == '+' || peek() == '*' || peek() == '?' || peek() == '{')) fail("nested or possessive quantifier");
        repeat->child = std::move(atom);
        return repeat;
    }

    // "{n}", "{n,}" or "{n,m}"; anything else is a literal '{' (as in Oniguruma), returning false.
    bool parse_braces(size_t& min, size_t& max) {
        size_t p = pos_ + 1;
        auto number = [&](size_t& out) {
            const size_t begin = p;
            out = 0;
            while (p < cps_.size() && cps_[p] >= '0' && cps_[p] <= '9') out = out * 10 + (cps_[p++] - '0');
            return p > begin;
        };
        if (!number(min)) return false;
        max = min;
        if (p < cps_.size() && cps_[p] == ',') {
            ++p;
            if (!number(max)) max = kUnbounded;
        }
        if (p >= cps_.size() || cps_[p] != '}') return false;
        if (max < min) fail("{n,m} with m < n");
        pos_ = p + 1;
        return true;
    }

    std::unique_ptr<Node> parse_atom(bool ignore_case) {
        auto node = std::make_unique<Node>();
        node->ignore_case = ignore_case;
        const char32_t c = cps_[pos_++];
        switch (c) {
            case '(':
                return parse_group(ignore_case);
            case '[':
                node->kind = NodeKind::Set;
                node->set = parse_class();
                return node;
            case '.':
                node->kind = NodeKind::Set;
                node->set.ranges = {{'\n', '\n'}};
                node->set.negated = true;
                return node;
            case '\\':
                parse_escape(*node, false);
                return node;
            case '*':
            case '+':
            case '?':
                --pos_;
                fail("a quantifier with nothing to repeat");
            case '^':
            case '$':
                --pos_;
                fail("anchors aren't supported");
            default:
                node->kind = NodeKind::Literal;
                node->literal = c;
                return node;
        }
    }

    std::unique_ptr<Node> parse_group(bool ignore_case) {
        auto node = std::make_unique<Node>();
        node->kind = NodeKind::Group;
        if (accept('?')) {
            if (accept(':')) {
            } else if (accept('=')) {
                node->kind = NodeKind::Lookahead;
            } else if (accept('!')) {
                node->kind = NodeKind::Lookahead;
                node->negate = true;
            } else if (accept('i')) {
                if (!accept(':')) fail("only (?i:...) is supported for flags");
                ignore_case = true;
            } else {
                fail("unsupported group syntax");
            }
        }
        node->branches = parse_alternatives(ignore_case);
        if (!accept(')')) fail("missing ')'");
        return node;
    }

    CharSet parse_class() {
        CharSet set;
        set.negated = accept('^');
        bool first = true;
        while (true) {
            if (done()) fail("missing ']'");
            if (peek() == ']' && !first) {
                ++pos_;
                return set;
            }
            first = false;
            char32_t lo = 0;
            if (accept('\\')) {
                Node escaped;
                parse_escape(escaped, true);
                if (escaped.kind == NodeKind::Set) {
                    merge(set, escaped.set);
                    continue;
                }
                lo = escaped.literal;
            } else {
                if (peek() == '[') fail("nested classes aren't supported");
                lo = cps_[pos_++];
            }
            char32_t hi = lo;
            if (pos_ + 1 < cps_.size() && peek() == '-' && cps_[pos_ + 1] != ']') {
                ++pos_;
                if (accept('\\')) {
                    Node escaped;
                    parse_escape(escaped, true);
                    if (escaped.kind != NodeKind::Literal) fail("a class shorthand can't end a range");
                    hi = escaped.literal;
                } else {
                    hi = cps_[pos_++];
                }
                if (hi < lo) fail("a range out of order");
            }
            set.ranges.push_back({lo, hi});
        }
    }

    static void merge(CharSet& into, const CharSet& from) {
        // from is a shorthand (\s, \p{..}, ...): never negated as a whole except \P / \S / \W,
        // which parse_escape expresses through the not_* fields.
        into.ranges.insert(into.ranges.end(), from.ranges.begin(), from.ranges.end());
        into.categories |= from.categories;
        into.not_categories.insert(into.not_categories.end(), from.not_categories.begin(), from.not_categories.end());
        into.space |= from.space;
        into.not_space |= from.not_space;
        into.word |= from.word;
        into.not_word |= from.not_word;
    }

    uint32_t parse_hex(size_t digits_min, size_t digits_max, bool braced) {
        uint32_t value = 0;
        size_t n = 0;
        while (!done() && n < digits_max) {
            const char32_t h = peek();
            int d = -1;
            if (h >= '0' && h <= '9') d = static_cast<int>(h - '0');
            if (h >= 'a' && h <= 'f') d = static_cast<int>(h - 'a' + 10);
            if (h >= 'A' && h <= 'F') d = static_cast<int>(h - 'A' + 10);
            if (d < 0) break;
            value = value * 16 + static_cast<uint32_t>(d);
            ++pos_;
            ++n;
        }
        if (n < digits_min) fail("a bad hex escape");
        if (braced && !accept('}')) fail("missing '}' in \\x{...}");
        if (value > 0x10FFFF) fail("a code point above U+10FFFF");
        return value;
    }

    void parse_escape(Node& node, bool in_class) {
        if (done()) fail("a trailing backslash");
        const char32_t c = cps_[pos_++];
        node.kind = NodeKind::Set;
        CharSet& s = node.set;
        switch (c) {
            case 's': s.space = true; return;
            case 'S': s.not_space = true; return;
            case 'd': s.categories = CategoryMask("Nd"); return;
            case 'D': s.not_categories.push_back(CategoryMask("Nd")); return;
            case 'w': s.word = true; return;
            case 'W': s.not_word = true; return;
            case 'p':
            case 'P': {
                if (!accept('{')) fail("\\p needs {...}");
                std::string name;
                bool negate = c == 'P';
                if (accept('^')) negate = !negate;
                while (!done() && peek() != '}') name.push_back(static_cast<char>(cps_[pos_++]));
                if (!accept('}')) fail("missing '}' in \\p{...}");
                const uint32_t mask = CategoryMask(name);
                if (mask == 0) fail("unknown category \\p{" + name + "}");
                if (negate) {
                    s.not_categories.push_back(mask);
                } else {
                    s.categories = mask;
                }
                return;
            }
            default:
                break;
        }
        node.kind = NodeKind::Literal;
        switch (c) {
            case 'n': node.literal = '\n'; return;
            case 'r': node.literal = '\r'; return;
            case 't': node.literal = '\t'; return;
            case 'f': node.literal = '\f'; return;
            case 'v': node.literal = '\v'; return;
            case 'a': node.literal = 0x07; return;
            case 'e': node.literal = 0x1B; return;
            case '0': node.literal = 0; return;
            case 'x':
                node.literal = accept('{') ? parse_hex(1, 8, true) : parse_hex(2, 2, false);
                return;
            case 'u':
                node.literal = parse_hex(4, 4, false);
                return;
            default:
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
                    --pos_;
                    fail(std::string("unsupported escape \\") + static_cast<char>(c) + (in_class ? " in a class" : ""));
                }
                node.literal = c;  // escaped punctuation, or any other character
        }
    }

    std::vector<char32_t> cps_;
    size_t pos_ = 0;
};

}  // namespace

// ---- Matcher -------------------------------------------------------------------------------

struct UnicodeRegex::Program {
    std::vector<Sequence> alternatives;
};

namespace {

/** @brief What to match after the current sequence ends: the rest of an enclosing sequence, or
 *         another iteration of a repeat. */
struct Continuation {
    const Sequence* seq = nullptr;
    size_t index = 0;
    const Continuation* next = nullptr;
    // Repeat iteration: non-null node, with the iteration count and where the iteration began.
    const Node* repeat = nullptr;
    size_t count = 0;
    size_t start = 0;
};

class Matcher {
public:
    explicit Matcher(const std::vector<char32_t>& subject) : s_(subject) {}

    /** @brief The end of a match of @p alternatives starting at @p pos, if any. */
    bool match(const std::vector<Sequence>& alternatives, size_t pos, size_t& end) {
        for (const Sequence& alt : alternatives) {
            if (run(alt, 0, pos, nullptr)) {
                end = end_;
                return true;
            }
        }
        return false;
    }

private:
    bool resume(const Continuation* k, size_t pos) {
        if (k == nullptr) {
            end_ = pos;
            return true;
        }
        if (k->repeat != nullptr) return repeat_from(*k->repeat, k->count, k->start, pos, *k->seq, k->index, k->next);
        return run(*k->seq, k->index, pos, k->next);
    }

    bool run(const Sequence& seq, size_t index, size_t pos, const Continuation* k) {
        while (index < seq.size()) {
            const Node& n = *seq[index];
            switch (n.kind) {
                case NodeKind::Literal:
                case NodeKind::Set:
                    if (pos >= s_.size() || !n.matches_char(s_[pos])) return false;
                    ++pos;
                    ++index;
                    continue;
                case NodeKind::Group: {
                    const Continuation after{&seq, index + 1, k};
                    for (const Sequence& b : n.branches) {
                        if (run(b, 0, pos, &after)) return true;
                    }
                    return false;
                }
                case NodeKind::Lookahead: {
                    const size_t saved = end_;
                    bool found = false;
                    for (const Sequence& b : n.branches) {
                        if (run(b, 0, pos, nullptr)) {
                            found = true;
                            break;
                        }
                    }
                    end_ = saved;
                    if (found == n.negate) return false;
                    ++index;
                    continue;
                }
                case NodeKind::Repeat:
                    return repeat(n, pos, seq, index + 1, k);
            }
        }
        return resume(k, pos);
    }

    bool repeat(const Node& n, size_t pos, const Sequence& seq, size_t next, const Continuation* k) {
        if (n.child->single_char()) {
            size_t count = 0;
            while (count < n.max && pos + count < s_.size() && n.child->matches_char(s_[pos + count])) ++count;
            if (count < n.min) return false;
            if (n.greedy) {
                for (size_t c = count + 1; c-- > n.min;) {
                    if (run(seq, next, pos + c, k)) return true;
                }
            } else {
                for (size_t c = n.min; c <= count; ++c) {
                    if (run(seq, next, pos + c, k)) return true;
                }
            }
            return false;
        }
        return repeat_from(n, 0, pos, pos, seq, next, k);
    }

    // General repeat: @p count iterations are done, the last one began at @p start and ended at @p pos.
    bool repeat_from(const Node& n, size_t count, size_t start, size_t pos, const Sequence& seq, size_t next,
                     const Continuation* k) {
        if (count > n.min && pos == start) return false;  // an empty iteration past the minimum: stop looping
        const bool can_more = count < n.max;
        const bool can_stop = count >= n.min;
        const Continuation again{&seq, next, k, &n, count + 1, pos};
        // A repeated atom that isn't one character is a group: the parser refuses nested
        // quantifiers and quantified lookaheads.
        auto more = [&] {
            if (!can_more) return false;
            for (const Sequence& b : n.child->branches) {
                if (run(b, 0, pos, &again)) return true;
            }
            return false;
        };
        if (n.greedy) return more() || (can_stop && run(seq, next, pos, k));
        return (can_stop && run(seq, next, pos, k)) || more();
    }

    const std::vector<char32_t>& s_;
    size_t end_ = 0;
};

}  // namespace

UnicodeRegex::UnicodeRegex(std::string_view pattern)
    : pattern_(pattern), program_(std::make_unique<Program>()) {
    program_->alternatives = Parser(pattern).parse();
}

UnicodeRegex::~UnicodeRegex() = default;
UnicodeRegex::UnicodeRegex(UnicodeRegex&&) noexcept = default;
UnicodeRegex& UnicodeRegex::operator=(UnicodeRegex&&) noexcept = default;

std::vector<UnicodeRegex::Match> UnicodeRegex::find_all(std::string_view text) const {
    std::vector<char32_t> cps;
    std::vector<size_t> byte_at;  // byte offset of each code point, plus the end
    for (size_t i = 0; i < text.size();) {
        const utf8::CodePoint c = utf8::Decode(text, i);
        if (c.length == 0) throw std::invalid_argument("UnicodeRegex::find_all: the text isn't valid UTF-8");
        cps.push_back(c.value);
        byte_at.push_back(i);
        i += c.length;
    }
    byte_at.push_back(text.size());
    Matcher matcher(cps);
    std::vector<Match> out;
    for (size_t pos = 0; pos < cps.size();) {
        size_t end = 0;
        if (matcher.match(program_->alternatives, pos, end) && end > pos) {
            out.push_back({byte_at[pos], byte_at[end]});
            pos = end;
        } else {
            ++pos;
        }
    }
    return out;
}

}  // namespace pulsatrix
