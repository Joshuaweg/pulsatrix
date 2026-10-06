#include "unicode.hpp"

#include <algorithm>
#include <iterator>

#include "utf8.hpp"

namespace pulsatrix::unicode {

namespace {

struct CategoryRange {
    char32_t first;
    char32_t last;
    uint8_t category;
};
struct CombiningRange {
    char32_t first;
    char32_t last;
    uint8_t ccc;
};
struct Decomposition {
    char32_t cp;
    char32_t parts[4];
    uint8_t length;
};
struct Composition {
    char32_t first;
    char32_t second;
    char32_t composite;
};

#include "unicode_data.inc"

// Hangul syllables (Unicode 15.0, section 3.12).
constexpr char32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr char32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

template <typename Range>
const Range* FindRange(const Range* begin, const Range* end, char32_t cp) {
    const Range* it = std::upper_bound(begin, end, cp, [](char32_t c, const Range& r) { return c < r.first; });
    if (it == begin) return nullptr;
    --it;
    return cp <= it->last ? it : nullptr;
}

const Decomposition* FindDecomposition(char32_t cp) {
    const auto* end = std::end(kDecompositions);
    const auto* it = std::lower_bound(std::begin(kDecompositions), end, cp,
                                      [](const Decomposition& d, char32_t c) { return d.cp < c; });
    return it != end && it->cp == cp ? it : nullptr;
}

char32_t Compose(char32_t first, char32_t second) {
    if (first >= kLBase && first < kLBase + kLCount && second >= kVBase && second < kVBase + kVCount) {
        return kSBase + ((first - kLBase) * kVCount + (second - kVBase)) * kTCount;
    }
    if (first >= kSBase && first < kSBase + kSCount && (first - kSBase) % kTCount == 0 && second > kTBase &&
        second < kTBase + kTCount) {
        return first + (second - kTBase);
    }
    const auto* end = std::end(kCompositions);
    const auto* it = std::lower_bound(std::begin(kCompositions), end, std::make_pair(first, second),
                                      [](const Composition& c, const std::pair<char32_t, char32_t>& key) {
                                          return std::make_pair(c.first, c.second) < key;
                                      });
    return it != end && it->first == first && it->second == second ? it->composite : 0;
}

/** @brief Whether NFC can change a text holding @p cp: it decomposes, has a nonzero combining
 *         class, or can be the second half of a composition. */
bool MayChange(char32_t cp) {
    if (cp < 0x300) return cp >= 0xC0 && FindDecomposition(cp) != nullptr;  // fast path: Latin-1
    if (cp >= kSBase && cp < kSBase + kSCount) return true;
    if ((cp >= kVBase && cp < kVBase + kVCount) || (cp > kTBase && cp < kTBase + kTCount)) return true;
    if (CombiningClass(cp) != 0 || FindDecomposition(cp) != nullptr) return true;
    static const std::vector<char32_t> seconds = [] {
        std::vector<char32_t> s;
        for (const Composition& c : kCompositions) s.push_back(c.second);
        std::sort(s.begin(), s.end());
        s.erase(std::unique(s.begin(), s.end()), s.end());
        return s;
    }();
    return std::binary_search(seconds.begin(), seconds.end(), cp);
}

void Decompose(char32_t cp, size_t begin, size_t end, std::vector<NfcChar>& out) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
        const char32_t s = cp - kSBase;
        out.push_back({kLBase + s / kNCount, begin, end});
        out.push_back({kVBase + (s % kNCount) / kTCount, begin, end});
        if (s % kTCount != 0) out.push_back({kTBase + s % kTCount, begin, end});
        return;
    }
    if (const Decomposition* d = FindDecomposition(cp)) {
        for (uint8_t i = 0; i < d->length; ++i) Decompose(d->parts[i], begin, end, out);
        return;
    }
    out.push_back({cp, begin, end});
}

}  // namespace

const char* Version() { return kUnicodeVersion; }

Category GetCategory(char32_t cp) {
    const CategoryRange* r = FindRange(std::begin(kCategoryRanges), std::end(kCategoryRanges), cp);
    return r ? static_cast<Category>(r->category) : Category::Cn;
}

bool IsWhiteSpace(char32_t cp) {
    // PropList.txt's White_Space (Unicode 15.0).
    return (cp >= 0x9 && cp <= 0xD) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

uint8_t CombiningClass(char32_t cp) {
    if (cp < 0x300) return 0;
    const CombiningRange* r = FindRange(std::begin(kCombiningRanges), std::end(kCombiningRanges), cp);
    return r ? r->ccc : 0;
}

char32_t FoldCase(char32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp - 'A' + 'a';
    if (cp == 0x17F) return 's';
    if (cp == 0x212A) return 'k';
    return cp;
}

std::vector<NfcChar> Nfc(std::string_view text) {
    bool may_change = false;
    for (size_t i = 0; i < text.size() && !may_change;) {
        const utf8::CodePoint c = utf8::Decode(text, i);
        may_change = MayChange(c.value);
        i += std::max<size_t>(1, c.length);
    }
    if (!may_change) return {};

    std::vector<NfcChar> chars;
    for (size_t i = 0; i < text.size();) {
        const utf8::CodePoint c = utf8::Decode(text, i);
        const size_t length = std::max<size_t>(1, c.length);
        Decompose(c.value, i, i + length, chars);
        i += length;
    }
    // Canonical ordering: a stable sort of each run of non-starters by combining class.
    for (size_t i = 0; i < chars.size();) {
        if (CombiningClass(chars[i].cp) == 0) {
            ++i;
            continue;
        }
        size_t j = i;
        while (j < chars.size() && CombiningClass(chars[j].cp) != 0) ++j;
        std::stable_sort(chars.begin() + static_cast<ptrdiff_t>(i), chars.begin() + static_cast<ptrdiff_t>(j),
                         [](const NfcChar& a, const NfcChar& b) { return CombiningClass(a.cp) < CombiningClass(b.cp); });
        i = j;
    }
    // Canonical composition: C joins the last starter S unless something between them is a starter
    // or has a combining class >= C's.
    std::vector<NfcChar> out;
    out.reserve(chars.size());
    ptrdiff_t starter = -1;
    int last_ccc = -1;  // -1: nothing between the starter and the current character
    for (const NfcChar& c : chars) {
        const int ccc = CombiningClass(c.cp);
        if (starter >= 0 && (last_ccc == -1 || (last_ccc != 0 && last_ccc < ccc))) {
            NfcChar& s = out[static_cast<size_t>(starter)];
            if (const char32_t composite = Compose(s.cp, c.cp)) {
                s.cp = composite;
                s.source_begin = std::min(s.source_begin, c.source_begin);
                s.source_end = std::max(s.source_end, c.source_end);
                continue;
            }
        }
        out.push_back(c);
        if (ccc == 0) {
            starter = static_cast<ptrdiff_t>(out.size()) - 1;
            last_ccc = -1;
        } else {
            last_ccc = ccc;
        }
    }
    return out;
}

}  // namespace pulsatrix::unicode
