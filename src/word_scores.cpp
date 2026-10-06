#include "pulsatrix/word_scores.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pulsatrix {

std::vector<Offset> SplitWords(std::string_view text, const UnicodeRegex& pattern) {
    std::vector<Offset> words;
    for (const UnicodeRegex::Match& m : pattern.find_all(text)) words.push_back({m.begin, m.end});
    return words;
}

std::vector<Offset> SplitWords(std::string_view text, WordSplit split) {
    static const UnicodeRegex kWordsAndPunctuation(R"(\w+|[^\w\s]+)");
    static const UnicodeRegex kWhitespace(R"(\S+)");
    return SplitWords(text, split == WordSplit::Whitespace ? kWhitespace : kWordsAndPunctuation);
}

WordScores AggregateToWords(std::string_view text, const std::vector<Offset>& token_offsets,
                            const std::vector<float>& token_scores, const std::vector<Offset>& words,
                            WordAggregation aggregation) {
    if (token_offsets.size() != token_scores.size()) {
        throw std::invalid_argument("AggregateToWords: one score per token is needed");
    }
    for (size_t j = 0; j < words.size(); ++j) {
        const Offset& w = words[j];
        if (w.begin > w.end || w.end > text.size() || (j > 0 && w.begin < words[j - 1].end)) {
            throw std::invalid_argument("AggregateToWords: words must lie in the text, in order, without overlapping");
        }
    }
    for (const Offset& o : token_offsets) {
        if (o.begin > o.end || o.end > text.size()) throw std::invalid_argument("AggregateToWords: a token offset lies outside the text");
    }

    WordScores out;
    out.words.resize(words.size());
    std::vector<double> weighted(words.size(), 0.0), weight(words.size(), 0.0);
    std::vector<float> best(words.size(), 0.0f);
    std::vector<bool> any(words.size(), false);
    double unassigned = 0.0;
    for (size_t i = 0; i < token_offsets.size(); ++i) {
        const Offset& o = token_offsets[i];
        const double score = token_scores[i];
        // The words this token overlaps, and how many of its bytes fall in each.
        const auto first = std::upper_bound(words.begin(), words.end(), o.begin,
                                            [](size_t pos, const Offset& w) { return pos < w.end; });
        std::vector<std::pair<size_t, size_t>> overlaps;  // (word, bytes)
        size_t in_words = 0;
        for (auto it = first; it != words.end() && it->begin < o.end; ++it) {
            const size_t bytes = std::min(o.end, it->end) - std::max(o.begin, it->begin);
            if (bytes == 0) continue;
            overlaps.emplace_back(static_cast<size_t>(it - words.begin()), bytes);
            in_words += bytes;
        }
        if (overlaps.empty()) {
            unassigned += score;
            continue;
        }
        for (const auto& [j, bytes] : overlaps) {
            const double share = static_cast<double>(bytes) / static_cast<double>(in_words);
            weighted[j] += share * score;
            weight[j] += share;
            const float s = token_scores[i];
            switch (aggregation) {
                case WordAggregation::Max:
                    best[j] = any[j] ? std::max(best[j], s) : s;
                    break;
                case WordAggregation::MaxAbs:
                    if (!any[j] || std::fabs(s) > std::fabs(best[j])) best[j] = s;
                    break;
                default:
                    break;
            }
            any[j] = true;
            out.words[j].tokens.push_back(i);
        }
    }
    for (size_t j = 0; j < words.size(); ++j) {
        WordScore& w = out.words[j];
        w.span = words[j];
        w.text = std::string(text.substr(words[j].begin, words[j].end - words[j].begin));
        switch (aggregation) {
            case WordAggregation::Sum:
                w.score = static_cast<float>(weighted[j]);
                break;
            case WordAggregation::Mean:
                w.score = weight[j] > 0.0 ? static_cast<float>(weighted[j] / weight[j]) : 0.0f;
                break;
            case WordAggregation::Max:
            case WordAggregation::MaxAbs:
                w.score = best[j];
                break;
        }
    }
    out.unassigned = static_cast<float>(unassigned);
    return out;
}

WordScores AggregateToWords(std::string_view text, const Encoding& encoding, const std::vector<float>& token_scores,
                            WordAggregation aggregation, WordSplit split) {
    return AggregateToWords(text, encoding.offsets, token_scores, SplitWords(text, split), aggregation);
}

}  // namespace pulsatrix
