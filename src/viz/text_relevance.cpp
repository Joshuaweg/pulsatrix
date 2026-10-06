#include "pulsatrix/viz/text_relevance.hpp"

#include <algorithm>
#include <stdexcept>

namespace pulsatrix {

namespace {

bool IsContinuation(char c) { return (static_cast<unsigned char>(c) & 0xC0) == 0x80; }

/** @brief Widens a byte range of @p text to whole characters. */
Offset ToCharacters(std::string_view text, Offset o) {
    while (o.begin > 0 && o.begin < text.size() && IsContinuation(text[o.begin])) --o.begin;
    while (o.end < text.size() && IsContinuation(text[o.end])) ++o.end;
    return o;
}

void Add(TokenRelevanceDocument& doc, std::string piece, float score, bool scored) {
    doc.tokens.push_back(std::move(piece));
    doc.relevance.push_back(score);
    doc.scored.push_back(scored);
}

/** @brief Drops the scored flags when every piece is scored, the document's default. */
void Tidy(TokenRelevanceDocument& doc) {
    for (bool b : doc.scored) {
        if (!b) return;
    }
    doc.scored.clear();
}

}  // namespace

TokenRelevanceDocument MakeTokenRelevanceDocument(std::string_view text, const Encoding& encoding,
                                                  const std::vector<float>& token_scores, std::string method,
                                                  std::string target, bool include_special_tokens) {
    if (token_scores.size() != encoding.size()) {
        throw std::invalid_argument("MakeTokenRelevanceDocument: needs one score per token");
    }
    TokenRelevanceDocument doc;
    doc.method = std::move(method);
    doc.target = std::move(target);
    double unassigned = 0.0;
    bool any_unassigned = false;
    size_t covered = 0;        // text before this is shown
    bool open_piece = false;   // the last piece is a token piece that may still grow
    Offset last{};             // its span
    for (size_t i = 0; i < encoding.size(); ++i) {
        const Offset o = encoding.offsets[i];
        if (o.begin > o.end || o.end > text.size()) {
            throw std::invalid_argument("MakeTokenRelevanceDocument: token " + std::to_string(i) + "'s offset lies outside the text");
        }
        if (o.begin == o.end) {  // inserted by the post-processor: BOS, EOS
            if (include_special_tokens) {
                Add(doc, encoding.tokens[i], token_scores[i], true);
            } else {
                unassigned += token_scores[i];
                any_unassigned = true;
            }
            open_piece = false;
            continue;
        }
        const Offset span = ToCharacters(text, o);
        if (open_piece && span.begin < last.end) {
            // Shares a character with the previous token: one piece, scores summed.
            last.end = std::max(last.end, span.end);
            doc.tokens.back() = std::string(text.substr(last.begin, last.end - last.begin));
            doc.relevance.back() += token_scores[i];
            covered = std::max(covered, last.end);
            continue;
        }
        if (span.begin > covered) Add(doc, std::string(text.substr(covered, span.begin - covered)), 0.0f, false);
        const size_t begin = std::max(span.begin, covered);
        Add(doc, std::string(text.substr(begin, span.end - begin)), token_scores[i], true);
        last = {begin, span.end};
        covered = std::max(covered, span.end);
        open_piece = true;
    }
    if (covered < text.size()) Add(doc, std::string(text.substr(covered)), 0.0f, false);
    if (any_unassigned) doc.unassigned = static_cast<float>(unassigned);
    Tidy(doc);
    return doc;
}

TokenRelevanceDocument MakeWordRelevanceDocument(std::string_view text, const WordScores& words, std::string method,
                                                 std::string target) {
    TokenRelevanceDocument doc;
    doc.method = std::move(method);
    doc.target = std::move(target);
    doc.granularity = "word";
    size_t covered = 0;
    for (const WordScore& w : words.words) {
        if (w.span.begin < covered || w.span.end < w.span.begin || w.span.end > text.size()) {
            throw std::invalid_argument("MakeWordRelevanceDocument: words must lie in the text, in order");
        }
        if (w.span.begin > covered) Add(doc, std::string(text.substr(covered, w.span.begin - covered)), 0.0f, false);
        Add(doc, std::string(text.substr(w.span.begin, w.span.end - w.span.begin)), w.score, true);
        covered = w.span.end;
    }
    if (covered < text.size()) Add(doc, std::string(text.substr(covered)), 0.0f, false);
    if (words.unassigned != 0.0f) doc.unassigned = words.unassigned;
    Tidy(doc);
    return doc;
}

}  // namespace pulsatrix
