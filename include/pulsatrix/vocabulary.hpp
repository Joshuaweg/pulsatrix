/** @file vocabulary.hpp
 *  @brief Token<->index lookup with a reserved <unk> fallback, plus a frequency-ranked builder.
 *  @ingroup dl_modules
 */
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pulsatrix {

/**
 * @brief Token<->index lookup table. Index 0 is always the reserved "<unk>" token --
 *        guaranteed by construction, not caller convention: the constructor takes the
 *        ranked list of *real* tokens and prepends "<unk>" itself.
 */
class Vocabulary {
public:
    static constexpr int64_t kUnkIndex = 0;
    static constexpr const char* kUnkToken = "<unk>";

    /** @param ranked_tokens Real (non-<unk>) tokens, in the order they should be indexed from 1. */
    explicit Vocabulary(std::vector<std::string> ranked_tokens);

    /** @brief The token's index, or kUnkIndex if the token isn't in this vocabulary. */
    [[nodiscard]] int64_t IndexOf(const std::string& token) const;

    /**
     * @brief The token at a given index.
     * @note PULSATRIX_ASSERT-gated, not throw -- internal invariant: every call site in
     *       this codebase passes an index already known to be in range (from IndexOf's own
     *       return value or a corpus's known vocabulary size), matching Shape::dim()'s
     *       classification.
     */
    [[nodiscard]] const std::string& TokenAt(int64_t index) const;

    /** @brief Total token count, including the reserved <unk> at index 0. */
    [[nodiscard]] int64_t size() const { return static_cast<int64_t>(tokens_.size()); }

private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int64_t> token_to_index_;
};

/**
 * @brief Builds a Vocabulary from a tokenized corpus, ranked by descending token frequency
 *        (ties broken by first-seen order, for determinism).
 * @param tokenized_corpus One token sequence per document/line (e.g. Tokenizer::Tokenize's
 *        output for each line of a corpus).
 * @param max_vocab_size If >= 0, keeps only the top max_vocab_size most frequent real
 *        tokens (the reserved <unk> at index 0 doesn't count against this limit). -1
 *        (default) keeps every distinct token seen.
 * @return The built Vocabulary. An empty corpus produces a Vocabulary containing only
 *         <unk> (size() == 1) -- not an error, just nothing to rank.
 */
[[nodiscard]] Vocabulary BuildVocabulary(const std::vector<std::vector<std::string>>& tokenized_corpus,
                                          int64_t max_vocab_size = -1);

}  // namespace pulsatrix
