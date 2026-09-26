#include "pulsatrix/vocabulary.hpp"

#include <algorithm>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {

Vocabulary::Vocabulary(std::vector<std::string> ranked_tokens) {
    tokens_.reserve(ranked_tokens.size() + 1);
    tokens_.emplace_back(kUnkToken);
    for (std::string& token : ranked_tokens) {
        tokens_.push_back(std::move(token));
    }
    for (size_t i = 0; i < tokens_.size(); ++i) {
        token_to_index_[tokens_[i]] = static_cast<int64_t>(i);
    }
}

int64_t Vocabulary::IndexOf(const std::string& token) const {
    auto it = token_to_index_.find(token);
    if (it == token_to_index_.end()) {
        return kUnkIndex;
    }
    return it->second;
}

const std::string& Vocabulary::TokenAt(int64_t index) const {
    PULSATRIX_ASSERT(index >= 0 && index < size());
    return tokens_[static_cast<size_t>(index)];
}

Vocabulary BuildVocabulary(const std::vector<std::vector<std::string>>& tokenized_corpus, int64_t max_vocab_size) {
    std::unordered_map<std::string, int64_t> counts;
    std::unordered_map<std::string, int64_t> first_seen;
    int64_t order = 0;

    for (const std::vector<std::string>& doc : tokenized_corpus) {
        for (const std::string& token : doc) {
            auto it = counts.find(token);
            if (it == counts.end()) {
                counts[token] = 1;
                first_seen[token] = order++;
            } else {
                ++it->second;
            }
        }
    }

    std::vector<std::string> ranked;
    ranked.reserve(counts.size());
    for (const auto& kv : counts) {
        ranked.push_back(kv.first);
    }
    std::sort(ranked.begin(), ranked.end(), [&](const std::string& a, const std::string& b) {
        if (counts.at(a) != counts.at(b)) {
            return counts.at(a) > counts.at(b);
        }
        return first_seen.at(a) < first_seen.at(b);
    });

    if (max_vocab_size >= 0 && static_cast<int64_t>(ranked.size()) > max_vocab_size) {
        ranked.resize(static_cast<size_t>(max_vocab_size));
    }

    return Vocabulary(std::move(ranked));
}

}  // namespace pulsatrix
