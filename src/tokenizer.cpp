#include "pulsatrix/tokenizer.hpp"

#include <cctype>

namespace pulsatrix {

std::vector<std::string> Tokenizer::Tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current;

    auto flush_current = [&]() {
        if (!current.empty()) {
            tokens.push_back(current);
            current.clear();
        }
    };

    for (char raw_c : text) {
        char c = static_cast<char>(std::tolower(static_cast<unsigned char>(raw_c)));
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '\'') {
            current.push_back(c);
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            flush_current();
        } else {
            flush_current();
            tokens.emplace_back(1, c);
        }
    }
    flush_current();
    return tokens;
}

}  // namespace pulsatrix
