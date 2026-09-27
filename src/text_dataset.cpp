#include "pulsatrix/text_dataset.hpp"

#include <fstream>
#include <stdexcept>

#include "pulsatrix/tokenizer.hpp"

namespace pulsatrix {

TextDataset::TextDataset(const std::string& corpus_path, const Vocabulary* vocabulary, DeviceBackend* backend)
    : vocabulary_(vocabulary), backend_(backend) {
    std::ifstream file(corpus_path);
    if (!file) {
        throw std::runtime_error("TextDataset: failed to open corpus file: " + corpus_path);
    }
    std::string line;
    while (std::getline(file, line)) {
        lines_.push_back(line);
    }
}

int64_t TextDataset::size() const {
    return static_cast<int64_t>(lines_.size());
}

Sample TextDataset::get(int64_t index) const {
    if (index < 0 || index >= size()) {
        throw std::out_of_range("TextDataset::get: index out of range");
    }
    std::vector<std::string> tokens = Tokenizer::Tokenize(lines_[static_cast<size_t>(index)]);
    std::vector<float> indices;
    indices.reserve(tokens.size());
    for (const std::string& token : tokens) {
        indices.push_back(static_cast<float>(vocabulary_->IndexOf(token)));
    }
    Tensor token_tensor(Shape({1, static_cast<int64_t>(indices.size())}), backend_, indices);
    return Sample{{std::move(token_tensor)}};
}

}  // namespace pulsatrix
