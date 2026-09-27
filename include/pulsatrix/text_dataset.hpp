/** @file text_dataset.hpp
 *  @brief Line-delimited corpus Dataset -- tokenizes and indexes each line into a Tensor.
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>
#include <vector>

#include "pulsatrix/dataset.hpp"
#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/vocabulary.hpp"

namespace pulsatrix {

/**
 * @brief Dataset over a line-delimited text corpus: each line becomes one sample, tokenized
 *        via Tokenizer::Tokenize and indexed via a Vocabulary into a (1, seq_len) float32
 *        Tensor of token indices -- Decision Point 6's resolved representation (token IDs
 *        as float32 values, the same integer-as-float32 pattern CsvDataset's label column
 *        and MnistDatasetAdapter's class label already use). seq_len varies per sample;
 *        no padding here (see PadCollate, Mission 10) -- an empty line produces a
 *        zero-element (1, 0) Tensor, a valid, non-error Tensor state.
 */
class TextDataset : public Dataset {
public:
    /**
     * @param corpus_path Path to a line-delimited text file.
     * @param vocabulary Vocabulary to index tokens through. Not owned; must outlive this
     *        Dataset.
     * @param backend Backend to allocate token-index Tensors through. Not owned.
     * @throws std::runtime_error if corpus_path can't be opened -- external boundary (file
     *         content, not an internal invariant).
     */
    TextDataset(const std::string& corpus_path, const Vocabulary* vocabulary, DeviceBackend* backend);

    [[nodiscard]] int64_t size() const override;

    /** @throws std::out_of_range if index is out of bounds. */
    [[nodiscard]] Sample get(int64_t index) const override;

private:
    std::vector<std::string> lines_;
    const Vocabulary* vocabulary_;
    DeviceBackend* backend_;
};

}  // namespace pulsatrix
