#include "pulsatrix/csv_dataset.hpp"

#include <stdexcept>

namespace pulsatrix {

namespace {

size_t FindColumn(const std::vector<std::string>& header, const std::string& name) {
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == name) {
            return i;
        }
    }
    throw std::runtime_error("CsvDataset: column not found: " + name);
}

float ParseFloatCell(const std::string& text) {
    try {
        size_t consumed = 0;
        float value = std::stof(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing characters after numeric value");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("CsvDataset: failed to parse float from cell: '" + text + "'");
    }
}

}  // namespace

CsvDataset::CsvDataset(const std::string& path, std::vector<std::string> feature_columns,
                        std::string label_column, DeviceBackend* backend)
    : table_(CsvReader::Load(path, /*has_header=*/true)), backend_(backend) {
    feature_col_indices_.reserve(feature_columns.size());
    for (const std::string& name : feature_columns) {
        feature_col_indices_.push_back(FindColumn(table_.header, name));
    }
    label_col_index_ = FindColumn(table_.header, label_column);
}

int64_t CsvDataset::size() const {
    return static_cast<int64_t>(table_.rows.size());
}

Sample CsvDataset::get(int64_t index) const {
    if (index < 0 || index >= size()) {
        throw std::out_of_range("CsvDataset::get: index out of range");
    }
    const std::vector<std::string>& row = table_.rows[static_cast<size_t>(index)];

    std::vector<float> feature_values;
    feature_values.reserve(feature_col_indices_.size());
    for (size_t col : feature_col_indices_) {
        feature_values.push_back(ParseFloatCell(row[col]));
    }
    float label_value = ParseFloatCell(row[label_col_index_]);

    Tensor features(Shape({1, static_cast<int64_t>(feature_values.size())}), backend_, feature_values);
    Tensor label(Shape({1}), backend_, {label_value});
    return Sample{{std::move(features), std::move(label)}};
}

}  // namespace pulsatrix
