#include "pulsatrix/csv_reader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pulsatrix {

CsvTable CsvReader::Load(const std::string& path, bool has_header) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("CsvReader::Load: failed to open file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    std::vector<std::vector<std::string>> raw_rows;
    std::vector<std::string> current_row;
    std::string current_field;
    bool in_quotes = false;

    auto end_field = [&]() {
        current_row.push_back(current_field);
        current_field.clear();
    };
    auto end_row = [&]() {
        end_field();
        raw_rows.push_back(std::move(current_row));
        current_row.clear();
    };

    size_t i = 0;
    size_t n = content.size();
    while (i < n) {
        char c = content[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < n && content[i + 1] == '"') {
                    current_field.push_back('"');
                    i += 2;
                } else {
                    in_quotes = false;
                    ++i;
                }
            } else {
                current_field.push_back(c);
                ++i;
            }
            continue;
        }
        if (c == '"') {
            in_quotes = true;
            ++i;
        } else if (c == ',') {
            end_field();
            ++i;
        } else if (c == '\r') {
            ++i;  // bare CR ignored; \r\n handled by the following \n
        } else if (c == '\n') {
            end_row();
            ++i;
        } else {
            current_field.push_back(c);
            ++i;
        }
    }
    if (in_quotes) {
        throw std::runtime_error("CsvReader::Load: unterminated quoted field in " + path);
    }
    // Trailing content with no final newline still forms a row; a file ending cleanly on
    // \n must NOT produce a spurious trailing empty row (current_row was already flushed
    // by the last end_row() call, in which case current_row is empty here).
    if (!current_field.empty() || !current_row.empty()) {
        end_row();
    }

    CsvTable table;
    size_t start_idx = 0;
    if (has_header) {
        if (raw_rows.empty()) {
            throw std::runtime_error("CsvReader::Load: file has no header row: " + path);
        }
        table.header = raw_rows[0];
        start_idx = 1;
    }
    size_t expected_field_count =
        has_header ? table.header.size() : (raw_rows.empty() ? 0 : raw_rows[0].size());
    for (size_t r = start_idx; r < raw_rows.size(); ++r) {
        if (raw_rows[r].size() != expected_field_count) {
            throw std::runtime_error("CsvReader::Load: row " + std::to_string(r) +
                                      " field count mismatch in " + path);
        }
        table.rows.push_back(std::move(raw_rows[r]));
    }
    return table;
}

}  // namespace pulsatrix
