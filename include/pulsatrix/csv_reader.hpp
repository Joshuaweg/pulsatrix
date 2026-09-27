/** @file csv_reader.hpp
 *  @brief Minimal hand-rolled CSV parser -- RFC-4180-ish, whole-file-at-once.
 *  @ingroup data_pipeline
 */
#pragma once

#include <string>
#include <vector>

namespace pulsatrix {

/** @brief One parsed CSV file's contents: raw string cells, row-major, plus an optional header. */
struct CsvTable {
    std::vector<std::string> header;                // empty if the file has no header row
    std::vector<std::vector<std::string>> rows;      // raw string cells, one vector per row
};

/**
 * @brief Minimal RFC-4180-ish CSV parser: comma-delimited, double-quote-quoted fields,
 *        "" as an escaped quote, \n or \r\n line endings. Whole file loaded into memory at
 *        once -- no streaming/chunked reading in this phase, matching MnistIdxLoader's own
 *        whole-file-at-once precedent.
 * @note Chosen over fetching Arrow C++ (campaign_exai_dl_library_data_pipeline Decision
 *       Point 1): Arrow's columnar/zero-copy value proposition doesn't apply to this
 *       row-oriented, float32-only, small-scale use case, and would break this project's
 *       minimal-FetchContent discipline (GoogleTest + optional pybind11 only, today).
 */
class CsvReader {
public:
    /**
     * @param path Path to the CSV file.
     * @param has_header Whether row 0 is a header (excluded from CsvTable::rows, stored in
     *        CsvTable::header instead).
     * @return The parsed table.
     * @throws std::runtime_error if the file can't be opened, a quoted field is never
     *         closed, has_header is true but the file has no rows at all, or any data
     *         row's field count doesn't match the header's (or, with no header, the first
     *         row's) field count -- external boundary: malformed file content, not an
     *         internal invariant.
     */
    [[nodiscard]] static CsvTable Load(const std::string& path, bool has_header = true);
};

}  // namespace pulsatrix
