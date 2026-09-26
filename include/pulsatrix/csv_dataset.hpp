/** @file csv_dataset.hpp
 *  @brief Dataset over a CSV file's numeric feature/label columns -- Phase 1's tabular reference case.
 *  @ingroup dl_modules
 */
#pragma once

#include <string>
#include <vector>

#include "pulsatrix/csv_reader.hpp"
#include "pulsatrix/dataset.hpp"
#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

/**
 * @brief Dataset over a CSV file's numeric columns: N named feature columns -> one
 *        (1, num_features) Tensor per row, plus one named label column -> one (1,) Tensor.
 *        Requires a header row (feature/label columns are resolved by name).
 */
class CsvDataset : public Dataset {
public:
    /**
     * @param path CSV file path.
     * @param feature_columns Column names to use as features, in order.
     * @param label_column Column name to use as the label.
     * @param backend Backend to allocate row Tensors through. Not owned.
     * @throws std::runtime_error if the file can't be loaded (see CsvReader::Load), or a
     *         named feature/label column doesn't exist in the header -- external boundary:
     *         column names are caller-supplied configuration, not internally derived.
     */
    CsvDataset(const std::string& path, std::vector<std::string> feature_columns, std::string label_column,
               DeviceBackend* backend);

    [[nodiscard]] int64_t size() const override;

    /**
     * @throws std::out_of_range if index is out of bounds.
     * @throws std::runtime_error if a feature or label cell fails to parse as a float --
     *         external boundary: cell content originates from the file, not internal state.
     */
    [[nodiscard]] Sample get(int64_t index) const override;

private:
    CsvTable table_;
    std::vector<size_t> feature_col_indices_;
    size_t label_col_index_;
    DeviceBackend* backend_;
};

}  // namespace pulsatrix
