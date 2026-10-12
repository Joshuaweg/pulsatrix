/** @file embedding_export.hpp
 *  @brief Hand embeddings or activations to UMAP and t-SNE tools, and bring their coordinates
 *         back as a projection document.
 *  @ingroup mech_interp
 *
 * pulsatrix doesn't implement UMAP or t-SNE. Both are expensive, have many settings that change
 * the picture, and are well served elsewhere. Both also distort the data: they keep each point's
 * nearest neighbors, but distances between clusters, cluster sizes and densities in their plots
 * carry little meaning, and an elongated or empty region can come from the settings alone. Use
 * PCA (pca.hpp) for a view whose distances mean something, and these for a view of the
 * neighborhoods.
 *
 * This header writes the data in the formats those tools read:
 *
 * - **`.npy`** (SerializeNpy(), WriteNpy()): numpy's own format, which `umap-learn`, `openTSNE`
 *   and scikit-learn's `TSNE` take after `np.load`.
 * - **TSV for the Embedding Projector** (ProjectorVectorsTsv(), ProjectorMetadataTsv()):
 *   projector.tensorflow.org runs UMAP, t-SNE and PCA in the browser on uploaded files, with
 *   nothing to install.
 *
 * ExportEmbeddings() writes all of them to a directory. ParseNpy() and ReadNpy() read the
 * coordinates the tools return, and MakeProjectionDocument() in viz/document.hpp turns them into a
 * `pulsatrix.projection.v1` document for the SVG, HTML and notebook views, which mark UMAP and
 * t-SNE plots as distorting distances.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief A tensor as a `.npy` file: format version 1.0, little-endian float32 (`<f4`), C order,
 *        with the tensor's shape.
 * @note Copies the tensor to the host first, whatever device it is on.
 */
[[nodiscard]] std::vector<uint8_t> SerializeNpy(const Tensor& x);

/**
 * @brief SerializeNpy() written to @p path.
 * @throws std::runtime_error if the file can't be written.
 */
void WriteNpy(const std::string& path, const Tensor& x);

/**
 * @brief A `.npy` file as a float32 tensor on @p backend, with the file's shape.
 *
 * Reads what numpy writes for a float array: format versions 1.0, 2.0 and 3.0, little-endian
 * float32 (`<f4`) or float64 (`<f8`, rounded to float32), in C or Fortran order, with up to 8
 * dimensions.
 * @throws std::invalid_argument for anything else: another dtype or byte order, a pickled object
 *         array, a malformed header, a size that doesn't match the data, or a float64 value too
 *         large for float32.
 */
[[nodiscard]] Tensor ParseNpy(const std::vector<uint8_t>& bytes, DeviceBackend* backend);

/**
 * @brief ParseNpy() of the file at @p path.
 * @throws std::runtime_error if the file can't be read; otherwise as ParseNpy().
 */
[[nodiscard]] Tensor ReadNpy(const std::string& path, DeviceBackend* backend);

/**
 * @brief The Embedding Projector's tensor file: a line per row of @p x, its values separated by
 *        tabs. Dimensions after the first are flattened.
 * @note Values are written with 9 significant digits, enough to read back the same float32.
 * @throws std::invalid_argument if x has rank below 2 or a NaN or infinite value, which the
 *         projector can't read.
 */
[[nodiscard]] std::string ProjectorVectorsTsv(const Tensor& x);

/** @brief One column of per-point metadata: a label, a class, a value as text. */
struct MetadataColumn {
    std::string name;
    std::vector<std::string> values;
};

/**
 * @brief The Embedding Projector's metadata file: a line per point, a column per entry of
 *        @p columns, separated by tabs.
 *
 * With two or more columns the first line holds their names, as the projector expects; with
 * one, it doesn't, since the projector would read the name as a point. Tabs and line breaks in
 * a value or name are replaced by spaces.
 * @throws std::invalid_argument if there are no columns, or they hold different numbers of
 *         values.
 */
[[nodiscard]] std::string ProjectorMetadataTsv(const std::vector<MetadataColumn>& columns);

/**
 * @brief Writes @p x for UMAP and t-SNE tools into @p directory, which must exist: `vectors.npy`,
 *        `vectors.tsv`, and `metadata.tsv` when there are columns.
 * @throws std::invalid_argument if a column doesn't have one value per row of x, or as
 *         ProjectorVectorsTsv() and ProjectorMetadataTsv() do; std::runtime_error if a file can't
 *         be written.
 */
void ExportEmbeddings(const std::string& directory, const Tensor& x, const std::vector<MetadataColumn>& columns = {});

}  // namespace pulsatrix
