/** @file image_grid_view.hpp
 *  @brief Scrollable grid of dataset image samples, texture-cached across frames.
 *  @ingroup visualization
 */
#pragma once

#include <string>
#include <vector>

#include "pulsatrix/dataset.hpp"
#include "pulsatrix/viz/texture_cache.hpp"

namespace pulsatrix {

/**
 * @brief Draws a grid of image samples from a Dataset whose first Sample field is a decoded
 *        (1,C,H,W) or (C,H,W) image tensor (e.g. ImageFolderDataset) -- the data-loading
 *        "preview what you're about to train on" touchpoint.
 * @note Deliberately not unit-tested -- see AttributionBarChart's note. Pixel-format
 *       conversion is plot_data.hpp's ToRgbImageBuffer (unit-tested); this only lays out
 *       already-cached GL textures.
 */
class ImageGridView {
public:
    /**
     * @param title A unique ImGui child id for this grid.
     * @param dataset A Dataset whose Sample::fields[0] is an image tensor.
     * @param cache Texture cache to upload through -- owned by the caller so it persists
     *        across frames instead of re-uploading every image every frame.
     * @param start_index First sample index to display.
     * @param count How many samples to display, in a fixed-column grid.
     * @param columns Number of images per row.
     * @param thumbnail_size Displayed width/height in pixels per thumbnail.
     * @param captions Optional per-thumbnail caption drawn under each image (e.g.
     *        "true 7 / pred 7"); captions[k] labels sample start_index + k. Thumbnails past
     *        the end of captions get no caption. nullptr (default) draws images only.
     */
    static void Draw(const char* title, const Dataset& dataset, TextureCache& cache, int64_t start_index,
                      int64_t count, int columns = 4, float thumbnail_size = 96.0f,
                      const std::vector<std::string>* captions = nullptr);
};

}  // namespace pulsatrix
