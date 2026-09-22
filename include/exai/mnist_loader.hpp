/** @file mnist_loader.hpp
 *  @brief Parses real MNIST IDX/ubyte files (fetched by tools/fetch_mnist.py) into Tensor
 *         images and integer labels.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "exai/device_backend.hpp"
#include "exai/tensor.hpp"

namespace exai {

/** @brief One IDX file pair's contents: parallel images/labels, same length. */
struct MnistDataset {
    /** @brief Shape (1, 28, 28), pixel values normalized to [0,1]. */
    std::vector<Tensor> images;
    /** @brief 0-9 class index, one per image, same order/length as images. */
    std::vector<int64_t> labels;
};

/**
 * @brief Reads MNIST's original IDX-format files directly -- no format conversion, no
 *        generic Dataset abstraction. MNIST-specific by deliberate scope decision (see
 *        plan_mnist_classification_training_example.md's Recon); if a future mission
 *        needs a second real dataset, generalize then.
 * @note Files are fetched once via tools/fetch_mnist.py to data/MNIST/raw/ (gitignored,
 *       not part of this repo's tracked source) -- callers/tests must handle their
 *       absence gracefully (GTEST_SKIP() in tests), not assume they exist.
 */
class MnistIdxLoader {
public:
    /**
     * @brief Loads one images/labels IDX file pair.
     * @param images_path Path to an IDX3 (images) file, e.g. "data/MNIST/raw/train-images-idx3-ubyte".
     * @param labels_path Path to an IDX1 (labels) file, e.g. "data/MNIST/raw/train-labels-idx1-ubyte".
     * @param backend Backend to allocate image Tensors through. Not owned; must outlive
     *        the returned dataset's Tensors.
     * @param max_count If >= 0, load at most this many images/labels (from the start of
     *        the file) rather than the full file -- for sizing a training subset without
     *        a separate truncation step. -1 (default) loads everything.
     * @return The parsed dataset.
     * @throws std::runtime_error if a file can't be opened, its IDX magic number doesn't
     *         match the expected type, or it's shorter than its own header claims --
     *         external boundary (file content, not an internal invariant).
     */
    [[nodiscard]] static MnistDataset Load(const std::string& images_path, const std::string& labels_path,
                                            DeviceBackend* backend, int64_t max_count = -1);
};

}  // namespace exai
