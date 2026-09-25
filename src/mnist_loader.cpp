#include "pulsatrix/mnist_loader.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace pulsatrix {
namespace {

constexpr uint32_t kImageMagic = 0x00000803;
constexpr uint32_t kLabelMagic = 0x00000801;

// IDX integers are big-endian, 4 bytes.
uint32_t ReadBigEndianUint32(std::ifstream& file, const std::string& path) {
    unsigned char bytes[4];
    file.read(reinterpret_cast<char*>(bytes), 4);
    if (!file) {
        throw std::runtime_error("MnistIdxLoader: unexpected end of file reading header: " + path);
    }
    return (static_cast<uint32_t>(bytes[0]) << 24) | (static_cast<uint32_t>(bytes[1]) << 16) |
           (static_cast<uint32_t>(bytes[2]) << 8) | static_cast<uint32_t>(bytes[3]);
}

}  // namespace

MnistDataset MnistIdxLoader::Load(const std::string& images_path, const std::string& labels_path,
                                   DeviceBackend* backend, int64_t max_count) {
    std::ifstream images_file(images_path, std::ios::binary);
    if (!images_file) {
        throw std::runtime_error("MnistIdxLoader: could not open images file: " + images_path);
    }
    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file) {
        throw std::runtime_error("MnistIdxLoader: could not open labels file: " + labels_path);
    }

    uint32_t image_magic = ReadBigEndianUint32(images_file, images_path);
    if (image_magic != kImageMagic) {
        throw std::runtime_error("MnistIdxLoader: unexpected images magic number in " + images_path);
    }
    uint32_t image_count = ReadBigEndianUint32(images_file, images_path);
    uint32_t rows = ReadBigEndianUint32(images_file, images_path);
    uint32_t cols = ReadBigEndianUint32(images_file, images_path);

    uint32_t label_magic = ReadBigEndianUint32(labels_file, labels_path);
    if (label_magic != kLabelMagic) {
        throw std::runtime_error("MnistIdxLoader: unexpected labels magic number in " + labels_path);
    }
    uint32_t label_count = ReadBigEndianUint32(labels_file, labels_path);

    if (image_count != label_count) {
        throw std::runtime_error("MnistIdxLoader: images/labels count mismatch (" + std::to_string(image_count) +
                                  " vs " + std::to_string(label_count) + ")");
    }

    int64_t count = static_cast<int64_t>(image_count);
    if (max_count >= 0) {
        count = std::min(count, max_count);
    }

    int64_t pixels_per_image = static_cast<int64_t>(rows) * static_cast<int64_t>(cols);
    std::vector<unsigned char> pixel_buffer(static_cast<size_t>(pixels_per_image));

    MnistDataset dataset;
    dataset.images.reserve(static_cast<size_t>(count));
    dataset.labels.reserve(static_cast<size_t>(count));

    for (int64_t i = 0; i < count; ++i) {
        images_file.read(reinterpret_cast<char*>(pixel_buffer.data()), pixels_per_image);
        if (!images_file) {
            throw std::runtime_error("MnistIdxLoader: unexpected end of file reading image " + std::to_string(i) +
                                      " in " + images_path);
        }

        // Leading 1 is the batch dim (campaign_exai_dl_library_batch_dimension_support) --
        // each loaded image is its own batch-of-1 example, matching Conv2DModule's now
        // rank-4 (N, in_channels, H, W) contract; the second 1 is the single MNIST channel.
        Tensor image(Shape({1, 1, static_cast<int64_t>(rows), static_cast<int64_t>(cols)}), backend);
        for (int64_t p = 0; p < pixels_per_image; ++p) {
            image.data()[p] = static_cast<float>(pixel_buffer[static_cast<size_t>(p)]) / 255.0f;
        }
        dataset.images.push_back(std::move(image));

        unsigned char label_byte = 0;
        labels_file.read(reinterpret_cast<char*>(&label_byte), 1);
        if (!labels_file) {
            throw std::runtime_error("MnistIdxLoader: unexpected end of file reading label " + std::to_string(i) +
                                      " in " + labels_path);
        }
        dataset.labels.push_back(static_cast<int64_t>(label_byte));
    }

    return dataset;
}

}  // namespace pulsatrix
