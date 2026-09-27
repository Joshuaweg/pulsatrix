#include "pulsatrix/video_frame_directory_dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include "pulsatrix/image_decoder.hpp"

namespace pulsatrix {

namespace {
namespace fs = std::filesystem;

std::vector<fs::path> SortedSubdirectories(const fs::path& dir) {
    std::vector<fs::path> subdirs;
    for (const fs::directory_entry& entry : fs::directory_iterator(dir)) {
        if (entry.is_directory()) {
            subdirs.push_back(entry.path());
        }
    }
    std::sort(subdirs.begin(), subdirs.end());
    return subdirs;
}

std::vector<fs::path> SortedFiles(const fs::path& dir) {
    std::vector<fs::path> files;
    for (const fs::directory_entry& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}
}  // namespace

VideoFrameDirectoryDataset::VideoFrameDirectoryDataset(const std::string& root_dir, DeviceBackend* backend)
    : backend_(backend) {
    if (!fs::exists(root_dir) || !fs::is_directory(root_dir)) {
        throw std::runtime_error("VideoFrameDirectoryDataset: root directory does not exist: " + root_dir);
    }

    std::vector<fs::path> class_dirs = SortedSubdirectories(root_dir);
    if (class_dirs.empty()) {
        throw std::runtime_error("VideoFrameDirectoryDataset: no class subdirectories found in: " + root_dir);
    }

    for (size_t i = 0; i < class_dirs.size(); ++i) {
        classes_.push_back(class_dirs[i].filename().string());

        std::vector<fs::path> clip_dirs = SortedSubdirectories(class_dirs[i]);
        for (const fs::path& clip_dir : clip_dirs) {
            if (SortedFiles(clip_dir).empty()) {
                throw std::runtime_error("VideoFrameDirectoryDataset: clip directory contains no frame files: " +
                                          clip_dir.string());
            }
            entries_.push_back({clip_dir.string(), static_cast<int64_t>(i)});
        }
    }
}

int64_t VideoFrameDirectoryDataset::size() const {
    return static_cast<int64_t>(entries_.size());
}

Sample VideoFrameDirectoryDataset::get(int64_t index) const {
    if (index < 0 || index >= size()) {
        throw std::out_of_range("VideoFrameDirectoryDataset::get: index out of range");
    }
    const Entry& entry = entries_[static_cast<size_t>(index)];

    std::vector<fs::path> frame_files = SortedFiles(entry.clip_dir);
    std::vector<Tensor> frames;
    frames.reserve(frame_files.size());
    for (const fs::path& frame_file : frame_files) {
        frames.push_back(ImageDecoder::DecodeFile(frame_file.string(), backend_));
    }

    Tensor clip = Tensor::Stack(frames, backend_);
    Tensor label(Shape({1}), backend_, {static_cast<float>(entry.label)});
    return Sample{{std::move(clip), std::move(label)}};
}

}  // namespace pulsatrix
