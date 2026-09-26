#include "pulsatrix/audio_folder_dataset.hpp"

#include <algorithm>
#include <filesystem>
#include <stdexcept>

#include "pulsatrix/wav_reader.hpp"

namespace pulsatrix {

namespace {
namespace fs = std::filesystem;
}  // namespace

AudioFolderDataset::AudioFolderDataset(const std::string& root_dir, DeviceBackend* backend) : backend_(backend) {
    if (!fs::exists(root_dir) || !fs::is_directory(root_dir)) {
        throw std::runtime_error("AudioFolderDataset: root directory does not exist: " + root_dir);
    }

    std::vector<fs::path> class_dirs;
    for (const fs::directory_entry& entry : fs::directory_iterator(root_dir)) {
        if (entry.is_directory()) {
            class_dirs.push_back(entry.path());
        }
    }
    std::sort(class_dirs.begin(), class_dirs.end());
    if (class_dirs.empty()) {
        throw std::runtime_error("AudioFolderDataset: no class subdirectories found in: " + root_dir);
    }

    for (size_t i = 0; i < class_dirs.size(); ++i) {
        classes_.push_back(class_dirs[i].filename().string());

        std::vector<fs::path> files;
        for (const fs::directory_entry& file_entry : fs::directory_iterator(class_dirs[i])) {
            if (file_entry.is_regular_file()) {
                files.push_back(file_entry.path());
            }
        }
        std::sort(files.begin(), files.end());

        for (const fs::path& file : files) {
            entries_.push_back({file.string(), static_cast<int64_t>(i)});
        }
    }
}

int64_t AudioFolderDataset::size() const {
    return static_cast<int64_t>(entries_.size());
}

Sample AudioFolderDataset::get(int64_t index) const {
    if (index < 0 || index >= size()) {
        throw std::out_of_range("AudioFolderDataset::get: index out of range");
    }
    const Entry& entry = entries_[static_cast<size_t>(index)];
    WavData wav = WavReader::Load(entry.path, backend_);
    Tensor label(Shape({1}), backend_, {static_cast<float>(entry.label)});
    return Sample{{std::move(wav.waveform), std::move(label)}};
}

}  // namespace pulsatrix
