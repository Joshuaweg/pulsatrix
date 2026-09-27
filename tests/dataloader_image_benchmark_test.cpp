#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "bmp_test_helper.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/image_folder_dataset.hpp"

// campaign_exai_dl_library_data_pipeline, Phase 2 Mission 7: closes Phase 2's exit gate by
// exercising DataLoader against realistically-sized images (not MNIST's tiny 28x28).
// Reports throughput for the record -- not a strict pass/fail perf gate, matching
// captum_benchmark_test.cpp's existing "benchmark test" precedent, since no parallel-
// prefetch mission has landed yet to hold this to a target.
namespace pulsatrix {
namespace {

namespace fs = std::filesystem;
using ::pulsatrix::test::WriteBmp;

TEST(DataLoaderImageBenchmarkTest, ReportsThroughputForRealisticImageSizes) {
    constexpr int kImageSize = 224;
    constexpr int kImagesPerClass = 10;

    fs::path root = fs::path(::testing::TempDir()) / "pulsatrix_dataloader_image_benchmark";
    fs::remove_all(root);
    fs::create_directories(root / "class_a");
    fs::create_directories(root / "class_b");

    std::vector<unsigned char> rgb(static_cast<size_t>(kImageSize) * static_cast<size_t>(kImageSize) * 3, 128);
    for (int i = 0; i < kImagesPerClass; ++i) {
        WriteBmp((root / "class_a" / (std::to_string(i) + ".bmp")).string(), kImageSize, kImageSize, rgb);
        WriteBmp((root / "class_b" / (std::to_string(i) + ".bmp")).string(), kImageSize, kImageSize, rgb);
    }

    CPUBackend backend;
    auto dataset = std::make_shared<ImageFolderDataset>(root.string(), &backend);
    DataLoaderOptions options;
    options.batch_size = 4;
    DataLoader loader(dataset, &backend, options);

    auto start = std::chrono::steady_clock::now();
    int64_t images_seen = 0;
    while (auto batch = loader.next_batch()) {
        images_seen += batch->size();
    }
    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    double images_per_sec = seconds > 0.0 ? static_cast<double>(images_seen) / seconds : 0.0;

    std::printf("[BENCHMARK] DataLoader over %lld %dx%d images (batch_size=%lld, synchronous): "
                "%.3fs total, %.1f images/sec\n",
                static_cast<long long>(images_seen), kImageSize, kImageSize,
                static_cast<long long>(options.batch_size), seconds, images_per_sec);

    // Correctness floor for this "benchmark test": every image must actually have been
    // visited exactly once, regardless of how fast/slow the run was.
    EXPECT_EQ(images_seen, kImagesPerClass * 2);

    fs::remove_all(root);
}

}  // namespace
}  // namespace pulsatrix
