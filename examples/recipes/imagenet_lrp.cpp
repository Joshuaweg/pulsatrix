// Recipe: LRP heatmaps for torchvision's ImageNet ResNet18 or VGG16 (roadmap KS-9). Loads the
// published weights (converted to safetensors), classifies a photo, and writes the Zennit-style
// heatmap of the top class next to the resized input.
//
//   imagenet_lrp_recipe {resnet18|vgg16} WEIGHTS.safetensors IMAGE OUT_DIR [--composite NAME] [--device hip]
//
// NAME: epsilon_plus (the default), epsilon_alpha2_beta1 or epsilon_gamma_box. IMAGE: any
// PNG/JPEG/BMP. See docs/recipes/interpretability/imagenet_lrp.md.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/image_decoder.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/vision_models.hpp"
#ifdef PULSATRIX_DEMO_WITH_HIP
#include "pulsatrix/hip_backend.hpp"
#endif

using namespace pulsatrix;

namespace {

constexpr int kSize = 224;
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

/** torchvision's eval transform, approximately: the short side to 256 (bilinear, no
 *  antialiasing), a centered 224x224 crop. Returns (3, 224, 224) in [0, 1]. */
std::vector<float> ResizeAndCrop(const std::vector<float>& img, int64_t channels, int64_t h, int64_t w) {
    const double scale = 256.0 / static_cast<double>(std::min(h, w));
    const double rh = static_cast<double>(h) * scale, rw = static_cast<double>(w) * scale;
    const double top = (rh - kSize) / 2.0, left = (rw - kSize) / 2.0;
    std::vector<float> out(3 * kSize * kSize);
    for (int c = 0; c < 3; ++c) {
        const int64_t src_c = channels == 1 ? 0 : c;
        for (int y = 0; y < kSize; ++y) {
            for (int x = 0; x < kSize; ++x) {
                const double sy = std::clamp((y + top + 0.5) / scale - 0.5, 0.0, static_cast<double>(h - 1));
                const double sx = std::clamp((x + left + 0.5) / scale - 0.5, 0.0, static_cast<double>(w - 1));
                const auto y0 = static_cast<int64_t>(sy), x0 = static_cast<int64_t>(sx);
                const int64_t y1 = std::min(y0 + 1, h - 1), x1 = std::min(x0 + 1, w - 1);
                const double fy = sy - static_cast<double>(y0), fx = sx - static_cast<double>(x0);
                auto at = [&](int64_t yy, int64_t xx) { return img[static_cast<size_t>((src_c * h + yy) * w + xx)]; };
                out[static_cast<size_t>((c * kSize + y) * kSize + x)] = static_cast<float>(
                    (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x1)) + fy * ((1 - fx) * at(y1, x0) + fx * at(y1, x1)));
            }
        }
    }
    return out;
}

/** A binary PPM: easy to write, and every image viewer and converter reads it. */
void WritePpm(const std::string& path, const std::vector<unsigned char>& rgb) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << kSize << " " << kSize << "\n255\n";
    f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (!f) throw std::runtime_error("could not write " + path);
}

/** Relevance summed over the colour channels, on a diverging map: red for evidence for the class,
 *  blue against, white for none. Scaled symmetrically by the 99.5th percentile of |relevance|,
 *  so a few extreme pixels don't wash the rest out. */
std::vector<unsigned char> Heatmap(const std::vector<float>& relevance) {
    std::vector<float> r(kSize * kSize, 0.0f);
    for (int c = 0; c < 3; ++c) {
        for (size_t i = 0; i < r.size(); ++i) r[i] += relevance[c * r.size() + i];
    }
    std::vector<float> magnitude(r.size());
    for (size_t i = 0; i < r.size(); ++i) magnitude[i] = std::fabs(r[i]);
    const auto q = magnitude.begin() + static_cast<std::ptrdiff_t>(0.995 * static_cast<double>(magnitude.size()));
    std::nth_element(magnitude.begin(), q, magnitude.end());
    const float peak = std::max(*q, 1e-12f);
    std::vector<unsigned char> rgb(3 * r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        const float t = std::clamp(r[i] / peak, -1.0f, 1.0f);
        const auto fade = static_cast<unsigned char>(255.0f * (1.0f - std::fabs(t)));
        rgb[3 * i] = t < 0 ? fade : 255;
        rgb[3 * i + 1] = fade;
        rgb[3 * i + 2] = t > 0 ? fade : 255;
    }
    return rgb;
}

LRP Composite(const std::string& name) {
    if (name == "epsilon_plus") return LRP::epsilon_plus();
    if (name == "epsilon_alpha2_beta1") return LRP::epsilon_alpha2_beta1();
    // The input box: the normalized image of [0, 1] pixels, widest over the three channels.
    if (name == "epsilon_gamma_box") return LRP::epsilon_gamma_box(-kMean[0] / kStd[0], (1.0f - kMean[2]) / kStd[2]);
    throw std::invalid_argument("unknown composite " + name);
}

template <typename Model>
int Explain(Model& model, const std::string& weights, const std::string& image, const std::string& out,
            const std::string& composite, DeviceBackend* backend) {
    LoadTorchvisionWeights(model, weights);
    model.set_training(false);

    // 1. The photo, resized and cropped as torchvision does, then normalized per channel.
    CPUBackend cpu;
    const Tensor decoded = ImageDecoder::DecodeFile(image, &cpu, 3);
    const std::vector<float> pixels =
        ResizeAndCrop(decoded.to_host_vector(), decoded.shape().dim(1), decoded.shape().dim(2), decoded.shape().dim(3));
    std::vector<float> normalized(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        const size_t c = i / (kSize * kSize);
        normalized[i] = (pixels[i] - kMean[c]) / kStd[c];
    }
    const Tensor x(Shape({1, 3, kSize, kSize}), backend, normalized, backend->device());

    // 2. Classify: the top five ImageNet class indices.
    Tensor logits = model.forward(x);
    logits.to(DeviceType::Cpu, &cpu);
    const std::vector<float> l = logits.to_host_vector();
    std::vector<int64_t> order(l.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + 5, order.end(), [&](int64_t a, int64_t b) { return l[a] > l[b]; });
    std::printf("top-5 ImageNet classes (index: logit):");
    for (int i = 0; i < 5; ++i) std::printf("  %lld: %.2f", static_cast<long long>(order[i]), l[order[i]]);
    std::printf("\n");

    // 3. Explain the top class. A ResNet's BatchNorms are folded into its convolutions while it
    //    is explained (Zennit's canonizer); the folds undo themselves when they go out of scope.
    std::vector<std::unique_ptr<BatchNormFold>> folds;
    if constexpr (std::is_same_v<Model, TorchvisionResNet>) folds = model.fold_batch_norms();
    ExplainerContext ctx(model.layers());
    Attribution a = Composite(composite).explain(ctx, x, LRPTarget{{order[0]}, {}, LRPSeed::OneHot}, backend);
    a.values.to(DeviceType::Cpu, &cpu);
    const std::vector<float> relevance = a.values.to_host_vector();
    const double total = std::accumulate(relevance.begin(), relevance.end(), 0.0);
    std::printf("%s relevance: total %.4f (the explained logit %.4f)\n", composite.c_str(), total, l[order[0]]);

    // 4. The input as the model saw it, and the heatmap.
    std::filesystem::create_directories(out);
    std::vector<unsigned char> rgb(3 * kSize * kSize);
    for (size_t i = 0; i < static_cast<size_t>(kSize * kSize); ++i) {
        for (size_t c = 0; c < 3; ++c) {
            rgb[3 * i + c] = static_cast<unsigned char>(std::lround(255.0f * std::clamp(pixels[c * kSize * kSize + i], 0.0f, 1.0f)));
        }
    }
    WritePpm(out + "/input.ppm", rgb);
    WritePpm(out + "/heatmap.ppm", Heatmap(relevance));
    std::printf("wrote %s/input.ppm and %s/heatmap.ppm\n", out.c_str(), out.c_str());
    return 0;
}

int Run(DeviceBackend* backend, char** argv, const std::string& composite) {
    const std::string arch = argv[1];
    if (arch == "resnet18") {
        TorchvisionResNet model(TorchvisionResNet::ResNet18(), backend);
        return Explain(model, argv[2], argv[3], argv[4], composite, backend);
    }
    if (arch == "vgg16") {
        TorchvisionVGG model(TorchvisionVGG::VGG16(), backend);
        return Explain(model, argv[2], argv[3], argv[4], composite, backend);
    }
    throw std::invalid_argument("the model must be resnet18 or vgg16, not " + arch);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: %s {resnet18|vgg16} WEIGHTS.safetensors IMAGE OUT_DIR [--composite NAME] [--device hip]\n", argv[0]);
        return 2;
    }
    std::string composite = "epsilon_plus";
    bool hip = false;
    for (int i = 5; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--composite") {
            composite = argv[i + 1];
        } else if (flag == "--device" && std::string(argv[i + 1]) == "hip") {
            hip = true;
        } else {
            std::fprintf(stderr, "unknown option %s %s\n", argv[i], argv[i + 1]);
            return 2;
        }
    }
    try {
        if (hip) {
#ifdef PULSATRIX_DEMO_WITH_HIP
            HIPBackend gpu;
            return Run(&gpu, argv, composite);
#else
            std::fprintf(stderr, "this build has no HIP backend (configure with -DPULSATRIX_ENABLE_HIP=ON)\n");
            return 1;
#endif
        }
        CPUBackend cpu;
        return Run(&cpu, argv, composite);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
