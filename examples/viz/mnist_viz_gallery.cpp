/** @file mnist_viz_gallery.cpp
 *  @brief Runs every widget in the visualization pack against real MNIST test-set digits:
 *         trains MnistConvNet live (TrainingDashboard over an ImPlotMetricsSink), previews
 *         the test set (DatasetStatisticsView + ImageGridView), then explains a selectable
 *         set of test digits with every explainer that applies to the model (Saliency,
 *         Integrated Gradients, Grad-CAM, four LRP rule sets, LIME, KernelSHAP) and shows
 *         them through ConfidenceMeter, SaliencyHeatmapView, AttributionBarChart,
 *         AttributionWaterfallChart, ExplanationScoreCard, AttributionBeeswarmView and
 *         CircuitGraphView.
 *  @note Requires real MNIST data in data/MNIST/raw/ (see tools/fetch_mnist.py). Run from
 *        the repository root, or pass --data <dir>.
 *  @note `--screenshot <dir>` renders every page (and every per-method variant) for a few
 *        frames, writes one PNG per page via glReadPixels + stb_image_write, then exits.
 *  @note Not a test -- see explanation_dashboard_demo.cpp's note on this module's testing
 *        strategy.
 *
 *  Explainer budgets (also printed and shown in the UI):
 *   - Integrated Gradients: 128 Riemann steps from an all-black (zero) baseline.
 *   - Grad-CAM: last (only) conv layer, 24x24 map; padded by 2 px of zeros on each side to
 *     the 28x28 input grid, which places each CAM cell exactly on its 5x5 receptive-field
 *     centre (valid convolution) -- no interpolation.
 *   - LIME: this library's LIME perturbs every feature with N(0, sigma^2) noise and weights
 *     samples by exp(-||delta||^2 / 2 sigma^2) with the *same* sigma, so its effective sample
 *     size shrinks like (3/4)^(d/2) with input dimension d -- on raw 784 pixels every weight
 *     underflows to 0 (exp(-392)). It is therefore run on 16 superpixels (4x4 grid of 7x7
 *     patches): feature z_k multiplies patch k's pixel intensities (z = 1 is the real image),
 *     2000 samples, sigma = 0.5, l2_lambda = 1e-4. Each patch coefficient is spread evenly
 *     over its 49 pixels, so the pixel map sums to the superpixel coefficients' sum.
 *   - KernelSHAP: this library's KernelSHAP enumerates all 2^n coalitions exactly (n <= 20),
 *     so it is run on the same 16 superpixels: present = original patch, absent = black
 *     (the MNIST background), i.e. 65,536 exact coalition evaluations per digit. SHAP values
 *     are spread over each patch's pixels the same way (efficiency: they sum to
 *     f(x) - f(black)).
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/attribution.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/explainer_stability.hpp"
#include "pulsatrix/grad_cam.hpp"
#include "pulsatrix/integrated_gradients.hpp"
#include "pulsatrix/kernel_shap.hpp"
#include "pulsatrix/lime.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/lrp_conservation.hpp"
#include "pulsatrix/mnist_classifier_example.hpp"
#include "pulsatrix/mnist_dataset_adapter.hpp"
#include "pulsatrix/mnist_loader.hpp"
#include "pulsatrix/saliency.hpp"
#include "pulsatrix/viz/attribution_bar_chart.hpp"
#include "pulsatrix/viz/attribution_beeswarm.hpp"
#include "pulsatrix/viz/attribution_waterfall.hpp"
#include "pulsatrix/viz/circuit_graph_view.hpp"
#include "pulsatrix/viz/confidence_meter.hpp"
#include "pulsatrix/viz/dataset_statistics_view.hpp"
#include "pulsatrix/viz/explanation_score_card.hpp"
#include "pulsatrix/viz/image_grid_view.hpp"
#include "pulsatrix/viz/plot_data.hpp"
#include "pulsatrix/viz/saliency_heatmap_view.hpp"
#include "pulsatrix/viz/texture_cache.hpp"
#include "pulsatrix/viz/training_dashboard.hpp"
#include "pulsatrix/viz/window.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace {

using namespace pulsatrix;
using Clock = std::chrono::steady_clock;

constexpr int64_t kSide = 28;
constexpr int64_t kPixels = kSide * kSide;
constexpr int64_t kClasses = 10;
constexpr int64_t kIgSteps = 128;
constexpr int64_t kPatch = 7;                         // superpixel side for LIME / KernelSHAP
constexpr int64_t kPatchGrid = kSide / kPatch;        // 4
constexpr int64_t kSuperpixels = kPatchGrid * kPatchGrid;  // 16
constexpr int64_t kLimeSamples = 2000;
constexpr float kLimeSigma = 0.5f;
constexpr float kLimeLambda = 1e-4f;
constexpr int kTopK = 12;

// Attribution/Tensor have no default constructor; placeholder values for not-yet-computed slots.
Attribution PlaceholderAttribution() {
    static CPUBackend backend;
    return Attribution{"", Tensor(Shape({1}), &backend), {}};
}

double SecondsSince(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------------------
struct Options {
    std::string screenshot_dir;
    std::string data_dir = "data/MNIST/raw";
    int images = 4;            // test digits explained with every method (selectable in the UI)
    int epochs = 1;
    int64_t train_count = 60000;  // the full MNIST training set: ~25 s for one epoch on CPU
    int64_t eval_count = 2000;  // held-out digits for the accuracy curve
    int beeswarm_images = 200;  // test digits behind each beeswarm
    int stats_count = 1000;     // test digits behind DatasetStatisticsView (recomputed per frame)
};

void PrintUsage() {
    std::printf(
        "usage: mnist_viz_gallery [--screenshot DIR] [--images N] [--epochs N] [--train N]\n"
        "                         [--eval N] [--beeswarm N] [--data DIR]\n");
}

Options ParseOptions(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                PrintUsage();
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--screenshot") {
            o.screenshot_dir = next();
        } else if (a == "--images") {
            o.images = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--epochs") {
            o.epochs = std::max(1, std::atoi(next().c_str()));
        } else if (a == "--train") {
            o.train_count = std::max<int64_t>(1, std::atoll(next().c_str()));
        } else if (a == "--eval") {
            o.eval_count = std::max<int64_t>(1, std::atoll(next().c_str()));
        } else if (a == "--beeswarm") {
            o.beeswarm_images = std::max(2, std::atoi(next().c_str()));
        } else if (a == "--data") {
            o.data_dir = next();
        } else if (a == "--help" || a == "-h") {
            PrintUsage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            PrintUsage();
            std::exit(2);
        }
    }
    return o;
}

// ---------------------------------------------------------------------------------------
// Attribution helpers
// ---------------------------------------------------------------------------------------
Tensor ImageTensor(const std::vector<float>& values, DeviceBackend* backend) {
    return Tensor(Shape({1, 1, kSide, kSide}), backend, values);
}

Attribution AsImageAttribution(const Attribution& attr, DeviceBackend* backend) {
    Attribution out = attr;
    out.values = ImageTensor(attr.values.to_host_vector(), backend);
    return out;
}

// Grad-CAM on a valid 5x5 conv yields a 24x24 map whose cell (h, w) is centred on input pixel
// (h + 2, w + 2) -- pad with zeros back onto the 28x28 grid.
Attribution GradCamToInputGrid(const Attribution& cam, DeviceBackend* backend) {
    int64_t h = cam.values.shape().dim(1);
    int64_t w = cam.values.shape().dim(2);
    int64_t off_y = (kSide - h) / 2;
    int64_t off_x = (kSide - w) / 2;
    std::vector<float> src = cam.values.to_host_vector();
    std::vector<float> out(static_cast<size_t>(kPixels), 0.0f);
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            out[static_cast<size_t>((y + off_y) * kSide + x + off_x)] = src[static_cast<size_t>(y * w + x)];
        }
    }
    Attribution a = cam;
    a.values = ImageTensor(out, backend);
    a.metadata["mapping"] = "24x24 CAM zero-padded 2px onto 28x28 (receptive-field centres)";
    return a;
}

int64_t SuperpixelOf(int64_t pixel) {
    int64_t r = pixel / kSide;
    int64_t c = pixel % kSide;
    return (r / kPatch) * kPatchGrid + (c / kPatch);
}

// Superpixel-space model: z (1, 16) scales each 7x7 patch of the digit.
Tensor MaskedImage(const std::vector<float>& image, const Tensor& z, DeviceBackend* backend) {
    std::vector<float> zv = z.to_host_vector();
    std::vector<float> out(static_cast<size_t>(kPixels));
    for (int64_t p = 0; p < kPixels; ++p) {
        out[static_cast<size_t>(p)] = image[static_cast<size_t>(p)] * zv[static_cast<size_t>(SuperpixelOf(p))];
    }
    return ImageTensor(out, backend);
}

Attribution SuperpixelsToPixels(const Attribution& sp, const std::string& note, DeviceBackend* backend) {
    std::vector<float> coeff = sp.values.to_host_vector();
    std::vector<float> out(static_cast<size_t>(kPixels));
    const float area = static_cast<float>(kPatch * kPatch);
    for (int64_t p = 0; p < kPixels; ++p) {
        out[static_cast<size_t>(p)] = coeff[static_cast<size_t>(SuperpixelOf(p))] / area;
    }
    Attribution a = sp;
    a.values = ImageTensor(out, backend);
    a.metadata["mapping"] = note;
    return a;
}

const std::string& PixelNamesCsv() {
    static const std::string names = [] {
        std::string s;
        for (int64_t p = 0; p < kPixels; ++p) {
            if (p > 0) s += ",";
            s += "px(r" + std::to_string(p / kSide) + " c" + std::to_string(p % kSide) + ")";
        }
        return s;
    }();
    return names;
}

std::string PixelName(int64_t p) {
    return "px(r" + std::to_string(p / kSide) + " c" + std::to_string(p % kSide) + ")";
}

// "patch rIcJ" = the 7x7 patch at patch-grid row I, column J (pixel rows 7I..7I+6).
std::string SuperpixelName(int64_t k) {
    return "patch r" + std::to_string(k / kPatchGrid) + "c" + std::to_string(k % kPatchGrid);
}

// (1, 16) superpixel attribution with patch names -- for LIME / KernelSHAP bars and waterfall,
// whose real features are the 16 patches (per-pixel bars would just repeat each patch value).
Attribution SuperpixelFeatures(const Attribution& sp, DeviceBackend* backend) {
    std::string names;
    for (int64_t k = 0; k < kSuperpixels; ++k) names += (k == 0 ? "" : ",") + SuperpixelName(k);
    return Attribution{sp.method, Tensor(Shape({1, kSuperpixels}), backend, sp.values.to_host_vector()),
                       {{"feature_names", names}}};
}

// (1, 784) with pixel feature names -- the shape AttributionBarChart / ExplanationScoreCard take.
Attribution FlattenWithPixelNames(const Attribution& a, DeviceBackend* backend) {
    Attribution out{a.method, Tensor(Shape({1, kPixels}), backend, a.values.to_host_vector()), {}};
    out.metadata["feature_names"] = PixelNamesCsv();
    return out;
}

// Rank-1 top-k pixels by |value| (descending) plus one aggregate "rest" bar, so the waterfall
// still cascades all the way to baseline + sum(all attributions).
Attribution TopKPlusRest(const Attribution& a, int k, DeviceBackend* backend) {
    std::vector<float> v = a.values.to_host_vector();
    std::vector<int64_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), int64_t{0});
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t x, int64_t y) {
        return std::abs(v[static_cast<size_t>(x)]) > std::abs(v[static_cast<size_t>(y)]);
    });
    std::vector<std::string> feature_names;
    auto it = a.metadata.find("feature_names");
    if (it != a.metadata.end()) {
        std::string item;
        for (char ch : it->second + ",") {
            if (ch == ',') {
                feature_names.push_back(item);
                item.clear();
            } else {
                item += ch;
            }
        }
    }
    auto name_of = [&](int64_t i) {
        return i < static_cast<int64_t>(feature_names.size()) ? feature_names[static_cast<size_t>(i)] : PixelName(i);
    };
    std::vector<float> vals;
    std::string names;
    float rest = 0.0f;
    k = std::min<int>(k, static_cast<int>(idx.size()));
    for (size_t i = 0; i < idx.size(); ++i) {
        float val = v[static_cast<size_t>(idx[i])];
        if (static_cast<int>(i) < k) {
            vals.push_back(val);
            names += (i == 0 ? "" : ",") + name_of(idx[i]);
        } else {
            rest += val;
        }
    }
    vals.push_back(rest);
    names += ",other " + std::to_string(idx.size() - static_cast<size_t>(k)) +
             (feature_names.empty() || feature_names.size() == static_cast<size_t>(kPixels) ? " px" : " features");
    Attribution out{a.method, Tensor(Shape({static_cast<int64_t>(vals.size())}), backend, vals), {}};
    out.metadata["feature_names"] = names;
    return out;
}

struct Stats {
    double sum = 0.0;
    float max_abs = 0.0f;
    float min = 0.0f;
    float max = 0.0f;
    bool finite = true;
};

Stats Summarize(const Attribution& a) {
    Stats s;
    std::vector<float> v = a.values.to_host_vector();
    if (!v.empty()) {
        s.min = s.max = v[0];
    }
    for (float x : v) {
        s.finite = s.finite && std::isfinite(x);
        s.sum += static_cast<double>(x);
        s.max_abs = std::max(s.max_abs, std::abs(x));
        s.min = std::min(s.min, x);
        s.max = std::max(s.max, x);
    }
    return s;
}

// ---------------------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------------------
enum Method {
    kSaliency,
    kIG,
    kGradCam,
    kLrpEpsilon,
    kLrpEpsPlus,
    kLrpAlpha2Beta1,
    kLrpGammaBox,
    kLime,
    kKernelShap,
    kMethodCount
};

const char* kMethodNames[kMethodCount] = {
    "Saliency (gradient)",
    "Integrated Gradients",
    "Grad-CAM",
    "LRP epsilon",
    "LRP EpsilonPlus",
    "LRP EpsilonAlpha2Beta1",
    "LRP EpsilonGammaBox(0,1)",
    "LIME (16 superpixels)",
    "KernelSHAP (16 superpixels)",
};

const char* kMethodSlugs[kMethodCount] = {
    "saliency", "integrated_gradients", "grad_cam", "lrp_epsilon", "lrp_epsilon_plus",
    "lrp_epsilon_alpha2_beta1", "lrp_epsilon_gamma_box", "lime", "kernel_shap",
};

const char* kMethodBudgets[kMethodCount] = {
    "1 backward pass",
    "128 steps, black baseline",
    "last conv layer, 24x24 -> 28x28 (zero pad 2px)",
    "LRP() default: epsilon = 1e-6 on every layer",
    "conv: alpha1beta0, dense: epsilon(1e-6)",
    "conv: alpha2beta1, dense: epsilon(1e-6)",
    "first conv: ZBox[0,1]",
    "2000 samples, sigma 0.5, l2 1e-4",
    "exact: 2^16 = 65536 coalitions, absent = black",
};

bool IsLrp(int m) { return m >= kLrpEpsilon && m <= kLrpGammaBox; }
bool IsCheap(int m) { return m <= kLrpGammaBox; }  // gradient / relevance methods

LRP MakeLrp(int m) {
    switch (m) {
        case kLrpEpsPlus:
            return LRP::epsilon_plus();
        case kLrpAlpha2Beta1:
            return LRP::epsilon_alpha2_beta1();
        case kLrpGammaBox:
            return LRP::epsilon_gamma_box(0.0f, 1.0f);
        default:
            return LRP(LRPRuleConfig{});
    }
}

struct MethodResult {
    Attribution attr = PlaceholderAttribution();       // (1, 1, 28, 28)
    Attribution flat = PlaceholderAttribution();       // (1, 784) pixel or (1, 16) superpixel features
    Attribution waterfall = PlaceholderAttribution();  // rank-1 top-k + rest
    float waterfall_baseline = 0.0f;
    std::string baseline_note;
    Stats stats;
    double seconds = 0.0;
    std::string extra;         // method-specific check printed under the panel
    bool done = false;
};

struct ImageRecord {
    int64_t test_index = 0;
    int64_t label = 0;
    int64_t pred = 0;
    std::vector<float> pixels;
    std::vector<float> logits;
    std::vector<float> probs;
    float logit_black = 0.0f;  // target logit on an all-black image (IG / SHAP baseline)
    MethodResult methods[kMethodCount];
    std::vector<ExplanationScoreCard::Input> lrp_cards;
    std::optional<CircuitGraph> circuit;
};

std::vector<float> Softmax(const std::vector<float>& logits) {
    float m = *std::max_element(logits.begin(), logits.end());
    std::vector<float> p(logits.size());
    float z = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        p[i] = std::exp(logits[i] - m);
        z += p[i];
    }
    for (float& x : p) x /= z;
    return p;
}

int64_t Argmax(const std::vector<float>& v) {
    return static_cast<int64_t>(std::max_element(v.begin(), v.end()) - v.begin());
}

class Gallery {
public:
    Gallery(Options options, DeviceBackend* backend, MnistDataset train, MnistDataset test)
        : opt_(std::move(options)),
          backend_(backend),
          train_(std::move(train)),
          test_(std::move(test)),
          net_(backend),
          optimizer_(0.001f, backend),
          loss_sink_(sink_) {
        MnistDataset preview;
        MnistDataset stats;
        for (size_t i = 0; i < test_.images.size(); ++i) {
            if (static_cast<int>(i) < kGridCount) {
                preview.images.push_back(test_.images[i]);
                preview.labels.push_back(test_.labels[i]);
            }
            if (static_cast<int>(i) < opt_.stats_count) {
                stats.images.push_back(test_.images[i]);
                stats.labels.push_back(test_.labels[i]);
            }
        }
        preview_dataset_.emplace(std::move(preview), backend);
        stats_dataset_.emplace(std::move(stats), backend);
        total_steps_ = static_cast<int64_t>(train_.images.size()) * opt_.epochs;
    }

    void Frame(VizWindow& window);
    void AfterRender(int w, int h, VizWindow& window);

private:
    // ---- phases ------------------------------------------------------------------------
    enum class Phase { Training, Explaining, Ready };

    // Forwards MnistConvNet::train_step's per-step "loss" scalar as a 100-step mean -- the raw
    // per-example cross-entropy is too noisy to read as a curve.
    class WindowedLossSink : public MetricsSink {
    public:
        explicit WindowedLossSink(ImPlotMetricsSink& out) : out_(out) {}
        void log_scalar(const std::string& tag, double value, int step) override {
            if (tag != "loss") {
                out_.log_scalar(tag, value, step);
                return;
            }
            sum_ += value;
            if (++count_ == kWindow) {
                out_.log_scalar("train_loss (mean of 100 steps)", sum_ / kWindow, step);
                sum_ = 0.0;
                count_ = 0;
            }
        }
        void log_histogram(const std::string& tag, const Tensor& values, int step) override {
            out_.log_histogram(tag, values, step);
        }

    private:
        static constexpr int kWindow = 100;
        ImPlotMetricsSink& out_;
        double sum_ = 0.0;
        int count_ = 0;
    };

    float EvaluateAccuracy(int64_t count) {
        int64_t n = std::min<int64_t>(count, static_cast<int64_t>(test_.images.size()));
        int64_t correct = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (net_.predict(test_.images[static_cast<size_t>(i)]) == test_.labels[static_cast<size_t>(i)]) {
                ++correct;
            }
        }
        return static_cast<float>(correct) / static_cast<float>(n);
    }

    void TrainSome(double budget_seconds);
    void FinishTraining();
    void BuildExplainTasks();
    void ExplainSome(double budget_seconds);
    void ComputeMethod(ImageRecord& rec, int m);
    void ComputeBeeswarm(int m);
    void PrintSummary();

    // ---- drawing -----------------------------------------------------------------------
    void DrawNav();
    void DrawPage();
    void DrawTrainingPage();
    void DrawDatasetPage();
    void DrawPredictionPage();
    void DrawHeatmapsPage();
    void DrawChartsPage();
    void DrawScoreCardsPage();
    void DrawBeeswarmPage();
    void DrawCircuitPage();
    void DrawImageSelector();
    void DrawDigit(const ImageRecord& rec, float size);

    enum Page { kTraining, kDataset, kPrediction, kHeatmaps, kCharts, kScoreCards, kBeeswarm, kCircuit, kPageCount };
    static constexpr const char* kPageNames[kPageCount] = {
        "Training", "Dataset", "Prediction", "Heatmaps (all methods)", "Bar + waterfall", "LRP score cards",
        "Beeswarm", "Circuit graph",
    };
    static constexpr const char* kPageSlugs[kPageCount] = {
        "training", "dataset", "prediction", "heatmaps", "charts", "lrp_score_cards", "beeswarm", "circuit",
    };
    static constexpr int kGridCount = 32;

    Options opt_;
    DeviceBackend* backend_;
    MnistDataset train_;
    MnistDataset test_;
    std::optional<MnistDatasetAdapter> preview_dataset_;
    std::optional<MnistDatasetAdapter> stats_dataset_;
    std::vector<std::string> grid_captions_;
    MnistConvNet net_;
    AdamOptimizer optimizer_;
    ImPlotMetricsSink sink_;
    WindowedLossSink loss_sink_;
    TextureCache textures_;

    Phase phase_ = Phase::Training;
    int64_t step_ = 0;
    int64_t total_steps_ = 0;
    float final_accuracy_ = 0.0f;
    int64_t final_accuracy_n_ = 0;
    double train_seconds_ = 0.0;

    std::vector<ImageRecord> records_;
    std::vector<std::function<void()>> tasks_;
    size_t next_task_ = 0;
    std::string current_task_label_;

    std::vector<Attribution> beeswarm_runs_[kMethodCount];
    std::vector<int64_t> beeswarm_pixels_[kMethodCount];

    int page_ = kTraining;
    int selected_image_ = 0;
    int selected_method_ = kLrpEpsPlus;

    // screenshot driver
    struct Shot {
        int page;
        int method;  // -1 when the page has no per-method variant
        std::string file;
        int image = 0;  // index into records_
    };
    std::vector<Shot> shots_;
    size_t shot_index_ = 0;
    int shot_frames_ = 0;
    std::vector<std::string> written_;
};

// ---------------------------------------------------------------------------------------
// Training / explanation work, spread across frames so the window stays live
// ---------------------------------------------------------------------------------------
void Gallery::TrainSome(double budget_seconds) {
    auto t0 = Clock::now();
    const int64_t n = static_cast<int64_t>(train_.images.size());
    const int64_t eval_every = std::max<int64_t>(1, n / 6);
    while (step_ < total_steps_ && SecondsSince(t0) < budget_seconds) {
        size_t i = static_cast<size_t>(step_ % n);
        (void)net_.train_step(train_.images[i], train_.labels[i], optimizer_, loss_sink_, static_cast<int>(step_));
        ++step_;
        if (step_ % eval_every == 0 || step_ == total_steps_) {
            sink_.log_scalar("test_accuracy (first " + std::to_string(opt_.eval_count) + " test digits)",
                             static_cast<double>(EvaluateAccuracy(opt_.eval_count)), static_cast<int>(step_));
        }
    }
    train_seconds_ += SecondsSince(t0);
    if (step_ >= total_steps_) {
        FinishTraining();
    }
}

void Gallery::FinishTraining() {
    std::vector<Module*> mods = net_.modules();
    for (const ParamRef& p : mods[0]->parameters()) {
        sink_.log_histogram(p.value->numel() > 8 ? "conv_kernel" : "conv_bias", *p.value, static_cast<int>(step_));
        break;  // kernel only
    }
    sink_.log_histogram("classifier_weight", net_.classifier_weight(), static_cast<int>(step_));

    auto t0 = Clock::now();
    final_accuracy_n_ = static_cast<int64_t>(test_.images.size());
    final_accuracy_ = EvaluateAccuracy(final_accuracy_n_);
    std::printf("Trained MnistConvNet (Conv2D(1,8,5x5)->ReLU->Flatten->Linear(4608,10), Adam lr=1e-3)\n");
    std::printf("  %lld steps (%d epoch(s) x %zu train digits) in %.1f s\n", static_cast<long long>(step_),
                opt_.epochs, train_.images.size(), train_seconds_);
    std::printf("  test accuracy: %.2f%% on all %lld MNIST test digits (eval %.1f s)\n\n",
                static_cast<double>(final_accuracy_) * 100.0, static_cast<long long>(final_accuracy_n_),
                SecondsSince(t0));
    std::fflush(stdout);

    for (int i = 0; i < kGridCount && i < static_cast<int>(test_.images.size()); ++i) {
        int64_t pred = net_.predict(test_.images[static_cast<size_t>(i)]);
        grid_captions_.push_back("t" + std::to_string(test_.labels[static_cast<size_t>(i)]) + " p" + std::to_string(pred) +
                                 (pred == test_.labels[static_cast<size_t>(i)] ? "" : " X"));
    }

    BuildExplainTasks();
    phase_ = Phase::Explaining;
}

void Gallery::BuildExplainTasks() {
    // Explain the first --images test digits, plus (if none of those is misclassified) the
    // first misclassified test digit -- an error case is the most interesting thing to explain.
    std::vector<int64_t> chosen;
    for (int64_t i = 0; i < static_cast<int64_t>(test_.images.size()) &&
                        static_cast<int>(chosen.size()) < opt_.images;
         ++i) {
        chosen.push_back(i);
    }
    bool has_error = false;
    for (int64_t i : chosen) {
        has_error = has_error || net_.predict(test_.images[static_cast<size_t>(i)]) != test_.labels[static_cast<size_t>(i)];
    }
    if (!has_error) {
        for (int64_t i = static_cast<int64_t>(chosen.size()); i < static_cast<int64_t>(test_.images.size()); ++i) {
            if (net_.predict(test_.images[static_cast<size_t>(i)]) != test_.labels[static_cast<size_t>(i)]) {
                chosen.push_back(i);
                break;
            }
        }
    }

    records_.resize(chosen.size());
    for (size_t r = 0; r < chosen.size(); ++r) {
        ImageRecord& rec = records_[r];
        rec.test_index = chosen[r];
        rec.label = test_.labels[static_cast<size_t>(chosen[r])];
        rec.pixels = test_.images[static_cast<size_t>(chosen[r])].to_host_vector();
        rec.logits = net_.forward(test_.images[static_cast<size_t>(chosen[r])]).to_host_vector();
        rec.probs = Softmax(rec.logits);
        rec.pred = Argmax(rec.logits);
        rec.logit_black = net_.forward(ImageTensor(std::vector<float>(kPixels, 0.0f), backend_))
                              .to_host_vector()[static_cast<size_t>(rec.pred)];
    }

    for (size_t r = 0; r < records_.size(); ++r) {
        for (int m = 0; m < kMethodCount; ++m) {
            tasks_.push_back([this, r, m]() {
                current_task_label_ = std::string(kMethodNames[m]) + " on test #" + std::to_string(records_[r].test_index);
                ComputeMethod(records_[r], m);
            });
        }
        tasks_.push_back([this, r]() {
            ImageRecord& rec = records_[r];
            ExplainerContext ctx(net_.modules());
            rec.circuit = ctx.build_circuit_graph(ImageTensor(rec.pixels, backend_));
        });
    }
    for (int m = 0; m < kMethodCount; ++m) {
        if (IsCheap(m)) {
            tasks_.push_back([this, m]() { ComputeBeeswarm(m); });
        }
    }
    tasks_.push_back([this]() { PrintSummary(); });
}

void Gallery::ComputeMethod(ImageRecord& rec, int m) {
    ExplainerContext ctx(net_.modules());
    Tensor input = ImageTensor(rec.pixels, backend_);
    const int64_t target = rec.pred;
    MethodResult& out = rec.methods[m];
    auto t0 = Clock::now();
    char buf[256];
    std::optional<Attribution> features;  // superpixel-space features (LIME / KernelSHAP only)

    switch (m) {
        case kSaliency:
            out.attr = AsImageAttribution(Saliency().explain(ctx, input, target, backend_), backend_);
            out.baseline_note = "no reference point: baseline 0";
            break;
        case kIG: {
            Tensor black = ImageTensor(std::vector<float>(kPixels, 0.0f), backend_);
            out.attr = AsImageAttribution(IntegratedGradients().explain(ctx, input, black, target, kIgSteps, backend_),
                                          backend_);
            out.waterfall_baseline = rec.logit_black;
            out.baseline_note = "baseline = logit on black image";
            Stats s = Summarize(out.attr);
            std::snprintf(buf, sizeof(buf), "completeness: sum %.3f vs f(x)-f(black) %.3f", s.sum,
                          static_cast<double>(rec.logits[static_cast<size_t>(target)] - rec.logit_black));
            out.extra = buf;
            break;
        }
        case kGradCam:
            out.attr = GradCamToInputGrid(GradCAM().explain(ctx, input, target, backend_), backend_);
            out.baseline_note = "no reference point: baseline 0";
            break;
        case kLime: {
            std::vector<float> pixels = rec.pixels;
            auto predict = [&](const Tensor& z) { return net_.forward(MaskedImage(pixels, z, backend_)); };
            Tensor ones(Shape({1, kSuperpixels}), backend_, std::vector<float>(kSuperpixels, 1.0f));
            Attribution sp = LIME().explain(predict, ones, target, kLimeSamples, kLimeSigma, kLimeLambda, 7u, backend_);
            features = SuperpixelFeatures(sp, backend_);
            out.attr = SuperpixelsToPixels(sp, "16 superpixel coefficients, each spread over its 7x7 patch", backend_);
            out.baseline_note = "local linear model around the image: baseline 0";
            break;
        }
        case kKernelShap: {
            std::vector<float> pixels = rec.pixels;
            auto predict = [&](const Tensor& z) { return net_.forward(MaskedImage(pixels, z, backend_)); };
            Tensor ones(Shape({1, kSuperpixels}), backend_, std::vector<float>(kSuperpixels, 1.0f));
            Tensor zeros(Shape({1, kSuperpixels}), backend_, std::vector<float>(kSuperpixels, 0.0f));
            Attribution sp = KernelSHAP().explain(predict, ones, zeros, target, backend_);
            features = SuperpixelFeatures(sp, backend_);
            out.attr = SuperpixelsToPixels(sp, "16 superpixel SHAP values, each spread over its 7x7 patch", backend_);
            out.waterfall_baseline = rec.logit_black;
            out.baseline_note = "baseline = logit on black image";
            Stats s = Summarize(out.attr);
            std::snprintf(buf, sizeof(buf), "efficiency: sum %.3f vs f(x)-f(black) %.3f", s.sum,
                          static_cast<double>(rec.logits[static_cast<size_t>(target)] - rec.logit_black));
            out.extra = buf;
            break;
        }
        default: {  // LRP variants
            LRP lrp = MakeLrp(m);
            std::vector<Attribution> runs;
            for (int k = 0; k < 3; ++k) {  // repeated runs, for the measured stability number
                runs.push_back(lrp.explain(ctx, input, target, backend_));
            }
            Attribution raw = runs.front();
            out.attr = AsImageAttribution(raw, backend_);
            out.baseline_note = "relevance conserves the target logit: baseline 0";
            std::vector<float> seed(kClasses, 0.0f);
            seed[static_cast<size_t>(target)] = rec.logits[static_cast<size_t>(target)];
            ConservationResult conservation =
                ComputeConservation(raw.values, Tensor(Shape({1, kClasses}), backend_, seed));
            StabilityResult stability = ComputeAttributionStability(runs);
            std::snprintf(buf, sizeof(buf), "conservation: in %.4f, out %.4f, delta %.4f (rules %s)",
                          static_cast<double>(conservation.relevance_in_sum),
                          static_cast<double>(conservation.relevance_out_sum),
                          static_cast<double>(conservation.delta()), raw.metadata["rules"].c_str());
            out.extra = buf;
            ExplanationScoreCard::Input card{
                "test #" + std::to_string(rec.test_index) + ": true " + std::to_string(rec.label) + ", pred " +
                    std::to_string(rec.pred),
                rec.probs[static_cast<size_t>(target)],
                FlattenWithPixelNames(out.attr, backend_),
                conservation,
                true,  // measured: variance over the 3 repeated runs above
                stability,
            };
            rec.lrp_cards.push_back(std::move(card));
            break;
        }
    }
    out.attr.metadata["budget"] = kMethodBudgets[m];
    out.seconds = SecondsSince(t0);
    out.stats = Summarize(out.attr);
    out.flat = features ? *features : FlattenWithPixelNames(out.attr, backend_);
    out.waterfall = TopKPlusRest(out.flat, kTopK, backend_);
    out.done = true;
}

void Gallery::ComputeBeeswarm(int m) {
    current_task_label_ = std::string("beeswarm: ") + kMethodNames[m] + " over " + std::to_string(opt_.beeswarm_images) +
                          " test digits";
    ExplainerContext ctx(net_.modules());
    Tensor black = ImageTensor(std::vector<float>(kPixels, 0.0f), backend_);
    std::vector<double> mean_abs(static_cast<size_t>(kPixels), 0.0);
    int n = std::min<int>(opt_.beeswarm_images, static_cast<int>(test_.images.size()));
    for (int i = 0; i < n; ++i) {
        const Tensor& x = test_.images[static_cast<size_t>(i)];
        int64_t target = net_.predict(x);
        Attribution a = PlaceholderAttribution();
        switch (m) {
            case kSaliency:
                a = Saliency().explain(ctx, x, target, backend_);
                break;
            case kIG:
                a = IntegratedGradients().explain(ctx, x, black, target, kIgSteps, backend_);
                break;
            case kGradCam:
                a = GradCamToInputGrid(GradCAM().explain(ctx, x, target, backend_), backend_);
                break;
            default:
                a = MakeLrp(m).explain(ctx, x, target, backend_);
                break;
        }
        a = AsImageAttribution(a, backend_);
        std::vector<float> v = a.values.to_host_vector();
        for (size_t p = 0; p < v.size(); ++p) mean_abs[p] += std::abs(v[p]);
        beeswarm_runs_[m].push_back(std::move(a));
    }
    std::vector<int64_t> idx(static_cast<size_t>(kPixels));
    std::iota(idx.begin(), idx.end(), int64_t{0});
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int64_t a, int64_t b) { return mean_abs[static_cast<size_t>(a)] > mean_abs[static_cast<size_t>(b)]; });
    beeswarm_pixels_[m].assign(idx.begin(), idx.begin() + 4);
}

void Gallery::ExplainSome(double budget_seconds) {
    auto t0 = Clock::now();
    while (next_task_ < tasks_.size() && SecondsSince(t0) < budget_seconds) {
        tasks_[next_task_++]();
    }
    if (next_task_ >= tasks_.size()) {
        phase_ = Phase::Ready;
        if (!opt_.screenshot_dir.empty()) {
            std::filesystem::create_directories(opt_.screenshot_dir);
            int n = 0;
            auto add = [&](int page, int method, const std::string& slug, int image = 0) {
                char name[32];
                std::snprintf(name, sizeof(name), "%02d_", ++n);
                shots_.push_back({page, method, opt_.screenshot_dir + "/" + name + slug + ".png", image});
            };
            // The last explained digit is a misclassified one whenever BuildExplainTasks found one.
            const int error_image = static_cast<int>(records_.size()) - 1;
            const bool has_error = records_.back().pred != records_.back().label;
            add(kTraining, -1, kPageSlugs[kTraining]);
            add(kDataset, -1, kPageSlugs[kDataset]);
            add(kPrediction, -1, kPageSlugs[kPrediction]);
            add(kHeatmaps, -1, kPageSlugs[kHeatmaps]);
            if (has_error) {
                add(kPrediction, -1, "prediction_misclassified", error_image);
                add(kHeatmaps, -1, "heatmaps_misclassified", error_image);
                add(kCharts, kLrpEpsPlus, "charts_lrp_epsilon_plus_misclassified", error_image);
            }
            for (int m = 0; m < kMethodCount; ++m) add(kCharts, m, std::string("charts_") + kMethodSlugs[m]);
            add(kScoreCards, -1, kPageSlugs[kScoreCards]);
            for (int m = 0; m < kMethodCount; ++m) {
                if (IsCheap(m)) add(kBeeswarm, m, std::string("beeswarm_") + kMethodSlugs[m]);
            }
            add(kCircuit, -1, kPageSlugs[kCircuit]);
        }
    }
}

void Gallery::PrintSummary() {
    for (const ImageRecord& rec : records_) {
        std::printf("Test digit #%lld: true %lld, predicted %lld (p = %.3f, logit %.3f)\n",
                    static_cast<long long>(rec.test_index), static_cast<long long>(rec.label),
                    static_cast<long long>(rec.pred), static_cast<double>(rec.probs[static_cast<size_t>(rec.pred)]),
                    static_cast<double>(rec.logits[static_cast<size_t>(rec.pred)]));
        for (int m = 0; m < kMethodCount; ++m) {
            const MethodResult& r = rec.methods[m];
            std::printf("  %-28s sum %10.4f  max|a| %9.5f  range [%9.5f, %9.5f]  %6.2fs%s\n", kMethodNames[m], r.stats.sum,
                        static_cast<double>(r.stats.max_abs), static_cast<double>(r.stats.min),
                        static_cast<double>(r.stats.max), r.seconds, r.stats.finite ? "" : "  NON-FINITE!");
            if (!r.extra.empty()) std::printf("  %-28s %s\n", "", r.extra.c_str());
        }
        if (rec.circuit) {
            std::printf("  circuit (zero-ablation effect on logits, L2):");
            for (const CircuitNode& node : rec.circuit->nodes()) {
                std::printf(" %s=%.3f", CircuitNodeDisplayLabel(node).c_str(), static_cast<double>(node.ablation_effect));
            }
            std::printf("\n");
        }
        std::printf("\n");
    }
    std::printf("Explainer budgets:\n");
    for (int m = 0; m < kMethodCount; ++m) std::printf("  %-28s %s\n", kMethodNames[m], kMethodBudgets[m]);
    std::printf("Beeswarms: %d test digits per cheap method (LIME/KernelSHAP excluded: ~1-20 s per digit)\n\n",
                opt_.beeswarm_images);
    std::fflush(stdout);
}

// ---------------------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------------------
void Gallery::Frame(VizWindow& /*window*/) {
    if (phase_ == Phase::Training) {
        TrainSome(0.03);
    } else if (phase_ == Phase::Explaining) {
        ExplainSome(0.05);
    }

    if (!shots_.empty() && shot_index_ < shots_.size()) {
        page_ = shots_[shot_index_].page;
        selected_image_ = shots_[shot_index_].image;
        if (shots_[shot_index_].method >= 0) selected_method_ = shots_[shot_index_].method;
    }

    const ImGuiViewport* vp = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(vp->WorkPos);
    ImGui::SetNextWindowSize(vp->WorkSize);
    ImGui::Begin("MNIST visualization gallery", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    if (phase_ != Phase::Ready) {
        if (phase_ == Phase::Training) {
            ImGui::Text("Training MnistConvNet on %zu MNIST digits x %d epoch(s) -- step %lld / %lld",
                        train_.images.size(), opt_.epochs, static_cast<long long>(step_),
                        static_cast<long long>(total_steps_));
            ImGui::ProgressBar(static_cast<float>(step_) / static_cast<float>(std::max<int64_t>(1, total_steps_)));
            ImGui::Separator();
            TrainingDashboard::Draw(sink_);
        } else {
            ImGui::Text("Explaining %zu test digits with %d methods (+ circuit graphs, beeswarms) -- %s",
                        records_.size(), static_cast<int>(kMethodCount), current_task_label_.c_str());
            ImGui::ProgressBar(static_cast<float>(next_task_) / static_cast<float>(std::max<size_t>(1, tasks_.size())));
        }
        ImGui::End();
        return;
    }

    ImGui::BeginChild("nav", ImVec2(250, 0), true);
    DrawNav();
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("page", ImVec2(0, 0), false);
    DrawPage();
    ImGui::EndChild();
    ImGui::End();
}

void Gallery::AfterRender(int w, int h, VizWindow& window) {
    if (phase_ != Phase::Ready || shots_.empty()) return;
    if (shot_index_ >= shots_.size()) {
        window.request_close();
        return;
    }
    // A few frames per page: ImPlot auto-fit and ImGui layout settle over the first frames.
    if (++shot_frames_ < 4) return;
    shot_frames_ = 0;
    std::vector<unsigned char> rgba(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data());
    for (size_t i = 3; i < rgba.size(); i += 4) rgba[i] = 255;
    stbi_flip_vertically_on_write(1);  // GL rows are bottom-up
    const std::string& file = shots_[shot_index_].file;
    if (stbi_write_png(file.c_str(), w, h, 4, rgba.data(), w * 4) != 0) {
        written_.push_back(file);
        std::printf("wrote %s (%dx%d)\n", file.c_str(), w, h);
    } else {
        std::fprintf(stderr, "failed to write %s\n", file.c_str());
    }
    std::fflush(stdout);
    ++shot_index_;
}

// ---------------------------------------------------------------------------------------
// Pages
// ---------------------------------------------------------------------------------------
void Gallery::DrawNav() {
    ImGui::TextUnformatted("MNIST viz gallery");
    ImGui::Text("test acc %.2f%% (n=%lld)", static_cast<double>(final_accuracy_) * 100.0,
                static_cast<long long>(final_accuracy_n_));
    ImGui::Separator();
    for (int p = 0; p < kPageCount; ++p) {
        if (ImGui::Selectable(kPageNames[p], page_ == p)) page_ = p;
    }
    ImGui::Separator();
    DrawImageSelector();
    ImGui::Separator();
    ImGui::TextUnformatted("Method (bar/waterfall,");
    ImGui::TextUnformatted("beeswarm):");
    for (int m = 0; m < kMethodCount; ++m) {
        if (ImGui::RadioButton(kMethodNames[m], selected_method_ == m)) selected_method_ = m;
    }
}

void Gallery::DrawImageSelector() {
    ImGui::TextUnformatted("Explained test digit:");
    if (ImGui::ArrowButton("##prev", ImGuiDir_Left)) {
        selected_image_ = (selected_image_ + static_cast<int>(records_.size()) - 1) % static_cast<int>(records_.size());
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("##img", &selected_image_, 0, static_cast<int>(records_.size()) - 1);
    ImGui::SameLine();
    if (ImGui::ArrowButton("##next", ImGuiDir_Right)) {
        selected_image_ = (selected_image_ + 1) % static_cast<int>(records_.size());
    }
    const ImageRecord& rec = records_[static_cast<size_t>(selected_image_)];
    DrawDigit(rec, 112.0f);
    ImGui::Text("test #%lld  true %lld  pred %lld", static_cast<long long>(rec.test_index),
                static_cast<long long>(rec.label), static_cast<long long>(rec.pred));
}

void Gallery::DrawDigit(const ImageRecord& rec, float size) {
    TextureId tex = textures_.GetOrUpload(rec.test_index, test_.images[static_cast<size_t>(rec.test_index)]);
    ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(tex)), ImVec2(size, size));
}

void Gallery::DrawPage() {
    ImGui::TextUnformatted(kPageNames[page_]);
    ImGui::Separator();
    switch (page_) {
        case kTraining:
            DrawTrainingPage();
            break;
        case kDataset:
            DrawDatasetPage();
            break;
        case kPrediction:
            DrawPredictionPage();
            break;
        case kHeatmaps:
            DrawHeatmapsPage();
            break;
        case kCharts:
            DrawChartsPage();
            break;
        case kScoreCards:
            DrawScoreCardsPage();
            break;
        case kBeeswarm:
            DrawBeeswarmPage();
            break;
        case kCircuit:
            DrawCircuitPage();
            break;
        default:
            break;
    }
}

void Gallery::DrawTrainingPage() {
    ImGui::Text("MnistConvNet: Conv2D(1,8,5x5) -> ReLU -> Flatten -> Linear(4608,10), Adam lr 1e-3, %d epoch(s) x %zu "
                "train digits, %.1f s. Final test accuracy %.2f%% on %lld digits.",
                opt_.epochs, train_.images.size(), train_seconds_, static_cast<double>(final_accuracy_) * 100.0,
                static_cast<long long>(final_accuracy_n_));
    ImGui::Separator();
    TrainingDashboard::Draw(sink_);
}

void Gallery::DrawDatasetPage() {
    float half = ImGui::GetContentRegionAvail().x * 0.5f;
    ImGui::BeginChild("stats", ImVec2(half, 0), true);
    ImGui::Text("DatasetStatisticsView -- first %lld MNIST test digits (field 0: pixels, field 1: label)",
                static_cast<long long>(stats_dataset_->size()));
    DatasetStatisticsView::Draw("mnist_stats", *stats_dataset_, 10);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("grid", ImVec2(0, 0), true);
    ImGui::Text("ImageGridView -- test digits #0-#%d, row-major (t = true label, p = prediction, X = error)", kGridCount - 1);
    ImageGridView::Draw("mnist_grid", *preview_dataset_, textures_, 0, kGridCount, 8, 76.0f, &grid_captions_);
    ImGui::EndChild();
}

void Gallery::DrawPredictionPage() {
    const ImageRecord& rec = records_[static_cast<size_t>(selected_image_)];
    DrawDigit(rec, 196.0f);
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::Text("Test digit #%lld -- true label %lld, predicted %lld", static_cast<long long>(rec.test_index),
                static_cast<long long>(rec.label), static_cast<long long>(rec.pred));
    ImGui::SetNextItemWidth(400);
    ConfidenceMeter::Draw("Confidence in the predicted class (softmax)", rec.probs[static_cast<size_t>(rec.pred)]);
    ImGui::EndGroup();
    ImGui::Separator();
    ImGui::TextUnformatted("ConfidenceMeter per class (10-way softmax over the logits):");
    if (ImGui::BeginTable("probs", 2, ImGuiTableFlags_SizingStretchSame)) {
        for (int64_t c = 0; c < kClasses; ++c) {
            ImGui::TableNextColumn();
            char label[64];
            std::snprintf(label, sizeof(label), "digit %lld%s%s   (logit %.2f)", static_cast<long long>(c),
                          c == rec.pred ? "  <- predicted" : "", c == rec.label ? "  <- true" : "",
                          static_cast<double>(rec.logits[static_cast<size_t>(c)]));
            ImGui::PushID(static_cast<int>(c));
            ConfidenceMeter::Draw(label, rec.probs[static_cast<size_t>(c)]);
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void Gallery::DrawHeatmapsPage() {
    const ImageRecord& rec = records_[static_cast<size_t>(selected_image_)];
    ImGui::Text("SaliencyHeatmapView for every explainer -- test #%lld (true %lld, pred %lld), target = predicted class. "
                "Signed maps: blue-white-red centred at 0; unsigned (Grad-CAM): viridis.",
                static_cast<long long>(rec.test_index), static_cast<long long>(rec.label),
                static_cast<long long>(rec.pred));
    constexpr int kCols = 4;  // 10 panels (input + 9 methods) -> 3 rows
    ImVec2 avail = ImGui::GetContentRegionAvail();
    float cell_h = avail.y / 3.0f - 8.0f;
    float map_h = std::min(cell_h - 40.0f, avail.x / kCols - 90.0f);
    if (ImGui::BeginTable("heatmaps", kCols, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_BordersInnerV)) {
        ImGui::TableNextColumn();
        ImGui::Text("Input digit (true %lld, pred %lld)", static_cast<long long>(rec.label),
                    static_cast<long long>(rec.pred));
        ImGui::Text("p(pred) = %.3f", static_cast<double>(rec.probs[static_cast<size_t>(rec.pred)]));
        DrawDigit(rec, map_h);
        for (int m = 0; m < kMethodCount; ++m) {
            ImGui::TableNextColumn();
            const MethodResult& r = rec.methods[m];
            ImGui::PushID(m);
            ImGui::TextUnformatted(kMethodNames[m]);
            ImGui::Text("sum %.3g  max|a| %.3g", r.stats.sum, static_cast<double>(r.stats.max_abs));
            ImGui::BeginChild("hm", ImVec2(0, map_h), false);
            SaliencyHeatmapView::Draw("##heat", r.attr);
            ImGui::EndChild();
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void Gallery::DrawChartsPage() {
    const ImageRecord& rec = records_[static_cast<size_t>(selected_image_)];
    const MethodResult& r = rec.methods[selected_method_];
    ImGui::Text("%s on test #%lld (true %lld, pred %lld) -- budget: %s", kMethodNames[selected_method_],
                static_cast<long long>(rec.test_index), static_cast<long long>(rec.label),
                static_cast<long long>(rec.pred), kMethodBudgets[selected_method_]);
    ImGui::Text("sum of attribution %.4f, max|a| %.4g, computed in %.2f s.  %s", r.stats.sum,
                static_cast<double>(r.stats.max_abs), r.seconds, r.extra.c_str());
    float h = ImGui::GetContentRegionAvail().y;
    float left_w = ImGui::GetContentRegionAvail().x * 0.30f;
    ImGui::BeginChild("chart_heat", ImVec2(left_w, h * 0.5f), true);
    ImGui::TextUnformatted("SaliencyHeatmapView");
    SaliencyHeatmapView::Draw("##chart_heat", r.attr);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("bars", ImVec2(0, h * 0.5f), true);
    ImGui::Text("AttributionBarChart -- top %d of %lld %s by |attribution|", kTopK,
                static_cast<long long>(r.flat.values.numel()),
                r.flat.values.numel() == kPixels ? "pixels" : "superpixels (patch rIcJ = 7x7 patch at grid row I, col J)");
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotDefaultSize, ImVec2(-1, ImGui::GetContentRegionAvail().y - 4));
    AttributionBarChart::Draw("##bars", r.flat, kTopK);
    ImPlot::PopStyleVar();
    ImGui::EndChild();
    ImGui::BeginChild("waterfall", ImVec2(0, 0), true);
    float final_value = r.waterfall_baseline + static_cast<float>(r.stats.sum);
    ImGui::Text("AttributionWaterfallChart -- top %d features + 'other' bar; start %.3f (%s) -> end %.3f; model logit for "
                "class %lld: %.3f",
                kTopK, static_cast<double>(r.waterfall_baseline), r.baseline_note.c_str(), static_cast<double>(final_value),
                static_cast<long long>(rec.pred), static_cast<double>(rec.logits[static_cast<size_t>(rec.pred)]));
    ImPlot::PushStyleVar(ImPlotStyleVar_PlotDefaultSize, ImVec2(-1, ImGui::GetContentRegionAvail().y - 4));
    AttributionWaterfallChart::Draw("##waterfall", r.waterfall, r.waterfall_baseline);
    ImPlot::PopStyleVar();
    ImGui::EndChild();
}

void Gallery::DrawScoreCardsPage() {
    const ImageRecord& rec = records_[static_cast<size_t>(selected_image_)];
    ImGui::Text("ExplanationScoreCard per LRP rule set (small multiples, shared bar scale). Conservation = sum of input "
                "relevance vs the seeded target logit %.4f; stability = variance over 3 repeated runs.",
                static_cast<double>(rec.logits[static_cast<size_t>(rec.pred)]));
    ScoreCardScaleContext shared;
    if (ImGui::BeginTable("cards", 2, ImGuiTableFlags_SizingStretchSame)) {
        for (size_t i = 0; i < rec.lrp_cards.size(); ++i) {
            ImGui::TableNextColumn();
            std::string title = std::string(kMethodNames[kLrpEpsilon + static_cast<int>(i)]);
            ExplanationScoreCard::Draw(title.c_str(), rec.lrp_cards[i], &shared);
        }
        ImGui::EndTable();
    }
}

void Gallery::DrawBeeswarmPage() {
    int m = selected_method_;
    if (!IsCheap(m)) {
        ImGui::TextWrapped("%s: not run across many digits -- %s costs ~1-20 s per digit on CPU. Pick a gradient/LRP "
                           "method on the left.",
                           kMethodNames[m], kMethodNames[m]);
        return;
    }
    ImGui::Text("AttributionBeeswarmView -- %s across the first %zu test digits (target = each digit's predicted "
                "class); the 4 pixels with the largest mean |attribution|. Each dot is one digit.",
                kMethodNames[m], beeswarm_runs_[m].size());
    float h = (ImGui::GetContentRegionAvail().y - 8.0f) / 2.0f;
    if (ImGui::BeginTable("swarms", 2, ImGuiTableFlags_SizingStretchSame)) {
        for (size_t i = 0; i < beeswarm_pixels_[m].size(); ++i) {
            ImGui::TableNextColumn();
            int64_t p = beeswarm_pixels_[m][i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::BeginChild("swarm", ImVec2(0, h), true);
            ImGui::TextUnformatted(PixelName(p).c_str());
            AttributionBeeswarmView::Draw(PixelName(p).c_str(), beeswarm_runs_[m], p);
            ImGui::EndChild();
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

void Gallery::DrawCircuitPage() {
    const ImageRecord& rec = records_[static_cast<size_t>(selected_image_)];
    ImGui::TextWrapped("CircuitGraphView -- ExplainerContext::build_circuit_graph on the full CNN for test #%lld: each "
                       "node is one layer output, sized/coloured by its zero-ablation effect (L2 change of the 10 "
                       "logits when that activation is replaced by zeros).",
                       static_cast<long long>(rec.test_index));
    if (!rec.circuit) return;
    if (ImGui::BeginTable("nodes", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("node");
        ImGui::TableSetupColumn("ablation effect");
        ImGui::TableHeadersRow();
        for (const CircuitNode& n : rec.circuit->nodes()) {
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(CircuitNodeDisplayLabel(n).c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%.4f", static_cast<double>(n.ablation_effect));
        }
        ImGui::EndTable();
    }
    ImGui::BeginChild("circuit", ImVec2(0, 0), true);
    CircuitGraphView::Draw("##circuit", *rec.circuit);
    ImGui::EndChild();
}

}  // namespace

int main(int argc, char** argv) {
    using namespace pulsatrix;
    Options opt = ParseOptions(argc, argv);
    CPUBackend backend;

    MnistDataset train;
    MnistDataset test;
    try {
        train = MnistIdxLoader::Load(opt.data_dir + "/train-images-idx3-ubyte", opt.data_dir + "/train-labels-idx1-ubyte",
                                     &backend, opt.train_count);
        test = MnistIdxLoader::Load(opt.data_dir + "/t10k-images-idx3-ubyte", opt.data_dir + "/t10k-labels-idx1-ubyte",
                                    &backend);
    } catch (const std::runtime_error& e) {
        std::fprintf(stderr,
                     "Could not load MNIST from %s: %s\nRun `python3 tools/fetch_mnist.py` first (or pass --data DIR).\n",
                     opt.data_dir.c_str(), e.what());
        return 1;
    }
    std::printf("MNIST viz gallery: %zu train digits x %d epoch(s), %zu test digits, explaining %d test digit(s)\n\n",
                train.images.size(), opt.epochs, test.images.size(), opt.images);
    std::fflush(stdout);

    try {
        VizWindow window("pulsatrix -- MNIST visualization gallery", 1680, 1000);
        Gallery gallery(opt, &backend, std::move(train), std::move(test));
        window.run([&]() { gallery.Frame(window); }, [&](int w, int h) { gallery.AfterRender(w, h, window); });
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mnist_viz_gallery: %s\n", e.what());
        return 1;
    }
    return 0;
}
