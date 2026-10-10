// Recipe: a notebook report written from C++ (roadmap NB-2). Trains a small MLP on XOR quadrants,
// explains it with LRP, and writes everything (text, code, the training curve, the model's output
// over the input plane and the explanations) as a Jupyter notebook and as an HTML page.
//
//   notebook_report_recipe [OUT_DIR]      (default: the current directory)
//
// Writes OUT_DIR/xor_report.ipynb and OUT_DIR/xor_report.html. See
// docs/recipes/visualization/notebook_report.md.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/token_cross_entropy_loss.hpp"
#include "pulsatrix/viz/report.hpp"

using namespace pulsatrix;

namespace {

/** He-initialized weights, (in, out) as LinearModule stores them, and zero biases. */
void Init(LinearModule& layer, int64_t in, int64_t out, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, std::sqrt(2.0f / static_cast<float>(in)));
    std::vector<float> w(static_cast<size_t>(in * out));
    for (float& v : w) v = normal(rng);
    layer.set_weight(w);
    layer.set_bias(std::vector<float>(static_cast<size_t>(out), 0.0f));
}

std::vector<float> Softmax(const Tensor& logits, CPUBackend& cpu) {
    Tensor p(logits.shape(), &cpu);
    cpu.softmax_rows(logits.data(), p.data(), static_cast<size_t>(logits.shape().dim(0)),
                     static_cast<size_t>(logits.shape().dim(1)));
    return p.to_host_vector();
}

std::string Point(float x, float y) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "(%.1f, %.1f)", x, y);
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out = argc > 1 ? argv[1] : ".";
    CPUBackend cpu;
    std::mt19937 rng(7);

    // 1. Data: 256 points in the unit square; the class is which XOR quadrant a point is in.
    constexpr int64_t kPoints = 256;
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::vector<float> xs(kPoints * 2), labels(kPoints);
    for (int64_t i = 0; i < kPoints; ++i) {
        xs[2 * i] = unit(rng);
        xs[2 * i + 1] = unit(rng);
        labels[i] = static_cast<float>((xs[2 * i] > 0.5f) != (xs[2 * i + 1] > 0.5f));
    }
    const Tensor x(Shape({kPoints, 2}), &cpu, xs), y(Shape({kPoints}), &cpu, labels);

    // 2. Model: Linear(2,16) -> ReLU -> Linear(16,16) -> ReLU -> Linear(16,2).
    LinearModule l1(2, 16, &cpu), l2(16, 16, &cpu), l3(16, 2, &cpu);
    ReluModule r1(&cpu), r2(&cpu);
    Init(l1, 2, 16, rng);
    Init(l2, 16, 16, rng);
    Init(l3, 16, 2, rng);
    SequentialModule model({&l1, &r1, &l2, &r2, &l3});

    // 3. Training, full batch, logging the loss and accuracy for the report.
    AdamOptimizer optimizer(0.02f, &cpu);
    TrainingLogDocument log;
    log.scalars = {{"accuracy", {}, {}}, {"loss", {}, {}}};  // sorted by tag
    float accuracy = 0.0f;
    for (int64_t step = 0; step <= 400; ++step) {
        optimizer.zero_grad(model);
        TokenCrossEntropyLoss loss(&cpu);
        const Tensor logits = model.forward(x);
        const float l = loss.forward(logits, y);
        (void)model.backward(loss.backward());
        optimizer.step(model);
        if (step % 10 == 0) {
            const std::vector<float> p = Softmax(logits, cpu);
            int64_t correct = 0;
            for (int64_t i = 0; i < kPoints; ++i) correct += (p[2 * i + 1] > 0.5f) == (labels[i] > 0.5f);
            accuracy = static_cast<float>(correct) / static_cast<float>(kPoints);
            log.scalars[0].steps.push_back(step);
            log.scalars[0].values.push_back(accuracy);
            log.scalars[1].steps.push_back(step);
            log.scalars[1].values.push_back(l);
        }
    }
    std::printf("trained: loss %.4f, accuracy %.1f%%\n", log.scalars[1].values.back(), 100.0f * accuracy);

    // 4. What the model learned: P(class 1) over a 40 x 40 grid of the unit square.
    constexpr int64_t kGrid = 40;
    std::vector<float> grid(kGrid * kGrid * 2);
    for (int64_t r = 0; r < kGrid; ++r) {
        for (int64_t c = 0; c < kGrid; ++c) {
            grid[2 * (r * kGrid + c)] = (static_cast<float>(c) + 0.5f) / kGrid;            // x across
            grid[2 * (r * kGrid + c) + 1] = 1.0f - (static_cast<float>(r) + 0.5f) / kGrid;  // y up
        }
    }
    const std::vector<float> p_grid = Softmax(model.forward(Tensor(Shape({kGrid * kGrid, 2}), &cpu, grid)), cpu);
    HeatmapDocument landscape;
    landscape.title = "P(class 1) over the unit square (x across, y up)";
    landscape.rows = kGrid;
    landscape.cols = kGrid;
    for (int64_t i = 0; i < kGrid * kGrid; ++i) landscape.values.push_back(p_grid[2 * i + 1]);

    // 5. Why: LRP relevance of x and y for class 1, at one point in each quadrant.
    const std::vector<float> probes = {0.2f, 0.2f, 0.2f, 0.8f, 0.8f, 0.2f, 0.8f, 0.8f};
    const Tensor probe(Shape({4, 2}), &cpu, probes);
    const std::vector<float> p_probe = Softmax(model.forward(probe), cpu);
    ExplainerContext ctx({&l1, &r1, &l2, &r2, &l3});
    const Attribution relevance = LRP().explain(ctx, probe, /*target_index=*/1, &cpu);
    HeatmapDocument why;
    why.title = "LRP relevance for class 1";
    why.rows = 4;
    why.cols = 2;
    why.values = relevance.values.to_host_vector();
    why.col_labels = {"x", "y"};
    for (int i = 0; i < 4; ++i) why.row_labels.push_back(Point(probes[2 * i], probes[2 * i + 1]));

    // 6. The report.
    Report report("XOR quadrants: training and explaining a small MLP");
    report.markdown(
        "A small MLP learns which XOR quadrant a point of the unit square is in: class 1 when exactly "
        "one of `x` and `y` is above 0.5. This report was written by `notebook_report_recipe`, from C++.");
    report.markdown("## Data and model");
    report.code(
        "// 256 points in the unit square; label = (x > 0.5) != (y > 0.5)\n"
        "SequentialModule model({&l1, &r1, &l2, &r2, &l3});  // Linear(2,16)-ReLU-Linear(16,16)-ReLU-Linear(16,2)");
    report.markdown("## Training");
    report.code(
        "for (int64_t step = 0; step <= 400; ++step) {\n"
        "    optimizer.zero_grad(model);\n"
        "    const float l = loss.forward(model.forward(x), y);\n"
        "    (void)model.backward(loss.backward());\n"
        "    optimizer.step(model);  // Adam, learning rate 0.02\n"
        "}")
        .show(log);
    char summary[96];
    std::snprintf(summary, sizeof summary, "final loss %.4f, training accuracy %.1f%%", log.scalars[1].values.back(),
                  100.0f * accuracy);
    report.text(summary);
    report.markdown("## What the model learned\n\nThe probability of class 1 at each point of the unit square:");
    report.show(landscape);
    report.markdown(
        "## Why: LRP\n\n"
        "Layer-wise Relevance Propagation splits the class-1 logit between the two inputs, for one point "
        "in each quadrant. Red pushes toward class 1, blue against.");
    report.code("Attribution r = LRP().explain(ctx, probe, /*target_index=*/1, &cpu);").show(why);
    report.markdown("The model's probabilities at those points, as a tensor (columns: class 0, class 1):");
    report.show(Tensor(Shape({4, 2}), &cpu, p_probe));

    report.save_ipynb(out + "/xor_report.ipynb");
    report.save_html(out + "/xor_report.html");
    std::printf("wrote %s/xor_report.ipynb and %s/xor_report.html (%zu cells)\n", out.c_str(), out.c_str(),
                report.num_cells());
    return 0;
}
