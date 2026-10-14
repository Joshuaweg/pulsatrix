// KD-1 on the GPU: DistillationLoss on the HIP backend gives the CPU's loss and gradient, with and
// without dropped rows. KD-3's and KD-4's pieces follow.
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "pulsatrix/born_again.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/distillation_loss.hpp"
#include "pulsatrix/feature_distillation.hpp"
#include "pulsatrix/hip_backend.hpp"
#include "pulsatrix/layer_student.hpp"
#include "pulsatrix/soft_decision_tree.hpp"

namespace pulsatrix {
namespace {

struct Outcome {
    float loss;
    std::vector<float> grad;
};

Outcome Distill(DeviceBackend* backend, DistillationDivergence divergence, bool with_targets) {
    const Shape shape({3, 5, 40});
    std::vector<float> s(static_cast<size_t>(shape.numel())), t(s.size());
    for (size_t k = 0; k < s.size(); ++k) {
        s[k] = static_cast<float>(2 * std::sin(0.7 * k + 0.3));
        t[k] = static_cast<float>(3 * std::cos(0.45 * k + 0.1));
    }
    std::vector<float> y(15);
    for (size_t k = 0; k < y.size(); ++k) y[k] = k % 4 == 1 ? -100.0f : static_cast<float>((7 * k) % 40);
    DistillationOptions o;
    o.temperature = 2.5f;
    o.alpha = 0.6f;
    o.divergence = divergence;
    DistillationLoss loss(backend, o);
    const Tensor st(shape, backend, s, backend->device()), tt(shape, backend, t, backend->device());
    const float value = with_targets ? loss.forward(st, tt, Tensor(Shape({3, 5}), backend, y, backend->device()))
                                     : loss.forward(st, tt);
    const Tensor grad = loss.backward();
    EXPECT_EQ(grad.device(), backend->device());
    return {value, grad.to_host_vector()};
}

TEST(DistillationLossHip, TheGpuGivesTheCpusLossAndGradient) {
    HIPBackend hip;
    CPUBackend cpu;
    for (DistillationDivergence divergence : {DistillationDivergence::Forward, DistillationDivergence::Reverse}) {
        for (bool with_targets : {false, true}) {
            const Outcome g = Distill(&hip, divergence, with_targets), c = Distill(&cpu, divergence, with_targets);
            EXPECT_NEAR(g.loss, c.loss, 1e-5 * std::abs(c.loss) + 1e-6);
            ASSERT_EQ(g.grad.size(), c.grad.size());
            for (size_t k = 0; k < g.grad.size(); ++k) EXPECT_NEAR(g.grad[k], c.grad[k], 1e-6) << k;
        }
    }
}

// KD-3: the hint loss (through its regressor) and attention transfer on the GPU give the CPU's.
std::vector<float> Wave(size_t n, double a, double b) {
    std::vector<float> v(n);
    for (size_t k = 0; k < n; ++k) v[k] = static_cast<float>(std::sin(a * static_cast<double>(k) + b));
    return v;
}

std::vector<std::vector<float>> FeatureLosses(DeviceBackend* backend) {
    const DeviceType d = backend->device();
    HintLoss hint(6, 10, backend, 4);
    const float h = hint.forward(Tensor(Shape({3, 4, 6}), backend, Wave(72, 0.7, 0.3), d), Tensor(Shape({3, 4, 10}), backend, Wave(120, 0.45, 0.1), d));
    AttentionTransferLoss at(backend);
    const float a = at.forward(Tensor(Shape({2, 3, 4, 5}), backend, Wave(120, 0.9, 0.2), d), Tensor(Shape({2, 6, 4, 5}), backend, Wave(240, 0.35, 0.6), d));
    const Tensor hg = hint.backward(), ag = at.backward();
    EXPECT_EQ(hg.device(), d);
    EXPECT_EQ(ag.device(), d);
    return {{h, a}, hg.to_host_vector(), hint.regressor()->weight_grad().to_host_vector(), ag.to_host_vector()};
}

TEST(FeatureDistillationHip, TheGpuGivesTheCpusLossesAndGradients) {
    HIPBackend hip;
    CPUBackend cpu;
    const std::vector<std::vector<float>> g = FeatureLosses(&hip), c = FeatureLosses(&cpu);
    for (size_t part = 0; part < g.size(); ++part) {
        ASSERT_EQ(g[part].size(), c[part].size());
        for (size_t k = 0; k < g[part].size(); ++k) EXPECT_NEAR(g[part][k], c[part][k], 1e-5 * std::abs(c[part][k]) + 1e-6) << part << " " << k;
    }
}

// KD-3: the soft decision tree with its gates on the GPU gives the CPU's output and gradients.
std::vector<std::vector<float>> SoftTree(DeviceBackend* backend) {
    SoftDecisionTreeOptions o;
    o.depth = 3;
    o.beta = 1.5f;
    o.penalty = 0.1f;
    o.seed = 5;
    SoftDecisionTree tree(4, 3, backend, o);
    const Tensor y = tree.forward(Tensor(Shape({6, 4}), backend, Wave(24, 0.37, 0.5), backend->device()));
    const Tensor dx = tree.backward(Tensor(Shape({6, 3}), backend, Wave(18, 0.6, 0.9), backend->device()));
    EXPECT_EQ(y.device(), backend->device());
    std::vector<NamedParamRef> params = tree.named_parameters();
    return {y.to_host_vector(), dx.to_host_vector(), params[0].ref.grad->to_host_vector(), params[2].ref.grad->to_host_vector()};
}

TEST(SoftDecisionTreeHip, TheGpuGivesTheCpusOutputAndGradients) {
    HIPBackend hip;
    CPUBackend cpu;
    const std::vector<std::vector<float>> g = SoftTree(&hip), c = SoftTree(&cpu);
    for (size_t part = 0; part < g.size(); ++part) {
        ASSERT_EQ(g[part].size(), c[part].size());
        for (size_t k = 0; k < g[part].size(); ++k) EXPECT_NEAR(g[part][k], c[part][k], 1e-5 * std::abs(c[part][k]) + 1e-6) << part << " " << k;
    }
}

// KD-4: a student built on the GPU from a GPU or a CPU teacher computes the teacher's logits; the
// tap and the cosine loss train it there; the ensemble and the permutation give the CPU's values.
TEST(LayerStudentHip, StudentsBuiltOnTheGpuComputeTheTeachersLogits) {
    HIPBackend hip;
    CPUBackend cpu;
    const std::string dir = std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/llama";
    std::unique_ptr<CausalLM> cpu_teacher = LoadCausalLM(dir, &cpu), hip_teacher = LoadCausalLM(dir, &hip);
    std::vector<float> ids(12);
    for (size_t k = 0; k < ids.size(); ++k) ids[k] = static_cast<float>((5 * k + 1) % 64);
    const Tensor cpu_ids(Shape({2, 6}), &cpu, ids), hip_ids(Shape({2, 6}), &hip, ids, DeviceType::Hip);
    const std::vector<float> expected = cpu_teacher->forward(cpu_ids).to_host_vector();
    for (CausalLM* teacher : {cpu_teacher.get(), hip_teacher.get()}) {
        std::unique_ptr<CausalLM> student = MakeStudentFromLayers(*teacher, {0, 1}, &hip);
        for (const NamedParamRef& p : student->named_parameters()) EXPECT_EQ(p.ref.value->device(), DeviceType::Hip) << p.name;
        const std::vector<float> got = student->forward(hip_ids).to_host_vector();
        ASSERT_EQ(got.size(), expected.size());
        for (size_t k = 0; k < got.size(); ++k) EXPECT_NEAR(got[k], expected[k], 1e-4) << k;
    }
}

std::vector<std::vector<float>> CosineAndTap(DeviceBackend* backend) {
    const DeviceType d = backend->device();
    HiddenStateCosineLoss loss(backend);
    const float value = loss.forward(Tensor(Shape({2, 3, 5}), backend, Wave(30, 0.7, 0.3), d), Tensor(Shape({2, 3, 5}), backend, Wave(30, 0.45, 0.1), d),
                                     {1, 1, 1, 1, 0, 1});
    const Tensor grad = loss.backward();
    EXPECT_EQ(grad.device(), d);
    HiddenStateTap tap(backend);
    (void)tap.forward(Tensor(Shape({2, 3, 5}), backend, Wave(30, 0.2, 0.0), d));
    tap.add_gradient(grad);
    const Tensor through = tap.backward(Tensor(Shape({2, 3, 5}), backend, Wave(30, 0.3, 0.6), d));
    EXPECT_EQ(through.device(), d);
    const Shape shape({2, 2, 4});
    const Tensor ensemble = EnsembleLogProbabilities({Tensor(shape, backend, Wave(16, 0.8, 0.1), d), Tensor(shape, backend, Wave(16, 0.3, 0.9), d)}, backend);
    const Tensor permuted = PermuteDarkKnowledge(Tensor(Shape({3, 5}), backend, Wave(15, 0.9, 0.2), d), 11, backend);
    EXPECT_EQ(ensemble.device(), d);
    EXPECT_EQ(permuted.device(), d);
    return {{value}, grad.to_host_vector(), through.to_host_vector(), ensemble.to_host_vector(), permuted.to_host_vector()};
}

TEST(LayerStudentHip, TheGpuGivesTheCpusCosineLossTapEnsembleAndPermutation) {
    HIPBackend hip;
    CPUBackend cpu;
    const auto g = CosineAndTap(&hip), c = CosineAndTap(&cpu);
    ASSERT_EQ(g.size(), c.size());
    for (size_t i = 0; i < g.size(); ++i) {
        ASSERT_EQ(g[i].size(), c[i].size()) << i;
        for (size_t k = 0; k < g[i].size(); ++k) EXPECT_NEAR(g[i][k], c[i][k], 1e-6) << i << " " << k;
    }
}

}  // namespace
}  // namespace pulsatrix
