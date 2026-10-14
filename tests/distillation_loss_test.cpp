// KD-1: DistillationLoss against tools/generate_distillation_reference_values.py (numpy, float64,
// gradients by central differences), and a student that learns a teacher from its soft labels.
#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/distillation_loss.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

const double kForwardLoss = 1.8041053929, kForwardSoft = 1.8041053929, kForwardHard = 0.0000000000;
const std::vector<double> kForwardGrad = {-0.1120270927, -0.0245700840, 0.0509931444, 0.0553939470, 0.0302100849, -0.0325751468, -0.0231008723, -0.0117270222, 0.0110227532, 0.0563802875, 0.1068858810, 0.0987087376, 0.0099444148, -0.0878701980, -0.1276688359, -0.1897523513, -0.1003094718, -0.0076470049, 0.0941547444, 0.2035540838};
const double kReverseLoss = 0.6405466441, kReverseSoft = 0.6405466441, kReverseHard = 0.0000000000;
const std::vector<double> kReverseGrad = {-0.0578877398, -0.0289815222, 0.0168521527, 0.0337173684, 0.0234309275, 0.0128688133, -0.0091541942, -0.0050582556, 0.0040228614, 0.0192412959, 0.0195290475, -0.0285807553, 0.0777461924, -0.0151962250, -0.0409841423, -0.0368003727, -0.0182559754, 0.0334905226, -0.0434788605, 0.0174308904, 0.0566729126, 0.0282595407, -0.0174404623, -0.0414440213, -0.0015759138, -0.0166460161, -0.0249878465, -0.0185962367, 0.0138625397, 0.0479434733, -0.0308970276, -0.0206912133, -0.0125950228, -0.0025238968, 0.0146606342, 0.0520465266};
const double kForwardTargetsLoss = 2.3023846072, kForwardTargetsSoft = 1.5206353027, kForwardTargetsHard = 2.6374200233;
const std::vector<double> kForwardTargetsGrad = {-0.0119087191, -0.0904996684, 0.0517445529, 0.0351191649, 0.0155446691, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0615265121, 0.0674860938, 0.0265120133, -0.1282098869, -0.0273147320, -0.1540538055, -0.0218045764, 0.0044133972, 0.0493415810, 0.1221034038, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0052817613, -0.0054644227, -0.1063539412, 0.0488019285, 0.0682981960, 0.0770474911, 0.0191336962, 0.0042666477, 0.0046464010, -0.1050942360, 0.0048834603, -0.0948924708, 0.0478599547, 0.0456426423, -0.0034935861};
const double kReverseTargetsLoss = 1.7329130535, kReverseTargetsSoft = 1.3452672092, kReverseTargetsHard = 2.6374200233;
const std::vector<double> kReverseTargetsGrad = {-0.0287322335, -0.0592302833, 0.0363022657, 0.0367053548, 0.0149548962, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0736022870, 0.0476073327, -0.0187103427, -0.0818519450, -0.0206473321, -0.0839558716, -0.0318442077, -0.0343509712, 0.0012682150, 0.1488828355, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000, -0.0052729703, -0.0106888116, -0.0609751154, 0.0198000738, 0.0571368234, 0.0265634972, 0.0055906504, 0.0014356505, 0.0037705764, -0.0373603743, 0.0018953742, -0.0286596218, 0.0488294318, 0.0172839277, -0.0393491117};

constexpr float kIgnore = static_cast<float>(TokenCrossEntropyLoss::kIgnoreIndex);

// The generator's logits: student 2 sin(0.7 k + 0.3), teacher 3 cos(0.45 k + 0.1).
Tensor Logits(const Shape& shape, bool teacher, DeviceBackend* backend) {
    std::vector<float> v(static_cast<size_t>(shape.numel()));
    for (size_t k = 0; k < v.size(); ++k) {
        const double x = static_cast<double>(k);
        v[k] = static_cast<float>(teacher ? 3 * std::cos(0.45 * x + 0.1) : 2 * std::sin(0.7 * x + 0.3));
    }
    return Tensor(shape, backend, v);
}

Tensor Targets(DeviceBackend* backend) {
    return Tensor(Shape({2, 4}), backend, {1, kIgnore, 3, 0, kIgnore, 2, 4, 1});
}

void ExpectGrad(const Tensor& grad, const std::vector<double>& expected) {
    const std::vector<float> g = grad.to_host_vector();
    ASSERT_EQ(g.size(), expected.size());
    for (size_t k = 0; k < g.size(); ++k) EXPECT_NEAR(g[k], expected[k], 2e-6) << k;
}

class DistillationLossTest : public ::testing::Test {
protected:
    CPUBackend backend;
};

TEST_F(DistillationLossTest, ForwardKlMatchesTheReference) {
    DistillationOptions o;
    o.temperature = 2.0f;
    DistillationLoss loss(&backend, o);
    const Shape shape({4, 5});
    EXPECT_NEAR(loss.forward(Logits(shape, false, &backend), Logits(shape, true, &backend)), kForwardLoss, 1e-5);
    EXPECT_NEAR(loss.soft_loss(), kForwardSoft, 1e-5);
    EXPECT_EQ(loss.hard_loss(), 0.0f);
    EXPECT_EQ(loss.num_rows(), 4);
    ExpectGrad(loss.backward(), kForwardGrad);
}

TEST_F(DistillationLossTest, ReverseKlOverTokensMatchesTheReference) {
    DistillationOptions o;
    o.temperature = 3.0f;
    o.divergence = DistillationDivergence::Reverse;
    DistillationLoss loss(&backend, o);
    const Shape shape({2, 3, 6});
    EXPECT_NEAR(loss.forward(Logits(shape, false, &backend), Logits(shape, true, &backend)), kReverseLoss, 1e-5);
    EXPECT_EQ(loss.num_rows(), 6);
    const Tensor grad = loss.backward();
    EXPECT_EQ(grad.shape(), shape);
    ExpectGrad(grad, kReverseGrad);
}

TEST_F(DistillationLossTest, TargetsMixInTheHardTermAndDropPadding) {
    const Shape shape({2, 4, 5});
    for (const DistillationDivergence divergence : {DistillationDivergence::Forward, DistillationDivergence::Reverse}) {
        const bool forward = divergence == DistillationDivergence::Forward;
        DistillationOptions o;
        o.temperature = forward ? 4.0f : 1.5f;
        o.alpha = forward ? 0.3f : 0.7f;
        o.divergence = divergence;
        DistillationLoss loss(&backend, o);
        const float value = loss.forward(Logits(shape, false, &backend), Logits(shape, true, &backend), Targets(&backend));
        EXPECT_NEAR(value, forward ? kForwardTargetsLoss : kReverseTargetsLoss, 1e-5);
        EXPECT_NEAR(loss.soft_loss(), forward ? kForwardTargetsSoft : kReverseTargetsSoft, 1e-5);
        EXPECT_NEAR(loss.hard_loss(), forward ? kForwardTargetsHard : kReverseTargetsHard, 1e-5);
        EXPECT_EQ(loss.num_rows(), 6);
        ExpectGrad(loss.backward(), forward ? kForwardTargetsGrad : kReverseTargetsGrad);
    }
}

TEST_F(DistillationLossTest, IsZeroWhenTheStudentIsTheTeacher) {
    DistillationLoss loss(&backend);
    const Tensor t = Logits(Shape({3, 7}), true, &backend);
    EXPECT_NEAR(loss.forward(t, t), 0.0f, 1e-6);
    for (float g : loss.backward().to_host_vector()) EXPECT_NEAR(g, 0.0f, 1e-7);
}

TEST_F(DistillationLossTest, TheTemperatureSquaredKeepsTheGradientsScale) {
    // Without T², the soft gradient shrinks like 1/T; with it, it stays the same order as T grows.
    const Shape shape({4, 5});
    std::vector<double> norms;
    for (float t : {1.0f, 4.0f, 16.0f}) {
        DistillationOptions o;
        o.temperature = t;
        DistillationLoss loss(&backend, o);
        (void)loss.forward(Logits(shape, false, &backend), Logits(shape, true, &backend));
        double sq = 0;
        for (float g : loss.backward().to_host_vector()) sq += static_cast<double>(g) * g;
        norms.push_back(std::sqrt(sq));
    }
    // At high T, T (p_s - p_t) -> (s - t - mean) / classes: a limit, not zero.
    EXPECT_GT(norms[2], 0.5 * norms[1]);
    EXPECT_LT(norms[2], 2.0 * norms[1]);
}

TEST_F(DistillationLossTest, AWindowNormalizerAccumulatesToTheWholeBatch) {
    const Shape shape({2, 4, 5});
    DistillationLoss whole(&backend);
    (void)whole.forward(Logits(shape, false, &backend), Logits(shape, true, &backend), Targets(&backend));
    const std::vector<float> expected = whole.backward().to_host_vector();
    const std::vector<float> s = Logits(shape, false, &backend).to_host_vector(), t = Logits(shape, true, &backend).to_host_vector();
    const std::vector<float> y = Targets(&backend).to_host_vector();
    std::vector<float> accumulated;
    float total = 0;
    for (int half = 0; half < 2; ++half) {
        const auto from = static_cast<std::ptrdiff_t>(half * 20);
        const Tensor sh(Shape({1, 4, 5}), &backend, std::vector<float>(s.begin() + from, s.begin() + from + 20));
        const Tensor th(Shape({1, 4, 5}), &backend, std::vector<float>(t.begin() + from, t.begin() + from + 20));
        const Tensor yh(Shape({1, 4}), &backend, std::vector<float>(y.begin() + half * 4, y.begin() + half * 4 + 4));
        DistillationLoss part(&backend);
        total += part.forward(sh, th, yh, 6.0f);
        const std::vector<float> g = part.backward().to_host_vector();
        accumulated.insert(accumulated.end(), g.begin(), g.end());
    }
    EXPECT_NEAR(total, whole.forward(Logits(shape, false, &backend), Logits(shape, true, &backend), Targets(&backend)), 1e-5);
    for (size_t k = 0; k < expected.size(); ++k) EXPECT_NEAR(accumulated[k], expected[k], 1e-6) << k;
}

TEST_F(DistillationLossTest, EveryRowIgnoredGivesZero) {
    DistillationLoss loss(&backend);
    const Shape shape({2, 3});
    const Tensor y(Shape({2}), &backend, {kIgnore, kIgnore});
    EXPECT_EQ(loss.forward(Logits(shape, false, &backend), Logits(shape, true, &backend), y), 0.0f);
    EXPECT_EQ(loss.num_rows(), 0);
    EXPECT_EQ(loss.backward().to_host_vector(), std::vector<float>(6, 0.0f));
}

TEST_F(DistillationLossTest, RejectsBadSettingsAndInputs) {
    DistillationOptions o;
    o.temperature = 0.0f;
    EXPECT_THROW(DistillationLoss(&backend, o), std::invalid_argument);
    o.temperature = 2.0f;
    o.alpha = 1.5f;
    EXPECT_THROW(DistillationLoss(&backend, o), std::invalid_argument);
    DistillationLoss loss(&backend);
    EXPECT_THROW((void)loss.backward(), std::logic_error);
    const Tensor s = Logits(Shape({2, 3}), false, &backend);
    EXPECT_THROW((void)loss.forward(s, Logits(Shape({3, 2}), true, &backend)), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(Logits(Shape({6}), false, &backend), Logits(Shape({6}), true, &backend)), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(s, s, -1.0f), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(s, s, Tensor(Shape({2}), &backend, {0, 3})), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(s, s, Tensor(Shape({3}), &backend, {0, 1, 2})), std::invalid_argument);
}

TEST_F(DistillationLossTest, AStudentLearnsTheTeacherFromItsSoftLabels) {
    LinearModule t1(4, 16, &backend), t2(16, 3, &backend);
    ReluModule relu(&backend);
    SequentialModule teacher({&t1, &relu, &t2});
    std::vector<float> w1, w2, x, ws;
    for (int k = 0; k < 64; ++k) w1.push_back(static_cast<float>(0.9 * std::sin(1.3 * k + 0.2)));
    for (int k = 0; k < 48; ++k) w2.push_back(static_cast<float>(0.8 * std::cos(0.9 * k + 0.4)));
    for (int k = 0; k < 12; ++k) ws.push_back(static_cast<float>(0.05 * std::sin(2.1 * k)));
    for (int k = 0; k < 256; ++k) x.push_back(static_cast<float>(std::sin(0.37 * k + 0.5 * (k % 4))));
    t1.set_weight(w1);
    t2.set_weight(w2);
    const Tensor inputs(Shape({64, 4}), &backend, x);
    const Tensor teacher_logits = teacher.forward(inputs);

    // A smaller student, 4 -> 8 -> 3, from a different start.
    LinearModule s1(4, 8, &backend), s2(8, 3, &backend);
    ReluModule student_relu(&backend);
    SequentialModule student({&s1, &student_relu, &s2});
    std::vector<float> v1, v2;
    for (int k = 0; k < 32; ++k) v1.push_back(static_cast<float>(0.5 * std::sin(2.1 * k + 0.7)));
    for (int k = 0; k < 24; ++k) v2.push_back(static_cast<float>(0.3 * std::cos(1.7 * k)));
    s1.set_weight(v1);
    s2.set_weight(v2);
    AdamOptimizer opt(0.02f, &backend);
    DistillationLoss loss(&backend);
    float first = 0, last = 0;
    for (int step = 0; step < 500; ++step) {
        opt.zero_grad(student);
        last = loss.forward(student.forward(inputs), teacher_logits);
        if (step == 0) first = last;
        (void)student.backward(loss.backward());
        opt.step(student);
    }
    EXPECT_LT(last, 0.2f * first);
    const std::vector<float> a = student.forward(inputs).to_host_vector(), b = teacher_logits.to_host_vector();
    int agree = 0;
    for (size_t r = 0; r < 64; ++r) {
        auto argmax = [&](const std::vector<float>& v) {
            size_t best = 3 * r;
            for (size_t c = 3 * r + 1; c < 3 * r + 3; ++c) best = v[c] > v[best] ? c : best;
            return best;
        };
        agree += argmax(a) == argmax(b);
    }
    EXPECT_GE(agree, 56);
}

}  // namespace
}  // namespace pulsatrix
