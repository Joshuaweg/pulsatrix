// KD-3: HintLoss and AttentionTransferLoss against
// tools/generate_feature_distillation_reference_values.py (numpy, gradients by central differences),
// and a student trained through a hint at its hidden layer.
#include "pulsatrix/feature_distillation.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/distillation_loss.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

namespace reference {
const double kHintLoss = 0.726511119728;
const std::vector<double> kHintGrad = {0.033706832769, -0.010508726389, 0.020819367175, 0.069293669303, 0.004710406731, 0.005605669207, -0.009900201714, -0.045511072910, -0.032122177918, 0.016543164783, 0.052386858462, 0.018739267249, 0.067287675709, -0.035849552771, -0.016623122545, 0.012748056111, -0.051928536693, 0.033937643695, 0.027311471917, -0.001391901483, 0.033010476019, -0.006890130577, 0.025813183147, -0.017806729480};
const std::vector<double> kHintWeightGrad = {-0.063342651846, 0.011586212267, 0.025453767427, -0.153782471313, -0.174368544603, 0.131773776557, 0.040340236651, 0.004809290799, -0.179435665770, -0.080689897430, 0.264914942427, 0.050121613804, -0.018097077170, -0.120697470296, 0.050938471163, 0.273462468614, 0.036330011055, -0.032492106339, -0.005193363850, 0.158609679224};
const double kAt2Loss = 0.002897749926;
const std::vector<double> kAt2Grad = {-0.000059140988, -0.001120763078, -0.000225745574, -0.000046351960, 0.000594958436, -0.000164389117, -0.000916680275, 0.000099303812, -0.000254166559, 0.000412396427, -0.000028672342, 0.000958853458, 0.000740715704, -0.000356255845, 0.000007411126, -0.000012312230, 0.001136929302, 0.000166954681, 0.000554261949, 0.000642628960, 0.000180049992, -0.000311670938, -0.000226651207, -0.000171877132, -0.000184739263, -0.000107019333, -0.001451077096, -0.000272982801, -0.000074687730, 0.000420398578, -0.000113108832, -0.000266074247, 0.000491568844, -0.000575470527, -0.000018493897, -0.000003808939, 0.001176774108, 0.000704425226, -0.000136256666, 0.000434590922, 0.000296921019, 0.000709160059, -0.000073577569, -0.000180634728, -0.000838931298, 0.000023055812, -0.000999660684, -0.000459814010, 0.000155531444, 0.000142796010, -0.000120702961, -0.001310240029, -0.000091533449, -0.000403729749, -0.000031834785, 0.000007282019, 0.000417157929, 0.000763332795, -0.000734207048, -0.000068693650, 0.000969360095, 0.000019806973, -0.000582487198, 0.000080723272, 0.000048431415, 0.000741332536, -0.000005491367, 0.001075430938, 0.000691507796, -0.000077118939, -0.000195949522, -0.000126552526, 0.001113425883, 0.000058733545, 0.000424955636, 0.000166091463, -0.000001987706, -0.000481017451, -0.001025401420, 0.000499889672, -0.000077015498, 0.000014904598, -0.001100247309, -0.000112087805, 0.000019948612, 0.000854693608, -0.000049191318, -0.000059118853, 0.000529237234, -0.000149349935, 0.000252107392, -0.000053889518, 0.001304021169, 0.000638256789, -0.000013286836, 0.000129259709, -0.000003949904, 0.000573931365, -0.000450781790, 0.000594801503, -0.001071069345, -0.000000123464, -0.000870536168, -0.000228750174, -0.000022086610, 0.000387404397, -0.000059472334, -0.001153505298, 0.000007420538, -0.000120117583, 0.000528890957, 0.000055384225, 0.000608708009, 0.000784169692, -0.000442502668, 0.000004613228, -0.000003228670, 0.001238970379, 0.000430083936, 0.000285625093};
const double kAt3Loss = 0.004191587800;
const std::vector<double> kAt3Grad = {-0.000013166441, -0.002269034955, -0.000543661013, -0.000023536899, 0.000805520913, -0.000615213830, -0.001008610915, 0.000046624610, -0.000630579079, 0.000842069877, -0.000010483042, 0.001388496947, 0.001384352583, -0.000345883562, 0.000013158737, -0.000254125350, 0.001987921055, 0.000077144766, 0.000783566630, 0.001134087017, 0.000122033049, -0.000175471332, -0.000548031815, -0.000323630337, -0.000077664398, -0.000260738066, -0.002527370228, -0.000352333507, -0.000054450423, 0.000875066019, -0.000163137582, -0.000106916845, 0.000609694545, -0.000902510157, -0.000081941271, -0.000024321062, 0.002129699963, 0.001373343154, -0.000047354507, 0.000518665180, 0.000331873926, 0.000908452218, -0.000057753801, -0.000357450188, -0.001601608483, 0.000012101545, -0.001199480287, -0.000999649551, 0.000236123467, 0.000100960267, -0.000185779090, -0.002592639767, -0.000021139916, -0.000444209078, -0.000242800550, 0.000088895152, 0.000267629030, 0.001612638819, -0.001374937323, -0.000012958626, 0.001566827141, -0.000375595690, -0.000556037699, 0.000048791401, -0.000048083681, 0.001208770573, 0.000001869686, 0.001786142446, 0.001261460571, -0.000005699071, -0.000131171250, -0.000311840572, 0.001713307769, 0.000010049718, 0.000734380353, -0.000034512646, 0.000081903388, -0.000397338326, -0.002112072663, 0.000783497424, -0.000009890245, -0.000212679142, -0.001983864919, -0.000094072459, -0.000008157720, 0.001606714335, 0.000150032245, -0.000005397629, 0.000738890871, -0.000021374306, 0.000217130783, -0.000056545700, 0.002350076724, 0.001186785763, -0.000000717922, -0.000020903065, 0.000323422891, 0.000565664224, -0.000408181911, 0.001109260015, -0.001912872356, 0.000000014594, -0.001241952844, -0.000391803790, 0.000010000036, 0.000330101006, 0.000219299506, -0.002054897796, 0.000000145261, -0.000013825954, 0.000955614041, 0.000059725962, 0.000512071718, 0.001791437033, -0.000796279710, -0.000000026625, 0.000216095100, 0.002636092463, 0.000371558730, 0.000255789057};
const std::vector<double> kAtMap = {0.208147808369, 0.174476903008, 0.259875878423, 0.254741119758, 0.171675416157, 0.214555559031, 0.278136369381, 0.206364820338};  // the first 8
}  // namespace reference

class FeatureDistillationTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // The generator's features: float32(sin(a k + b)) of the flat index k.
    Tensor Features(const Shape& shape, double a, double b) {
        std::vector<float> v(static_cast<size_t>(shape.numel()));
        for (size_t k = 0; k < v.size(); ++k) v[k] = static_cast<float>(std::sin(a * static_cast<double>(k) + b));
        return Tensor(shape, &backend, v);
    }

    static void ExpectNear(const Tensor& got, const std::vector<double>& expected, double tolerance) {
        const std::vector<float> g = got.to_host_vector();
        ASSERT_EQ(g.size(), expected.size());
        for (size_t k = 0; k < g.size(); ++k) EXPECT_NEAR(g[k], expected[k], tolerance) << k;
    }
};

TEST_F(FeatureDistillationTest, HintLossMatchesTheReference) {
    HintLoss loss(4, 5, &backend, 3);
    ASSERT_NE(loss.regressor(), nullptr);
    EXPECT_NEAR(loss.forward(Features(Shape({2, 3, 4}), 0.7, 0.3), Features(Shape({2, 3, 5}), 0.45, 0.1)), reference::kHintLoss, 1e-6);
    const Tensor grad = loss.backward();
    EXPECT_EQ(grad.shape(), Shape({2, 3, 4}));
    ExpectNear(grad, reference::kHintGrad, 1e-6);
    ExpectNear(loss.regressor()->weight_grad(), reference::kHintWeightGrad, 1e-6);
}

TEST_F(FeatureDistillationTest, HintLossWithoutARegressorIsTheMeanSquaredError) {
    HintLoss loss(&backend);
    EXPECT_EQ(loss.regressor(), nullptr);
    const Tensor s = Features(Shape({3, 4}), 0.7, 0.3), t = Features(Shape({3, 4}), 0.45, 0.1);
    const std::vector<float> a = s.to_host_vector(), b = t.to_host_vector();
    double mse = 0;
    for (size_t k = 0; k < a.size(); ++k) mse += (a[k] - b[k]) * (a[k] - b[k]) / 12.0;
    EXPECT_NEAR(loss.forward(s, t), mse, 1e-6);
    const std::vector<float> g = loss.backward().to_host_vector();
    for (size_t k = 0; k < a.size(); ++k) EXPECT_NEAR(g[k], 2 * (a[k] - b[k]) / 12.0, 1e-7);
    EXPECT_THROW((void)loss.forward(s, Features(Shape({3, 5}), 0.1, 0.0)), std::invalid_argument);
}

TEST_F(FeatureDistillationTest, AttentionTransferMatchesTheAuthorsCode) {
    const Tensor s = Features(Shape({2, 3, 4, 5}), 0.9, 0.2), t = Features(Shape({2, 6, 4, 5}), 0.35, 0.6);
    AttentionTransferLoss squared(&backend);
    EXPECT_NEAR(squared.forward(s, t), reference::kAt2Loss, 1e-9);
    ExpectNear(squared.backward(), reference::kAt2Grad, 1e-9);
    AttentionTransferLoss cubed(&backend, 3.0);
    EXPECT_NEAR(cubed.forward(s, t), reference::kAt3Loss, 1e-9);
    ExpectNear(cubed.backward(), reference::kAt3Grad, 1e-9);
    const std::vector<float> map = AttentionMap(s, &backend).to_host_vector();
    ASSERT_EQ(map.size(), 40u);
    for (size_t k = 0; k < 8; ++k) EXPECT_NEAR(map[k], reference::kAtMap[k], 1e-7) << k;
}

TEST_F(FeatureDistillationTest, AttentionTransferIgnoresScaleAndChannelCount) {
    const Tensor s = Features(Shape({2, 3, 4, 5}), 0.9, 0.2);
    // The same per-channel maps repeated over twice the channels, at three times the size.
    const std::vector<float> v = s.to_host_vector();
    std::vector<float> wide;
    for (int i = 0; i < 2; ++i) {
        for (int copy = 0; copy < 2; ++copy) {
            for (int k = 0; k < 60; ++k) wide.push_back(3.0f * v[static_cast<size_t>(i * 60 + k)]);
        }
    }
    AttentionTransferLoss loss(&backend);
    EXPECT_NEAR(loss.forward(s, Tensor(Shape({2, 6, 4, 5}), &backend, wide)), 0.0, 1e-12);
    for (float g : loss.backward().to_host_vector()) EXPECT_NEAR(g, 0.0f, 1e-9);
}

TEST_F(FeatureDistillationTest, RejectsBadInputs) {
    AttentionTransferLoss loss(&backend);
    EXPECT_THROW((void)loss.backward(), std::logic_error);
    EXPECT_THROW((void)loss.forward(Features(Shape({2, 3}), 0.1, 0), Features(Shape({2, 3}), 0.2, 0)), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(Features(Shape({2, 3, 4}), 0.1, 0), Features(Shape({3, 3, 4}), 0.2, 0)), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(Features(Shape({2, 3, 4}), 0.1, 0), Features(Shape({2, 3, 5}), 0.2, 0)), std::invalid_argument);
    EXPECT_THROW(AttentionTransferLoss(&backend, 0.0), std::invalid_argument);
    EXPECT_THROW((void)AttentionMap(Features(Shape({2, 3}), 0.1, 0), &backend), std::invalid_argument);
    HintLoss hint(4, 5, &backend);
    EXPECT_THROW((void)hint.backward(), std::logic_error);
    EXPECT_THROW((void)hint.forward(Features(Shape({3, 5}), 0.1, 0), Features(Shape({3, 5}), 0.2, 0)), std::invalid_argument);
    EXPECT_THROW((void)hint.forward(Features(Shape({3, 4}), 0.1, 0), Features(Shape({2, 5}), 0.2, 0)), std::invalid_argument);
    EXPECT_THROW(HintLoss(0, 5, &backend), std::invalid_argument);
}

TEST_F(FeatureDistillationTest, AHintAtTheHiddenLayerTrainsTheStudentsFirstHalf) {
    // Teacher 4 -> 16 -> 3; student 4 -> 8 -> 3, run as two parts so the hint's gradient can join
    // the output loss's at the hidden layer.
    LinearModule t1(4, 16, &backend), t2(16, 3, &backend), s1(4, 8, &backend), s2(8, 3, &backend);
    ReluModule t_relu(&backend), s_relu(&backend);
    SequentialModule teacher_front({&t1, &t_relu}), student_front({&s1, &s_relu});
    std::vector<float> w1, w2, v1, v2, x;
    for (int k = 0; k < 64; ++k) w1.push_back(static_cast<float>(0.9 * std::sin(1.3 * k + 0.2)));
    for (int k = 0; k < 48; ++k) w2.push_back(static_cast<float>(0.8 * std::cos(0.9 * k + 0.4)));
    for (int k = 0; k < 32; ++k) v1.push_back(static_cast<float>(0.5 * std::sin(2.1 * k + 0.7)));
    for (int k = 0; k < 24; ++k) v2.push_back(static_cast<float>(0.3 * std::cos(1.7 * k)));
    for (int k = 0; k < 256; ++k) x.push_back(static_cast<float>(std::sin(0.37 * k + 0.5 * (k % 4))));
    t1.set_weight(w1);
    t2.set_weight(w2);
    s1.set_weight(v1);
    s2.set_weight(v2);
    const Tensor inputs(Shape({64, 4}), &backend, x);
    const Tensor teacher_hidden = teacher_front.forward(inputs);
    const Tensor teacher_logits = t2.forward(teacher_hidden);

    HintLoss hint(8, 16, &backend, 1);
    DistillationLoss distill(&backend);
    AdamOptimizer opt(0.02f, &backend), regressor_opt(0.02f, &backend);
    float first_hint = 0, last_hint = 0, first_kd = 0, last_kd = 0;
    for (int step = 0; step < 400; ++step) {
        opt.zero_grad(student_front);
        opt.zero_grad(s2);
        regressor_opt.zero_grad(*hint.regressor());
        const Tensor hidden = student_front.forward(inputs);
        const float kd = distill.forward(s2.forward(hidden), teacher_logits);
        const float h = hint.forward(hidden, teacher_hidden);
        if (step == 0) first_kd = kd, first_hint = h;
        last_kd = kd, last_hint = h;
        // The output loss's gradient at the hidden layer, plus the hint's.
        std::vector<float> g = s2.backward(distill.backward()).to_host_vector();
        const std::vector<float> gh = hint.backward().to_host_vector();
        for (size_t k = 0; k < g.size(); ++k) g[k] += gh[k];
        (void)student_front.backward(Tensor(hidden.shape(), &backend, g));
        opt.step(student_front);
        opt.step(s2);
        regressor_opt.step(*hint.regressor());
    }
    EXPECT_LT(last_hint, 0.5f * first_hint);
    EXPECT_LT(last_kd, 0.05f * first_kd);
}

}  // namespace
}  // namespace pulsatrix
