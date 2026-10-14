// KD-2: CompareExplanations against tools/generate_explanation_agreement_reference_values.py
// (scipy, numpy, the same shuffles), and two students with the teacher's outputs, one of which uses
// different evidence.
#include "pulsatrix/explanation_agreement.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {
namespace {

namespace reference {
const std::vector<float> kTeacher = {0.198669329f, 0.891207337f, 0.909297407f, 0.239249334f, -0.611857891f, -0.999923229f, -0.631266654f, 0.215119988f, 0.898708105f, 0.90217185f, 0.222889915f, -0.625070632f, -0.999990225f, -0.618137121f, 0.23150982f, 0.905954719f, 0.894791186f, 0.206467479f, -0.638106704f, -0.999774456f, -0.604832828f, 0.247834206f, 0.912945271f, 0.8871575f, 0.189986676f, -0.650962293f, -0.999275982f, -0.591357529f, 0.264088511f, 0.919677615f, 0.879273057f, 0.173452154f, -0.663633883f, -0.998495042f, -0.577715039f, 0.280268162f, 0.926150024f, 0.871140003f, 0.156868592f, -0.676117837f, -0.997431755f, -0.563909233f, 0.296368569f, 0.93236053f, 0.862760663f, 0.140240684f, -0.68841064f, -0.996086478f, -0.549943984f, 0.312385201f, 0.938307464f, 0.854137361f, 0.123573124f, -0.700508773f, -0.994459629f, -0.535823226f, 0.3283135f, 0.943989098f, 0.845272601f, 0.106870621f, -0.7124089f, -0.992551565f, -0.521551013f, 0.344148964f, 0.949403882f, 0.836168826f, 0.0901379138f, -0.724107563f, -0.990362883f, -0.507131338f, 0.359887153f, 0.954550207f};
const std::vector<float> kStudent = {0.699999988f, 0.600000024f, 0.200000003f, 0.400000006f, -0.100000001f, -1.10000002f, -0.899999976f, 0.600000024f, 1.0f, 0.300000012f, 0.0f, -0.0f, -0.800000012f, -1.0f, 0.300000012f, 1.20000005f, 0.5f, -0.200000003f, -0.200000003f, -0.5f, -0.899999976f, -0.0f, 1.20000005f, 0.800000012f, -0.300000012f, -0.5f, -0.300000012f, -0.600000024f, -0.200000003f, 1.0f, 1.10000002f, -0.200000003f, -0.800000012f, -0.300000012f, -0.300000012f, -0.300000012f, 0.699999988f, 1.20000005f, 0.0f, -1.0f, -0.600000024f, -0.0f, -0.100000001f, 0.400000006f, 1.10000002f, 0.300000012f, -1.0f, -0.899999976f, 0.100000001f, 0.200000003f, 0.300000012f, 0.800000012f, 0.600000024f, -0.800000012f, -1.20000005f, -0.100000001f, 0.600000024f, 0.300000012f, 0.5f, 0.600000024f, -0.5f, -1.29999995f, -0.300000012f, 0.800000012f, 0.600000024f, 0.200000003f, 0.400000006f, -0.200000003f, -1.20000005f, -0.600000024f, 0.800000012f, 0.899999976f};
const std::vector<double> kSpearman = {0.256141927625, 0.514097255099, 0.559390088764, 0.459651952314, 0.205325493516, 0.397193569897};
const std::vector<double> kTopK = {0.333333333333, 0.333333333333, 0.333333333333, 0.000000000000, 0.333333333333, 1.000000000000};
const std::vector<double> kCosine = {0.790099526396, 0.915654608488, 0.885058382367, 0.903192265145, 0.807971463777, 0.879707229048};
const double kCka = 0.909979124499;
const double kMismatched = -0.066560684476;
const double kPValue = 0.010000000000;  // 2 of 200
}  // namespace reference

class ExplanationAgreementTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Row i of a (rows x 12) list as one attribution each.
    std::vector<Attribution> Maps(const std::vector<float>& values, int64_t features = 12) {
        std::vector<Attribution> out;
        for (size_t i = 0; i < values.size() / static_cast<size_t>(features); ++i) {
            const auto from = values.begin() + static_cast<std::ptrdiff_t>(i * static_cast<size_t>(features));
            out.push_back(Attribution{"test", Tensor(Shape({1, features}), &backend, std::vector<float>(from, from + features)), {}});
        }
        return out;
    }
};

TEST_F(ExplanationAgreementTest, MatchesScipyAndNumpy) {
    AgreementOptions o;
    o.top_fraction = 0.25;
    o.permutations = 199;
    o.seed = 5;
    const AgreementReport r = CompareExplanations(Maps(reference::kTeacher), Maps(reference::kStudent), o);
    EXPECT_EQ(r.inputs, 6);
    EXPECT_EQ(r.features, 12);
    EXPECT_EQ(r.top_k, 3);
    ASSERT_EQ(r.per_input.size(), 6u);
    double spearman = 0, top = 0, cosine = 0;
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(r.per_input[i].spearman, reference::kSpearman[i], 1e-9) << i;
        EXPECT_NEAR(r.per_input[i].top_k_overlap, reference::kTopK[i], 1e-12) << i;
        EXPECT_NEAR(r.per_input[i].cosine, reference::kCosine[i], 1e-9) << i;
        spearman += reference::kSpearman[i] / 6, top += reference::kTopK[i] / 6, cosine += reference::kCosine[i] / 6;
    }
    EXPECT_NEAR(r.spearman, spearman, 1e-9);
    EXPECT_NEAR(r.top_k_overlap, top, 1e-12);
    EXPECT_NEAR(r.cosine, cosine, 1e-9);
    EXPECT_NEAR(r.cka, reference::kCka, 1e-6);
    EXPECT_NEAR(r.mismatched_spearman, reference::kMismatched, 1e-9);
    EXPECT_EQ(r.permutations, 199);
    EXPECT_NEAR(r.p_value, reference::kPValue, 1e-12);
}

TEST_F(ExplanationAgreementTest, MapsThatLookAlikeForEveryInputAreNotEvidence) {
    // Every map is the same bump plus a little noise of its own: teacher and student agree on each
    // input, but no more than a student's map of one input agrees with the teacher's of another.
    std::vector<float> t, s;
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 12; ++j) {
            const double bump = std::exp(-0.5 * (j - 6) * (j - 6) / 4.0);
            t.push_back(static_cast<float>(bump + 0.05 * std::sin(3.1 * (12 * i + j))));
            s.push_back(static_cast<float>(bump + 0.05 * std::cos(2.3 * (12 * i + j) + 1)));
        }
    }
    const AgreementReport r = CompareExplanations(Maps(t), Maps(s));
    EXPECT_GT(r.spearman, 0.8);
    EXPECT_GT(r.mismatched_spearman, 0.8);
    EXPECT_GT(r.p_value, 0.05);
}

TEST_F(ExplanationAgreementTest, SameOutputsDifferentEvidence) {
    // Features 4..7 copy features 0..3, so a student can compute the teacher's function from either
    // copy. One student uses the teacher's copy; the other the duplicate.
    LinearModule teacher(8, 3, &backend), same(8, 3, &backend), other(8, 3, &backend);
    std::vector<float> wt(24), ws(24), wo(24);
    for (int c = 0; c < 3; ++c) {
        for (int j = 0; j < 4; ++j) {
            const auto w = static_cast<float>(std::sin(1.9 * (4 * c + j) + 0.4));
            wt[static_cast<size_t>(3 * j + c)] = w;
            ws[static_cast<size_t>(3 * j + c)] = w * 1.01f;
            wo[static_cast<size_t>(3 * (j + 4) + c)] = w;
        }
    }
    teacher.set_weight(wt);
    same.set_weight(ws);
    other.set_weight(wo);
    std::vector<Tensor> inputs;
    for (int i = 0; i < 12; ++i) {
        std::vector<float> x(8);
        for (int j = 0; j < 4; ++j) x[static_cast<size_t>(j)] = x[static_cast<size_t>(j + 4)] = static_cast<float>(std::sin(0.83 * (4 * i + j) + 0.1));
        inputs.emplace_back(Shape({1, 8}), &backend, x);
    }
    // Gradient x input of the teacher's predicted class, for whichever model is explained.
    auto explainer = [&](LinearModule& m) -> ExplainFn {
        return [&](const Tensor& x) {
            const std::vector<float> y = teacher.forward(x).to_host_vector();
            std::vector<float> seed(3, 0.0f);
            seed[static_cast<size_t>(std::max_element(y.begin(), y.end()) - y.begin())] = 1.0f;
            (void)m.forward(x);
            const std::vector<float> g = m.backward(Tensor(Shape({1, 3}), &backend, seed)).to_host_vector();
            std::vector<float> v = x.to_host_vector();
            for (size_t k = 0; k < v.size(); ++k) v[k] *= g[k];
            return Attribution{"input_x_gradient", Tensor(x.shape(), &backend, v), {}};
        };
    };
    for (const Tensor& x : inputs) {
        const std::vector<float> a = teacher.forward(x).to_host_vector(), b = other.forward(x).to_host_vector();
        for (size_t c = 0; c < 3; ++c) EXPECT_NEAR(a[c], b[c], 1e-6);
    }
    AgreementOptions o;
    o.top_fraction = 0.25;
    const AgreementReport agree = ExplanationAgreement(inputs, explainer(teacher), explainer(same), o);
    EXPECT_GT(agree.spearman, 0.99);
    EXPECT_EQ(agree.top_k_overlap, 1.0);
    EXPECT_GT(agree.cka, 0.99);
    EXPECT_LT(agree.p_value, 0.01);
    const AgreementReport differ = ExplanationAgreement(inputs, explainer(teacher), explainer(other), o);
    EXPECT_LT(differ.spearman, 0.0);
    EXPECT_EQ(differ.top_k_overlap, 0.0);
    EXPECT_EQ(differ.cosine, 0.0);
}

TEST_F(ExplanationAgreementTest, SignedValuesSeeAFlippedSign) {
    const std::vector<float> t = reference::kTeacher;
    std::vector<float> s(t.size());
    for (size_t k = 0; k < t.size(); ++k) s[k] = -t[k];
    EXPECT_NEAR(CompareExplanations(Maps(t), Maps(s)).spearman, 1.0, 1e-12);
    AgreementOptions o;
    o.magnitudes = false;
    const AgreementReport r = CompareExplanations(Maps(t), Maps(s), o);
    EXPECT_NEAR(r.spearman, -1.0, 1e-12);
    EXPECT_NEAR(r.cosine, -1.0, 1e-6);
}

TEST_F(ExplanationAgreementTest, TooFewInputsLeaveCkaAndTheTestEmpty) {
    const std::vector<float> three(reference::kTeacher.begin(), reference::kTeacher.begin() + 36);
    const AgreementReport r = CompareExplanations(Maps(three), Maps(three));
    EXPECT_TRUE(std::isnan(r.cka));  // the debiased estimator needs 4
    AgreementOptions o;
    o.estimator = CkaEstimator::Biased;
    EXPECT_NEAR(CompareExplanations(Maps(three), Maps(three), o).cka, 1.0, 1e-9);

    const std::vector<float> one(reference::kTeacher.begin(), reference::kTeacher.begin() + 12);
    const AgreementReport single = CompareExplanations(Maps(one), Maps(one));
    EXPECT_TRUE(std::isnan(single.mismatched_spearman));
    EXPECT_TRUE(std::isnan(single.p_value));
    const std::string json = WriteJsonCompact(ToJson(single));
    EXPECT_NE(json.find("\"p_value\":null"), std::string::npos) << json;
    EXPECT_NE(json.find("pulsatrix.explanation_agreement.v1"), std::string::npos);
}

TEST_F(ExplanationAgreementTest, ConstantMapsCorrelateZero) {
    const std::vector<float> varied(reference::kTeacher.begin(), reference::kTeacher.begin() + 24);
    const AgreementReport r = CompareExplanations(Maps(std::vector<float>(24, 0.5f)), Maps(varied));
    EXPECT_EQ(r.spearman, 0.0);
    EXPECT_EQ(r.mismatched_spearman, 0.0);
}

TEST_F(ExplanationAgreementTest, RejectsBadInputs) {
    const std::vector<Attribution> t = Maps(reference::kTeacher);
    EXPECT_THROW((void)CompareExplanations({}, {}), std::invalid_argument);
    EXPECT_THROW((void)CompareExplanations(t, std::vector<Attribution>(t.begin(), t.begin() + 5)), std::invalid_argument);
    std::vector<Attribution> wrong = t;
    wrong[3] = Attribution{"test", Tensor(Shape({1, 11}), &backend, std::vector<float>(11, 1.0f)), {}};
    EXPECT_THROW((void)CompareExplanations(t, wrong), std::invalid_argument);
    EXPECT_THROW((void)CompareExplanations(Maps({1.0f}, 1), Maps({1.0f}, 1)), std::invalid_argument);
    AgreementOptions o;
    o.top_fraction = 0.0;
    EXPECT_THROW((void)CompareExplanations(t, t, o), std::invalid_argument);
    o.top_fraction = 0.5;
    o.permutations = -1;
    EXPECT_THROW((void)CompareExplanations(t, t, o), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
