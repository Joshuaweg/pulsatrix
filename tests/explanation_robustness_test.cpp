// KS-4: explanation sensitivity against numpy implementations of Quantus's MaxSensitivity,
// AvgSensitivity and LocalLipschitzEstimate on the same samples
// (tools/generate_explanation_robustness_reference_values.py).
#include "pulsatrix/explanation_robustness.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

namespace reference {
const double kMaxSensitivity = 1.5561111685;
const double kAvgSensitivity = 0.8217057759;
const double kLocalLipschitz = 6.3089242130;
const double kAdversarialSensitivity = 0.2656195297;  // at eps 0.0115429912
}  // namespace reference

class ExplanationRobustnessTest : public ::testing::Test {
protected:
    CPUBackend backend;
    LinearModule l1{4, 8, &backend};
    ReluModule relu{&backend};
    LinearModule l2{8, 3, &backend};
    SequentialModule net{{&l1, &relu, &l2}};

    void SetUp() override {
        std::vector<float> w1, b1, w2, b2;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 8; ++j) w1.push_back(static_cast<float>(0.7 * std::sin(1.3 * i + 0.7 * j + 0.2)));
        }
        for (int j = 0; j < 8; ++j) b1.push_back(static_cast<float>(0.1 * std::cos(j)));
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 3; ++k) w2.push_back(static_cast<float>(0.6 * std::cos(0.9 * j + 1.1 * k + 0.4)));
        }
        for (int k = 0; k < 3; ++k) b2.push_back(static_cast<float>(0.05 * k));
        l1.set_weight(w1);
        l1.set_bias(b1);
        l2.set_weight(w2);
        l2.set_bias(b2);
        net.set_training(false);
    }

    Tensor Input() {
        std::vector<float> x;
        for (int i = 0; i < 4; ++i) x.push_back(static_cast<float>(0.5 + 0.4 * std::sin(0.8 * 2 + 1.9 * i + 0.3)));
        return Tensor(Shape({1, 4}), &backend, x);
    }

    PredictFn Predict() {
        return [this](const Tensor& x) { return net.forward(x); };
    }

    // Gradient x input of class 2's logit, the class the network predicts for Input().
    ExplainFn InputTimesGradient() {
        return [this](const Tensor& x) {
            (void)net.forward(x);
            const std::vector<float> g = net.backward(Tensor(Shape({1, 3}), &backend, {0.0f, 0.0f, 1.0f})).to_host_vector();
            std::vector<float> v = x.to_host_vector();
            for (size_t k = 0; k < v.size(); ++k) v[k] *= g[k];
            return Attribution{"input_x_gradient", Tensor(x.shape(), &backend, v), {}};
        };
    }

    static SensitivityOptions Options() {
        SensitivityOptions o;
        o.radius = 0.3;
        o.samples = 12;
        o.seed = 7;
        return o;
    }
};

TEST_F(ExplanationRobustnessTest, MatchesQuantusDefinitionsOnTheSameSamples) {
    const DifferentiableClassifier classifier = ClassifierFrom(net);
    const SensitivityReport r = ExplanationSensitivity(Input(), InputTimesGradient(), Predict(), &classifier, &backend, Options());
    EXPECT_EQ(r.samples, 12);
    EXPECT_EQ(r.kept, 9);  // three samples changed the prediction
    EXPECT_NEAR(r.max_sensitivity, reference::kMaxSensitivity, 1e-5);
    EXPECT_NEAR(r.avg_sensitivity, reference::kAvgSensitivity, 1e-5);
    EXPECT_NEAR(r.local_lipschitz, reference::kLocalLipschitz, 1e-4);
    ASSERT_TRUE(r.adversarial_sensitivity.has_value());
    EXPECT_NEAR(*r.adversarial_sensitivity, reference::kAdversarialSensitivity, 1e-5);
    EXPECT_NEAR(r.adversarial_epsilon, r.radius / 8, 1e-12);  // halved three times before the prediction stayed
    const JsonValue doc = ToJson(r);
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.explanation_sensitivity.v1");
    EXPECT_NE(doc.find("adversarial_sensitivity"), nullptr);
}

TEST_F(ExplanationRobustnessTest, AnExplanationThatIgnoresTheInputIsInsensitive) {
    const ExplainFn constant = [this](const Tensor& x) {
        return Attribution{"constant", Tensor(x.shape(), &backend, {1.0f, 2.0f, 3.0f, 4.0f}), {}};
    };
    const SensitivityReport r = ExplanationSensitivity(Input(), constant, Predict(), nullptr, &backend, Options());
    EXPECT_EQ(r.max_sensitivity, 0.0);
    EXPECT_EQ(r.local_lipschitz, 0.0);
    EXPECT_FALSE(r.adversarial_sensitivity.has_value());  // no classifier given
}

TEST_F(ExplanationRobustnessTest, InputsAreChecked) {
    SensitivityOptions o = Options();
    const Tensor two(Shape({2, 4}), &backend, std::vector<float>(8, 0.5f));
    EXPECT_THROW((void)ExplanationSensitivity(two, InputTimesGradient(), Predict(), nullptr, &backend, o), std::invalid_argument);
    o.radius = 0;
    EXPECT_THROW((void)ExplanationSensitivity(Input(), InputTimesGradient(), Predict(), nullptr, &backend, o), std::invalid_argument);
    o = Options();
    o.samples = 0;
    EXPECT_THROW((void)ExplanationSensitivity(Input(), InputTimesGradient(), Predict(), nullptr, &backend, o), std::invalid_argument);
    o = Options();
    int calls = 0;
    const ExplainFn shrinking = [&](const Tensor&) {
        ++calls;
        return Attribution{"x", Tensor(Shape({1, calls == 1 ? 4 : 3}), &backend), {}};
    };
    EXPECT_THROW((void)ExplanationSensitivity(Input(), shrinking, Predict(), nullptr, &backend, o), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
