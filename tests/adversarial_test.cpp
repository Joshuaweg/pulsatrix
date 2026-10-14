// KS-4: FGSM and PGD against a numpy implementation of torchattacks' updates
// (tools/generate_adversarial_reference_values.py), and their guarantees: the budget, the valid
// range, the seed, and the optimum on a linear model.
#include "pulsatrix/adversarial.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

namespace reference {
    const std::vector<int64_t> kCleanPredicted = {0, 0, 2, 1, 0, 0};
    const std::vector<double> kCleanLoss = {0.6679860813, 1.2161288450, 1.0242964241, 0.9990520729, 1.0346568778, 2.0325562735};

    // FgsmLinf
    const std::vector<float> kFgsmLinf = {
        0.51820811f, 0.72339859f, 0.27268915f, 0.48823381f,
        0.95648292f, 0.65644798f, 0.00701896f, 0.59764537f,
        0.77852001f, 0.35525683f, 0.37972579f, 0.98716787f,
        0.77095196f, 0.00252360f, 0.48604801f, 0.94183955f,
        0.45968670f, 0.29089420f, 0.94017467f, 0.48915597f,
        0.03353363f, 0.56676424f, 0.98795590f, 0.38239155f,
    };
    const std::vector<int64_t> kFgsmLinfPredicted = {0, 0, 1, 1, 0, 0};
    const std::vector<double> kFgsmLinfLoss = {0.7379770335, 1.3164666247, 1.2115267544, 1.1631041828, 1.1103543604, 2.2574147033};

    // FgmL2
    const std::vector<float> kFgmL2 = {
        0.50875151f, 0.62710648f, 0.17712994f, 0.58690173f,
        1.00000000f, 0.65262169f, 0.00000000f, 0.52963581f,
        0.69505710f, 0.32628085f, 0.50118638f, 0.93462475f,
        0.80140621f, 0.00000000f, 0.37110643f, 0.88476725f,
        0.41067831f, 0.41164077f, 0.90728196f, 0.40431165f,
        0.00000000f, 0.56024041f, 1.00000000f, 0.30883175f,
    };
    const std::vector<int64_t> kFgmL2Predicted = {0, 0, 0, 1, 0, 0};
    const std::vector<double> kFgmL2Loss = {0.8534348131, 1.3528238707, 1.3307130417, 1.2575710838, 1.1391849019, 2.2841809437};

    // PgdLinf
    const std::vector<float> kPgdLinf = {
        0.51820811f, 0.72339859f, 0.07268915f, 0.48823381f,
        0.95648292f, 0.65644798f, 0.00701896f, 0.59764537f,
        0.77852001f, 0.35525683f, 0.37972579f, 0.98716787f,
        0.77095196f, 0.00252360f, 0.48604801f, 0.94183955f,
        0.45968670f, 0.29089420f, 0.94017467f, 0.48915597f,
        0.03353363f, 0.56676424f, 0.98795590f, 0.38239155f,
    };
    const std::vector<int64_t> kPgdLinfPredicted = {0, 0, 1, 1, 0, 0};
    const std::vector<double> kPgdLinfLoss = {0.7877933103, 1.3164666247, 1.2115267544, 1.1631041828, 1.1103543604, 2.2574147033};

    // PgdL2
    const std::vector<float> kPgdL2 = {
        0.56142646f, 0.45094264f, 0.03020774f, 0.68446254f,
        1.00000000f, 0.82646631f, 0.00000000f, 0.32666108f,
        0.53552533f, 0.24919769f, 0.61947885f, 1.00000000f,
        0.89919135f, 0.00000000f, 0.17973652f, 0.95731035f,
        0.40241849f, 0.57107610f, 0.97081299f, 0.29492928f,
        0.00000000f, 0.78186730f, 1.00000000f, 0.36712271f,
    };
    const std::vector<int64_t> kPgdL2Predicted = {0, 0, 0, 2, 0, 0};
    const std::vector<double> kPgdL2Loss = {1.0328487781, 1.4763623421, 1.5464112897, 1.4834690742, 1.2347194613, 2.3851291581};

    // PgdTargeted
    const std::vector<float> kPgdTargeted = {
        0.41820811f, 0.62339859f, 0.37268915f, 0.58823381f,
        1.00000000f, 0.35644798f, 0.00000000f, 0.89764537f,
        0.67852001f, 0.45525683f, 0.47972579f, 0.68716787f,
        0.47095196f, 0.00000000f, 0.38604801f, 1.00000000f,
        0.55968670f, 0.00000000f, 0.64017467f, 0.38915597f,
        0.00000000f, 0.66676424f, 1.00000000f, 0.08239155f,
    };
    const std::vector<int64_t> kPgdTargetedPredicted = {0, 2, 0, 1, 0, 0};
    const std::vector<double> kPgdTargetedLoss = {1.1352075061, 0.8517126369, 0.8753516764, 0.8781554080, 1.3671587722, 0.4506425306};
}  // namespace reference

// The generator's network: Linear(4, 8) -> ReLU -> Linear(8, 3), weights from formulas.
class AdversarialTest : public ::testing::Test {
protected:
    CPUBackend backend;
    LinearModule l1{4, 8, &backend};
    ReluModule relu{&backend};
    LinearModule l2{8, 3, &backend};
    SequentialModule net{{&l1, &relu, &l2}};
    std::vector<int64_t> labels = {0, 1, 2, 0, 1, 2};
    std::vector<int64_t> targets = {1, 2, 0, 1, 2, 0};

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

    Tensor Inputs() {
        std::vector<float> x;
        for (int n = 0; n < 6; ++n) {
            for (int i = 0; i < 4; ++i) x.push_back(static_cast<float>(0.5 + 0.4 * std::sin(0.8 * n + 1.9 * i + 0.3)));
        }
        return Tensor(Shape({6, 4}), &backend, x);
    }

    static AttackOptions Deterministic(double epsilon, AttackNorm norm, double step, int64_t steps) {
        AttackOptions o;
        o.epsilon = epsilon;
        o.norm = norm;
        o.step_size = step;
        o.steps = steps;
        o.random_start = false;
        o.clip_min = 0.0;
        o.clip_max = 1.0;
        return o;
    }

    static void ExpectMatches(const AttackResult& r, const std::vector<float>& x, const std::vector<int64_t>& predicted,
                              const std::vector<double>& loss) {
        const std::vector<float> got = r.adversarial.to_host_vector();
        ASSERT_EQ(got.size(), x.size());
        for (size_t k = 0; k < x.size(); ++k) EXPECT_NEAR(got[k], x[k], 2e-6) << k;
        EXPECT_EQ(r.adversarial_predicted, predicted);
        for (size_t i = 0; i < loss.size(); ++i) EXPECT_NEAR(r.adversarial_loss[i], loss[i], 1e-5) << i;
    }
};

TEST_F(AdversarialTest, CleanPredictionsAndLossesMatchNumpy) {
    const AttackResult r = Fgsm(ClassifierFrom(net), Inputs(), labels, &backend, Deterministic(0.1, AttackNorm::Linf, 0, 1));
    EXPECT_EQ(r.clean_predicted, reference::kCleanPredicted);
    for (size_t i = 0; i < labels.size(); ++i) EXPECT_NEAR(r.clean_loss[i], reference::kCleanLoss[i], 1e-5);
    EXPECT_NEAR(r.clean_accuracy, 2.0 / 6.0, 1e-12);  // inputs 0 and 2
}

TEST_F(AdversarialTest, FgsmMatchesNumpy) {
    ExpectMatches(Fgsm(ClassifierFrom(net), Inputs(), labels, &backend, Deterministic(0.1, AttackNorm::Linf, 0, 1)),
                  reference::kFgsmLinf, reference::kFgsmLinfPredicted, reference::kFgsmLinfLoss);
    ExpectMatches(Fgsm(ClassifierFrom(net), Inputs(), labels, &backend, Deterministic(0.3, AttackNorm::L2, 0, 1)),
                  reference::kFgmL2, reference::kFgmL2Predicted, reference::kFgmL2Loss);
}

TEST_F(AdversarialTest, PgdMatchesNumpy) {
    ExpectMatches(Pgd(ClassifierFrom(net), Inputs(), labels, &backend, Deterministic(0.1, AttackNorm::Linf, 0.025, 10)),
                  reference::kPgdLinf, reference::kPgdLinfPredicted, reference::kPgdLinfLoss);
    ExpectMatches(Pgd(ClassifierFrom(net), Inputs(), labels, &backend, Deterministic(0.5, AttackNorm::L2, 0.1, 10)),
                  reference::kPgdL2, reference::kPgdL2Predicted, reference::kPgdL2Loss);
    const AttackResult t =
        Pgd(ClassifierFrom(net), Inputs(), targets, &backend, Deterministic(0.2, AttackNorm::Linf, 0.02, 20), /*targeted=*/true);
    ExpectMatches(t, reference::kPgdTargeted, reference::kPgdTargetedPredicted, reference::kPgdTargetedLoss);
    EXPECT_TRUE(t.targeted);
    for (size_t i = 0; i < targets.size(); ++i) EXPECT_EQ(t.success[i], t.adversarial_predicted[i] == targets[i]);
}

TEST_F(AdversarialTest, AnExplainerContextGivesTheSameAttack) {
    ExplainerContext ctx({&l1, &relu, &l2});
    ExpectMatches(Pgd(ClassifierFrom(ctx), Inputs(), labels, &backend, Deterministic(0.5, AttackNorm::L2, 0.1, 10)),
                  reference::kPgdL2, reference::kPgdL2Predicted, reference::kPgdL2Loss);
}

TEST_F(AdversarialTest, EveryPerturbationStaysInTheBudgetAndTheRange) {
    for (AttackNorm norm : {AttackNorm::Linf, AttackNorm::L2}) {
        for (uint64_t seed : {1u, 2u, 3u}) {
            AttackOptions o = Deterministic(norm == AttackNorm::Linf ? 0.15 : 0.4, norm, 0, 7);
            o.random_start = true;
            o.seed = seed;
            const AttackResult r = Pgd(ClassifierFrom(net), Inputs(), labels, &backend, o);
            const std::vector<float> x0 = Inputs().to_host_vector(), x = r.adversarial.to_host_vector();
            for (size_t i = 0; i < 6; ++i) {
                double linf = 0, l2 = 0;
                for (size_t j = 0; j < 4; ++j) {
                    const double d = static_cast<double>(x[i * 4 + j]) - x0[i * 4 + j];
                    linf = std::max(linf, std::abs(d));
                    l2 += d * d;
                    EXPECT_GE(x[i * 4 + j], 0.0f);
                    EXPECT_LE(x[i * 4 + j], 1.0f);
                }
                EXPECT_LE(norm == AttackNorm::Linf ? linf : std::sqrt(l2), o.epsilon + 1e-6);
                EXPECT_NEAR(r.perturbation[i], norm == AttackNorm::Linf ? linf : std::sqrt(l2), 1e-9);
            }
        }
    }
}

TEST_F(AdversarialTest, TheRandomStartFollowsTheSeed) {
    AttackOptions o = Deterministic(0.2, AttackNorm::Linf, 0, 1);
    o.random_start = true;
    o.seed = 5;
    const std::vector<float> a = Pgd(ClassifierFrom(net), Inputs(), labels, &backend, o).adversarial.to_host_vector();
    const std::vector<float> b = Pgd(ClassifierFrom(net), Inputs(), labels, &backend, o).adversarial.to_host_vector();
    o.seed = 6;
    const std::vector<float> c = Pgd(ClassifierFrom(net), Inputs(), labels, &backend, o).adversarial.to_host_vector();
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

// On a linear two-class model the loss gradient's direction is the same everywhere, u = w_other -
// w_label, so the best attack is known: x + eps sign(u) under L∞, x + eps u / |u| under L2.
TEST_F(AdversarialTest, PgdFindsTheOptimumOfALinearModel) {
    LinearModule linear(3, 2, &backend);
    linear.set_weight({0.5f, -0.25f, -1.0f, 0.75f, 0.2f, 0.3f});  // (in, out): w0 = (0.5, -1, 0.2), w1 = (-0.25, 0.75, 0.3)
    const Tensor x(Shape({2, 3}), &backend, {0.1f, 0.2f, 0.3f, -0.4f, 0.5f, 0.0f});
    const std::vector<int64_t> y = {0, 1};
    const double u[3] = {-0.75, 1.75, 0.1};  // w1 - w0, to push class 0 toward 1; class 1 goes the other way
    const double norm_u = std::sqrt(0.75 * 0.75 + 1.75 * 1.75 + 0.1 * 0.1);
    for (AttackNorm norm : {AttackNorm::Linf, AttackNorm::L2}) {
        AttackOptions o;
        o.epsilon = 0.3;
        o.norm = norm;
        o.steps = 100;
        o.step_size = 0.1;  // under L2 the random start's part off the optimum shrinks by eps / (eps + step) a step
        o.seed = 11;
        const std::vector<float> got = Pgd(ClassifierFrom(linear), x, y, &backend, o).adversarial.to_host_vector();
        for (int i = 0; i < 2; ++i) {
            const double side = i == 0 ? 1.0 : -1.0;
            for (int j = 0; j < 3; ++j) {
                const double step = norm == AttackNorm::Linf ? (u[j] > 0 ? 0.3 : -0.3) : 0.3 * u[j] / norm_u;
                EXPECT_NEAR(got[static_cast<size_t>(i * 3 + j)], x.to_host_vector()[static_cast<size_t>(i * 3 + j)] + side * step, 1e-4)
                    << (norm == AttackNorm::Linf ? "linf " : "l2 ") << i << "," << j;
            }
        }
    }
}

TEST_F(AdversarialTest, RobustAccuracyFallsWithTheBudget) {
    AttackOptions o;
    o.clip_min = 0.0;
    o.clip_max = 1.0;
    o.steps = 20;
    const RobustnessCurve c = RobustAccuracyCurve(ClassifierFrom(net), Inputs(), labels, {0.01, 0.1, 0.4}, &backend, o);
    ASSERT_EQ(c.points.size(), 3u);
    EXPECT_NEAR(c.clean_accuracy, 2.0 / 6.0, 1e-12);
    EXPECT_LE(c.points[0].robust_accuracy, c.clean_accuracy);
    EXPECT_LE(c.points[2].robust_accuracy, c.points[0].robust_accuracy);
    EXPECT_DOUBLE_EQ(c.points[2].robust_accuracy, 0.0);
    EXPECT_DOUBLE_EQ(c.points[2].flip_rate, 1.0);
    EXPECT_GT(c.points[2].mean_perturbation, 0.0);
    EXPECT_LE(c.points[2].mean_perturbation, 0.4 + 1e-6);
    const JsonValue doc = ToJson(c);
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.robustness_curve.v1");
    EXPECT_EQ(doc.find("points")->as_array().size(), 3u);
}

TEST_F(AdversarialTest, TheResultBecomesADocument) {
    const AttackOptions o = Deterministic(0.1, AttackNorm::Linf, 0, 1);
    const AttackResult r = Fgsm(ClassifierFrom(net), Inputs(), labels, &backend, o);
    const JsonValue doc = ToJson(r, o);
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.attack.v1");
    EXPECT_EQ(doc.find("norm")->as_string(), "linf");
    EXPECT_EQ(doc.find("inputs")->as_array().size(), 6u);
}

TEST_F(AdversarialTest, InputsAreChecked) {
    const DifferentiableClassifier m = ClassifierFrom(net);
    AttackOptions o;
    EXPECT_THROW((void)Pgd(m, Inputs(), {0, 1}, &backend, o), std::invalid_argument);
    EXPECT_THROW((void)Pgd(m, Inputs(), {0, 1, 2, 3, 0, 1}, &backend, o), std::invalid_argument);
    o.epsilon = 0;
    EXPECT_THROW((void)Fgsm(m, Inputs(), labels, &backend, o), std::invalid_argument);
    o.epsilon = 0.1;
    o.steps = 0;
    EXPECT_THROW((void)Pgd(m, Inputs(), labels, &backend, o), std::invalid_argument);
    o.steps = 5;
    o.clip_min = 1;
    o.clip_max = 0;
    EXPECT_THROW((void)Pgd(m, Inputs(), labels, &backend, o), std::invalid_argument);
    EXPECT_THROW((void)RobustAccuracyCurve(m, Inputs(), labels, {}, &backend), std::invalid_argument);
    EXPECT_THROW((void)RobustAccuracyCurve(m, Inputs(), labels, {0.1, -1}, &backend), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
