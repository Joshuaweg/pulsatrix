// KS-3: Monte Carlo dropout, deep ensembles and the uncertainty decomposition. The decomposition,
// the Gaussian mixture and the Gaussian NLL are checked against numpy
// (tools/generate_uncertainty_reference_values.py); the sampling on networks whose answer is known.

#include "pulsatrix/uncertainty.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/dropout_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lora.hpp"
#include "pulsatrix/sequential_module.hpp"
#include "pulsatrix/softmax_module.hpp"

namespace pulsatrix {
namespace {

// ---- Reference values: tools/generate_uncertainty_reference_values.py output, verbatim ----
const std::vector<double> kMean = {0.6651682674884796, 0.24178613424301149, 0.09304560571908951, 0.33783624656498434, 0.31406692415475845, 0.3480968207120895, 0.21197389513254167, 0.18804856725037097, 0.5999775350093841, 0.3036216601729393, 0.415064936876297, 0.2813134118914604, 0.1305170115083456, 0.7812543690204621, 0.0882286123931408, 0.29452995769679546, 0.4099125262349844, 0.2955575123429298};
const std::vector<double> kTotal = {0.8354156118944532, 1.0976919509710752, 0.9495793032551558, 1.0836696415657519, 0.6728251720849883, 1.0858432667273532};
const std::vector<double> kAleatoric = {0.6482287540594833, 0.607032328610421, 0.6851374259029138, 0.6585112764128696, 0.5625924140174176, 0.7693821417745315};
const std::vector<double> kEpistemic = {0.1871868578349699, 0.4906596223606542, 0.264441877352242, 0.4251583651528823, 0.11023275806757071, 0.31646112495282175};
const std::vector<int64_t> kPredicted = {0, 2, 2, 1, 1, 1};
const std::vector<double> kAgreement = {0.8, 0.2, 0.6, 0.4, 1.0, 0.6};

const std::vector<double> kMixtureMean = {0.463659427156978, 0.8902280978510635, 0.9168850525550631, 0.5255263373192237, -0.11749763506152112};
const std::vector<double> kMixtureVariance = {0.19672795166847218, 0.1556836029789076, 0.1526186989834305, 0.19621394787707214, 0.16299480204140074};
const std::vector<double> kMixtureAleatoric = {0.17364860524616807, 0.14168363331143102, 0.13327305413185997, 0.17223809790481254, 0.14821831768031332};
const std::vector<double> kMixtureEpistemic = {0.023079346422304067, 0.013999969667476617, 0.019345644851570507, 0.023975849972259492, 0.014776484361087406};

constexpr double kNll = -0.3568997667327969;
// ---- end of reference values ----

// An identity layer that records whether it ran in training mode.
class ModeProbe : public Module {
public:
    std::vector<bool> seen;
    Tensor backward(const Tensor& g) override { return g; }
    Tensor propagate_relevance(const Tensor& r, const LRPRuleConfig&) override { return r; }
    OpType op_type() const override { return OpType::Elementwise; }

protected:
    Tensor forward_impl(const Tensor& x) override {
        seen.push_back(is_training());
        return x;
    }
};

class UncertaintyTest : public ::testing::Test {
protected:
    CPUBackend backend;

    // Uneven weights: a linear layer starts at zero.
    static std::vector<float> Pattern(int64_t n, float scale) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = scale * std::sin(0.73f * static_cast<float>(i) + 0.2f);
        return v;
    }

    // The generator's samples (its docstring gives the formula).
    std::vector<Tensor> Samples() {
        std::vector<Tensor> out;
        for (int t = 0; t < 5; ++t) {
            std::vector<float> p;
            for (int i = 0; i < 6; ++i) {
                double l[3], peak = -1e300, sum = 0;
                for (int k = 0; k < 3; ++k) {
                    l[k] = 1.8 * std::sin(0.4 + 0.9 * i + 1.7 * k + 0.6 * t * (1 + i % 2) + 0.3 * i * k);
                    peak = std::max(peak, l[k]);
                }
                for (double v : l) sum += std::exp(v - peak);
                for (double v : l) p.push_back(static_cast<float>(std::exp(v - peak) / sum));
            }
            out.emplace_back(Shape({6, 3}), &backend, p);
        }
        return out;
    }
};

TEST_F(UncertaintyTest, DecompositionMatchesItsDefinition) {
    const UncertaintyDecomposition d = DecomposeUncertainty(Samples());
    EXPECT_EQ(d.samples, 5);
    const std::vector<float> mean = d.mean.to_host_vector();
    for (size_t j = 0; j < kMean.size(); ++j) EXPECT_NEAR(mean[j], kMean[j], 1e-7) << j;
    EXPECT_EQ(d.predicted, kPredicted);
    for (size_t i = 0; i < kTotal.size(); ++i) {
        EXPECT_NEAR(d.total[i], kTotal[i], 1e-12) << i;
        EXPECT_NEAR(d.aleatoric[i], kAleatoric[i], 1e-12) << i;
        EXPECT_NEAR(d.epistemic[i], kEpistemic[i], 1e-12) << i;
        EXPECT_DOUBLE_EQ(d.agreement[i], kAgreement[i]) << i;
    }
    const JsonValue doc = ToJson(d);
    EXPECT_EQ(doc.find("schema")->as_string(), "pulsatrix.uncertainty.v1");
    EXPECT_EQ(doc.find("epistemic")->as_array().size(), 6u);
}

TEST_F(UncertaintyTest, AgreementIsAleatoricAndDisagreementEpistemic) {
    // Samples that agree on a 50/50 prediction: all of it is aleatoric.
    const Tensor even(Shape({1, 2}), &backend, {0.5f, 0.5f});
    const UncertaintyDecomposition same = DecomposeUncertainty({even, even, even});
    EXPECT_NEAR(same.total[0], std::log(2.0), 1e-12);
    EXPECT_NEAR(same.aleatoric[0], std::log(2.0), 1e-12);
    EXPECT_DOUBLE_EQ(same.epistemic[0], 0.0);
    // Confident samples that disagree: the same total, all epistemic.
    const Tensor a(Shape({1, 2}), &backend, {1.0f, 0.0f}), b(Shape({1, 2}), &backend, {0.0f, 1.0f});
    const UncertaintyDecomposition split = DecomposeUncertainty({a, b});
    EXPECT_NEAR(split.total[0], std::log(2.0), 1e-12);
    EXPECT_DOUBLE_EQ(split.aleatoric[0], 0.0);
    EXPECT_NEAR(split.epistemic[0], std::log(2.0), 1e-12);
    EXPECT_DOUBLE_EQ(split.agreement[0], 0.5);  // the tie goes to class 0, which one sample predicts
}

TEST_F(UncertaintyTest, DecompositionInputsAreChecked) {
    const Tensor p(Shape({1, 2}), &backend, {0.5f, 0.5f});
    EXPECT_THROW((void)DecomposeUncertainty({p}), std::invalid_argument);
    EXPECT_THROW((void)DecomposeUncertainty({p, Tensor(Shape({1, 3}), &backend, {0.2f, 0.3f, 0.5f})}), std::invalid_argument);
    EXPECT_THROW((void)DecomposeUncertainty({Tensor(Shape({2}), &backend, {0.5f, 0.5f}), Tensor(Shape({2}), &backend, {0.5f, 0.5f})}),
                 std::invalid_argument);
    EXPECT_THROW((void)DecomposeUncertainty({p, Tensor(Shape({1, 2}), &backend, {-0.5f, 1.5f})}), std::invalid_argument);
}

TEST_F(UncertaintyTest, GaussianMixtureMatchesLakshminarayanan) {
    std::vector<std::vector<double>> mu(4), var(4);
    for (int m = 0; m < 4; ++m) {
        for (int j = 0; j < 5; ++j) {
            mu[m].push_back(std::sin(0.5 + 0.7 * j) + 0.2 * std::cos(1.3 * m + j));
            var[m].push_back(0.1 + 0.05 * (1 + std::sin(m + 2.0 * j)));
        }
    }
    const GaussianMixturePrediction g = CombineGaussians(mu, var);
    for (size_t i = 0; i < kMixtureMean.size(); ++i) {
        EXPECT_NEAR(g.mean[i], kMixtureMean[i], 1e-15) << i;
        EXPECT_NEAR(g.variance[i], kMixtureVariance[i], 1e-14) << i;
        EXPECT_NEAR(g.aleatoric[i], kMixtureAleatoric[i], 1e-15) << i;
        EXPECT_NEAR(g.epistemic[i], kMixtureEpistemic[i], 1e-15) << i;
    }
    // Means alone: the variance is the members' disagreement.
    const GaussianMixturePrediction plain = CombineGaussians({{1.0, 5.0}, {3.0, 5.0}});
    EXPECT_DOUBLE_EQ(plain.variance[0], 1.0);
    EXPECT_DOUBLE_EQ(plain.variance[1], 0.0);
    EXPECT_DOUBLE_EQ(plain.aleatoric[0], 0.0);
    EXPECT_THROW((void)CombineGaussians({{1.0}}), std::invalid_argument);
    EXPECT_THROW((void)CombineGaussians({{1.0}, {1.0, 2.0}}), std::invalid_argument);
    EXPECT_THROW((void)CombineGaussians({{1.0}, {2.0}}, {{0.1}, {-0.1}}), std::invalid_argument);
    EXPECT_THROW((void)CombineGaussians({{1.0}, {2.0}}, {{0.1}}), std::invalid_argument);
}

TEST_F(UncertaintyTest, GaussianNllMatchesPyTorchsDefinitionAndItsGradient) {
    const std::vector<float> x = {0.3f, -1.2f, 1.5f, 0.4f, -0.7f, 2.5f, 0.0f, -4.0f};
    const Tensor prediction(Shape({4, 2}), &backend, x);
    const Tensor target(Shape({4, 1}), &backend, {0.5f, 1.0f, -2.0f, 0.1f});
    GaussianNLLLoss loss(&backend);
    EXPECT_NEAR(loss.forward(prediction, target), kNll, 1e-6);
    const std::vector<float> grad = loss.backward().to_host_vector();
    // Central differences in double, through the same formula.
    for (size_t j = 0; j < x.size(); ++j) {
        const auto at = [&](double delta) {
            double total = 0;
            for (int i = 0; i < 4; ++i) {
                double mu = x[2 * i], s = x[2 * i + 1];
                if (j == static_cast<size_t>(2 * i)) mu += delta;
                if (j == static_cast<size_t>(2 * i + 1)) s += delta;
                const double y = target.to_host_vector()[i], v = loss.variance(s);
                total += 0.5 * (std::log(v) + (y - mu) * (y - mu) / v);
            }
            return total / 4;
        };
        EXPECT_NEAR(grad[j], (at(1e-6) - at(-1e-6)) / 2e-6, 1e-5) << j;
    }
    EXPECT_NEAR(loss.variance(-40.0), 1e-6, 1e-12);  // softplus doesn't underflow to a zero variance
    EXPECT_NEAR(loss.variance(40.0), 40.0 + 1e-6, 1e-9);  // or overflow
    EXPECT_THROW((void)loss.forward(Tensor(Shape({4, 3}), &backend), target), std::invalid_argument);
    EXPECT_THROW((void)loss.forward(prediction, Tensor(Shape({3}), &backend)), std::invalid_argument);
    EXPECT_THROW(GaussianNLLLoss(&backend, 0.0), std::invalid_argument);
    GaussianNLLLoss fresh(&backend);
    EXPECT_THROW((void)fresh.backward(), std::logic_error);
}

// A linear layer after dropout: inverted dropout keeps the expected input, so the mean of many
// samples approaches the eval-mode output.
TEST_F(UncertaintyTest, McDropoutAveragesToTheDeterministicNetwork) {
    LinearModule l0(4, 8, &backend), l1(8, 3, &backend);
    DropoutModule dropout(0.3f, &backend, 7);
    l0.set_weight(Pattern(32, 0.6f));
    l1.set_weight(Pattern(24, 0.5f));
    ModeProbe probe;
    SequentialModule model({&l0, &probe, &dropout, &l1});
    model.set_training(false);
    const Tensor x(Shape({2, 4}), &backend, {0.5f, -1.0f, 1.5f, 0.2f, -0.3f, 0.8f, 0.1f, -1.4f});
    const std::vector<float> eval = model.forward(x).to_host_vector();
    probe.seen.clear();
    const auto predict = [&](const Tensor& in) { return model.forward(in); };
    const std::vector<Tensor> samples = McDropoutSamples(model, predict, x, 4000);
    ASSERT_EQ(samples.size(), 4000u);
    EXPECT_NE(samples[0].to_host_vector(), samples[1].to_host_vector());
    std::vector<double> mean(eval.size(), 0.0);
    for (const Tensor& s : samples) {
        const std::vector<float> v = s.to_host_vector();
        for (size_t j = 0; j < v.size(); ++j) mean[j] += v[j] / 4000.0;
    }
    for (size_t j = 0; j < eval.size(); ++j) EXPECT_NEAR(mean[j], eval[j], 0.03) << j;
    // Only dropout ran in training mode, and every mode is back as it was.
    for (bool training : probe.seen) EXPECT_FALSE(training);
    EXPECT_FALSE(model.is_training());
    EXPECT_FALSE(dropout.is_training());
    model.set_training(true);
    (void)McDropoutSamples(model, predict, x, 2);
    EXPECT_TRUE(l0.is_training());
    EXPECT_TRUE(probe.is_training());
}

TEST_F(UncertaintyTest, McDropoutSamplesLoraAdaptersWithDropout) {
    LinearModule l0(4, 3, &backend);
    l0.set_weight(Pattern(12, 0.4f));
    LoRAConfig config;
    config.rank = 2;
    config.dropout = 0.5f;
    config.seed = 3;
    l0.attach_lora(config);
    // B starts at zero, so the adapter adds nothing until B is set.
    for (const NamedParamRef& p : l0.named_parameters()) {
        if (p.name == "lora_B.weight") *p.ref.value = Tensor(p.ref.value->shape(), &backend, Pattern(p.ref.value->numel(), 0.7f));
    }
    SoftmaxModule softmax(&backend);
    SequentialModule model({&l0, &softmax});
    const Tensor x(Shape({1, 4}), &backend, {1.0f, -0.5f, 0.3f, 2.0f});
    const std::vector<Tensor> samples = McDropoutSamples(model, [&](const Tensor& in) { return model.forward(in); }, x, 8);
    bool varied = false;
    for (const Tensor& s : samples) varied = varied || s.to_host_vector() != samples[0].to_host_vector();
    EXPECT_TRUE(varied);
    const UncertaintyDecomposition d = DecomposeUncertainty(samples);
    EXPECT_GT(d.epistemic[0], 0.0);
}

TEST_F(UncertaintyTest, McDropoutNeedsDropoutAndTwoSamples) {
    LinearModule l0(2, 2, &backend);
    DropoutModule off(0.0f, &backend, 1);
    SequentialModule plain({&l0, &off});
    const Tensor x(Shape({1, 2}), &backend, {1.0f, 2.0f});
    const auto predict = [&](const Tensor& in) { return plain.forward(in); };
    EXPECT_THROW((void)McDropoutSamples(plain, predict, x, 10), std::invalid_argument);
    DropoutModule on(0.5f, &backend, 1);
    SequentialModule model({&l0, &on});
    EXPECT_THROW((void)McDropoutSamples(model, [&](const Tensor& in) { return model.forward(in); }, x, 1), std::invalid_argument);
    // A throwing predict leaves the modes as they were.
    model.set_training(false);
    EXPECT_THROW((void)McDropoutSamples(model, [](const Tensor&) -> Tensor { throw std::runtime_error("boom"); }, x, 2),
                 std::runtime_error);
    EXPECT_FALSE(on.is_training());
}

TEST_F(UncertaintyTest, EnsembleSamplesRunEachMember) {
    const Tensor x(Shape({1, 2}), &backend, {1.0f, 2.0f});
    const auto constant = [this](float a) {
        return [this, a](const Tensor&) { return Tensor(Shape({1, 2}), &backend, {a, 1.0f - a}); };
    };
    const std::vector<Tensor> s = EnsembleSamples({constant(0.9f), constant(0.2f), constant(0.6f)}, x);
    ASSERT_EQ(s.size(), 3u);
    EXPECT_FLOAT_EQ(s[1].to_host_vector()[0], 0.2f);
    EXPECT_THROW((void)EnsembleSamples({constant(0.5f)}, x), std::invalid_argument);
}

}  // namespace
}  // namespace pulsatrix
