#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/batch_norm_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/null_model_baseline.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/sequential_module.hpp"

namespace pulsatrix {
namespace {

std::vector<float> values_of(const Tensor& t) { return std::vector<float>(t.data(), t.data() + t.numel()); }

class NullModelBaselineTest : public ::testing::Test {
protected:
    CPUBackend backend;
    LinearModule l0{4, 6, &backend};
    ReluModule relu{&backend};
    LinearModule l1{6, 2, &backend};
    SequentialModule model{{&l0, &relu, &l1}};
    Tensor x{Shape({1, 4}), &backend, {0.5f, -1, 2, 1}};

    void SetUp() override {
        l0.set_weight({0.3f, -0.2f, 0.5f, 0.1f, 0.4f, -0.6f, 0.2f, 0.7f, -0.1f, 0.3f, 0.5f, 0.2f,
                       -0.4f, 0.6f, 0.1f, 0.2f, -0.3f, 0.4f, 0.5f, 0.1f, 0.3f, -0.2f, 0.6f, 0.4f});
        l0.set_bias({0.1f, -0.1f, 0.2f, 0.0f, 0.05f, -0.2f});
        l1.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.6f, 0.1f, 0.3f, 0.7f, -0.2f, 0.5f, 0.4f, -0.1f});
    }

    std::vector<std::vector<float>> snapshot() {
        std::vector<std::vector<float>> out;
        for (ParamRef p : model.parameters()) out.push_back(values_of(*p.value));
        return out;
    }

    ExplainFn gradient_times_input = [this](const Tensor& in) {
        for (ParamRef p : model.parameters()) p.grad->fill(0.0f);
        Tensor out = model.forward(in);
        Tensor g = model.backward(Tensor(out.shape(), &backend, {1.0f, 0.0f}));
        std::vector<float> gv = values_of(g), xv = values_of(in);
        for (size_t i = 0; i < gv.size(); ++i) gv[i] *= xv[i];
        return Attribution{"gxi", Tensor(in.shape(), &backend, gv), {}};
    };
};

// The generic form: any analysis of the model, any result type (an explainer's map, a probe's
// accuracy, a featurizer's statistics).
TEST_F(NullModelBaselineTest, RunsAnyAnalysisOnTheTrainedAndTheNullModel) {
    auto output0 = [&] { return model.forward(x).data()[0]; };
    const float trained = output0();
    const auto before = snapshot();
    NullModelComparison<float> c = NullModelBaseline(model, output0, /*seed=*/1);
    EXPECT_EQ(c.trained, trained);
    EXPECT_NE(c.null_model, trained);
    EXPECT_EQ(snapshot(), before);  // restored exactly
    EXPECT_EQ(NullModelBaseline(model, output0, 1).null_model, c.null_model);  // reproducible
    EXPECT_NE(NullModelBaseline(model, output0, 2).null_model, c.null_model);
}

TEST_F(NullModelBaselineTest, RestoresTheModelWhenTheAnalysisThrows) {
    const auto before = snapshot();
    int calls = 0;
    auto failing = [&] {
        if (++calls == 2) throw std::runtime_error("analysis failed on the null model");
        return 0;
    };
    EXPECT_THROW((void)NullModelBaseline(model, failing, 1), std::runtime_error);
    EXPECT_EQ(snapshot(), before);
}

TEST_F(NullModelBaselineTest, AttributionsReportHowSimilarTheNullExplanationIs) {
    AttributionNullReport real = NullModelBaseline(model, gradient_times_input, x, 1);
    EXPECT_EQ(values_of(real.trained.values), values_of(gradient_times_input(x).values));
    EXPECT_LT(real.rank_similarity, 0.9f);
    // A "method" that ignores the model looks identical on a random network: the warning sign the
    // baseline rule exists for.
    ExplainFn input_itself = [](const Tensor& in) { return Attribution{"input", Tensor(in), {}}; };
    EXPECT_NEAR(NullModelBaseline(model, input_itself, x, 1).rank_similarity, 1.0f, 1e-6f);
}

TEST_F(NullModelBaselineTest, ReinitializationKeepsEachTensorsScaleAndFlags) {
    LinearModule big(64, 64, &backend);
    std::vector<float> w(64 * 64);
    for (size_t i = 0; i < w.size(); ++i) w[i] = 0.05f * static_cast<float>(static_cast<int>(i % 13) - 6);
    big.set_weight(w);
    big.set_requires_grad(false, "weight");
    double mean = 0, sq = 0;
    for (float v : w) mean += v;
    mean /= w.size();
    for (float v : w) sq += (v - mean) * (v - mean);
    const double std_before = std::sqrt(sq / w.size());

    ReinitializeParameters(big, 3);
    const std::vector<float> r = values_of(big.weight());
    double rmean = 0, rsq = 0;
    for (float v : r) rmean += v;
    rmean /= r.size();
    for (float v : r) rsq += (v - rmean) * (v - rmean);
    EXPECT_NEAR(std::sqrt(rsq / r.size()), std_before, 0.05 * std_before);
    EXPECT_NEAR(rmean, 0.0, 0.05 * std_before);
    EXPECT_EQ(values_of(big.bias()), std::vector<float>(64, 0.0f));  // a zero tensor has nothing to match
    EXPECT_FALSE(big.weight().requires_grad());
}

TEST_F(NullModelBaselineTest, BuffersAreLeftAlone) {
    BatchNormModule bn(2, &backend);
    bn.set_running_mean({0.5f, -0.5f});
    ReinitializeParameters(bn, 4);
    EXPECT_EQ(values_of(bn.running_mean()), (std::vector<float>{0.5f, -0.5f}));
}

TEST_F(NullModelBaselineTest, SnapshotRestoresOnScopeExit) {
    const auto before = snapshot();
    {
        ParameterSnapshot saved(model);
        ReinitializeParameters(model, 9);
        EXPECT_NE(snapshot(), before);
    }
    EXPECT_EQ(snapshot(), before);
}

}  // namespace
}  // namespace pulsatrix
