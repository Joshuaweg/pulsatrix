#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <utility>

#include "exai/conv2d_module.hpp"
#include "exai/cpu_backend.hpp"
#include "exai/flatten_module.hpp"
#include "exai/linear_module.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/relu_module.hpp"

// Phase 4 Mission 0: charter's "not just a spot-check on one architecture" LRP
// completeness requirement, turned into a systematic TEST_P suite over every module type
// this codebase has, plus a genuinely new end-to-end conservation test. Reuses each
// module's existing, already-proven-correct weight/input/relevance setups from Phase
// 1-3's own per-module tests (LinearModuleTest, Conv2DModuleTest, ReluModuleTest,
// FlattenModuleTest) -- this mission doesn't re-derive their math, it systematizes it.
namespace exai {
namespace {

struct ConservationCase {
    std::string name;
    std::function<std::pair<float, float>()> run;  // returns {sum(relevance_in), sum(relevance_out)}
};

class LRPConservationTest : public ::testing::TestWithParam<ConservationCase> {};

TEST_P(LRPConservationTest, RelevanceIsConserved) {
    auto [sum_in, sum_out] = GetParam().run();
    EXPECT_NEAR(sum_in, sum_out, 1e-2f);
}

std::vector<ConservationCase> AllModuleTypeCases() {
    std::vector<ConservationCase> cases;

    cases.push_back({"LinearModule", [] {
                          CPUBackend backend;
                          LinearModule linear(3, 2, &backend);
                          linear.set_weight({1.0f, -2.0f, 3.0f, 0.5f, 2.5f, -1.0f});
                          linear.set_bias({0.1f, -0.2f});
                          Tensor x(Shape({1, 3}), &backend, {1.0f, 2.0f, 0.5f});
                          (void)linear.forward(x);
                          Tensor relevance_out(Shape({1, 2}), &backend, {4.0f, 6.0f});
                          LRPRuleConfig config;
                          Tensor relevance_in = linear.propagate_relevance(relevance_out, config);
                          float sum_in = relevance_in.data()[0] + relevance_in.data()[1] + relevance_in.data()[2];
                          float sum_out = relevance_out.data()[0] + relevance_out.data()[1];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"Conv2DModule", [] {
                          CPUBackend backend;
                          Conv2DModule conv(1, 2, 2, 2, &backend);
                          conv.set_kernel({1.0f, -0.5f, 0.5f, 2.0f, -1.0f, 1.5f, 0.5f, -0.5f});
                          conv.set_bias({0.1f, -0.2f});
                          Tensor input(Shape({1, 1, 3, 3}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
                          (void)conv.forward(input);
                          Tensor relevance_out(Shape({1, 2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 1.5f, 2.5f, 3.5f});
                          LRPRuleConfig config;
                          Tensor relevance_in = conv.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"ReluModule", [] {
                          CPUBackend backend;
                          ReluModule relu(&backend);
                          Tensor x(Shape({3}), &backend, {-1.0f, 2.0f, 3.0f});
                          (void)relu.forward(x);
                          Tensor relevance_out(Shape({3}), &backend, {0.1f, 5.0f, -2.5f});
                          LRPRuleConfig config;
                          Tensor relevance_in = relu.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    cases.push_back({"FlattenModule", [] {
                          CPUBackend backend;
                          FlattenModule flatten(&backend);
                          Tensor input(Shape({2, 2, 2}), &backend, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
                          (void)flatten.forward(input);
                          Tensor relevance_out(Shape({8}), &backend, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f});
                          LRPRuleConfig config;
                          Tensor relevance_in = flatten.propagate_relevance(relevance_out, config);
                          float sum_in = 0.0f;
                          for (int64_t i = 0; i < relevance_in.numel(); ++i) sum_in += relevance_in.data()[i];
                          float sum_out = 0.0f;
                          for (int64_t i = 0; i < relevance_out.numel(); ++i) sum_out += relevance_out.data()[i];
                          return std::make_pair(sum_in, sum_out);
                      }});

    return cases;
}

INSTANTIATE_TEST_SUITE_P(AllModuleTypes, LRPConservationTest, ::testing::ValuesIn(AllModuleTypeCases()),
                          [](const ::testing::TestParamInfo<ConservationCase>& info) { return info.param.name; });

// The mission's actual new deliverable: no end-to-end conservation test existed anywhere
// in this codebase before this mission (grep-confirmed at campaign activation). Chains
// propagate_relevance backward through a real multi-module network -- LinearModule ->
// ReluModule -> LinearModule, the same network shape mission_module_graph_wiring.md and
// mission_explainer_context.md already used -- mirroring Module::forward_traced's
// chaining shape (Phase 2 Mission 0) but for relevance instead of activations.
TEST(LRPConservationEndToEndTest, ConservationHoldsAcrossFullNetworkChain) {
    CPUBackend backend;

    LinearModule linear1(3, 4, &backend);
    linear1.set_weight({0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.1f, 0.4f, -0.6f, 0.3f});
    linear1.set_bias({0.1f, -0.1f, 0.2f, 0.0f});

    ReluModule relu(&backend);

    LinearModule linear2(4, 2, &backend);
    linear2.set_weight({0.5f, -0.3f, 0.2f, 0.4f, -0.1f, 0.6f, 0.3f, -0.5f});
    linear2.set_bias({0.05f, -0.05f});

    Tensor input(Shape({1, 3}), &backend, {0.5f, -0.3f, 1.2f});
    Tensor h1 = linear1.forward(input);
    Tensor h2 = relu.forward(h1);
    (void)linear2.forward(h2);

    Tensor relevance_seed(Shape({1, 2}), &backend, {4.0f, 6.0f});
    LRPRuleConfig config;

    Tensor relevance_h2 = linear2.propagate_relevance(relevance_seed, config);
    Tensor relevance_h1 = relu.propagate_relevance(relevance_h2, config);
    Tensor relevance_input = linear1.propagate_relevance(relevance_h1, config);

    float sum_seed = relevance_seed.data()[0] + relevance_seed.data()[1];
    float sum_input = 0.0f;
    for (int64_t i = 0; i < relevance_input.numel(); ++i) {
        sum_input += relevance_input.data()[i];
    }

    EXPECT_NEAR(sum_input, sum_seed, 1e-2f)
        << "end-to-end LRP conservation violated across the full network chain";
}

}  // namespace
}  // namespace exai
