#include <gtest/gtest.h>

#include "exai/cpu_backend.hpp"
#include "exai/module.hpp"

namespace exai {
namespace {

// Minimal concrete subclass -- Module is abstract. This deliberately does NOT itself
// validate input, so passing tests prove the NVI wrapper's precondition check runs
// regardless of what forward_impl() does, not that forward_impl() happens to check too.
class TestDoubleModule : public Module {
public:
    Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig&) override {
        return Tensor(relevance_out);  // pass-through, not a real rule -- this is a test double
    }

protected:
    Tensor forward_impl(const Tensor& input) override {
        return Tensor(input);  // identity, no validation of its own
    }
};

class ModuleTest : public ::testing::Test {
protected:
    CPUBackend backend;
    TestDoubleModule module;
};

TEST_F(ModuleTest, ForwardDelegatesToForwardImpl) {
    Tensor input(Shape({3}), &backend, {1.0f, 2.0f, 3.0f});
    Tensor output = module.forward(input);

    EXPECT_FLOAT_EQ(output.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 3.0f);
}

using ModuleDeathTest = ModuleTest;

TEST_F(ModuleDeathTest, ForwardAbortsOnEmptyInputEvenThoughForwardImplDoesNotCheck) {
#ifdef NDEBUG
    GTEST_SKIP() << "EXAI_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    Tensor empty(Shape({0}), &backend);
    EXPECT_DEATH({ (void)module.forward(empty); }, "EXAI_ASSERT failed");
}

TEST_F(ModuleTest, PropagateRelevanceIsCallableThroughBasePointer) {
    Module& base = module;
    Tensor relevance_out(Shape({2}), &backend, {0.5f, 0.5f});
    LRPRuleConfig config;

    Tensor relevance_in = base.propagate_relevance(relevance_out, config);

    EXPECT_FLOAT_EQ(relevance_in.data()[0], 0.5f);
    EXPECT_FLOAT_EQ(relevance_in.data()[1], 0.5f);
}

TEST_F(ModuleTest, DefaultParametersIsEmpty) {
    // TestDoubleModule doesn't override parameters() -- the base default (no parameters)
    // is what a parameterless module type like ReluModule relies on too.
    EXPECT_TRUE(module.parameters().empty());
}

}  // namespace
}  // namespace exai
