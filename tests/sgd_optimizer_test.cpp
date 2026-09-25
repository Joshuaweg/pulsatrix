#include <gtest/gtest.h>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

namespace pulsatrix {
namespace {

TEST(SGDOptimizerTest, StepUpdatesParametersByLearningRateTimesGradient) {
    CPUBackend backend;
    LinearModule linear(1, 1, &backend);
    linear.set_weight({2.0f});
    linear.set_bias({1.0f});

    // Set gradients directly through parameters() rather than via forward()/backward(),
    // to isolate the optimizer's own update logic from module-specific gradient math.
    auto params = linear.parameters();
    params[0].grad->data()[0] = 4.0f;  // weight_grad
    params[1].grad->data()[0] = 2.0f;  // bias_grad

    SGDOptimizer opt(0.1f);
    opt.step(linear);

    EXPECT_FLOAT_EQ(linear.weight().data()[0], 1.6f);  // 2.0 - 0.1*4.0
    EXPECT_FLOAT_EQ(linear.bias().data()[0], 0.8f);    // 1.0 - 0.1*2.0
}

TEST(SGDOptimizerTest, ZeroGradResetsAllGradientsToZero) {
    CPUBackend backend;
    LinearModule linear(1, 1, &backend);
    auto params = linear.parameters();
    params[0].grad->data()[0] = 5.0f;
    params[1].grad->data()[0] = 3.0f;

    SGDOptimizer opt(0.1f);
    opt.zero_grad(linear);

    EXPECT_FLOAT_EQ(linear.weight_grad().data()[0], 0.0f);
    EXPECT_FLOAT_EQ(linear.bias_grad().data()[0], 0.0f);
}

TEST(SGDOptimizerTest, StepOnParameterlessModuleIsSafeNoOp) {
    // A module whose parameters() is the Module default (empty) must not crash.
    class NoParamModule : public Module {
    public:
        Tensor propagate_relevance(const Tensor& r, const LRPRuleConfig&) override { return Tensor(r); }
        Tensor backward(const Tensor& grad_output) override { return Tensor(grad_output); }
        [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }

    protected:
        Tensor forward_impl(const Tensor& input) override { return Tensor(input); }
    };

    NoParamModule m;
    SGDOptimizer opt(0.1f);
    EXPECT_NO_THROW(opt.step(m));
    EXPECT_NO_THROW(opt.zero_grad(m));
}

// step() dereferences Tensor::data() directly in a raw host loop -- undefined behavior on
// a CUDA-backed Tensor. Phase 1.5 Mission 2 (mission_host_loop_guards.md) guards it with
// PULSATRIX_ASSERT. No real GPU needed: see LinearModuleDeathTest for the mislabeled-Tensor
// testing pattern this reuses. LinearModule can't be used here -- its parameters are always
// constructed DeviceType::Cpu regardless of backend -- so this test defines its own minimal
// Module double whose parameter is explicitly tagged Cuda. zero_grad() is NOT guarded --
// confirmed safe, it routes through Tensor::fill() -> DeviceBackend::fill(), not a raw loop.
TEST(SGDOptimizerDeathTest, StepAbortsOnNonCpuParameter) {
#ifdef NDEBUG
    GTEST_SKIP() << "PULSATRIX_ASSERT is a no-op under NDEBUG (Release) by design -- see assert.hpp";
#endif
    class CudaParamModule : public Module {
    public:
        explicit CudaParamModule(DeviceBackend* backend)
            : value_(Shape({1}), backend, {1.0f}, DeviceType::Cuda),
              grad_(Shape({1}), backend, {1.0f}, DeviceType::Cuda) {}
        Tensor propagate_relevance(const Tensor& r, const LRPRuleConfig&) override { return Tensor(r); }
        Tensor backward(const Tensor& grad_output) override { return Tensor(grad_output); }
        [[nodiscard]] OpType op_type() const override { return OpType::Elementwise; }
        std::vector<ParamRef> parameters() override { return {{&value_, &grad_}}; }

    protected:
        Tensor forward_impl(const Tensor& input) override { return Tensor(input); }

    private:
        Tensor value_;
        Tensor grad_;
    };

    CPUBackend backend;
    CudaParamModule m(&backend);
    SGDOptimizer opt(0.1f);
    EXPECT_DEATH({ opt.step(m); }, "PULSATRIX_ASSERT failed");
}

}  // namespace
}  // namespace pulsatrix
