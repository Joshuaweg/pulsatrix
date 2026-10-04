#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <type_traits>

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {
namespace {

class MockDeviceBackend : public DeviceBackend {
public:
    MOCK_METHOD(DeviceType, device, (), (const, noexcept, override));
    MOCK_METHOD(void*, allocate, (size_t bytes), (override));
    MOCK_METHOD(void, free, (void* ptr), (noexcept, override));
    MOCK_METHOD(void, copy, (void* dst, const void* src, size_t bytes, CopyDirection dir), (override));
    MOCK_METHOD(void, fill, (void* ptr, float value, size_t n), (override));
    MOCK_METHOD(void, gemm, (const float* a, const float* b, float* out, size_t m, size_t k, size_t n), (override));
    MOCK_METHOD(void, elementwise, (ElementwiseOp op, const float* in, float* out, size_t n), (override));
    MOCK_METHOD(void, add, (const float* a, const float* b, float* out, size_t n), (override));
    MOCK_METHOD(void, mul, (const float* a, const float* b, float* out, size_t n), (override));
    MOCK_METHOD(void, gemm_ex,
                (const float* a, bool transpose_a, const float* b, bool transpose_b, float* out, size_t m, size_t k,
                 size_t n, float beta),
                (override));
    MOCK_METHOD(void, column_sums, (const float* in, float* out, size_t rows, size_t cols, float beta), (override));
    MOCK_METHOD(void, add_row_vector, (const float* in, const float* row, float* out, size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, elementwise_backward,
                (ElementwiseOp op, const float* x, const float* grad_out, float* grad_in, size_t n), (override));
    MOCK_METHOD(void, axpby, (float alpha, const float* x, float beta, const float* y, float* out, size_t n),
                (override));
    MOCK_METHOD(float, dot, (const float* a, const float* b, size_t n), (override));
    MOCK_METHOD(void, softmax_rows, (const float* in, float* out, size_t rows, size_t cols), (override));
    MOCK_METHOD(void, softmax_rows_backward, (const float* y, const float* dy, float* dx, size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, logsumexp_rows, (const float* in, float* out, size_t rows, size_t cols), (override));
    MOCK_METHOD(void, adam_step,
                (float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1, float beta2,
                 float eps, float bias_correction1, float bias_correction2),
                (override));
    MOCK_METHOD(float, sum, (const float* in, size_t n), (override));
    MOCK_METHOD(void, dropout_forward,
                (const float* in, float* out, float* mask, size_t n, float p, float scale, uint64_t seed,
                 uint64_t offset),
                (override));
    MOCK_METHOD(void, bce_with_logits, (const float* logits, const float* target, float* out, size_t n), (override));
    MOCK_METHOD(void, bce_with_logits_grad,
                (const float* logits, const float* target, float* grad, size_t n, float scale), (override));
    MOCK_METHOD(void, layer_norm_forward,
                (const float* in, const float* gamma, const float* beta, float* xhat, float* out, float* row_std,
                 size_t rows, size_t cols, float eps),
                (override));
    MOCK_METHOD(void, layer_norm_backward,
                (const float* grad_out, const float* gamma, const float* xhat, const float* row_std, float* grad_in,
                 size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, rms_norm_forward,
                (const float* in, const float* gamma, float* out, float* row_rms, size_t rows, size_t cols, float eps),
                (override));
    MOCK_METHOD(void, rms_norm_backward,
                (const float* grad_out, const float* gamma, const float* in, const float* row_rms, float* grad_in,
                 float* gamma_terms, size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, rope_rotate,
                (const float* in, const float* cos_table, const float* sin_table, float* out, size_t num_slices,
                 size_t seq_len, size_t head_dim, bool inverse),
                (override));
    MOCK_METHOD(void, permute_0213, (const float* in, float* out, size_t d0, size_t d1, size_t d2, size_t d3),
                (override));
    MOCK_METHOD(void, gather_rows, (const float* table, const float* indices, float* out, size_t count, size_t dim),
                (override));
    MOCK_METHOD(void, scatter_add_rows,
                (const float* src, const float* indices, float* table, size_t count, size_t dim), (override));
    MOCK_METHOD(void, tanh_gaussian_forward,
                (const float* mean, const float* log_std, const float* eps, float* action, float* std_cache,
                 float* log_prob, size_t rows, size_t cols, float stabilizer, double half_log_two_pi),
                (override));
    MOCK_METHOD(void, tanh_gaussian_backward,
                (const float* action, const float* std_cache, const float* eps, const float* grad_action,
                 const float* grad_log_prob, float* grad_mean, float* grad_log_std, size_t n, float stabilizer),
                (override));
    MOCK_METHOD(void, lrp_linear,
                (const float* x, const float* w, const float* z, const float* r, float* r_in, size_t rows,
                 size_t in_features, size_t out_features, float eps),
                (override));
    MOCK_METHOD(void, lrp_residual_split,
                (const float* a, const float* b, const float* r, float* r_a, float* r_b, size_t n, float eps),
                (override));
    MOCK_METHOD(void, lrp_bilinear_elementwise,
                (const float* a, const float* b, const float* r, float* r_out, size_t n, float eps), (override));
    MOCK_METHOD(void, lrp_bilinear_matmul,
                (const float* a, const float* b, const float* o, const float* r_o, float* r_a, float* r_b,
                 size_t slices, size_t m, size_t p, size_t q, float eps, bool b_transposed),
                (override));
    MOCK_METHOD(void, lrp_softmax_rows,
                (const float* x, const float* y, const float* r, float* r_in, size_t rows, size_t cols), (override));
    MOCK_METHOD(void, lrp_rope,
                (const float* x, const float* y, const float* r, const float* cos_table, const float* sin_table,
                 float* r_in, size_t slices, size_t seq_len, size_t head_dim, float eps),
                (override));
    MOCK_METHOD(void, logic_pointwise,
                (LogicOp op, int norm, const float* a, const float* b, const float* g_or_r, const float* y,
                 float* out_a, float* out_b, size_t n, float eps),
                (override));
    MOCK_METHOD(void, aggregator_forward,
                (const float* x, float* mean_pow, float* out, size_t n, size_t cols, float p), (override));
    MOCK_METHOD(void, aggregator_backward,
                (const float* x, const float* mean_pow, const float* grad_out, float* grad_in, size_t n, size_t cols,
                 float p),
                (override));
    MOCK_METHOD(void, im2col, (const float* in, float* col, size_t n, size_t c, size_t h, size_t w, size_t kh,
                size_t kw), (override));
    MOCK_METHOD(void, col2im_add, (const float* col, float* out, size_t n, size_t c, size_t h, size_t w, size_t kh,
                size_t kw), (override));
    MOCK_METHOD(void, add_channel_vector, (const float* in, const float* vec, float* out, size_t n, size_t c, size_t
                inner), (override));
    MOCK_METHOD(void, lrp_conv, (const float* col, const float* kernel, const float* pre_bias, const float* r,
                float* r_col, size_t n, size_t out_channels, size_t p, size_t q, float eps), (override));
    MOCK_METHOD(void, lrp_stabilized_divide,
                (const float* r, const float* denom, const float* gate, float* out, size_t n, float eps,
                 LrpGate gate_mode),
                (override));
    MOCK_METHOD(void, max_pool_forward, (const float* in, float* out, float* argmax, size_t planes, size_t h, size_t
                w, size_t kh, size_t kw), (override));
    MOCK_METHOD(void, max_unpool, (const float* src, const float* argmax, float* dst, size_t planes, size_t h,
                size_t w, size_t kh, size_t kw), (override));
    MOCK_METHOD(void, avg_pool_forward, (const float* in, float* out, size_t planes, size_t h, size_t w, size_t kh,
                size_t kw), (override));
    MOCK_METHOD(void, avg_pool_backward, (const float* grad_out, float* grad_in, size_t planes, size_t h, size_t w,
                size_t kh, size_t kw), (override));
    MOCK_METHOD(void, lrp_avg_pool, (const float* x, const float* r, float* r_in, size_t planes, size_t h, size_t w,
                size_t kh, size_t kw, float eps), (override));
    MOCK_METHOD(void, batch_norm_forward, (const float* in, const float* gamma, const float* beta, float* xhat,
                float* out, float* channel_std, size_t n, size_t c, size_t spatial, float eps), (override));
    MOCK_METHOD(void, batch_norm_backward, (const float* grad_out, const float* gamma, const float* xhat, const
                float* channel_std, float* grad_in, float* gamma_grad, float* beta_grad, size_t n, size_t c, size_t
                spatial), (override));
    MOCK_METHOD(void, group_norm_forward, (const float* in, const float* gamma, const float* beta, float* xhat,
                float* out, float* group_std, size_t n, size_t c, size_t spatial, size_t num_groups, float eps),
                (override));
    MOCK_METHOD(void, group_norm_backward, (const float* grad_out, const float* gamma, const float* xhat, const
                float* group_std, float* grad_in, float* gamma_grad, float* beta_grad, size_t n, size_t c, size_t
                spatial, size_t num_groups), (override));
    MOCK_METHOD(void, copy_2d,
                (float* dst, size_t dst_stride, const float* src, size_t src_stride, size_t rows, size_t cols),
                (override));
    MOCK_METHOD(void, accumulate_rows, (const float* in, float* out, size_t rows, size_t cols), (override));
    MOCK_METHOD(void, recurrent_cell, (RecurrentCellOp op, const RecurrentCellArgs& args, size_t n), (override));
    MOCK_METHOD(void, gru_lrp_hprev,
                (const float* h_prev, const float* w_hn, const float* hn, const float* r_term_b, const float* direct,
                 float* r_hprev, size_t rows, size_t hidden, float eps),
                (override));
    MOCK_METHOD(void, ssm_pass, (SsmPassOp op, const SsmPassArgs& args), (override));
    MOCK_METHOD(void, rl_rows, (RlRowOp op, const RlRowArgs& args), (override));
    MOCK_METHOD(void, batch_norm_update_running,
                (const float* in, float* running_mean, float* running_var, size_t n, size_t c, size_t spatial,
                 float momentum),
                (override));
    MOCK_METHOD(void, batch_norm_eval_forward,
                (const float* in, const float* gamma, const float* beta, const float* running_mean,
                 const float* running_var, float* xhat, float* out, float* channel_std, size_t n, size_t c,
                 size_t spatial, float eps),
                (override));
    MOCK_METHOD(void, batch_norm_eval_backward,
                (const float* grad_out, const float* gamma, const float* xhat, const float* channel_std,
                 float* grad_in, float* gamma_grad, float* beta_grad, size_t n, size_t c, size_t spatial),
                (override));
    MOCK_METHOD(void, top_k_rows,
                (const float* in, float* values, float* indices, size_t rows, size_t cols, size_t k, bool largest),
                (override));
    MOCK_METHOD(void, aggregator_lrp,
                (const float* x, const float* mean_pow, const float* r_out, float* r_in, size_t n, size_t cols,
                 float p, float eps),
                (override));
};

TEST(DeviceBackendInterface, HasVirtualDestructor) {
    static_assert(std::has_virtual_destructor_v<DeviceBackend>,
                  "DeviceBackend must have a virtual destructor -- it is deleted via base pointer");
}

TEST(DeviceBackendInterface, IsCallableThroughBasePointer) {
    MockDeviceBackend mock;
    DeviceBackend& backend = mock;

    EXPECT_CALL(mock, allocate(128)).WillOnce(::testing::Return(reinterpret_cast<void*>(0x1)));
    void* ptr = backend.allocate(128);
    EXPECT_EQ(ptr, reinterpret_cast<void*>(0x1));
}

TEST(DeviceBackendInterface, EveryPureVirtualIsMockable) {
    // If MockDeviceBackend compiled at all, every pure-virtual method above was
    // successfully overridden -- this test exists so the interface's completeness
    // is verified by the build itself, not just by inspection.
    MockDeviceBackend mock;
    EXPECT_CALL(mock, free(::testing::_)).Times(1);
    EXPECT_CALL(mock, copy(::testing::_, ::testing::_, 0, CopyDirection::HostToHost)).Times(1);
    EXPECT_CALL(mock, fill(::testing::_, 0.0f, 0)).Times(1);
    EXPECT_CALL(mock, gemm(::testing::_, ::testing::_, ::testing::_, 0, 0, 0)).Times(1);
    EXPECT_CALL(mock, elementwise(ElementwiseOp::Relu, ::testing::_, ::testing::_, 0)).Times(1);
    EXPECT_CALL(mock, add(::testing::_, ::testing::_, ::testing::_, 0)).Times(1);

    mock.free(nullptr);
    mock.copy(nullptr, nullptr, 0, CopyDirection::HostToHost);
    mock.fill(nullptr, 0.0f, 0);
    mock.gemm(nullptr, nullptr, nullptr, 0, 0, 0);
    mock.elementwise(ElementwiseOp::Relu, nullptr, nullptr, 0);
    mock.add(nullptr, nullptr, nullptr, 0);
}

}  // namespace
}  // namespace pulsatrix
