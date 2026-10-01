/** @file cpu_backend.hpp
 *  @brief CPU implementation of DeviceBackend -- the first, reference DeviceBackend implementation.
 *  @ingroup dl_modules
 */
#pragma once

#include "pulsatrix/device_backend.hpp"

namespace pulsatrix {

/**
 * @brief CPU-resident DeviceBackend implementation. Reference implementation every other
 *        backend (CUDABackend, HIPBackend) is checked for numerical equivalence against.
 */
class CPUBackend : public DeviceBackend {
public:
    [[nodiscard]] DeviceType device() const noexcept override { return DeviceType::Cpu; }
    [[nodiscard]] void* allocate(size_t bytes) override;
    void free(void* ptr) noexcept override;
    void copy(void* dst, const void* src, size_t bytes, CopyDirection dir) override;
    void fill(void* ptr, float value, size_t n) override;
    void gemm(const float* a, const float* b, float* out, size_t m, size_t k, size_t n) override;
    void elementwise(ElementwiseOp op, const float* in, float* out, size_t n) override;
    void add(const float* a, const float* b, float* out, size_t n) override;
    void mul(const float* a, const float* b, float* out, size_t n) override;
    void gemm_ex(const float* a, bool transpose_a, const float* b, bool transpose_b, float* out, size_t m, size_t k,
                 size_t n, float beta) override;
    void column_sums(const float* in, float* out, size_t rows, size_t cols, float beta) override;
    void add_row_vector(const float* in, const float* row, float* out, size_t rows, size_t cols) override;
    void elementwise_backward(ElementwiseOp op, const float* x, const float* grad_out, float* grad_in,
                              size_t n) override;
    void axpby(float alpha, const float* x, float beta, const float* y, float* out, size_t n) override;
    [[nodiscard]] float dot(const float* a, const float* b, size_t n) override;
    void softmax_rows(const float* in, float* out, size_t rows, size_t cols) override;
    void softmax_rows_backward(const float* y, const float* dy, float* dx, size_t rows, size_t cols) override;
    void logsumexp_rows(const float* in, float* out, size_t rows, size_t cols) override;
    void adam_step(float* param, const float* grad, float* m, float* v, size_t n, float lr, float beta1, float beta2,
                   float eps, float bias_correction1, float bias_correction2) override;
    [[nodiscard]] float sum(const float* in, size_t n) override;
    void dropout_forward(const float* in, float* out, float* mask, size_t n, float p, float scale, uint64_t seed,
                         uint64_t offset) override;
    void bce_with_logits(const float* logits, const float* target, float* out, size_t n) override;
    void bce_with_logits_grad(const float* logits, const float* target, float* grad, size_t n, float scale) override;
    void layer_norm_forward(const float* in, const float* gamma, const float* beta, float* xhat, float* out,
                            float* row_std, size_t rows, size_t cols, float eps) override;
    void layer_norm_backward(const float* grad_out, const float* gamma, const float* xhat, const float* row_std,
                             float* grad_in, size_t rows, size_t cols) override;
    void rms_norm_forward(const float* in, const float* gamma, float* out, float* row_rms, size_t rows, size_t cols,
                          float eps) override;
    void rms_norm_backward(const float* grad_out, const float* gamma, const float* in, const float* row_rms,
                           float* grad_in, float* gamma_terms, size_t rows, size_t cols) override;
    void rope_rotate(const float* in, const float* cos_table, const float* sin_table, float* out, size_t num_slices,
                     size_t seq_len, size_t head_dim, bool inverse) override;
    void permute_0213(const float* in, float* out, size_t d0, size_t d1, size_t d2, size_t d3) override;
    void gather_rows(const float* table, const float* indices, float* out, size_t count, size_t dim) override;
    void scatter_add_rows(const float* src, const float* indices, float* table, size_t count, size_t dim) override;
    void tanh_gaussian_forward(const float* mean, const float* log_std, const float* eps, float* action,
                               float* std_cache, float* log_prob, size_t rows, size_t cols, float stabilizer,
                               double half_log_two_pi) override;
    void tanh_gaussian_backward(const float* action, const float* std_cache, const float* eps,
                                const float* grad_action, const float* grad_log_prob, float* grad_mean,
                                float* grad_log_std, size_t n, float stabilizer) override;
    void lrp_linear(const float* x, const float* w, const float* z, const float* r, float* r_in, size_t rows,
                    size_t in_features, size_t out_features, float eps) override;
    void lrp_residual_split(const float* a, const float* b, const float* r, float* r_a, float* r_b, size_t n,
                            float eps) override;
    void lrp_bilinear_elementwise(const float* a, const float* b, const float* r, float* r_out, size_t n,
                                  float eps) override;
    void lrp_bilinear_matmul(const float* a, const float* b, const float* o, const float* r_o, float* r_a, float* r_b,
                             size_t slices, size_t m, size_t p, size_t q, float eps, bool b_transposed) override;
    void lrp_softmax_rows(const float* x, const float* y, const float* r, float* r_in, size_t rows,
                          size_t cols) override;
    void lrp_rope(const float* x, const float* y, const float* r, const float* cos_table, const float* sin_table,
                  float* r_in, size_t slices, size_t seq_len, size_t head_dim, float eps) override;
    void logic_pointwise(LogicOp op, int norm, const float* a, const float* b, const float* g_or_r, const float* y,
                         float* out_a, float* out_b, size_t n, float eps) override;
    void aggregator_forward(const float* x, float* mean_pow, float* out, size_t n, size_t cols, float p) override;
    void aggregator_backward(const float* x, const float* mean_pow, const float* grad_out, float* grad_in, size_t n,
                             size_t cols, float p) override;
    void aggregator_lrp(const float* x, const float* mean_pow, const float* r_out, float* r_in, size_t n, size_t cols,
                        float p, float eps) override;
};

}  // namespace pulsatrix
