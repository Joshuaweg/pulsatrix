#include "pulsatrix/multihead_attention_module.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/**
 * @brief head_dim used by the member initializer list, which runs *before* the constructor
 *        body's validation can throw. Falls back to 1 for any invalid configuration so every
 *        sub-object still constructs against a legal Shape; the body then throws the real,
 *        specific error. Same defensive-fallback pattern as RNNModule's
 *        `input_size > 0 ? input_size : 1` initializers.
 */
[[nodiscard]] int64_t safe_head_dim(int64_t d_model, int64_t num_heads) {
    if (d_model <= 0 || num_heads <= 0 || d_model % num_heads != 0) {
        return 1;
    }
    return d_model / num_heads;
}

[[nodiscard]] int64_t safe_positive(int64_t value) { return value > 0 ? value : 1; }

/**
 * @brief `(N, L, num_heads*head_dim)` -> `(N, num_heads, L, head_dim)`.
 * @note A genuine permutation, not a reshape: num_heads moves from inside the last axis to
 *       in front of L. No Tensor permute utility exists in this codebase, so this is a raw
 *       host loop, in the same style as RNNModule's per-timestep scatter/gather loops.
 */
void split_heads(const float* src, float* dst, int64_t N, int64_t L, int64_t num_heads, int64_t head_dim) {
    const int64_t d_model = num_heads * head_dim;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t l = 0; l < L; ++l) {
                const int64_t src_off = (n * L + l) * d_model + h * head_dim;
                const int64_t dst_off = ((n * num_heads + h) * L + l) * head_dim;
                for (int64_t e = 0; e < head_dim; ++e) {
                    dst[dst_off + e] = src[src_off + e];
                }
            }
        }
    }
}

/** @brief The exact inverse of split_heads: `(N, num_heads, L, head_dim)` -> `(N, L, d_model)`. */
void merge_heads(const float* src, float* dst, int64_t N, int64_t L, int64_t num_heads, int64_t head_dim) {
    const int64_t d_model = num_heads * head_dim;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t l = 0; l < L; ++l) {
                const int64_t dst_off = (n * L + l) * d_model + h * head_dim;
                const int64_t src_off = ((n * num_heads + h) * L + l) * head_dim;
                for (int64_t e = 0; e < head_dim; ++e) {
                    dst[dst_off + e] = src[src_off + e];
                }
            }
        }
    }
}

/**
 * @brief Transposes a (rows x cols) row-major block into a (cols x rows) row-major block --
 *        same helper shape as LinearModule's/RNNModule's own transpose() (DeviceBackend::gemm
 *        has no transpose flag), but writing into a caller-owned buffer since this one is
 *        called once per (n, h) slice rather than once per call.
 */
void transpose_into(const float* src, float* dst, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

/**
 * @brief AttnLRP Eq. 15 (Achtibat et al. 2024) -- the bilinear/"uniform" rule for a matmul
 *        `O = A @ B`, where BOTH operands are inputs carrying relevance.
 *
 *   `R_A[i,j] += (A[i,j]*B[j,k] / (2*O[i,k] + eps*sign(O[i,k]))) * R_O[i,k]`
 *   `R_B[j,k] += (A[i,j]*B[j,k] / (2*O[i,k] + eps*sign(O[i,k]))) * R_O[i,k]`
 *
 * @param a (M x P) row-major.
 * @param b (P x Q) row-major.
 * @param o (M x Q) row-major -- the forward product, the rule's denominator.
 * @param r_o (M x Q) relevance at the output.
 * @param r_a (M x P) accumulator for A's relevance. Caller zero-fills.
 * @param r_b (P x Q) accumulator for B's relevance. Caller zero-fills.
 * @param eps Stabilizer, signed to match o (sign(0) == +1, same convention as every other
 *        rule in this codebase).
 * @note **The `2*` in the denominator is the whole point of this rule and is not this
 *       codebase's usual additive epsilon rule.** Eq. 15 splits each output's relevance
 *       evenly between the two operands (each operand's shares sum to `R_O/2` rather than to
 *       `R_O`), which is what makes a bilinear product -- where both factors are activations,
 *       not one activation and one fixed weight -- attributable at all. Substituting the
 *       familiar `1*` denominator used by LinearModule/RNNModule/RoPEModule would double the
 *       total relevance handed to the two operands.
 */
void bilinear_lrp_eq15(const float* a, const float* b, const float* o, const float* r_o, float* r_a, float* r_b,
                       int64_t M, int64_t P, int64_t Q, float eps) {
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < Q; ++k) {
            const float o_ik = o[i * Q + k];
            const float denom = 2.0f * o_ik + eps * ((o_ik >= 0.0f) ? 1.0f : -1.0f);
            const float scaled_r = r_o[i * Q + k] / denom;
            for (int64_t j = 0; j < P; ++j) {
                const float contribution = a[i * P + j] * b[j * Q + k] * scaled_r;
                r_a[i * P + j] += contribution;
                r_b[j * Q + k] += contribution;
            }
        }
    }
}

/**
 * @brief A reshaped copy of t. Tensor::reshape is an in-place, non-const metadata-only
 *        operation on a contiguous row-major buffer, so this copy exists solely to keep the
 *        caller's tensor const -- the element order is byte-identical either way, which is
 *        what makes the "(N, L, d_model) viewed as (N*L, d_model) for LinearModule" trick
 *        valid in the first place.
 */
[[nodiscard]] Tensor reshaped(const Tensor& t, Shape new_shape) {
    Tensor out(t);
    out.reshape(std::move(new_shape));
    return out;
}

}  // namespace

MultiHeadAttentionModule::MultiHeadAttentionModule(int64_t d_model, int64_t num_heads, DeviceBackend* backend,
                                                   bool use_rope, bool use_qk_norm)
    : d_model_(d_model),
      num_heads_(num_heads),
      head_dim_(safe_head_dim(d_model, num_heads)),
      use_rope_(use_rope),
      use_qk_norm_(use_qk_norm),
      backend_(backend),
      q_proj_(safe_positive(d_model), safe_positive(d_model), backend),
      k_proj_(safe_positive(d_model), safe_positive(d_model), backend),
      v_proj_(safe_positive(d_model), safe_positive(d_model), backend),
      out_proj_(safe_positive(d_model), safe_positive(d_model), backend),
      softmax_(backend),
      last_q_(Shape({0}), backend),
      last_k_(Shape({0}), backend),
      last_v_(Shape({0}), backend),
      last_scores_raw_(Shape({0}), backend),
      last_attn_(Shape({0}), backend),
      last_context_(Shape({0}), backend) {
    // External boundary (constructor arguments can originate from Phase 5's Python bindings
    // with no upstream validation), same convention as every other module's constructor.
    if (d_model <= 0) {
        throw std::invalid_argument("MultiHeadAttentionModule: d_model must be positive");
    }
    if (num_heads <= 0) {
        throw std::invalid_argument("MultiHeadAttentionModule: num_heads must be positive");
    }
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("MultiHeadAttentionModule: d_model must be divisible by num_heads");
    }
    // Checked here rather than left to RoPEModule's own constructor so the message names the
    // real cause (the caller chose d_model/num_heads, not head_dim directly).
    if (use_rope && head_dim_ % 2 != 0) {
        throw std::invalid_argument(
            "MultiHeadAttentionModule: use_rope requires an even head_dim (d_model / num_heads)");
    }

    if (use_rope_) {
        // Separate instances for Q and K -- see the header's caching note; a shared instance
        // would leave only K's activations cached for propagate_relevance.
        q_rope_ = std::make_unique<RoPEModule>(head_dim_, backend);
        k_rope_ = std::make_unique<RoPEModule>(head_dim_, backend);
    }
    if (use_qk_norm_) {
        q_norm_ = std::make_unique<RMSNormModule>(head_dim_, backend);
        k_norm_ = std::make_unique<RMSNormModule>(head_dim_, backend);
        // RMSNormModule zero-initializes gamma; a zero gamma here would annihilate Q and K
        // and make attention uniform regardless of the input. See the header's note.
        const std::vector<float> ones(static_cast<size_t>(head_dim_), 1.0f);
        q_norm_->set_gamma(ones);
        k_norm_->set_gamma(ones);
    }
}

Tensor MultiHeadAttentionModule::forward_impl(const Tensor& input) {
    // Dereferences Tensor::data() directly in raw host loops -- not backend-generic.
    // See mission_host_loop_guards.md.
    PULSATRIX_ASSERT(input.device() == DeviceType::Cpu);

    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("MultiHeadAttentionModule::forward: input must be rank-3 (N, L, d_model)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t H = num_heads_;
    const int64_t D = head_dim_;

    // --- Step 1: Q/K/V projections ------------------------------------------------------
    // LinearModule takes rank-2 (batch, features); (N, L, d_model) is the same row-major
    // buffer as (N*L, d_model), so this is a pure re-view.
    const Tensor flat_input = reshaped(input, Shape({N * L, d_model_}));
    Tensor q_flat = q_proj_.forward(flat_input);
    Tensor k_flat = k_proj_.forward(flat_input);
    Tensor v_flat = v_proj_.forward(flat_input);

    // --- Step 2: split heads ------------------------------------------------------------
    Tensor q(Shape({N, H, L, D}), backend_);
    Tensor k(Shape({N, H, L, D}), backend_);
    Tensor v(Shape({N, H, L, D}), backend_);
    split_heads(q_flat.data(), q.data(), N, L, H, D);
    split_heads(k_flat.data(), k.data(), N, L, H, D);
    split_heads(v_flat.data(), v.data(), N, L, H, D);

    // --- Step 3: QK-Norm (optional) -----------------------------------------------------
    // (N, H, L, D) -> (N*H*L, D) is a pure reshape: the normalized axis is already last.
    if (use_qk_norm_) {
        const Shape rows({N * H * L, D});
        Tensor q_norm_out = q_norm_->forward(reshaped(q, rows));
        Tensor k_norm_out = k_norm_->forward(reshaped(k, rows));
        q = reshaped(q_norm_out, Shape({N, H, L, D}));
        k = reshaped(k_norm_out, Shape({N, H, L, D}));
    }

    // --- Step 4: RoPE (optional) --------------------------------------------------------
    // RoPEModule is already rank-agnostic over (..., L, head_dim) -- no reshape needed.
    if (use_rope_) {
        q = q_rope_->forward(q);
        k = k_rope_->forward(k);
    }

    // --- Step 5: scores = Q @ K^T / sqrt(head_dim) --------------------------------------
    // No batched-gemm primitive exists; loop the N*H independent 2D slices explicitly. Each
    // slice is a contiguous span of the row-major buffer, so the slice pointer can go
    // straight into gemm without a gather.
    Tensor scores_raw(Shape({N, H, L, L}), backend_);
    Tensor scores(Shape({N, H, L, L}), backend_);
    std::vector<float> k_transposed(static_cast<size_t>(D * L));
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const float* q_slice = q.data() + nh * L * D;
        const float* k_slice = k.data() + nh * L * D;
        float* scores_raw_slice = scores_raw.data() + nh * L * L;

        transpose_into(k_slice, k_transposed.data(), L, D);
        backend_->gemm(q_slice, k_transposed.data(), scores_raw_slice, static_cast<size_t>(L),
                       static_cast<size_t>(D), static_cast<size_t>(L));
        for (int64_t i = 0; i < L * L; ++i) {
            scores.data()[nh * L * L + i] = scores_raw_slice[i] * inv_sqrt_d;
        }
    }

    // --- Step 6: softmax over the last axis ---------------------------------------------
    // SoftmaxModule is rank-agnostic over the last axis -- (N, H, L, L) needs no reshape.
    Tensor attn = softmax_.forward(scores);

    // --- Step 7: context = Attn @ V -----------------------------------------------------
    Tensor context(Shape({N, H, L, D}), backend_);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        backend_->gemm(attn.data() + nh * L * L, v.data() + nh * L * D, context.data() + nh * L * D,
                       static_cast<size_t>(L), static_cast<size_t>(L), static_cast<size_t>(D));
    }

    // --- Step 8: merge heads ------------------------------------------------------------
    Tensor merged(Shape({N * L, d_model_}), backend_);
    merge_heads(context.data(), merged.data(), N, L, H, D);

    // --- Step 9: output projection ------------------------------------------------------
    Tensor out_flat = out_proj_.forward(merged);

    last_N_ = N;
    last_L_ = L;
    last_q_ = std::move(q);
    last_k_ = std::move(k);
    last_v_ = std::move(v);
    last_scores_raw_ = std::move(scores_raw);
    last_attn_ = std::move(attn);
    last_context_ = std::move(context);
    has_forwarded_ = true;

    return reshaped(out_flat, Shape({N, L, d_model_}));
}

Tensor MultiHeadAttentionModule::backward(const Tensor& grad_output) {
    if (!has_forwarded_) {
        throw std::logic_error("MultiHeadAttentionModule::backward: called before any forward()");
    }
    const int64_t N = last_N_;
    const int64_t L = last_L_;
    const int64_t H = num_heads_;
    const int64_t D = head_dim_;
    if (grad_output.rank() != 3 || grad_output.shape().dim(0) != N || grad_output.shape().dim(1) != L ||
        grad_output.shape().dim(2) != d_model_) {
        throw std::invalid_argument(
            "MultiHeadAttentionModule::backward: grad_output must be rank-3 (N, L, d_model) matching the cached "
            "forward shape");
    }
    // Raw host loops -- see the header's note and mission_host_loop_guards.md.
    PULSATRIX_ASSERT(grad_output.device() == DeviceType::Cpu);

    // --- Step 9' : output projection ----------------------------------------------------
    Tensor grad_merged = out_proj_.backward(reshaped(grad_output, Shape({N * L, d_model_})));

    // --- Step 8' : merge heads inverse (pure data movement) -----------------------------
    Tensor grad_context(Shape({N, H, L, D}), backend_);
    split_heads(grad_merged.data(), grad_context.data(), N, L, H, D);

    // --- Step 7' : context = Attn @ V ---------------------------------------------------
    // Standard matmul backward, per (n, h) slice: dA = dC @ B^T, dB = A^T @ dC.
    Tensor grad_attn(Shape({N, H, L, L}), backend_);
    Tensor grad_v(Shape({N, H, L, D}), backend_);
    std::vector<float> scratch_a(static_cast<size_t>(L * L));
    std::vector<float> scratch_b(static_cast<size_t>(L * D));
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const float* dc = grad_context.data() + nh * L * D;
        // dAttn = dContext @ V^T  ((L,D) @ (D,L) -> (L,L))
        transpose_into(last_v_.data() + nh * L * D, scratch_b.data(), L, D);
        backend_->gemm(dc, scratch_b.data(), grad_attn.data() + nh * L * L, static_cast<size_t>(L),
                       static_cast<size_t>(D), static_cast<size_t>(L));
        // dV = Attn^T @ dContext  ((L,L) @ (L,D) -> (L,D))
        transpose_into(last_attn_.data() + nh * L * L, scratch_a.data(), L, L);
        backend_->gemm(scratch_a.data(), dc, grad_v.data() + nh * L * D, static_cast<size_t>(L),
                       static_cast<size_t>(L), static_cast<size_t>(D));
    }

    // --- Step 6' : softmax --------------------------------------------------------------
    Tensor grad_scores = softmax_.backward(grad_attn);

    // --- Step 5' : scores = Q @ K^T / sqrt(head_dim) ------------------------------------
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    Tensor grad_scores_raw(grad_scores);
    for (int64_t i = 0; i < grad_scores_raw.numel(); ++i) {
        grad_scores_raw.data()[i] *= inv_sqrt_d;
    }
    Tensor grad_q(Shape({N, H, L, D}), backend_);
    Tensor grad_k(Shape({N, H, L, D}), backend_);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const float* ds = grad_scores_raw.data() + nh * L * L;
        // O[i,k] = sum_j Q[i,j] K[k,j]  ->  dQ = dO @ K, dK = dO^T @ Q.
        backend_->gemm(ds, last_k_.data() + nh * L * D, grad_q.data() + nh * L * D, static_cast<size_t>(L),
                       static_cast<size_t>(L), static_cast<size_t>(D));
        transpose_into(ds, scratch_a.data(), L, L);
        backend_->gemm(scratch_a.data(), last_q_.data() + nh * L * D, grad_k.data() + nh * L * D,
                       static_cast<size_t>(L), static_cast<size_t>(L), static_cast<size_t>(D));
    }

    // --- Step 4' : RoPE -----------------------------------------------------------------
    if (use_rope_) {
        grad_q = q_rope_->backward(grad_q);
        grad_k = k_rope_->backward(grad_k);
    }

    // --- Step 3' : QK-Norm --------------------------------------------------------------
    if (use_qk_norm_) {
        const Shape rows({N * H * L, D});
        Tensor gq = q_norm_->backward(reshaped(grad_q, rows));
        Tensor gk = k_norm_->backward(reshaped(grad_k, rows));
        grad_q = reshaped(gq, Shape({N, H, L, D}));
        grad_k = reshaped(gk, Shape({N, H, L, D}));
    }

    // --- Step 2' : split-heads inverse --------------------------------------------------
    Tensor grad_q_flat(Shape({N * L, d_model_}), backend_);
    Tensor grad_k_flat(Shape({N * L, d_model_}), backend_);
    Tensor grad_v_flat(Shape({N * L, d_model_}), backend_);
    merge_heads(grad_q.data(), grad_q_flat.data(), N, L, H, D);
    merge_heads(grad_k.data(), grad_k_flat.data(), N, L, H, D);
    merge_heads(grad_v.data(), grad_v_flat.data(), N, L, H, D);

    // --- Step 1' : Q/K/V projections ----------------------------------------------------
    // All three read the same input tensor, so the three input gradients sum.
    Tensor grad_input = q_proj_.backward(grad_q_flat);
    grad_input.accumulate(k_proj_.backward(grad_k_flat));
    grad_input.accumulate(v_proj_.backward(grad_v_flat));

    return reshaped(grad_input, Shape({N, L, d_model_}));
}

Tensor MultiHeadAttentionModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    if (!has_forwarded_) {
        throw std::logic_error("MultiHeadAttentionModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_N_;
    const int64_t L = last_L_;
    const int64_t H = num_heads_;
    const int64_t D = head_dim_;
    if (relevance_out.rank() != 3 || relevance_out.shape().dim(0) != N || relevance_out.shape().dim(1) != L ||
        relevance_out.shape().dim(2) != d_model_) {
        throw std::invalid_argument(
            "MultiHeadAttentionModule::propagate_relevance: relevance_out must be rank-3 (N, L, d_model) matching "
            "the cached forward shape");
    }
    // Raw host loops -- see the header's note and mission_host_loop_guards.md.
    PULSATRIX_ASSERT(relevance_out.device() == DeviceType::Cpu);

    // --- Step 9' : output projection (LinearModule's own epsilon rule) ------------------
    Tensor r_merged = out_proj_.propagate_relevance(reshaped(relevance_out, Shape({N * L, d_model_})), config);

    // --- Step 8' : merge-heads inverse (pure index mapping, no epsilon) -----------------
    Tensor r_context(Shape({N, H, L, D}), backend_);
    split_heads(r_merged.data(), r_context.data(), N, L, H, D);

    // --- Step 7' : Eq. 15 on context = Attn @ V -----------------------------------------
    Tensor r_attn(Shape({N, H, L, L}), backend_);
    Tensor r_v(Shape({N, H, L, D}), backend_);
    r_attn.fill(0.0f);  // bilinear_lrp_eq15 accumulates with +=
    r_v.fill(0.0f);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        bilinear_lrp_eq15(last_attn_.data() + nh * L * L, last_v_.data() + nh * L * D,
                          last_context_.data() + nh * L * D, r_context.data() + nh * L * D,
                          r_attn.data() + nh * L * L, r_v.data() + nh * L * D, L, L, D, config.epsilon);
    }

    // --- Step 6' : softmax (AttnLRP Eq. 13, SoftmaxModule's own rule) -------------------
    Tensor r_scores = softmax_.propagate_relevance(r_attn, config);

    // --- Step 5' : Eq. 15 on scores_raw = Q @ K^T ---------------------------------------
    // The 1/sqrt(head_dim) scale is a positive constant, under which the epsilon rule is
    // exactly the identity, so r_scores is also the relevance of the *raw* product -- which
    // is what the cached denominator (last_scores_raw_) is.
    Tensor r_q(Shape({N, H, L, D}), backend_);
    Tensor r_k(Shape({N, H, L, D}), backend_);
    r_q.fill(0.0f);
    r_k.fill(0.0f);
    std::vector<float> k_transposed(static_cast<size_t>(D * L));
    std::vector<float> r_k_transposed(static_cast<size_t>(D * L));
    for (int64_t nh = 0; nh < N * H; ++nh) {
        // B is K^T, so B's relevance comes back in (head_dim, L) layout and must be
        // transposed into K's own (L, head_dim) layout -- exactly as the value itself was.
        transpose_into(last_k_.data() + nh * L * D, k_transposed.data(), L, D);
        std::fill(r_k_transposed.begin(), r_k_transposed.end(), 0.0f);
        bilinear_lrp_eq15(last_q_.data() + nh * L * D, k_transposed.data(), last_scores_raw_.data() + nh * L * L,
                          r_scores.data() + nh * L * L, r_q.data() + nh * L * D, r_k_transposed.data(), L, D, L,
                          config.epsilon);
        transpose_into(r_k_transposed.data(), r_k.data() + nh * L * D, D, L);
    }

    // --- Step 4' : RoPE (its own epsilon rule) ------------------------------------------
    if (use_rope_) {
        r_q = q_rope_->propagate_relevance(r_q, config);
        r_k = k_rope_->propagate_relevance(r_k, config);
    }

    // --- Step 3' : QK-Norm (AttnLRP identity rule) --------------------------------------
    if (use_qk_norm_) {
        const Shape rows({N * H * L, D});
        Tensor rq = q_norm_->propagate_relevance(reshaped(r_q, rows), config);
        Tensor rk = k_norm_->propagate_relevance(reshaped(r_k, rows), config);
        r_q = reshaped(rq, Shape({N, H, L, D}));
        r_k = reshaped(rk, Shape({N, H, L, D}));
    }

    // --- Step 2' : split-heads inverse --------------------------------------------------
    Tensor r_q_flat(Shape({N * L, d_model_}), backend_);
    Tensor r_k_flat(Shape({N * L, d_model_}), backend_);
    Tensor r_v_flat(Shape({N * L, d_model_}), backend_);
    merge_heads(r_q.data(), r_q_flat.data(), N, L, H, D);
    merge_heads(r_k.data(), r_k_flat.data(), N, L, H, D);
    merge_heads(r_v.data(), r_v_flat.data(), N, L, H, D);

    // --- Step 1' : Q/K/V projections ----------------------------------------------------
    // All three projections read the same input, so their input relevances sum -- the same
    // fan-in accumulation the gradient path does.
    Tensor relevance_in = q_proj_.propagate_relevance(r_q_flat, config);
    relevance_in.accumulate(k_proj_.propagate_relevance(r_k_flat, config));
    relevance_in.accumulate(v_proj_.propagate_relevance(r_v_flat, config));

    return reshaped(relevance_in, Shape({N, L, d_model_}));
}

std::vector<ParamRef> MultiHeadAttentionModule::parameters() {
    std::vector<ParamRef> params;
    for (Module* sub : {static_cast<Module*>(&q_proj_), static_cast<Module*>(&k_proj_),
                        static_cast<Module*>(&v_proj_), static_cast<Module*>(&out_proj_)}) {
        std::vector<ParamRef> sub_params = sub->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    if (use_qk_norm_) {
        std::vector<ParamRef> q_params = q_norm_->parameters();
        params.insert(params.end(), q_params.begin(), q_params.end());
        std::vector<ParamRef> k_params = k_norm_->parameters();
        params.insert(params.end(), k_params.begin(), k_params.end());
    }
    return params;
}

void MultiHeadAttentionModule::set_training(bool training) {
    Module::set_training(training);
    q_proj_.set_training(training);
    k_proj_.set_training(training);
    v_proj_.set_training(training);
    out_proj_.set_training(training);
    softmax_.set_training(training);
    if (use_rope_) {
        q_rope_->set_training(training);
        k_rope_->set_training(training);
    }
    if (use_qk_norm_) {
        q_norm_->set_training(training);
        k_norm_->set_training(training);
    }
}

}  // namespace pulsatrix
