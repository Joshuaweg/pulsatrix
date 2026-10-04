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
    // Device-generic (GPU-native-kernels Mission 2): projections, permutes, per-head gemms,
    // softmax and scaling all run through DeviceBackend.

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
    // (N, L, H, D) -> (N, H, L, D): the head axis moves in front of L.
    const DeviceType device = input.device();
    const auto n = static_cast<size_t>(N), l = static_cast<size_t>(L), h = static_cast<size_t>(H),
               d = static_cast<size_t>(D);
    Tensor q(Shape({N, H, L, D}), backend_, device);
    Tensor k(Shape({N, H, L, D}), backend_, device);
    Tensor v(Shape({N, H, L, D}), backend_, device);
    backend_->permute_0213(q_flat.data(), q.data(), n, l, h, d);
    backend_->permute_0213(k_flat.data(), k.data(), n, l, h, d);
    backend_->permute_0213(v_flat.data(), v.data(), n, l, h, d);

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
    Tensor scores_raw(Shape({N, H, L, L}), backend_, device);
    Tensor scores(Shape({N, H, L, L}), backend_, device);
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    for (int64_t nh = 0; nh < N * H; ++nh) {
        // K read transposed in place by gemm_ex -- no host transpose copy.
        backend_->gemm_ex(q.data() + nh * L * D, false, k.data() + nh * L * D, true, scores_raw.data() + nh * L * L,
                          l, d, l, 0.0f);
    }
    backend_->axpby(inv_sqrt_d, scores_raw.data(), 0.0f, nullptr, scores.data(), static_cast<size_t>(scores.numel()));

    // --- Step 6: softmax over the last axis ---------------------------------------------
    // SoftmaxModule is rank-agnostic over the last axis -- (N, H, L, L) needs no reshape.
    Tensor attn = softmax_.forward(scores);

    // --- Step 7: context = Attn @ V -----------------------------------------------------
    Tensor context(Shape({N, H, L, D}), backend_, device);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        backend_->gemm(attn.data() + nh * L * L, v.data() + nh * L * D, context.data() + nh * L * D, l, l, d);
    }

    // --- Step 8: merge heads ------------------------------------------------------------
    // (N, H, L, D) -> (N, L, H, D) == (N*L, d_model).
    Tensor merged(Shape({N * L, d_model_}), backend_, device);
    backend_->permute_0213(context.data(), merged.data(), n, h, l, d);

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

    // --- Step 9' : output projection ----------------------------------------------------
    Tensor grad_merged = out_proj_.backward(reshaped(grad_output, Shape({N * L, d_model_})));

    // --- Step 8' : merge heads inverse (pure data movement) -----------------------------
    const DeviceType device = grad_output.device();
    const auto n = static_cast<size_t>(N), l = static_cast<size_t>(L), h = static_cast<size_t>(H),
               d = static_cast<size_t>(D);
    Tensor grad_context(Shape({N, H, L, D}), backend_, device);
    backend_->permute_0213(grad_merged.data(), grad_context.data(), n, l, h, d);

    // --- Step 7' : context = Attn @ V ---------------------------------------------------
    // Standard matmul backward, per (n, h) slice: dA = dC @ V^T, dV = A^T @ dC, with the
    // transposed operand read in place by gemm_ex.
    Tensor grad_attn(Shape({N, H, L, L}), backend_, device);
    Tensor grad_v(Shape({N, H, L, D}), backend_, device);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const float* dc = grad_context.data() + nh * L * D;
        backend_->gemm_ex(dc, false, last_v_.data() + nh * L * D, true, grad_attn.data() + nh * L * L, l, d, l, 0.0f);
        backend_->gemm_ex(last_attn_.data() + nh * L * L, true, dc, false, grad_v.data() + nh * L * D, l, l, d, 0.0f);
    }

    // --- Step 6' : softmax --------------------------------------------------------------
    Tensor grad_scores = softmax_.backward(grad_attn);

    // --- Step 5' : scores = Q @ K^T / sqrt(head_dim) ------------------------------------
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
    Tensor grad_scores_raw(grad_scores.shape(), backend_, device);
    backend_->axpby(inv_sqrt_d, grad_scores.data(), 0.0f, nullptr, grad_scores_raw.data(),
                    static_cast<size_t>(grad_scores.numel()));
    Tensor grad_q(Shape({N, H, L, D}), backend_, device);
    Tensor grad_k(Shape({N, H, L, D}), backend_, device);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const float* ds = grad_scores_raw.data() + nh * L * L;
        // O[i,k] = sum_j Q[i,j] K[k,j]  ->  dQ = dO @ K, dK = dO^T @ Q.
        backend_->gemm(ds, last_k_.data() + nh * L * D, grad_q.data() + nh * L * D, l, l, d);
        backend_->gemm_ex(ds, true, last_q_.data() + nh * L * D, false, grad_k.data() + nh * L * D, l, l, d, 0.0f);
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
    Tensor grad_q_flat(Shape({N * L, d_model_}), backend_, device);
    Tensor grad_k_flat(Shape({N * L, d_model_}), backend_, device);
    Tensor grad_v_flat(Shape({N * L, d_model_}), backend_, device);
    backend_->permute_0213(grad_q.data(), grad_q_flat.data(), n, h, l, d);
    backend_->permute_0213(grad_k.data(), grad_k_flat.data(), n, h, l, d);
    backend_->permute_0213(grad_v.data(), grad_v_flat.data(), n, h, l, d);

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
    // Device-generic (GPU-native-kernels Mission 3): permutes for head split/merge; both
    // bilinear products via lrp_bilinear_matmul (one GPU thread per relevance element, each
    // summed in the original loop's order); K read transposed in place.
    const DeviceType device = relevance_out.device();
    const auto n = static_cast<size_t>(N), l = static_cast<size_t>(L), h = static_cast<size_t>(H),
               d = static_cast<size_t>(D);

    Tensor r_merged = out_proj_.propagate_relevance(reshaped(relevance_out, Shape({N * L, d_model_})), config);
    Tensor r_context(Shape({N, H, L, D}), backend_, device);
    backend_->permute_0213(r_merged.data(), r_context.data(), n, l, h, d);

    // context = Attn @ V
    Tensor r_attn(Shape({N, H, L, L}), backend_, device);
    Tensor r_v(Shape({N, H, L, D}), backend_, device);
    backend_->lrp_bilinear_matmul(last_attn_.data(), last_v_.data(), last_context_.data(), r_context.data(),
                                  r_attn.data(), r_v.data(), n * h, l, l, d, config.epsilon, /*b_transposed=*/false);

    Tensor r_scores = softmax_.propagate_relevance(r_attn, config);

    // scores_raw = Q @ K^T, K stored (L, D) == K^T read transposed; r_k comes back in K's layout.
    Tensor r_q(Shape({N, H, L, D}), backend_, device);
    Tensor r_k(Shape({N, H, L, D}), backend_, device);
    backend_->lrp_bilinear_matmul(last_q_.data(), last_k_.data(), last_scores_raw_.data(), r_scores.data(),
                                  r_q.data(), r_k.data(), n * h, l, d, l, config.epsilon, /*b_transposed=*/true);

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
    Tensor r_q_flat(Shape({N * L, d_model_}), backend_, device);
    Tensor r_k_flat(Shape({N * L, d_model_}), backend_, device);
    Tensor r_v_flat(Shape({N * L, d_model_}), backend_, device);
    backend_->permute_0213(r_q.data(), r_q_flat.data(), n, h, l, d);
    backend_->permute_0213(r_k.data(), r_k_flat.data(), n, h, l, d);
    backend_->permute_0213(r_v.data(), r_v_flat.data(), n, h, l, d);

    // --- Step 1' : Q/K/V projections ----------------------------------------------------
    // All three projections read the same input, so their input relevances sum -- the same
    // fan-in accumulation the gradient path does.
    Tensor relevance_in = q_proj_.propagate_relevance(r_q_flat, config);
    relevance_in.accumulate(k_proj_.propagate_relevance(r_k_flat, config));
    relevance_in.accumulate(v_proj_.propagate_relevance(r_v_flat, config));

    return reshaped(relevance_in, Shape({N, L, d_model_}));
}

std::vector<NamedParamRef> MultiHeadAttentionModule::named_parameters() {
    std::vector<NamedParamRef> params;
    append_named_parameters(params, "q_proj", q_proj_);
    append_named_parameters(params, "k_proj", k_proj_);
    append_named_parameters(params, "v_proj", v_proj_);
    append_named_parameters(params, "out_proj", out_proj_);
    if (use_qk_norm_) {
        append_named_parameters(params, "q_norm", *q_norm_);
        append_named_parameters(params, "k_norm", *k_norm_);
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
