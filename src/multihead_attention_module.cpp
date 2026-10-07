#include "pulsatrix/multihead_attention_module.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "pulsatrix/assert.hpp"

namespace pulsatrix {
namespace {

/**
 * @brief The config with its 0 defaults filled in, or a throw naming the problem. Runs in the
 *        member initializer list, before any sub-module is built against its sizes.
 */
[[nodiscard]] AttentionConfig resolve(AttentionConfig c) {
    // External boundary (constructor arguments can originate from Phase 5's Python bindings
    // with no upstream validation), same convention as every other module's constructor.
    if (c.d_model <= 0) {
        throw std::invalid_argument("MultiHeadAttentionModule: d_model must be positive");
    }
    if (c.num_heads <= 0) {
        throw std::invalid_argument("MultiHeadAttentionModule: num_heads must be positive");
    }
    if (c.num_kv_heads < 0 || c.head_dim < 0) {
        throw std::invalid_argument("MultiHeadAttentionModule: num_kv_heads and head_dim must not be negative");
    }
    if (c.num_kv_heads == 0) {
        c.num_kv_heads = c.num_heads;
    }
    if (c.num_heads % c.num_kv_heads != 0) {
        throw std::invalid_argument("MultiHeadAttentionModule: num_heads must be a multiple of num_kv_heads");
    }
    if (c.head_dim == 0) {
        if (c.d_model % c.num_heads != 0) {
            throw std::invalid_argument("MultiHeadAttentionModule: d_model must be divisible by num_heads");
        }
        c.head_dim = c.d_model / c.num_heads;
    }
    // Checked here rather than left to RoPEModule's own constructor so the message names the
    // real cause (the caller chose d_model/num_heads, not head_dim directly).
    if (c.use_rope && c.head_dim % 2 != 0) {
        throw std::invalid_argument(
            "MultiHeadAttentionModule: use_rope requires an even head_dim (d_model / num_heads)");
    }
    return c;
}

[[nodiscard]] AttentionConfig legacy_config(int64_t d_model, int64_t num_heads, bool use_rope, bool use_qk_norm) {
    AttentionConfig c;
    c.d_model = d_model;
    c.num_heads = num_heads;
    c.use_rope = use_rope;
    c.use_qk_norm = use_qk_norm;
    return c;
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
    : MultiHeadAttentionModule(legacy_config(d_model, num_heads, use_rope, use_qk_norm), backend) {}

MultiHeadAttentionModule::MultiHeadAttentionModule(const AttentionConfig& config, DeviceBackend* backend)
    : config_(resolve(config)),
      d_model_(config_.d_model),
      num_heads_(config_.num_heads),
      num_kv_heads_(config_.num_kv_heads),
      head_dim_(config_.head_dim),
      use_rope_(config_.use_rope),
      use_qk_norm_(config_.use_qk_norm),
      backend_(backend),
      q_proj_(d_model_, num_heads_ * head_dim_, backend, config_.qkv_bias),
      k_proj_(d_model_, num_kv_heads_ * head_dim_, backend, config_.qkv_bias),
      v_proj_(d_model_, num_kv_heads_ * head_dim_, backend, config_.qkv_bias),
      out_proj_(num_heads_ * head_dim_, d_model_, backend, config_.out_bias),
      softmax_(backend),
      key_keep_(Shape({0}), backend),
      last_key_keep_(Shape({0}), backend),
      last_q_(Shape({0}), backend),
      last_k_(Shape({0}), backend),
      last_v_(Shape({0}), backend),
      last_scores_raw_(Shape({0}), backend),
      last_attn_(Shape({0}), backend),
      last_context_(Shape({0}), backend),
      last_key_relevance_(Shape({0}), backend),
      last_value_relevance_(Shape({0}), backend) {
    if (use_rope_) {
        // Separate instances for Q and K -- see the header's caching note; a shared instance
        // would leave only K's activations cached for propagate_relevance.
        if (config_.rope_inverse_frequencies.empty()) {
            q_rope_ = std::make_unique<RoPEModule>(head_dim_, backend, config_.rope_base, config_.rope_layout);
            k_rope_ = std::make_unique<RoPEModule>(head_dim_, backend, config_.rope_base, config_.rope_layout);
        } else {
            q_rope_ = std::make_unique<RoPEModule>(head_dim_, backend, config_.rope_inverse_frequencies, config_.rope_layout);
            k_rope_ = std::make_unique<RoPEModule>(head_dim_, backend, config_.rope_inverse_frequencies, config_.rope_layout);
        }
    }
    if (use_qk_norm_) {
        q_norm_ = std::make_unique<RMSNormModule>(head_dim_, backend, backend->device(), config_.qk_norm_eps);
        k_norm_ = std::make_unique<RMSNormModule>(head_dim_, backend, backend->device(), config_.qk_norm_eps);
        q_norm_->set_weight_offset(config_.qk_norm_weight_offset);
        k_norm_->set_weight_offset(config_.qk_norm_weight_offset);
        // RMSNormModule zero-initializes gamma; a zero gamma here would annihilate Q and K
        // and make attention uniform regardless of the input. See the header's note.
        const std::vector<float> ones(static_cast<size_t>(head_dim_), 1.0f);
        q_norm_->set_gamma(ones);
        k_norm_->set_gamma(ones);
    }
}

void MultiHeadAttentionModule::set_key_padding_mask(const Tensor& key_keep) {
    if (key_keep.rank() != 2) {
        throw std::invalid_argument("MultiHeadAttentionModule::set_key_padding_mask: key_keep must be rank-2 (N, L)");
    }
    // Copied through the host so a mask built on the CPU works with a GPU module.
    key_keep_ = Tensor(key_keep.shape(), backend_, key_keep.to_host_vector());
    has_key_keep_ = true;
}

void MultiHeadAttentionModule::clear_key_padding_mask() {
    key_keep_ = Tensor(Shape({0}), backend_);
    has_key_keep_ = false;
}

void MultiHeadAttentionModule::set_position_offset(int64_t offset) {
    if (offset < 0) {
        throw std::invalid_argument("MultiHeadAttentionModule::set_position_offset: offset must be non-negative");
    }
    position_offset_ = offset;
    if (use_rope_) {
        q_rope_->set_position_offset(offset);
        k_rope_->set_position_offset(offset);
    }
}

void MultiHeadAttentionModule::repeat_kv(Tensor& t) const {
    const int64_t group = num_heads_ / num_kv_heads_;
    if (group == 1) {
        return;
    }
    const Tensor& src = t;
    const int64_t N = src.shape().dim(0);
    const int64_t L = src.shape().dim(2);
    const auto slice = static_cast<size_t>(L * head_dim_);
    Tensor out(Shape({N, num_heads_, L, head_dim_}), backend_, src.device());
    // Head h reads K/V head h / group. Row r = (n, kv head) of src is copied to rows
    // r * group + g of out, so one strided copy per g covers every (n, kv head).
    for (int64_t g = 0; g < group; ++g) {
        backend_->copy_2d(out.data() + g * static_cast<int64_t>(slice), slice * static_cast<size_t>(group),
                          src.data(), slice, static_cast<size_t>(N * num_kv_heads_), slice);
    }
    t = std::move(out);
}

void MultiHeadAttentionModule::sum_kv_groups(Tensor& t) const {
    const int64_t group = num_heads_ / num_kv_heads_;
    if (group == 1) {
        return;
    }
    const Tensor& src = t;
    const int64_t N = src.shape().dim(0);
    const int64_t L = src.shape().dim(2);
    const auto slice = static_cast<size_t>(L * head_dim_);
    Tensor out(Shape({N, num_kv_heads_, L, head_dim_}), backend_, src.device());
    for (int64_t r = 0; r < N * num_kv_heads_; ++r) {
        backend_->accumulate_rows(src.data() + r * group * static_cast<int64_t>(slice),
                                  out.data() + r * static_cast<int64_t>(slice), static_cast<size_t>(group), slice);
    }
    t = std::move(out);
}

float MultiHeadAttentionModule::score_scale() const {
    return config_.score_scale > 0.0f ? config_.score_scale : 1.0f / std::sqrt(static_cast<float>(head_dim_));
}

void MultiHeadAttentionModule::fill_masked(Tensor& scores, float value) const {
    if (!config_.causal && !last_has_key_keep_) {
        return;
    }
    const auto n = static_cast<size_t>(last_N_), l = static_cast<size_t>(last_L_);
    backend_->attention_mask_fill(scores.data(), last_has_key_keep_ ? last_key_keep_.data() : nullptr, n,
                                  static_cast<size_t>(num_heads_), l, l, config_.causal, /*q_offset=*/0,
                                  static_cast<size_t>(config_.sliding_window), value);
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
    const int64_t Hkv = num_kv_heads_;
    const int64_t D = head_dim_;
    if (has_key_keep_ && (key_keep_.shape().dim(0) != N || key_keep_.shape().dim(1) != L)) {
        throw std::invalid_argument(
            "MultiHeadAttentionModule::forward: the key padding mask must be (N, L) for an (N, L, d_model) input");
    }

    // --- Step 1: Q/K/V projections ------------------------------------------------------
    // LinearModule takes rank-2 (batch, features); (N, L, d_model) is the same row-major
    // buffer as (N*L, d_model), so this is a pure re-view.
    const Tensor flat_input = reshaped(input, Shape({N * L, d_model_}));
    Tensor q_flat = q_proj_.forward(flat_input);
    Tensor k_flat = k_proj_.forward(flat_input);
    Tensor v_flat = v_proj_.forward(flat_input);

    // --- Step 2: split heads ------------------------------------------------------------
    // (N, L, H, D) -> (N, H, L, D): the head axis moves in front of L. K and V have Hkv heads.
    const DeviceType device = input.device();
    const auto n = static_cast<size_t>(N), l = static_cast<size_t>(L), h = static_cast<size_t>(H),
               hkv = static_cast<size_t>(Hkv), d = static_cast<size_t>(D);
    Tensor q(Shape({N, H, L, D}), backend_, device);
    Tensor k(Shape({N, Hkv, L, D}), backend_, device);
    Tensor v(Shape({N, Hkv, L, D}), backend_, device);
    backend_->permute_0213(q_flat.data(), q.data(), n, l, h, d);
    backend_->permute_0213(k_flat.data(), k.data(), n, l, hkv, d);
    backend_->permute_0213(v_flat.data(), v.data(), n, l, hkv, d);

    // --- Step 3: QK-Norm (optional) -----------------------------------------------------
    // (N, H, L, D) -> (N*H*L, D) is a pure reshape: the normalized axis is already last.
    if (use_qk_norm_) {
        Tensor q_norm_out = q_norm_->forward(reshaped(q, Shape({N * H * L, D})));
        Tensor k_norm_out = k_norm_->forward(reshaped(k, Shape({N * Hkv * L, D})));
        q = reshaped(q_norm_out, Shape({N, H, L, D}));
        k = reshaped(k_norm_out, Shape({N, Hkv, L, D}));
    }

    // --- Step 4: RoPE (optional) --------------------------------------------------------
    // RoPEModule is already rank-agnostic over (..., L, head_dim) -- no reshape needed.
    if (use_rope_) {
        q = q_rope_->forward(q);
        k = k_rope_->forward(k);
    }

    // --- Step 4b: GQA -- repeat each K/V head across its group of query heads -----------
    repeat_kv(k);
    repeat_kv(v);

    // The masks this pass uses, cached first so fill_masked() reads them here and in the
    // backward and relevance passes alike.
    last_N_ = N;
    last_L_ = L;
    last_has_key_keep_ = has_key_keep_;
    last_key_keep_ = has_key_keep_ ? key_keep_ : Tensor(Shape({0}), backend_);

    // --- Step 5: scores = Q @ K^T / sqrt(head_dim) --------------------------------------
    // No batched-gemm primitive exists; loop the N*H independent 2D slices explicitly. Each
    // slice is a contiguous span of the row-major buffer, so the slice pointer can go
    // straight into gemm without a gather.
    Tensor scores_raw(Shape({N, H, L, L}), backend_, device);
    Tensor scores(Shape({N, H, L, L}), backend_, device);
    const float inv_sqrt_d = score_scale();
    for (int64_t nh = 0; nh < N * H; ++nh) {
        // K read transposed in place by gemm_ex -- no host transpose copy.
        backend_->gemm_ex(q.data() + nh * L * D, false, k.data() + nh * L * D, true, scores_raw.data() + nh * L * L,
                          l, d, l, 0.0f);
    }
    backend_->axpby(inv_sqrt_d, scores_raw.data(), 0.0f, nullptr, scores.data(), static_cast<size_t>(scores.numel()));
    fill_masked(scores, std::numeric_limits<float>::lowest());

    // --- Step 6: softmax over the last axis ---------------------------------------------
    // SoftmaxModule is rank-agnostic over the last axis -- (N, H, L, L) needs no reshape.
    Tensor attn = softmax_.forward(scores);

    // --- Step 7: context = Attn @ V -----------------------------------------------------
    Tensor context(Shape({N, H, L, D}), backend_, device);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        backend_->gemm(attn.data() + nh * L * L, v.data() + nh * L * D, context.data() + nh * L * D, l, l, d);
    }

    // --- Step 8: merge heads ------------------------------------------------------------
    // (N, H, L, D) -> (N, L, H, D) == (N*L, H*D).
    Tensor merged(Shape({N * L, H * D}), backend_, device);
    backend_->permute_0213(context.data(), merged.data(), n, h, l, d);

    // --- Step 9: output projection ------------------------------------------------------
    Tensor out_flat = out_proj_.forward(merged);

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
    require_device(grad_output, *compute_device(), "MultiHeadAttentionModule::backward");
    if (!has_forwarded_) {
        throw std::logic_error("MultiHeadAttentionModule::backward: called before any forward()");
    }
    const int64_t N = last_N_;
    const int64_t L = last_L_;
    const int64_t H = num_heads_;
    const int64_t Hkv = num_kv_heads_;
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
               hkv = static_cast<size_t>(Hkv), d = static_cast<size_t>(D);
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
    // A masked score is a constant, so it passes no gradient on to Q and K.
    fill_masked(grad_scores, 0.0f);

    // --- Step 5' : scores = Q @ K^T / sqrt(head_dim) ------------------------------------
    const float inv_sqrt_d = score_scale();
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

    // --- Step 4b' : GQA -- each shared K/V head sums its group's gradients -------------
    sum_kv_groups(grad_k);
    sum_kv_groups(grad_v);

    // --- Step 4' : RoPE -----------------------------------------------------------------
    if (use_rope_) {
        grad_q = q_rope_->backward(grad_q);
        grad_k = k_rope_->backward(grad_k);
    }

    // --- Step 3' : QK-Norm --------------------------------------------------------------
    if (use_qk_norm_) {
        Tensor gq = q_norm_->backward(reshaped(grad_q, Shape({N * H * L, D})));
        Tensor gk = k_norm_->backward(reshaped(grad_k, Shape({N * Hkv * L, D})));
        grad_q = reshaped(gq, Shape({N, H, L, D}));
        grad_k = reshaped(gk, Shape({N, Hkv, L, D}));
    }

    // --- Step 2' : split-heads inverse --------------------------------------------------
    Tensor grad_q_flat(Shape({N * L, H * D}), backend_, device);
    Tensor grad_k_flat(Shape({N * L, Hkv * D}), backend_, device);
    Tensor grad_v_flat(Shape({N * L, Hkv * D}), backend_, device);
    backend_->permute_0213(grad_q.data(), grad_q_flat.data(), n, h, l, d);
    backend_->permute_0213(grad_k.data(), grad_k_flat.data(), n, hkv, l, d);
    backend_->permute_0213(grad_v.data(), grad_v_flat.data(), n, hkv, l, d);

    // --- Step 1' : Q/K/V projections ----------------------------------------------------
    // All three read the same input tensor, so the three input gradients sum.
    Tensor grad_input = q_proj_.backward(grad_q_flat);
    grad_input.accumulate(k_proj_.backward(grad_k_flat));
    grad_input.accumulate(v_proj_.backward(grad_v_flat));

    return reshaped(grad_input, Shape({N, L, d_model_}));
}

Tensor MultiHeadAttentionModule::propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) {
    require_device(relevance_out, *compute_device(), "MultiHeadAttentionModule::propagate_relevance");
    if (!has_forwarded_) {
        throw std::logic_error("MultiHeadAttentionModule::propagate_relevance: called before any forward()");
    }
    const int64_t N = last_N_;
    const int64_t L = last_L_;
    const int64_t H = num_heads_;
    const int64_t Hkv = num_kv_heads_;
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
               hkv = static_cast<size_t>(Hkv), d = static_cast<size_t>(D);

    Tensor r_merged = out_proj_.propagate_relevance(reshaped(relevance_out, Shape({N * L, d_model_})), config);
    Tensor r_context(Shape({N, H, L, D}), backend_, device);
    backend_->permute_0213(r_merged.data(), r_context.data(), n, l, h, d);

    // context = Attn @ V
    Tensor r_attn(Shape({N, H, L, L}), backend_, device);
    Tensor r_v(Shape({N, H, L, D}), backend_, device);
    backend_->lrp_bilinear_matmul(last_attn_.data(), last_v_.data(), last_context_.data(), r_context.data(),
                                  r_attn.data(), r_v.data(), n * h, l, l, d, config.epsilon, /*b_transposed=*/false);

    Tensor r_scores = softmax_.propagate_relevance(r_attn, config);
    // A masked score is a constant (the fill value), not a product of Q and K: no relevance
    // flows through it. Its attention weight is 0, so this only matters for fully masked rows.
    fill_masked(r_scores, 0.0f);

    // scores_raw = Q @ K^T, K stored (L, D) == K^T read transposed; r_k comes back in K's layout.
    Tensor r_q(Shape({N, H, L, D}), backend_, device);
    Tensor r_k(Shape({N, H, L, D}), backend_, device);
    backend_->lrp_bilinear_matmul(last_q_.data(), last_k_.data(), last_scores_raw_.data(), r_scores.data(),
                                  r_q.data(), r_k.data(), n * h, l, d, l, config.epsilon, /*b_transposed=*/true);

    // GQA: a shared K/V head's relevance is the sum over the query heads that read it.
    sum_kv_groups(r_k);
    sum_kv_groups(r_v);

    if (use_rope_) {
        r_q = q_rope_->propagate_relevance(r_q, config);
        r_k = k_rope_->propagate_relevance(r_k, config);
    }

    // --- Step 3' : QK-Norm (AttnLRP identity rule) --------------------------------------
    if (use_qk_norm_) {
        Tensor rq = q_norm_->propagate_relevance(reshaped(r_q, Shape({N * H * L, D})), config);
        Tensor rk = k_norm_->propagate_relevance(reshaped(r_k, Shape({N * Hkv * L, D})), config);
        r_q = reshaped(rq, Shape({N, H, L, D}));
        r_k = reshaped(rk, Shape({N, Hkv, L, D}));
    }

    // --- Step 2' : split-heads inverse --------------------------------------------------
    Tensor r_q_flat(Shape({N * L, H * D}), backend_, device);
    Tensor r_k_flat(Shape({N * L, Hkv * D}), backend_, device);
    Tensor r_v_flat(Shape({N * L, Hkv * D}), backend_, device);
    backend_->permute_0213(r_q.data(), r_q_flat.data(), n, h, l, d);
    backend_->permute_0213(r_k.data(), r_k_flat.data(), n, hkv, l, d);
    backend_->permute_0213(r_v.data(), r_v_flat.data(), n, hkv, l, d);

    // --- Step 1' : Q/K/V projections ----------------------------------------------------
    // All three projections read the same input, so their input relevances sum -- the same
    // fan-in accumulation the gradient path does.
    Tensor relevance_in = q_proj_.propagate_relevance(r_q_flat, config);
    relevance_in.accumulate(k_proj_.propagate_relevance(r_k_flat, config));
    relevance_in.accumulate(v_proj_.propagate_relevance(r_v_flat, config));
    last_key_relevance_ = std::move(r_k_flat);
    last_value_relevance_ = std::move(r_v_flat);
    has_relevance_ = true;

    return reshaped(relevance_in, Shape({N, L, d_model_}));
}

const Tensor& MultiHeadAttentionModule::key_relevance() const {
    if (!has_relevance_) throw std::logic_error("MultiHeadAttentionModule::key_relevance: no propagate_relevance() yet");
    return last_key_relevance_;
}

const Tensor& MultiHeadAttentionModule::value_relevance() const {
    if (!has_relevance_) throw std::logic_error("MultiHeadAttentionModule::value_relevance: no propagate_relevance() yet");
    return last_value_relevance_;
}

KVCache MultiHeadAttentionModule::MakeKVCache(int64_t batch, int64_t max_length) const {
    return KVCache(batch, num_kv_heads_, head_dim_, max_length, backend_);
}

Tensor MultiHeadAttentionModule::forward_cached(const Tensor& input, KVCache& cache) {
    require_device(input, *compute_device(), "MultiHeadAttentionModule::forward_cached");
    if (input.rank() != 3 || input.shape().dim(2) != d_model_) {
        throw std::invalid_argument("MultiHeadAttentionModule::forward_cached: input must be rank-3 (N, L, d_model)");
    }
    const int64_t N = input.shape().dim(0);
    const int64_t L = input.shape().dim(1);
    const int64_t H = num_heads_;
    const int64_t Hkv = num_kv_heads_;
    const int64_t D = head_dim_;
    if (cache.batch() != N || cache.num_kv_heads() != Hkv || cache.head_dim() != D) {
        throw std::invalid_argument("MultiHeadAttentionModule::forward_cached: the cache doesn't fit this layer and input");
    }
    const int64_t past = cache.length();
    const int64_t total = past + L;
    const int64_t max_len = cache.max_length();
    if (total > max_len) {
        throw std::invalid_argument("MultiHeadAttentionModule::forward_cached: " + std::to_string(total) +
                                    " positions don't fit in a cache of " + std::to_string(max_len));
    }
    if (has_key_keep_ && (key_keep_.shape().dim(0) != N || key_keep_.shape().dim(1) != total)) {
        throw std::invalid_argument(
            "MultiHeadAttentionModule::forward_cached: the key padding mask must be (N, cached + new positions)");
    }
    has_forwarded_ = false;  // nothing below is kept for backward or relevance

    const Tensor flat_input = reshaped(input, Shape({N * L, d_model_}));
    Tensor q_flat = q_proj_.forward(flat_input);
    Tensor k_flat = k_proj_.forward(flat_input);
    Tensor v_flat = v_proj_.forward(flat_input);
    const DeviceType device = input.device();
    const auto n = static_cast<size_t>(N), l = static_cast<size_t>(L), h = static_cast<size_t>(H),
               hkv = static_cast<size_t>(Hkv), d = static_cast<size_t>(D), t = static_cast<size_t>(total);
    Tensor q(Shape({N, H, L, D}), backend_, device);
    Tensor k(Shape({N, Hkv, L, D}), backend_, device);
    Tensor v(Shape({N, Hkv, L, D}), backend_, device);
    backend_->permute_0213(q_flat.data(), q.data(), n, l, h, d);
    backend_->permute_0213(k_flat.data(), k.data(), n, l, hkv, d);
    backend_->permute_0213(v_flat.data(), v.data(), n, l, hkv, d);
    if (use_qk_norm_) {
        Tensor q_norm_out = q_norm_->forward(reshaped(q, Shape({N * H * L, D})));
        Tensor k_norm_out = k_norm_->forward(reshaped(k, Shape({N * Hkv * L, D})));
        q = reshaped(q_norm_out, Shape({N, H, L, D}));
        k = reshaped(k_norm_out, Shape({N, Hkv, L, D}));
    }
    if (use_rope_) {
        q_rope_->set_position_offset(position_offset_ + past);
        k_rope_->set_position_offset(position_offset_ + past);
        q = q_rope_->forward(q);
        k = k_rope_->forward(k);
        q_rope_->set_position_offset(position_offset_);
        k_rope_->set_position_offset(position_offset_);
    }

    // Append the new keys and values: each (n, kv head) slice of the cache is contiguous, so one
    // strided copy writes all of them at positions past .. total - 1.
    const auto slice = static_cast<size_t>(max_len) * d;
    backend_->copy_2d(cache.keys().data() + past * D, slice, k.data(), l * d, n * hkv, l * d);
    backend_->copy_2d(cache.values().data() + past * D, slice, v.data(), l * d, n * hkv, l * d);
    cache.advance(L);

    // Scores against every cached position; query head hh reads K/V head hh / group in place.
    const int64_t group = H / Hkv;
    Tensor scores(Shape({N, H, L, total}), backend_, device);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t kv = (nh / H) * Hkv + (nh % H) / group;
        backend_->gemm_ex(q.data() + nh * L * D, false, cache.keys().data() + kv * max_len * D, true,
                          scores.data() + nh * L * total, l, d, t, 0.0f);
    }
    const float inv_sqrt_d = score_scale();
    backend_->axpby(inv_sqrt_d, scores.data(), 0.0f, nullptr, scores.data(), static_cast<size_t>(scores.numel()));
    if (config_.causal || has_key_keep_) {
        backend_->attention_mask_fill(scores.data(), has_key_keep_ ? key_keep_.data() : nullptr, n, h, l, t,
                                      config_.causal, static_cast<size_t>(past), static_cast<size_t>(config_.sliding_window),
                                      std::numeric_limits<float>::lowest());
    }
    backend_->softmax_rows(scores.data(), scores.data(), n * h * l, t);
    Tensor context(Shape({N, H, L, D}), backend_, device);
    for (int64_t nh = 0; nh < N * H; ++nh) {
        const int64_t kv = (nh / H) * Hkv + (nh % H) / group;
        backend_->gemm(scores.data() + nh * L * total, cache.values().data() + kv * max_len * D,
                       context.data() + nh * L * D, l, t, d);
    }
    Tensor merged(Shape({N * L, H * D}), backend_, device);
    backend_->permute_0213(context.data(), merged.data(), n, h, l, d);
    return reshaped(out_proj_.forward(merged), Shape({N, L, d_model_}));
}

std::vector<NamedBufferRef> MultiHeadAttentionModule::named_buffers() {
    std::vector<NamedBufferRef> result;
    append_named_buffers(result, "q_proj", q_proj_);
    append_named_buffers(result, "k_proj", k_proj_);
    append_named_buffers(result, "v_proj", v_proj_);
    append_named_buffers(result, "out_proj", out_proj_);
    if (use_qk_norm_) {
        append_named_buffers(result, "q_norm", *q_norm_);
        append_named_buffers(result, "k_norm", *k_norm_);
    }
    return result;
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
