/** @file block_sparse_featurizer.hpp
 *  @brief Block-sparse featurizers (FEAT-6; Fel et al., "Structuring Sparsity: Block-Sparse
 *         Featurizers Capture Visual Concept Manifolds", arXiv 2606.25234): features are small
 *         subspaces (blocks of directions) instead of single directions, so a concept that lives
 *         on a low-dimensional manifold, such as an angle or a lighting direction, is one feature.
 *  @ingroup mech_interp
 *  @note Implemented after the paper and its reference code (github.com/goodfire-ai/
 *        block-sparse-featurizer). Where the two differ, the code is followed: a γ per block in
 *        the Grassmannian variant, and a block JumpReLU (not a soft threshold) in the group-lasso
 *        variant. Tournament top-k is after Jerpelea and Ananthram, arXiv 2608.27515.
 */
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "pulsatrix/featurizer.hpp"
#include "pulsatrix/linear_module.hpp"

namespace pulsatrix {

enum class BsfVariant {
    /** @brief A free encoder: `z = Π_k(x W_enc + b_enc)`, decoder atoms of unit norm. */
    Vanilla,
    /** @brief Tied, orthonormal frames: `z_g = γ_g x D_gᵀ` with `D_g D_gᵀ = I`, then Π_k. */
    Grassmannian,
    /** @brief A free encoder and a learned threshold per block: a block fires when its norm is
     *         above θ_g, under an L0 penalty whose weight is set by dual ascent to hit target_l0. */
    GroupLasso,
};

enum class BlockSelection {
    /** @brief Π_k: each input keeps its k blocks of largest norm. */
    TopK,
    /** @brief Tournament top-k (Jerpelea and Ananthram): blocks are taken by norm, but one that
     *         overlaps an already-taken block loses a duel and is skipped; see BlockSparseOptions. */
    Tournament,
};

struct BlockSparseOptions {
    BsfVariant variant = BsfVariant::Vanilla;
    /** @brief b, the directions per block (the paper's main runs use 3). */
    int64_t block_size = 3;
    /** @brief Blocks kept per input (Vanilla, Grassmannian). */
    int64_t k = 16;
    BlockSelection selection = BlockSelection::TopK;
    /** @brief Tournament: a candidate block duels an accepted one when their subspace overlap
     *         `|Q_g Q_hᵀ|²_F / b` exceeds duel_overlap; it loses and is skipped, and the accepted
     *         block gains a win. Above verdict_overlap the verdict is permanent: the block with
     *         more wins keeps the feature and the other is never selected again. */
    double duel_overlap = 0.1;
    double verdict_overlap = 0.5;
    /** @brief Training steps of plain top-k before duels begin. */
    int64_t tournament_warmup = 0;
    /** @brief GroupLasso: the mean number of active blocks per input to hold, by dual ascent on
     *         log λ with step dual_lr (the multiplier starts at initial_lambda). */
    double target_l0 = 16;
    float initial_lambda = 1e-2f;
    float dual_lr = 3e-2f;
    /** @brief GroupLasso: θ_g = softplus(gain · raw_g); the straight-through Gaussian kernel's
     *         bandwidth is this multiple of the block norms' standard deviation (a moving average). */
    float gain = 10.0f;
    float bandwidth = 0.25f;
    uint64_t seed = 0;
};

/**
 * @brief A block-sparse featurizer: G blocks of b directions in a d-dimensional space. Codes are
 *        `(N, G·b)`, block g's in columns `[g·b, (g+1)·b)`, signed; the reconstruction is
 *        `x̂ = z D`, D's rows the atoms. A block is active when its codes aren't all zero, and its
 *        norm says how strongly; its coordinates say where on the concept's subspace the input is.
 *
 * - **No decoder bias,** as in the paper: center the data first (and scale it so the mean
 *   squared norm is d, the reference code's convention).
 * - **Training** minimizes `mean((x̂ - x)²)`, plus `λ · L0` for GroupLasso (`sparsity`). Only
 *   active blocks pass gradient. GroupLasso's gate has the reference code's straight-through
 *   estimator: a Gaussian kernel K at `‖a_g‖ - θ_g` sends `+K` to the block norm and `-K` to θ_g,
 *   from both the reconstruction and the penalty. Its θ starts on the first training batch at
 *   the norm quantile that makes target_l0 blocks fire.
 * - **normalize_decoder()** gives Vanilla and GroupLasso atoms unit norm, and restores the
 *   Grassmannian frames' orthonormality by QR (Gram–Schmidt in row order), as Appendix D of the
 *   paper does; the reference code orthonormalizes by QR inside every forward pass instead.
 * - **Relevance (LRP).** The selection is a fixed gate: relevance reaches the selected blocks'
 *   codes through the decoder's rule and goes back through the encoder's (the Grassmannian γ, a
 *   constant gain per block, passes it unchanged). This is the block top-k rule the roadmap
 *   asked for.
 *
 * A Module from input to reconstruction. Parameters: `encoder.*` and `decoder.weight` (Vanilla,
 * GroupLasso); `decoder.weight` (the frames) and `gamma` (Grassmannian); and `threshold_raw`
 * (GroupLasso). GroupLasso's bandwidth, λ and initialization flag are buffers. The tournament's
 * wins and verdicts are not saved.
 */
class BlockSparseFeaturizer : public Module, public Featurizer {
public:
    /** @throws std::invalid_argument for sizes < 1, k outside [1, num_blocks], target_l0 outside
     *          (0, num_blocks], a non-positive gain or bandwidth, or overlaps outside (0, 1]. */
    BlockSparseFeaturizer(int64_t dim, int64_t num_blocks, DeviceBackend* backend, BlockSparseOptions options = {});

    [[nodiscard]] int64_t num_blocks() const { return G_; }
    [[nodiscard]] const BlockSparseOptions& options() const { return options_; }
    /** @brief Block @p g's frame, `b x d` row-major (its b atoms). */
    [[nodiscard]] std::vector<float> block_frame(int64_t g);
    /** @brief The Grassmannian γ per block (empty for the other variants). */
    [[nodiscard]] std::vector<float> gamma() const;
    /** @brief GroupLasso's thresholds θ per block, and the current λ. */
    [[nodiscard]] std::vector<float> thresholds() const;
    [[nodiscard]] float l0_multiplier() const;
    [[nodiscard]] LinearModule& encoder() { return encoder_; }
    [[nodiscard]] LinearModule& decoder() { return decoder_; }

    // ---- Featurizer ----------------------------------------------------------------------
    [[nodiscard]] int64_t input_dim() const override { return d_; }
    /** @brief The codes per input, `num_blocks · block_size`. */
    [[nodiscard]] int64_t num_features() const override { return G_ * b_; }
    [[nodiscard]] int64_t block_size() const override { return b_; }
    [[nodiscard]] Tensor encode(const Tensor& x) override;
    [[nodiscard]] Tensor decode(const Tensor& codes) override;
    FeaturizerLoss loss_and_backward(const Tensor& x, std::vector<float>* codes = nullptr) override;
    void normalize_decoder() override;
    /** @brief Atom @p i (block i / b, direction i % b), d values. */
    [[nodiscard]] std::vector<float> decoder_direction(int64_t i) override;
    [[nodiscard]] Module& parameters_module() override { return *this; }

    // ---- Module: input to reconstruction -------------------------------------------------
    /** @brief Backward with the selection held fixed. */
    [[nodiscard]] Tensor backward(const Tensor& grad_output) override;
    [[nodiscard]] Tensor propagate_relevance(const Tensor& relevance_out, const LRPRuleConfig& config) override;
    [[nodiscard]] OpType op_type() const override { return OpType::Composite; }
    [[nodiscard]] std::vector<NamedParamRef> named_parameters() override;
    [[nodiscard]] std::vector<NamedBufferRef> named_buffers() override;
    [[nodiscard]] std::optional<DeviceType> compute_device() const override { return backend_->device(); }
    void set_training(bool training) override;
    void release_activations() override;

protected:
    /** @throws std::invalid_argument unless the input is (N, dim), N > 0. */
    [[nodiscard]] Tensor forward_impl(const Tensor& input) override;

private:
    struct Pass {
        std::vector<float> a;      ///< (N, G·b): pre-activations (Grassmannian: x Dᵀ, before γ)
        std::vector<float> z;      ///< (N, G·b): codes
        std::vector<float> gate;   ///< (N, G): 1 for a selected block
        std::vector<float> norm;   ///< (N, G): GroupLasso's ‖a_g‖
    };
    Pass Run(const Tensor& x, bool training);
    void Check(const Tensor& x, const char* who) const;
    /** @brief The Grassmannian encoder is the decoder's transpose; copy it over. */
    void SyncTiedEncoder();
    /** @brief Blocks to keep for one input, by norm, under the selection rule. */
    std::vector<int64_t> Select(const std::vector<float>& norms, const std::vector<std::vector<float>>* frames, bool training);
    /** @brief Orthonormal bases of every block's span, for the tournament. */
    std::vector<std::vector<float>> Frames();
    /** @brief Back through the gate (and γ) into the encoder, and the tied frames' gradient;
     *         returns the input's gradient. */
    Tensor BackwardThroughEncoder(const Pass& p, const std::vector<float>& g_z, double l0_weight, bool straight_through);

    int64_t d_, G_, b_;
    DeviceBackend* backend_;
    BlockSparseOptions options_;
    LinearModule encoder_;  ///< (d -> G·b); Grassmannian: no bias, tied to the decoder
    LinearModule decoder_;  ///< (G·b -> d), no bias
    Tensor gamma_, gamma_grad_;                ///< (G): Grassmannian
    Tensor threshold_raw_, threshold_raw_grad_;  ///< (G): GroupLasso
    Tensor bandwidth_, log_lambda_, initialized_;  ///< (1) buffers: GroupLasso
    int64_t steps_ = 0;
    std::vector<int64_t> wins_;             ///< tournament: duels won per block
    std::map<int64_t, int64_t> lost_to_;    ///< tournament: permanent loser -> winner
    Pass last_;
    int64_t last_rows_ = 0;
};

/**
 * @brief Each block's code geometry over @p x, from the covariance of its codes on the inputs
 *        where it is active: stable rank `‖C‖²_F / ‖C‖²_2`, participation ratio `(tr C)² / ‖C‖²_F`
 *        and effective rank (the exponential of the eigenvalues' entropy). The paper finds a
 *        block's dimension saturates at the concept's (2 to 4 for DINOv3) however large b is.
 */
struct BlockGeometry {
    std::vector<double> firing_rate;        ///< per block
    std::vector<double> stable_rank;        ///< per block; 0 where it fired on fewer than 2 inputs
    std::vector<double> participation_ratio;
    std::vector<double> effective_rank;
    /** @brief Means over blocks that fired on at least 2 inputs, weighted by firing rate. */
    double mean_stable_rank = 0, mean_participation_ratio = 0, mean_effective_rank = 0;
};

/** @throws std::invalid_argument for a batch that isn't (N, input_dim). Works for any featurizer:
 *          a single-direction featurizer's blocks have rank 1. */
[[nodiscard]] BlockGeometry MeasureBlockGeometry(Featurizer& featurizer, const Tensor& x, int64_t batch = 4096);

/**
 * @brief The paper's minimum description length (Appendix C), in bits per input at distortion
 *        `δ = distortion_fraction · (mean per-dimension variance of x)`:
 *        - support: `log2 C(G, ℓ)` for G features (blocks) and ℓ active per input on average;
 *        - code: `Σ_g p_g Σ_j ½ log2(1 + σ²_gj / δ)`, σ²_gj the eigenvalues of block g's code
 *          covariance where it is active and p_g its firing rate;
 *        - residual: `Σ_a ½ log2(1 + λ_a / δ)` over the residual covariance's eigenvalues;
 *        - dictionary: `½ · G·b(d - b) / N · log2 N` (the free parameters of G points on the
 *          Grassmannian Gr(b, d)).
 *        A shorter description at the same distortion is the paper's evidence that blocks fit
 *        the activations better than single directions.
 */
struct DescriptionLength {
    double support = 0, code = 0, residual = 0, dictionary = 0, total = 0;
    double delta = 0;
};

/** @throws std::invalid_argument for a batch that isn't (N, input_dim) with N >= 2, or a
 *          distortion_fraction outside (0, 1). */
[[nodiscard]] DescriptionLength MeasureDescriptionLength(Featurizer& featurizer, const Tensor& x, double distortion_fraction = 0.1,
                                                         int64_t batch = 4096);

}  // namespace pulsatrix
