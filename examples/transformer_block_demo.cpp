/** @file transformer_block_demo.cpp
 *  @brief Standalone demo: builds a small TransformerBlock, runs a forward pass over a
 *         toy 2-position sequence, and prints its attention weights and LRP relevance
 *         attribution -- Phase 6 Phase 3's literal exit-gate deliverable, shown running
 *         end to end.
 *  @note Not a test -- tests/transformer_block_test.cpp (including its
 *        ValidatesAgainstIndependentAttnLRPReference case) is the real acceptance
 *        criterion. This exists so a human can watch a real multi-head-attention +
 *        SwiGLU + residual block run, not just read test assertions.
 */
#include <cstdio>

#include "exai/cpu_backend.hpp"
#include "exai/lrp_rule_config.hpp"
#include "exai/tensor.hpp"
#include "exai/transformer_block.hpp"

int main() {
    using namespace exai;

    CPUBackend backend;

    // d_model=4, num_heads=2 (head_dim=2), d_ff=6, RoPE on, QK-Norm off -- a small but
    // representative block: every sub-mechanism (RMSNorm x2, RoPE, multi-head scaled
    // dot-product attention, SwiGLU) actually engages.
    TransformerBlock block(4, 2, 6, &backend);
    block.norm1().set_gamma({1.0f, 1.0f, 1.0f, 1.0f});
    block.mha().q_proj().set_weight({0.3f, -0.2f, 0.1f, 0.4f, -0.1f, 0.2f, 0.3f, -0.4f, 0.2f, 0.1f, -0.3f, 0.4f,
                                      -0.2f, 0.3f, 0.1f, -0.1f});
    block.mha().k_proj().set_weight({0.2f, 0.1f, -0.3f, 0.4f, 0.1f, -0.2f, 0.3f, 0.2f, -0.1f, 0.4f, 0.2f, -0.3f,
                                      0.3f, -0.1f, 0.4f, 0.2f});
    block.mha().v_proj().set_weight({0.4f, -0.1f, 0.2f, 0.3f, -0.2f, 0.3f, 0.1f, -0.4f, 0.3f, 0.2f, -0.1f, 0.4f,
                                      0.1f, -0.3f, 0.2f, 0.4f});
    block.mha().out_proj().set_weight({0.5f, -0.2f, 0.3f, 0.1f, -0.3f, 0.4f, 0.2f, -0.1f, 0.1f, 0.3f, -0.4f, 0.2f,
                                        -0.2f, 0.1f, 0.3f, 0.4f});
    block.norm2().set_gamma({1.0f, 1.0f, 1.0f, 1.0f});
    block.swiglu().gate_proj().set_weight({0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.2f, 0.3f, -0.3f,
                                            0.4f, -0.1f, 0.4f, 0.2f, -0.3f, 0.5f, 0.1f, -0.2f, 0.3f, 0.4f, -0.2f,
                                            0.1f, 0.2f});
    block.swiglu().up_proj().set_weight({-0.1f, 0.4f, 0.2f, -0.3f, 0.5f, 0.1f, -0.2f, 0.3f, 0.4f, -0.2f, 0.1f, 0.2f,
                                          0.3f, -0.2f, 0.5f, 0.1f, -0.4f, 0.2f, 0.6f, -0.1f, 0.2f, 0.3f, -0.3f, 0.4f});
    block.swiglu().down_proj().set_weight({0.6f, -0.2f, 0.3f, 0.1f, -0.5f, 0.4f, 0.2f, -0.3f, 0.1f, 0.3f, -0.4f, 0.2f,
                                            0.4f, -0.1f, 0.3f, 0.2f, -0.2f, 0.3f, 0.1f, -0.4f, 0.2f, -0.3f, 0.4f, 0.1f});

    Tensor x(Shape({1, 2, 4}), &backend, {0.6f, -0.9f, 1.1f, -0.7f, 0.4f, -0.7f, 0.8f, 0.6f});
    Tensor out = block.forward(x);

    std::printf("TransformerBlock demo -- d_model=4, num_heads=2, d_ff=6, RoPE on\n\n");
    std::printf("input  (position 0): [%.4f, %.4f, %.4f, %.4f]\n", x.data()[0], x.data()[1], x.data()[2],
                x.data()[3]);
    std::printf("input  (position 1): [%.4f, %.4f, %.4f, %.4f]\n", x.data()[4], x.data()[5], x.data()[6],
                x.data()[7]);
    std::printf("output (position 0): [%.4f, %.4f, %.4f, %.4f]\n", out.data()[0], out.data()[1], out.data()[2],
                out.data()[3]);
    std::printf("output (position 1): [%.4f, %.4f, %.4f, %.4f]\n\n", out.data()[4], out.data()[5], out.data()[6],
                out.data()[7]);

    const Tensor& attn = block.mha().last_attention_weights();  // (N, num_heads, L, L)
    std::printf("attention weights (head 0): [[%.4f, %.4f], [%.4f, %.4f]]\n", attn.data()[0], attn.data()[1],
                attn.data()[2], attn.data()[3]);
    std::printf("attention weights (head 1): [[%.4f, %.4f], [%.4f, %.4f]]\n\n", attn.data()[4], attn.data()[5],
                attn.data()[6], attn.data()[7]);

    // Seed relevance at the output equal to the output itself (this codebase's standard
    // "explain what produced this exact activation" convention) and propagate back to the
    // input -- the module's real LRP rule, composed from RMSNorm/attention/SwiGLU/residual.
    Tensor relevance_in = block.propagate_relevance(out, LRPRuleConfig{});
    std::printf("relevance (position 0): [%.4f, %.4f, %.4f, %.4f]\n", relevance_in.data()[0], relevance_in.data()[1],
                relevance_in.data()[2], relevance_in.data()[3]);
    std::printf("relevance (position 1): [%.4f, %.4f, %.4f, %.4f]\n\n", relevance_in.data()[4],
                relevance_in.data()[5], relevance_in.data()[6], relevance_in.data()[7]);

    float sum_out = 0.0f;
    float sum_in = 0.0f;
    for (int64_t i = 0; i < out.numel(); ++i) {
        sum_out += out.data()[i];
    }
    for (int64_t i = 0; i < relevance_in.numel(); ++i) {
        sum_in += relevance_in.data()[i];
    }
    std::printf("sum(relevance_out) = %.4f, sum(relevance_in) = %.4f (gap dominated by attention's own\n", sum_out,
                sum_in);
    std::printf("known non-conservation -- see mission_multihead_attention.md/mission_transformer_block.md)\n");

    return 0;
}
