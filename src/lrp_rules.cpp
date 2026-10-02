#include "lrp_rules.hpp"

#include <stdexcept>
#include <string>

#include "pulsatrix/tensor.hpp"

namespace pulsatrix {
namespace lrp_rules {

namespace {

Tensor scratch(const AffineOp& op, size_t n) {
    return Tensor(Shape({static_cast<int64_t>(n)}), op.backend, op.device);
}

// out = max(in, 0)
Tensor positive_part(const AffineOp& op, const float* in, size_t n) {
    Tensor out = scratch(op, n);
    op.backend->elementwise(ElementwiseOp::Relu, in, out.data(), n);
    return out;
}

// out = min(in, 0) = in - max(in, 0), exact (one of the two terms is always 0).
Tensor negative_part(const AffineOp& op, const float* in, size_t n) {
    Tensor out = positive_part(op, in, n);
    op.backend->axpby(1.0f, in, -1.0f, out.data(), out.data(), n);
    return out;
}

// out = a + scale * b (Zennit's GammaMod: param + gamma * param.clamp(...)).
Tensor plus_scaled(const AffineOp& op, const float* a, float scale, const float* b, size_t n) {
    Tensor out = scratch(op, n);
    op.backend->axpby(1.0f, a, scale, b, out.data(), n);
    return out;
}

// out (output-shaped) = forward(x, w) [+ bias when non-null].
Tensor affine(const AffineOp& op, const float* x, const float* w, const float* bias) {
    Tensor out = scratch(op, op.output_numel);
    op.forward(x, w, out.data());
    if (bias != nullptr) {
        op.add_bias(out.data(), bias, out.data());
    }
    return out;
}

// out = r / stab(denom), gated by the sign of gate.
Tensor divide(const AffineOp& op, const float* r, const float* denom, const float* gate, LrpGate mode, float eps) {
    Tensor out = scratch(op, op.output_numel);
    op.backend->lrp_stabilized_divide(r, denom, gate, out.data(), op.output_numel, eps, mode);
    return out;
}

// acc = (first ? 0 : acc) + scale * (x_mod * backward(g, w)).
void add_term(const AffineOp& op, const float* x_mod, const float* g, const float* w, float scale, float* acc,
              bool first) {
    Tensor term = scratch(op, op.input_numel);
    op.backward(g, w, term.data());
    op.backend->mul(x_mod, term.data(), term.data(), op.input_numel);
    op.backend->axpby(scale, term.data(), first ? 0.0f : 1.0f, acc, acc, op.input_numel);
}

void epsilon_with_bias(const AffineOp& op, const float* x, const float* w, const float* b, const float* pre_bias,
                       const float* r, float* r_in, float eps) {
    Tensor z = scratch(op, op.output_numel);
    op.add_bias(pre_bias, b, z.data());
    Tensor g = divide(op, r, z.data(), nullptr, LrpGate::None, eps);
    add_term(op, x, g.data(), w, 1.0f, r_in, true);
}

void alpha_beta(const AffineOp& op, const float* x, const float* w, const float* b, const float* r, float* r_in,
                const LRPRuleConfig& config) {
    const size_t n_in = op.input_numel, n_w = op.weight_numel, n_b = op.bias_numel, n_out = op.output_numel;
    Tensor xp = positive_part(op, x, n_in), xn = negative_part(op, x, n_in);
    Tensor wp = positive_part(op, w, n_w), wn = negative_part(op, w, n_w);
    Tensor bp = positive_part(op, b, n_b), bn = negative_part(op, b, n_b);

    // z+ = (x+ W+ + b+) + x- W-
    Tensor z_pos = affine(op, xp.data(), wp.data(), bp.data());
    Tensor z_pos_rest = affine(op, xn.data(), wn.data(), nullptr);
    op.backend->add(z_pos.data(), z_pos_rest.data(), z_pos.data(), n_out);
    Tensor g_pos = divide(op, r, z_pos.data(), nullptr, LrpGate::None, config.epsilon);

    Tensor positive = scratch(op, n_in);
    add_term(op, xp.data(), g_pos.data(), wp.data(), 1.0f, positive.data(), true);
    add_term(op, xn.data(), g_pos.data(), wn.data(), 1.0f, positive.data(), false);
    if (config.beta == 0.0f) {
        // alpha * positive - 0 * negative: the negative branch cannot contribute.
        op.backend->axpby(config.alpha, positive.data(), 0.0f, positive.data(), r_in, n_in);
        return;
    }

    // z- = (x+ W- + b-) + x- W+
    Tensor z_neg = affine(op, xp.data(), wn.data(), bn.data());
    Tensor z_neg_rest = affine(op, xn.data(), wp.data(), nullptr);
    op.backend->add(z_neg.data(), z_neg_rest.data(), z_neg.data(), n_out);
    Tensor g_neg = divide(op, r, z_neg.data(), nullptr, LrpGate::None, config.epsilon);

    Tensor negative = scratch(op, n_in);
    add_term(op, xp.data(), g_neg.data(), wn.data(), 1.0f, negative.data(), true);
    add_term(op, xn.data(), g_neg.data(), wp.data(), 1.0f, negative.data(), false);
    op.backend->axpby(config.alpha, positive.data(), -config.beta, negative.data(), r_in, n_in);
}

void gamma(const AffineOp& op, const float* x, const float* w, const float* b, const float* pre_bias, const float* r,
           float* r_in, const LRPRuleConfig& config) {
    const size_t n_in = op.input_numel, n_w = op.weight_numel, n_b = op.bias_numel, n_out = op.output_numel;
    const float g = config.gamma;
    Tensor xp = positive_part(op, x, n_in), xn = negative_part(op, x, n_in);
    Tensor wa = plus_scaled(op, w, g, positive_part(op, w, n_w).data(), n_w);
    Tensor wb = plus_scaled(op, w, g, negative_part(op, w, n_w).data(), n_w);
    Tensor ba = plus_scaled(op, b, g, positive_part(op, b, n_b).data(), n_b);
    Tensor bb = plus_scaled(op, b, g, negative_part(op, b, n_b).data(), n_b);

    // The original output z = xW + b selects which pair of modified outputs explains each unit.
    Tensor z = scratch(op, n_out);
    op.add_bias(pre_bias, b, z.data());

    Tensor den_pos = affine(op, xp.data(), wa.data(), ba.data());   // o0
    Tensor o1 = affine(op, xn.data(), wb.data(), nullptr);
    op.backend->add(den_pos.data(), o1.data(), den_pos.data(), n_out);
    Tensor den_neg = affine(op, xp.data(), wb.data(), bb.data());   // o2
    Tensor o3 = affine(op, xn.data(), wa.data(), nullptr);
    op.backend->add(den_neg.data(), o3.data(), den_neg.data(), n_out);

    Tensor g_pos = divide(op, r, den_pos.data(), z.data(), LrpGate::Positive, config.epsilon);
    Tensor g_neg = divide(op, r, den_neg.data(), z.data(), LrpGate::Negative, config.epsilon);

    add_term(op, xp.data(), g_pos.data(), wa.data(), 1.0f, r_in, true);
    add_term(op, xn.data(), g_pos.data(), wb.data(), 1.0f, r_in, false);
    add_term(op, xp.data(), g_neg.data(), wb.data(), 1.0f, r_in, false);
    add_term(op, xn.data(), g_neg.data(), wa.data(), 1.0f, r_in, false);
}

void zbox(const AffineOp& op, const float* x, const float* w, const float* pre_bias, const float* r, float* r_in,
          const LRPRuleConfig& config) {
    const size_t n_in = op.input_numel, n_w = op.weight_numel, n_out = op.output_numel;
    Tensor wp = positive_part(op, w, n_w), wn = negative_part(op, w, n_w);
    Tensor low = scratch(op, n_in), high = scratch(op, n_in);
    low.fill(config.low);
    high.fill(config.high);

    // den = xW - L W+ - H W-
    Tensor den = scratch(op, n_out);
    Tensor low_out = affine(op, low.data(), wp.data(), nullptr);
    Tensor high_out = affine(op, high.data(), wn.data(), nullptr);
    op.backend->axpby(1.0f, pre_bias, -1.0f, low_out.data(), den.data(), n_out);
    op.backend->axpby(1.0f, den.data(), -1.0f, high_out.data(), den.data(), n_out);
    Tensor g = divide(op, r, den.data(), nullptr, LrpGate::None, config.epsilon);

    add_term(op, x, g.data(), w, 1.0f, r_in, true);
    add_term(op, low.data(), g.data(), wp.data(), -1.0f, r_in, false);
    add_term(op, high.data(), g.data(), wn.data(), -1.0f, r_in, false);
}

}  // namespace

void validate(const LRPRuleConfig& config, const char* module_name) {
    if (config.rule == LRPRule::AlphaBeta) {
        if (!(config.alpha >= 0.0f) || !(config.beta >= 0.0f)) {
            throw std::invalid_argument(std::string(module_name) +
                                        "::propagate_relevance: AlphaBeta rule needs alpha >= 0 and beta >= 0");
        }
        if (config.alpha - config.beta != 1.0f) {
            throw std::invalid_argument(std::string(module_name) +
                                        "::propagate_relevance: AlphaBeta rule needs alpha - beta == 1");
        }
    }
}

void apply(const AffineOp& op, const float* x, const float* w, const float* b, const float* pre_bias, const float* r,
           float* r_in, const LRPRuleConfig& config) {
    switch (config.rule) {
        case LRPRule::Epsilon:
            epsilon_with_bias(op, x, w, b, pre_bias, r, r_in, config.epsilon);
            return;
        case LRPRule::AlphaBeta:
            alpha_beta(op, x, w, b, r, r_in, config);
            return;
        case LRPRule::Gamma:
            gamma(op, x, w, b, pre_bias, r, r_in, config);
            return;
        case LRPRule::ZBox:
            zbox(op, x, w, pre_bias, r, r_in, config);
            return;
    }
    throw std::invalid_argument("lrp_rules::apply: unknown LRPRule");
}

}  // namespace lrp_rules
}  // namespace pulsatrix
