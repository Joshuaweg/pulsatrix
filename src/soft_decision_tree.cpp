// Frosst and Hinton's soft decision tree (KD-3): sigmoid gates route each input to leaf
// distributions, and the output is the log of their mixture.

#include "pulsatrix/soft_decision_tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "portable_random.hpp"

namespace pulsatrix {

namespace {

int64_t CheckedInner(int depth) {
    if (depth < 1 || depth > 16) throw std::invalid_argument("SoftDecisionTree: depth must be between 1 and 16");
    return (int64_t{1} << depth) - 1;
}

int NodeDepth(int64_t node) {
    int d = 0;
    for (int64_t k = node + 1; k > 1; k >>= 1) ++d;
    return d;
}

}  // namespace

SoftDecisionTree::SoftDecisionTree(int64_t inputs, int64_t classes, DeviceBackend* backend, SoftDecisionTreeOptions options)
    : backend_(backend),
      inputs_(inputs),
      classes_(classes),
      options_(options),
      gates_(inputs, CheckedInner(options.depth), backend),
      leaves_(Shape({int64_t{1} << options.depth, std::max<int64_t>(classes, 1)}), backend),
      leaves_grad_(Shape({int64_t{1} << options.depth, std::max<int64_t>(classes, 1)}), backend) {
    if (inputs < 1 || classes < 2) throw std::invalid_argument("SoftDecisionTree: needs at least 1 input and 2 classes");
    if (!(std::isfinite(options.beta) && options.beta > 0)) throw std::invalid_argument("SoftDecisionTree: beta must be finite and positive");
    if (!(std::isfinite(options.penalty) && options.penalty >= 0)) throw std::invalid_argument("SoftDecisionTree: penalty must be finite and >= 0");
    const double bound = 1.0 / std::sqrt(static_cast<double>(inputs));
    PortableRng rng{options.seed};
    std::vector<float> w(static_cast<size_t>(inputs * num_inner())), b(static_cast<size_t>(num_inner()));
    for (float& v : w) v = static_cast<float>((2 * rng.uniform() - 1) * bound);
    for (float& v : b) v = static_cast<float>((2 * rng.uniform() - 1) * bound);
    gates_.set_weight(w);
    gates_.set_bias(b);
    leaves_.fill(0.0f);
    leaves_grad_.fill(0.0f);
}

std::vector<double> SoftDecisionTree::LeafDistributions() const {
    const std::vector<float> phi = leaves_.to_host_vector();
    const auto c = static_cast<size_t>(classes_);
    std::vector<double> q(phi.size());
    for (size_t l = 0; l < static_cast<size_t>(num_leaves()); ++l) {
        double max = phi[l * c];
        for (size_t k = 1; k < c; ++k) max = std::max(max, static_cast<double>(phi[l * c + k]));
        double sum = 0;
        for (size_t k = 0; k < c; ++k) sum += q[l * c + k] = std::exp(phi[l * c + k] - max);
        for (size_t k = 0; k < c; ++k) q[l * c + k] /= sum;
    }
    return q;
}

Tensor SoftDecisionTree::forward_impl(const Tensor& input) {
    if (input.rank() != 2 || input.shape().dim(1) != inputs_) {
        throw std::invalid_argument("SoftDecisionTree::forward: input must be (N, " + std::to_string(inputs_) + ")");
    }
    const int64_t n = input.shape().dim(0), inner = num_inner(), leaves = num_leaves(), c = classes_;
    const std::vector<float> z = gates_.forward(input).to_host_vector();
    p_.assign(static_cast<size_t>(n * inner), 0.0);
    reach_.assign(static_cast<size_t>(n * inner), 0.0);
    reach_leaf_.assign(static_cast<size_t>(n * leaves), 0.0);
    for (int64_t r = 0; r < n; ++r) {
        double* p = &p_[static_cast<size_t>(r * inner)];
        double* reach = &reach_[static_cast<size_t>(r * inner)];
        double* leaf = &reach_leaf_[static_cast<size_t>(r * leaves)];
        reach[0] = 1.0;
        for (int64_t i = 0; i < inner; ++i) {
            p[i] = 1.0 / (1.0 + std::exp(-static_cast<double>(options_.beta) * z[static_cast<size_t>(r * inner + i)]));
            const double right = reach[i] * p[i], left = reach[i] * (1.0 - p[i]);
            const int64_t l = 2 * i + 1, rr = 2 * i + 2;
            if (l < inner) {
                reach[l] = left, reach[rr] = right;
            } else {
                leaf[l - inner] = left, leaf[rr - inner] = right;
            }
        }
    }
    q_ = LeafDistributions();
    mixture_.assign(static_cast<size_t>(n * c), 0.0);
    std::vector<float> out(static_cast<size_t>(n * c));
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t k = 0; k < c; ++k) {
            double m = 0;
            for (int64_t l = 0; l < leaves; ++l) m += reach_leaf_[static_cast<size_t>(r * leaves + l)] * q_[static_cast<size_t>(l * c + k)];
            m = std::max(m, std::numeric_limits<double>::min());
            mixture_[static_cast<size_t>(r * c + k)] = m;
            out[static_cast<size_t>(r * c + k)] = static_cast<float>(std::log(m));
        }
    }

    penalty_ = 0.0;
    if (options_.penalty > 0) {
        for (int64_t i = 0; i < inner; ++i) {
            double a = 0, b = 0;
            for (int64_t r = 0; r < n; ++r) {
                a += reach_[static_cast<size_t>(r * inner + i)] * p_[static_cast<size_t>(r * inner + i)];
                b += reach_[static_cast<size_t>(r * inner + i)];
            }
            if (!(b > 0)) continue;
            const double alpha = a / b;
            penalty_ -= options_.penalty * std::ldexp(1.0, -NodeDepth(i)) * 0.5 * (std::log(alpha) + std::log(1.0 - alpha));
        }
    }
    batch_ = n;
    input_device_ = input.device();
    has_forwarded_ = true;
    return Tensor(Shape({n, c}), backend_, out, input.device());
}

Tensor SoftDecisionTree::backward(const Tensor& grad_output) {
    if (!has_forwarded_) throw std::logic_error("SoftDecisionTree::backward called before forward");
    const int64_t n = batch_, inner = num_inner(), leaves = num_leaves(), c = classes_;
    if (grad_output.rank() != 2 || grad_output.shape().dim(0) != n || grad_output.shape().dim(1) != c) {
        throw std::invalid_argument("SoftDecisionTree::backward: grad_output must be (N, classes) matching the last forward");
    }
    const std::vector<float> g = grad_output.to_host_vector();
    // y = log m: dL/dm = g / m.
    std::vector<double> gm(static_cast<size_t>(n * c));
    for (size_t k = 0; k < gm.size(); ++k) gm[k] = g[k] / mixture_[k];

    // The leaves: dL/dQ = Σ_n P_nℓ gm_n, then through each leaf's softmax.
    std::vector<float> leaf_grad = leaves_grad_.to_host_vector();
    for (int64_t l = 0; l < leaves; ++l) {
        std::vector<double> dq(static_cast<size_t>(c), 0.0);
        for (int64_t r = 0; r < n; ++r) {
            const double reach = reach_leaf_[static_cast<size_t>(r * leaves + l)];
            for (int64_t k = 0; k < c; ++k) dq[static_cast<size_t>(k)] += reach * gm[static_cast<size_t>(r * c + k)];
        }
        double dot = 0;
        for (int64_t k = 0; k < c; ++k) dot += q_[static_cast<size_t>(l * c + k)] * dq[static_cast<size_t>(k)];
        for (int64_t k = 0; k < c; ++k) {
            leaf_grad[static_cast<size_t>(l * c + k)] += static_cast<float>(q_[static_cast<size_t>(l * c + k)] * (dq[static_cast<size_t>(k)] - dot));
        }
    }
    leaves_grad_ = Tensor(leaves_grad_.shape(), backend_, leaf_grad, leaves_grad_.device());

    // The penalty's pieces per inner node: dC/dα and Σ_n reach.
    std::vector<double> dalpha(static_cast<size_t>(inner), 0.0), alpha(static_cast<size_t>(inner), 0.0), mass(static_cast<size_t>(inner), 0.0);
    if (options_.penalty > 0) {
        for (int64_t i = 0; i < inner; ++i) {
            double a = 0, b = 0;
            for (int64_t r = 0; r < n; ++r) {
                a += reach_[static_cast<size_t>(r * inner + i)] * p_[static_cast<size_t>(r * inner + i)];
                b += reach_[static_cast<size_t>(r * inner + i)];
            }
            if (!(b > 0)) continue;
            const double al = a / b;
            alpha[static_cast<size_t>(i)] = al;
            mass[static_cast<size_t>(i)] = b;
            dalpha[static_cast<size_t>(i)] = -options_.penalty * std::ldexp(1.0, -NodeDepth(i)) * 0.5 * (1.0 / al - 1.0 / (1.0 - al));
        }
    }

    // W_j = d(loss + C)/d(reach of node j), bottom-up: a leaf's is Σ_k Q_ℓk gm_k; an inner node's
    // is its penalty term plus its children's, weighted by the branch probabilities.
    std::vector<float> dz(static_cast<size_t>(n * inner));
    std::vector<double> w(static_cast<size_t>(inner + leaves));
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t l = 0; l < leaves; ++l) {
            double a = 0;
            for (int64_t k = 0; k < c; ++k) a += q_[static_cast<size_t>(l * c + k)] * gm[static_cast<size_t>(r * c + k)];
            w[static_cast<size_t>(inner + l)] = a;
        }
        for (int64_t i = inner - 1; i >= 0; --i) {
            const auto at = static_cast<size_t>(r * inner + i);
            const double p = p_[at], reach = reach_[at];
            const double wl = w[static_cast<size_t>(2 * i + 1)], wr = w[static_cast<size_t>(2 * i + 2)];
            double dp = reach * (wr - wl);
            double own = 0;
            if (mass[static_cast<size_t>(i)] > 0) {
                dp += dalpha[static_cast<size_t>(i)] * reach / mass[static_cast<size_t>(i)];
                own = dalpha[static_cast<size_t>(i)] * (p - alpha[static_cast<size_t>(i)]) / mass[static_cast<size_t>(i)];
            }
            w[static_cast<size_t>(i)] = own + p * wr + (1.0 - p) * wl;
            dz[at] = static_cast<float>(dp * options_.beta * p * (1.0 - p));
        }
    }
    return gates_.backward(Tensor(Shape({n, inner}), backend_, dz, backend_->device()));
}

Tensor SoftDecisionTree::propagate_relevance(const Tensor&, const LRPRuleConfig&) {
    throw std::logic_error("SoftDecisionTree::propagate_relevance: LRP isn't defined for a soft decision tree; explain it by its paths or with gradients");
}

std::vector<NamedParamRef> SoftDecisionTree::named_parameters() {
    std::vector<NamedParamRef> out;
    append_named_parameters(out, "gates", gates_);
    out.push_back({"leaves", {&leaves_, &leaves_grad_}});
    return out;
}

std::vector<NamedModuleRef> SoftDecisionTree::named_children() { return {{"gates", &gates_}}; }

std::vector<int64_t> SoftDecisionTree::MostProbableLeaf() const {
    if (!has_forwarded_) throw std::logic_error("SoftDecisionTree::MostProbableLeaf called before forward");
    const int64_t inner = num_inner();
    std::vector<int64_t> out;
    for (int64_t r = 0; r < batch_; ++r) {
        int64_t node = 0;
        while (node < inner) node = p_[static_cast<size_t>(r * inner + node)] > 0.5 ? 2 * node + 2 : 2 * node + 1;
        out.push_back(node - inner);
    }
    return out;
}

std::vector<std::pair<int64_t, bool>> SoftDecisionTree::PathTo(int64_t leaf) const {
    if (leaf < 0 || leaf >= num_leaves()) throw std::invalid_argument("SoftDecisionTree::PathTo: no such leaf");
    std::vector<std::pair<int64_t, bool>> path;
    for (int64_t node = leaf + num_inner(); node > 0;) {
        const int64_t parent = (node - 1) / 2;
        path.emplace_back(parent, node == 2 * parent + 2);
        node = parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

}  // namespace pulsatrix
