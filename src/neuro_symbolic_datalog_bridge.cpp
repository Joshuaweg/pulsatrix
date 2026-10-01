#include "pulsatrix/neuro_symbolic_datalog_bridge.hpp"

#include <cmath>
#include <stdexcept>

#include "pulsatrix/assert.hpp"
#include "pulsatrix/datalog_weighted_engine.hpp"

namespace pulsatrix::datalog {

namespace {

// Sigmoid squash + its elementary backward -- file-local glue, same convention as
// neuro_symbolic_toy_kb.cpp's own identically-named helpers (not shared across files: each
// toy-KB-style class owns its own small, private squashing glue, per that file's own header
// note on why sigmoid is not a library-wide Module).
Tensor sigmoid(const Tensor& z, DeviceBackend* backend) {
    // Device-generic (GPU-native-kernels Mission 3): the same 1 / (1 + exp(-z)).
    Tensor y(z.shape(), backend, z.device());
    backend->elementwise(ElementwiseOp::Sigmoid, z.data(), y.data(), static_cast<size_t>(z.numel()));
    return y;
}

// grad_z = (grad_y * y) * (1 - y), composed in the original evaluation order.
Tensor sigmoid_backward(float grad_y, const Tensor& y, DeviceBackend* backend) {
    const auto n = static_cast<size_t>(y.numel());
    Tensor grad_z(y.shape(), backend, y.device());
    backend->axpby(grad_y, y.data(), 0.0f, nullptr, grad_z.data(), n);
    Tensor one_minus_y(y.shape(), backend, y.device());
    one_minus_y.fill(1.0f);
    backend->axpby(-1.0f, y.data(), 1.0f, one_minus_y.data(), one_minus_y.data(), n);
    backend->mul(grad_z.data(), one_minus_y.data(), grad_z.data(), n);
    return grad_z;
}

Term C(const std::string& value) { return Term::make_constant(value); }
Term V(const std::string& name) { return Term::make_variable(name); }

}  // namespace

NeuralPredicateDatalogBridge::NeuralPredicateDatalogBridge(DeviceBackend* backend)
    : backend_(backend),
      predicate_(1, 1, backend),
      last_x_(Shape({0}), backend),
      last_s_(Shape({0}), backend) {
    // Same small, non-zero initial weights as Phase 1 Mission 2's predicate A -- deliberate
    // continuity with this campaign's established toy-KB initialization precedent.
    predicate_.set_weight({0.6f});
    predicate_.set_bias({0.0f});
}

std::vector<Rule> NeuralPredicateDatalogBridge::diamond_ancestor_program() {
    // ancestor(X,Y) :- edge(X,Y).
    Rule base(Atom("ancestor", {V("X"), V("Y")}), {Atom("edge", {V("X"), V("Y")})});
    // ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
    Rule recursive(Atom("ancestor", {V("X"), V("Y")}),
                    {Atom("edge", {V("X"), V("Z")}), Atom("ancestor", {V("Z"), V("Y")})});
    return {base, recursive};
}

WeightedFactDatabase<double> NeuralPredicateDatalogBridge::constant_edge_facts() {
    WeightedFactDatabase<double> db;
    db.set(Atom("edge", {C("a"), C("c")}), 0.4);
    db.set(Atom("edge", {C("b"), C("d")}), 0.6);
    db.set(Atom("edge", {C("c"), C("d")}), 0.3);
    return db;
}

NeuralPredicateQueryResult NeuralPredicateDatalogBridge::evaluate(const Tensor& x) {
    if (x.rank() != 2 || x.shape().dim(0) != 1 || x.shape().dim(1) != 1) {
        throw std::invalid_argument("NeuralPredicateDatalogBridge::evaluate: x must have shape (1, 1)");
    }

    Tensor z = predicate_.forward(x);  // (1, 1)
    Tensor s = sigmoid(z, backend_);   // (1, 1) -- edge(a,b)'s weight

    last_x_ = x;
    last_s_ = s;
    has_evaluated_ = true;

    // Host boundary: the datalog engine runs on the host in double, so the single truth value
    // crosses over -- one element read, valid on any device (GPU-native-kernels Mission 3).
    const double s_value = static_cast<double>(s.read_element(0));

    WeightedFactDatabase<DualNumber<double>> dual_facts;
    // Seed edge(a,b) with derivative 1 w.r.t. itself; every constant fact carries derivative 0.
    dual_facts.set(Atom("edge", {C("a"), C("b")}), DualNumber<double>{s_value, 1.0});
    dual_facts.set(Atom("edge", {C("a"), C("c")}), DualNumber<double>{0.4, 0.0});
    dual_facts.set(Atom("edge", {C("b"), C("d")}), DualNumber<double>{0.6, 0.0});
    dual_facts.set(Atom("edge", {C("c"), C("d")}), DualNumber<double>{0.3, 0.0});

    WeightedFactDatabase<DualNumber<double>> dual_result =
        naive_evaluate_weighted<DualSemiring<double>>(diamond_ancestor_program(), dual_facts);

    DualNumber<double> query = dual_result.weight_of(Atom("ancestor", {C("a"), C("d")}), DualSemiring<double>::zero());

    last_grad_wrt_s_ = query.grad;

    return NeuralPredicateQueryResult{query.value, query.grad};
}

NeuralPredicateRelevanceResult NeuralPredicateDatalogBridge::propagate_relevance(double relevance_seed,
                                                                                   const LRPRuleConfig& config) {
    if (!has_evaluated_) {
        throw std::logic_error("NeuralPredicateDatalogBridge::propagate_relevance: evaluate() has not been called yet");
    }

    // Host boundary, as in evaluate(): one element read.
    const double s_value = static_cast<double>(last_s_.read_element(0));
    WeightedFactDatabase<double> facts = constant_edge_facts();
    facts.set(Atom("edge", {C("a"), C("b")}), s_value);
    WeightedFactDatabase<double> fixpoint = naive_evaluate_weighted<RealSemiring<double>>(diamond_ancestor_program(), facts);

    RelevanceResult datalog_relevance = propagate_relevance_weighted(
        diamond_ancestor_program(), fixpoint, Atom("ancestor", {C("a"), C("d")}), relevance_seed,
        static_cast<double>(config.epsilon));

    const double r_s = datalog_relevance.base_facts.at(Atom("edge", {C("a"), C("b")}));

    // Continue through the predicate's own chain: sigmoid pass-through (Phase 2 Mission 0's
    // Sigmoid Design Decision precedent -- no gradient-shaped rule for a monotonic bijective
    // single-input nonlinearity), then LinearModule's real, unmodified propagate_relevance().
    Tensor r_z(last_s_.shape(), backend_, last_s_.device());  // zero-initialized
    r_z.write_element(0, static_cast<float>(r_s));
    Tensor r_x = predicate_.propagate_relevance(r_z, config);

    return NeuralPredicateRelevanceResult{datalog_relevance.base_facts, r_x};
}

void NeuralPredicateDatalogBridge::backward() {
    if (!has_evaluated_) {
        throw std::logic_error("NeuralPredicateDatalogBridge::backward: evaluate() has not been called yet");
    }

    // dQ/ds (last_grad_wrt_s_, from evaluate()'s DualSemiring pass) -> dQ/dz, via sigmoid's own
    // closed-form derivative -- then into the real LinearModule::backward(), exactly mirroring
    // ToyKnowledgeBase::backward()'s own hand-chained-Module::backward() precedent.
    Tensor grad_z = sigmoid_backward(static_cast<float>(last_grad_wrt_s_), last_s_, backend_);
    (void)predicate_.backward(grad_z);  // accumulates weight_grad()/bias_grad(); grad-wrt-x unused
}

}  // namespace pulsatrix::datalog
