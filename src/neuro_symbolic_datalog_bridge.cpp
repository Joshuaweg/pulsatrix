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
    PULSATRIX_ASSERT(z.device() == DeviceType::Cpu);
    Tensor y(z.shape(), backend, z.device());
    const int64_t n = z.numel();
    for (int64_t i = 0; i < n; ++i) {
        y.data()[i] = 1.0f / (1.0f + std::exp(-z.data()[i]));
    }
    return y;
}

Tensor sigmoid_backward(float grad_y, const Tensor& y, DeviceBackend* backend) {
    PULSATRIX_ASSERT(y.device() == DeviceType::Cpu);
    Tensor grad_z(y.shape(), backend, y.device());
    const int64_t n = y.numel();
    for (int64_t i = 0; i < n; ++i) {
        const float yv = y.data()[i];
        grad_z.data()[i] = grad_y * yv * (1.0f - yv);
    }
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

    const double s_value = static_cast<double>(s.at({0, 0}));

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
