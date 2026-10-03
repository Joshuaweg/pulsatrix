# Neuro-Symbolic Reasoning

Use this section when you want a neural network to work together with logical rules. You can
train a network so its outputs satisfy rules such as "if A then not B". You can also feed a
network's output into a rule-based (Datalog) program and trace the result back to the network's
input.

There are two layers:

- A **fuzzy-logic core** treats truth as a number in `[0, 1]` instead of true/false, so logical
  rules become differentiable and you can train against them with gradient descent. It follows
  the design of Logic Tensor Networks.
- A **Datalog engine** derives new facts from rules. A neural network's output can serve as the
  weight of one input fact.

## Which part should I use?

| Goal | Use |
|---|---|
| Train a network so its predictions obey logical constraints | Fuzzy-logic core: `ConjunctionModule`, `DisjunctionModule`, `NegationModule`, `AggregatorModule`, `SatisfactionLoss` |
| Derive facts from rules and a fact list (true/false) | `FactDatabase` + `semi_naive_evaluate` |
| Derive facts whose confidence is a number | `WeightedFactDatabase<T>` + `semi_naive_evaluate_weighted` |
| Explain which input facts a derived fact depends on | `propagate_relevance_weighted` (Datalog LRP) |
| Connect a network's output to a Datalog derivation, end to end | `NeuralPredicateDatalogBridge` |

## What's inside

**Fuzzy-logic core.** Each operator works on truth degrees in `[0, 1]` and is a `Module`
with `forward()`/`backward()`:

- `ConjunctionModule` / `DisjunctionModule`: fuzzy AND / OR. A *t-norm* is a fuzzy AND
  formula; a *t-conorm* is the matching OR. Pick one with `ConjunctionModule::TNorm` or
  `DisjunctionModule::TConorm` (`Product`, `Lukasiewicz`, `Godel`).
- `NegationModule`: fuzzy NOT (`1 - x`).
- `AggregatorModule`: a differentiable "for all" over many examples, using the p-mean
  `agg_p(x) = (mean(x^p))^(1/p)`.
- `SatisfactionLoss`: `loss = 1 - agg_p(truth_values)`. Minimizing it maximizes how well the
  rules hold.
- `ToyKnowledgeBase`: a small worked example that chains the above into one trainable rule,
  `A(x) -> not(B(x))`.

**Datalog engine.** Datalog is a simple rule language: `head :- body1, body2, ...` means "the
head is true when every body atom is true". The engine supports rules without function symbols,
where every head variable also appears in the body (*safe* rules), so evaluation always
terminates.

- `Term` / `Atom` / `Rule`: the building blocks of a rule.
- `FactDatabase`: a set of true facts.
- `naive_evaluate` / `semi_naive_evaluate`: apply rules until no new facts appear. Both give
  identical results; semi-naive is faster because each round only revisits newly derived facts.
- `WeightedFactDatabase<T>`, `naive_evaluate_weighted`, `semi_naive_evaluate_weighted`: the same
  engine with a number attached to each fact. The template argument is a *semiring* (a pair of
  "combine alternatives" and "combine requirements" operations), such as `BooleanSemiring` or
  `RealSemiring<T>`.
- `DualSemiring<T>`: a semiring that also carries a derivative. One evaluation gives a derived
  fact's weight and its derivative with respect to an input fact's weight.
- `propagate_relevance_weighted` (`datalog_lrp.hpp`): an LRP rule for Datalog derivations. It
  splits a derived fact's relevance across its alternative derivations and then across each
  derivation's body facts. For LRP itself, see [LRP](../interpretability/lrp.md).

**Neural-predicate bridge.** `NeuralPredicateDatalogBridge` uses a small network
(`LinearModule` + sigmoid) to produce one Datalog fact's weight. A derived fact's weight, and
its LRP relevance, then depend on the network's output.

Full API reference: [Doxygen: Neuro-Symbolic Reasoning](../api/group__neuro__symbolic.html)

## How to implement

### Training a rule with the fuzzy-logic core

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/neuro_symbolic_toy_kb.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

using namespace pulsatrix;

CPUBackend backend;
ToyKnowledgeBase kb(&backend, /*p=*/2.0f);  // rule A(x) -> not(B(x)), as not(A(x)) or not(B(x))
SGDOptimizer optimizer(0.1f);

Tensor x(Shape({4, 1}), &backend, {-1.5f, -0.5f, 0.5f, 1.5f});  // 4 example inputs
for (int epoch = 0; epoch < 500; ++epoch) {
    float loss = kb.train_step(x, optimizer);  // forward + backward + SGD update
}
```

**What's happening:** `ToyKnowledgeBase` has two small networks, `A(x)` and `B(x)` (each a
`LinearModule` followed by a sigmoid). It rewrites `A(x) -> not(B(x))` as
`not(A(x)) or not(B(x))`, using the rule `P -> Q` equals `not(P) or Q`. Two `NegationModule`s
and a `DisjunctionModule` (Product t-conorm) compute the rule's truth for each input.
`SatisfactionLoss` then turns those into one loss. Every step has `forward()`/`backward()`, so
the whole rule trains with ordinary gradient descent and no separate solver.

To run it: `cmake --build build --target neuro_symbolic_toy_kb_demo`. The demo
(`examples/neuro_symbolic_toy_kb_demo.cpp`) trains for 500 epochs and prints the loss and
satisfaction as it goes.

### Deriving facts with the Datalog engine

```cpp
#include "pulsatrix/datalog_engine.hpp"

using namespace pulsatrix::datalog;

Term X = Term::make_variable("X"), Y = Term::make_variable("Y"), Z = Term::make_variable("Z");
auto c = [](const char* name) { return Term::make_constant(name); };

// ancestor(X,Y) :- edge(X,Y).
// ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
std::vector<Rule> rules = {
    Rule(Atom("ancestor", {X, Y}), {Atom("edge", {X, Y})}),
    Rule(Atom("ancestor", {X, Y}), {Atom("edge", {X, Z}), Atom("ancestor", {Z, Y})}),
};

FactDatabase facts;
facts.insert(Atom("edge", {c("a"), c("b")}));
facts.insert(Atom("edge", {c("b"), c("c")}));

FactDatabase result = semi_naive_evaluate(rules, facts);
bool derived = result.contains(Atom("ancestor", {c("a"), c("c")}));  // true
```

**What's happening:** `semi_naive_evaluate` applies the rules repeatedly until no new facts
appear. The result holds the input facts plus every derived `ancestor` fact. Input facts must be
*ground*, meaning they contain no variables; `insert()` throws otherwise.

### Bridging a neural predicate into a Datalog derivation

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/neuro_symbolic_datalog_bridge.hpp"

using namespace pulsatrix::datalog;
using pulsatrix::CPUBackend;
using pulsatrix::Shape;
using pulsatrix::Tensor;

CPUBackend backend;
NeuralPredicateDatalogBridge bridge(&backend);

// edge(a,b)'s weight comes from a neural predicate; edge(a,c), edge(b,d), edge(c,d) are
// constants. Program: ancestor(X,Y) :- edge(X,Y).  ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
Tensor x(Shape({1, 1}), &backend, {0.5f});
NeuralPredicateQueryResult query = bridge.evaluate(x);
// query.query_weight: ancestor(a,d)'s derived weight
// query.grad_wrt_predicate_output: its derivative w.r.t. edge(a,b)'s weight

bridge.backward();  // accumulates the predicate's LinearModule weight/bias gradients

NeuralPredicateRelevanceResult relevance = bridge.propagate_relevance(/*relevance_seed=*/1.0);
// relevance.base_fact_relevance: each base fact's share of the relevance
// relevance.relevance_wrt_x: edge(a,b)'s share, carried on into the raw input x
```

**What's happening:** `evaluate()` runs weighted Datalog evaluation with
`DualSemiring<double>`. In one pass it computes `ancestor(a,d)`'s weight and its derivative with
respect to the network's output.

`propagate_relevance()` then runs the Datalog LRP rule back through the derivation. Because
`edge(a,b)` comes from the network, its relevance continues through the sigmoid and
`LinearModule::propagate_relevance()` into the input `x`. No new rule is needed at the seam:
the bridge composes the Datalog rule with `LinearModule`'s epsilon rule (see
[LRP](../interpretability/lrp.md)).

Relevance is conserved end to end: the other base facts' relevance plus the relevance reaching
`x` sums to the seed.

Recipe: [Datalog LRP bridge](../recipes/neuro-symbolic/datalog_lrp_bridge.md).

## Recipes

- [Datalog LRP bridge](../recipes/neuro-symbolic/datalog_lrp_bridge.md)
