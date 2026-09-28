# Neuro-Symbolic Reasoning

Where [Ad-hoc Interpretability](../interpretability/index.md) and per-layer LRP explain a
neural network's own computation, neuro-symbolic reasoning goes the other direction: it lets a
neural network's outputs participate directly in symbolic, rule-based reasoning, and traces
relevance *through* that symbolic structure back to the network. Two layers build on each
other here — a differentiable fuzzy-logic core (Logic Tensor Networks-shaped) for training a
network against first-order-logic-style constraints, and a Datalog engine that a neural
predicate's output can feed into as one weighted fact among purely symbolic ones.

## What's inside

**Differentiable fuzzy-logic core** — real-valued (fuzzy) truth degrees in `[0, 1]`, combined
via selectable t-norm/t-conorm operators and trained through ordinary gradient descent:

- `ConjunctionModule` / `DisjunctionModule` — t-norm/t-conorm logical AND/OR, selectable via
  `ConjunctionModule::TNorm` (`Product`, `Lukasiewicz`, `Godel`)
- `NegationModule` — fuzzy NOT
- `AggregatorModule` — differentiable p-mean quantifier (`agg_p(x) = (mean(x^p))^(1/p)`),
  standing in for a fuzzy ∀/∃ over a batch of groundings
- `SatisfactionLoss` — `loss = 1 - agg_p(truth_values)`, the standard "Real Logic" training
  objective: maximizing a knowledge base's satisfaction is the same as minimizing this loss
- `ToyKnowledgeBase` — a small, hand-traceable worked example chaining the above into a
  trainable rule, `A(x) -> not(B(x))`

**Datalog engine** — function-symbol-free, range-restricted (safe) rules, evaluated bottom-up
to a guaranteed-terminating fixpoint:

- `Term` / `Atom` / `Rule` — a rule's building blocks (`head :- body1, body2, ...`)
- `FactDatabase` — boolean semiring: a fact is simply present or absent
- `naive_evaluate` / `semi_naive_evaluate` — bottom-up fixpoint evaluation (identical results,
  semi-naive only re-scans facts newly derived in the previous round)
- `WeightedFactDatabase<T>` / a generic `Semiring` abstraction / `naive_evaluate_weighted` /
  `semi_naive_evaluate_weighted` — the same engine generalized to real-valued (weighted) facts
- `DualSemiring<T>` — a forward-mode-AD provenance semiring: differentiates a derived fact's
  weight with respect to any base fact's weight in one evaluation pass
- `propagate_relevance_weighted` (`datalog_lrp.hpp`) — a hand-derived LRP rule for the
  `(+, x)`-shaped provenance-semiring circuit, splitting relevance across derivation paths
  (⊕, epsilon rule) and a match's body atoms (⊗, bilinear split)

**Neural-predicate bridge** — `NeuralPredicateDatalogBridge` wires a real `LinearModule` +
sigmoid neural predicate in as one Datalog base fact's weight, so a derived query fact's weight
(and its LRP relevance) is a genuine function of a neural network's output.

Full API reference: [Doxygen: Neuro-Symbolic Reasoning](../api/group__neuro__symbolic.html)

## How to implement

### Training a rule with the fuzzy-logic core

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/neuro_symbolic_toy_kb.hpp"
#include "pulsatrix/sgd_optimizer.hpp"

using namespace pulsatrix;

CPUBackend backend;
ToyKnowledgeBase kb(&backend, /*p=*/2.0f);  // rule: A(x) -> not(B(x)), via De Morgan
SGDOptimizer optimizer(0.1f);

Tensor x(Shape({4, 1}), &backend, {-1.5f, -0.5f, 0.5f, 1.5f});  // 4 synthetic groundings
float loss = kb.train_step(x, optimizer);  // forward + backward + SGD update, one call
```

**What's happening:** `ToyKnowledgeBase` expresses `A(x) -> not(B(x))` via the propositional
identity `P -> Q == not(P) or Q`, chaining two neural predicates (`LinearModule` + sigmoid)
through `NegationModule`/`DisjunctionModule` (Product t-conorm), then `SatisfactionLoss`
aggregates the per-grounding rule truth degrees into one scalar loss. Because every step is a
real `Module` with `forward()`/`backward()`, the whole rule trains end-to-end via ordinary
gradient descent — no separate symbolic solver.

### Bridging a neural predicate into a Datalog derivation

```cpp
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/neuro_symbolic_datalog_bridge.hpp"

using namespace pulsatrix::datalog;
using pulsatrix::CPUBackend;

CPUBackend backend;
NeuralPredicateDatalogBridge bridge(&backend);

// edge(a,b)'s weight comes from a neural predicate; edge(a,c)/edge(b,d)/edge(c,d) are
// constants. Program: ancestor(X,Y) :- edge(X,Y).  ancestor(X,Y) :- edge(X,Z), ancestor(Z,Y).
Tensor x(Shape({1, 1}), &backend, {0.5f});
NeuralPredicateQueryResult query = bridge.evaluate(x);  // ancestor(a,d)'s weight + its
                                                         // gradient w.r.t. the predicate output

bridge.backward();  // accumulates the predicate's LinearModule weight/bias gradients

NeuralPredicateRelevanceResult relevance = bridge.propagate_relevance(/*seed=*/1.0);
// relevance.base_fact_relevance: every base fact's relevance in the Datalog derivation
// relevance.relevance_wrt_x: edge(a,b)'s relevance, continued through the predicate's own
//                            sigmoid + LinearModule::propagate_relevance() into raw input x
```

**What's happening:** `evaluate()` runs the ordinary weighted-Datalog fixpoint evaluation
under `DualSemiring<double>`, which computes `ancestor(a,d)`'s derived weight and its exact
local derivative with respect to the neural predicate's output in a single pass (forward-mode
AD over the derivation, not a new engine). `propagate_relevance()` then runs the hand-derived
`(+, x)`-circuit LRP rule back through the derivation and, since `edge(a,b)` is the one
neural-predicate-weighted base fact, continues on through that predicate's own real `Module`
chain — composing two already-proven-correct rules (the Datalog circuit rule and
`LinearModule`'s epsilon rule) rather than inventing a new one for the seam. Relevance is
conserved end-to-end: every other base fact's relevance plus the relevance reaching the
predicate's raw input sums back to the seeded relevance.

Recipe: [Datalog LRP bridge](../recipes/neuro-symbolic/datalog_lrp_bridge.md).

## Recipes

- [Datalog LRP bridge](../recipes/neuro-symbolic/datalog_lrp_bridge.md)
