# Recipe: Datalog LRP bridge

**What you'll build:** a `NeuralPredicateDatalogBridge` wiring a real neural predicate's output
into a Datalog derivation (`ancestor(a,d)` over a diamond-graph `edge`/`ancestor` program), then
LRP relevance traced from the derived query fact back through the Datalog circuit and into the
predicate's own raw input — reusing the hand-derived closed form from
`tests/neuro_symbolic_datalog_bridge_test.cpp`.

CMake target: `datalog_lrp_bridge_recipe`
(`examples/recipes/datalog_lrp_bridge.cpp`).

## Code

```cpp
NeuralPredicateDatalogBridge bridge(&backend);

Tensor x(Shape({1, 1}), &backend, {0.5f});
NeuralPredicateQueryResult query = bridge.evaluate(x);
bridge.backward();

NeuralPredicateRelevanceResult relevance = bridge.propagate_relevance(/*seed=*/1.0);
// relevance.base_fact_relevance: every base fact's share
// relevance.relevance_wrt_x: edge(a,b)'s relevance, continued into the predicate's raw input
```

Full source: [`examples/recipes/datalog_lrp_bridge.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/datalog_lrp_bridge.cpp).

## Expected output

```
Datalog LRP bridge recipe -- ancestor(a,d) via a neural-predicate-weighted edge

ancestor(a,d) query weight:        0.464666
d(ancestor(a,d))/d(edge(a,b)):      0.600000

predicate weight grad:              0.073338
predicate bias grad:                0.146675

Relevance seeded at ancestor(a,d) = 1.00, traced to every base fact:
  edge(c, d): 0.129124
  edge(a, c): 0.129124
  edge(b, d): 0.370873
  edge(a, b): 0.370874
  predicate's raw input x (edge(a,b)'s relevance, continued through sigmoid +
  LinearModule::propagate_relevance): 0.370872

conservation check: other base facts + relevance at x = 0.999993 (should equal seed 1.00)
```

## What's happening

`edge(a,b)`'s weight is a real `LinearModule(1,1)` + sigmoid neural predicate's output; the
other three edges are constants. `ancestor(a,d)` is reachable via two derivation paths
(`edge(a,b)*edge(b,d)` and `edge(a,c)*edge(c,d)`), so its weight is `0.6*s + 0.12` where `s` is
the predicate's sigmoid output — `evaluate()` computes this and its exact derivative with
respect to `s` in one pass via a forward-mode-AD provenance semiring (`DualSemiring<double>`).
`propagate_relevance()` then splits the seeded relevance across the two derivation paths
(weighted-sum/epsilon-rule split) and each path's two factors (bilinear split) — the same
composition Layer-wise Relevance Propagation uses for `+` and `x` elsewhere in this codebase,
reapplied to a Datalog derivation instead of a `Module` chain. Because `edge(a,b)` is the one
neural-predicate-weighted fact, its relevance doesn't stop there: it continues on through the
predicate's own `LinearModule::propagate_relevance()`, giving a genuine end-to-end trace from a
symbolic query back to the neural network's raw input. The conservation check confirms this:
every other base fact's relevance plus the relevance reaching `x` sums back to the seeded
relevance (up to the small epsilon-rule absorption).

See also: [Neuro-Symbolic Reasoning](../../neuro-symbolic/index.md#bridging-a-neural-predicate-into-a-datalog-derivation).
