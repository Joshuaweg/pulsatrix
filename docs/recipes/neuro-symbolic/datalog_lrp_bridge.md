# Recipe: Datalog LRP Bridge

**What you'll build:** a `NeuralPredicateDatalogBridge` that feeds a neural predicate's output
into a Datalog derivation: `ancestor(a,d)` over a diamond-shaped `edge`/`ancestor` program. LRP
relevance is then traced from the derived query fact, back through the Datalog rules, and into
the predicate's own input. The numbers match the hand-derived closed form in
`tests/neuro_symbolic_datalog_bridge_test.cpp`.

CMake target: `datalog_lrp_bridge_recipe`
(`examples/recipes/datalog_lrp_bridge.cpp`).

Run it: `./build/datalog_lrp_bridge_recipe` (Windows:
`build\Release\datalog_lrp_bridge_recipe.exe`).

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

**Forward: the query weight and its gradient.** The weight of `edge(a,b)` is the output of a
neural predicate: a `LinearModule(1,1)` followed by a sigmoid. The other three edges are
constants. `ancestor(a,d)` can be derived along two paths, `edge(a,b)*edge(b,d)` and
`edge(a,c)*edge(c,d)`. So its weight is `0.6*s + 0.12`, where `s` is the predicate's sigmoid
output.

`evaluate()` computes this weight and its exact derivative with respect to `s` in one pass. It
does this by carrying a value and its derivative together through the derivation
(forward-mode automatic differentiation, via `DualSemiring<double>`).

**Backward: relevance.** `propagate_relevance()` splits the seeded relevance in two stages:

1. across the two derivation paths, in proportion to each path's contribution (the epsilon rule
   for a sum), and
2. between the two factors within each path (the rule for a product).

These are the same rules LRP uses for `+` and `×` elsewhere in this codebase, applied to a
Datalog derivation instead of a `Module` chain.

`edge(a,b)` is the one fact that comes from the neural predicate, so its relevance keeps going.
It continues through the predicate's own `LinearModule::propagate_relevance()`, giving an
end-to-end trace from a symbolic query back to the network's input. The conservation check
confirms this. The other base facts' relevance plus the relevance reaching `x` adds back up to
the seed, minus the small amount the epsilon rule absorbs.

See also: [Neuro-Symbolic Reasoning](../../neuro-symbolic/index.md#bridging-a-neural-predicate-into-a-datalog-derivation).
