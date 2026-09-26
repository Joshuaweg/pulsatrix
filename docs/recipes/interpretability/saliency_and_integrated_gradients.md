# Recipe: Saliency and Integrated Gradients

**What you'll build:** both gradient-based explainers run against the same small
`Linear(2,4) -> ReLU -> Linear(4,1)` network, for one input — and a check that Integrated
Gradients' attributions actually sum to `F(x) - F(baseline)`.

CMake target: `saliency_and_ig_recipe`
(`examples/recipes/saliency_and_integrated_gradients.cpp`).

## Code

```cpp
ExplainerContext ctx({&linear1, &relu, &linear2});
Tensor input(Shape({1, 2}), &backend, {1.0f, 1.0f});
Tensor baseline(Shape({1, 2}), &backend);  // zero-initialized

Saliency saliency;
Attribution sal = saliency.explain(ctx, input, /*target_index=*/0, &backend);

IntegratedGradients ig;
Attribution ig_attr = ig.explain(ctx, input, baseline, /*target_index=*/0, /*steps=*/200, &backend);
```

Full source: [`examples/recipes/saliency_and_integrated_gradients.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/saliency_and_integrated_gradients.cpp).

## Expected output

```
input (1, 1) -> output 0.3200

saliency:             d(out)/d(in) = [0.4200, -0.1000]
integrated gradients: IG = [0.4200, -0.1000], sum=0.3200 (F(x)-F(baseline)=0.3200)
```

## What's happening

`Saliency::explain()` seeds a one-hot vector at `target_index` and calls
`ctx.backward_pass()` — the gradient that comes back *is* the saliency map.
`IntegratedGradients` instead averages the gradient along the straight-line path from
`baseline` to `input` over 200 steps. On this particular input the two happen to agree
(the ReLU doesn't clip along the path here), but IG's defining property is the completeness
axiom: the printed `sum` of its attributions matches `F(x) - F(baseline)` exactly, a
guarantee raw Saliency does not carry when a path crosses a ReLU's kink.

See also: [Ad-hoc Interpretability — Deep Learning Approaches](../../interpretability/deep-learning-approaches.md#saliency).
