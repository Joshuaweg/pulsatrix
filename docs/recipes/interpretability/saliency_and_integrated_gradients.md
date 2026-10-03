# Recipe: Saliency and Integrated Gradients

**What you'll build:** both gradient-based explainers run on the same small
`Linear(2,4) -> ReLU -> Linear(4,1)` network for one input. The recipe also checks that the
Integrated Gradients attributions sum to `F(x) - F(baseline)`.

CMake target: `saliency_and_ig_recipe`
(`examples/recipes/saliency_and_integrated_gradients.cpp`).

Run it: `./build/saliency_and_ig_recipe` (Windows: `build\Release\saliency_and_ig_recipe.exe`).

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
Saliency and Integrated Gradients recipe -- Linear(2,4)->ReLU->Linear(4,1)

input (1, 1) -> output 0.3200

saliency:             d(out)/d(in) = [0.4200, -0.1000]
integrated gradients: IG = [0.4200, -0.1000], sum=0.3200 (F(x)-F(baseline)=0.3200)
...
```

## What's happening

`Saliency::explain()` seeds a one-hot vector at `target_index` and calls
`ctx.backward_pass()`. The gradient that comes back *is* the saliency map.

`IntegratedGradients` instead averages the gradient along the straight-line path from
`baseline` to `input`, over 200 steps. On this input the two methods happen to agree, because
the ReLU doesn't switch on or off along the path.

IG's defining property is completeness: the printed `sum` of its attributions equals
`F(x) - F(baseline)`. Saliency has no such guarantee. It is the gradient at a single point, so
it can miss parts of the path where a ReLU switches on or off.

See also: [Gradient-Based Explainers](../../interpretability/deep-learning-approaches.md#saliency).
