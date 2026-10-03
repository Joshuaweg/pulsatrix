# Recipe: KernelSHAP Basics

**What you'll build:** `KernelSHAP` explaining a purely linear network. For a linear model the
true Shapley values have a known closed form, so you can check the explainer's output exactly.

CMake target: `kernel_shap_basics_recipe`
(`examples/recipes/kernel_shap_basics.cpp`).

Run it: `./build/kernel_shap_basics_recipe` (Windows: `build\Release\kernel_shap_basics_recipe.exe`).

## Code

```cpp
LinearModule linear(3, 1, &backend);
linear.set_weight({2.0f, -3.0f, 5.0f});
linear.set_bias({100.0f});  // deliberately large/irrelevant

ExplainerContext ctx({&linear});
auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

Tensor input(Shape({1, 3}), &backend, {1.0f, 2.0f, 3.0f});
Tensor baseline(Shape({1, 3}), &backend, {0.0f, 0.0f, 0.0f});

KernelSHAP shap;
Attribution attr = shap.explain(predict, input, baseline, /*target_index=*/0, &backend);
```

Full source: [`examples/recipes/kernel_shap_basics.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/kernel_shap_basics.cpp).

## Expected output

```
KernelSHAP recipe -- linear network f(x) = 2*x0 - 3*x1 + 5*x2 + 100

feature  phi (SHAP)    w*(x-b)
x0           2.0000     2.0000
x1          -6.0000    -6.0000
x2          15.0000    15.0000
...
```

## What's happening

For a linear model, the Shapley value of feature `i` is exactly `w_i * (x_i - baseline_i)`.
`KernelSHAP::explain()` evaluates every coalition (each mix of "present" and "baseline"
features). It weights each coalition by the SHAP kernel and solves a weighted regression.

On a linear model this recovers the closed form exactly: the `phi` column matches `w*(x-b)` to
four decimal places. The large bias appears in every coalition's `f(S)`, so it cancels out.

See also: [Model-Agnostic Explainers](../../interpretability/model-agnostic.md#kernelshap).
