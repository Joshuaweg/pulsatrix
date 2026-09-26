# Recipe: LIME Basics

**What you'll build:** `LIME` explaining a purely linear network, where weighted least
squares on noise-free linear data recovers the true weights almost exactly.

CMake target: `lime_basics_recipe` (`examples/recipes/lime_basics.cpp`).

## Code

```cpp
LinearModule linear(3, 2, &backend);
linear.set_weight({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
linear.set_bias({100.0f, 100.0f});  // deliberately large/irrelevant

ExplainerContext ctx({&linear});
auto predict = [&ctx](const Tensor& x) { return ctx.forward_pass(x); };

Tensor input(Shape({1, 3}), &backend, {1.0f, 1.0f, 1.0f});

LIME lime;
Attribution attr = lime.explain(predict, input, /*target_index=*/1, /*num_samples=*/300,
                                 /*sigma=*/1.0f, /*l2_lambda=*/0.0f, /*seed=*/42, &backend);
```

Full source: [`examples/recipes/lime_basics.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/lime_basics.cpp).

## Expected output

```
LIME recipe -- output 1 of a linear network, true weight column [2, 4, 6]

feature    LIME coeff  true weight
x0             2.0000          2.0
x1             4.0000          4.0
x2             6.0000          6.0
```

## What's happening

`LIME::explain()` draws 300 Gaussian perturbations of the input (std. dev. `sigma=1.0`),
weights each by an exponential locality kernel of the same width, and fits a local linear
surrogate `g(z) = f(x) + w^T(z - x)`. Against a network that is already exactly linear, `w`
matches the true weight column to four decimal places, regardless of the large, irrelevant
bias — the surrogate only ever sees output *differences*, so any constant term cancels.

See also: [Ad-hoc Interpretability — Model-Agnostic](../../interpretability/model-agnostic.md#lime).
