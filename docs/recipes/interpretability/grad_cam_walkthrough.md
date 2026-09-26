# Recipe: Grad-CAM Walkthrough

**What you'll build:** Grad-CAM's pipeline mechanics — graph wiring, activation/gradient
caching, per-channel weighting — on a small `Conv2DModule -> ReluModule -> FlattenModule ->
LinearModule` network with a synthetic "filled circle" input.

CMake target: `grad_cam_walkthrough_recipe`
(`examples/recipes/grad_cam_walkthrough.cpp`).

!!! note
    This is a synthetic, untrained-network demo (randomly-initialized weights, a hand-drawn
    blob instead of a real digit) — it demonstrates Grad-CAM's mechanics, not "what a real
    model learned." See `examples/mnist_training_demo.cpp` for a real trained classifier.

## Code

```cpp
Conv2DModule conv(1, /*out_channels=*/4, /*kh=*/3, /*kw=*/3, &backend);
ReluModule relu(&backend);
FlattenModule flatten(&backend);
LinearModule classifier(/*in=*/400, /*out=*/3, &backend);

ExplainerContext ctx({&conv, &relu, &flatten, &classifier});
Tensor output = ctx.forward_pass(input);  // input: (1, 1, 12, 12) synthetic "digit"

GradCAM gradcam;
Attribution attr = gradcam.explain(ctx, input, predicted_class, &backend);
```

Full source: [`examples/recipes/grad_cam_walkthrough.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/grad_cam_walkthrough.cpp).

## Expected output

```
class 0 score:  -0.1510
class 1 score:   0.1241
class 2 score:  -0.0904

predicted class 1 (meaningless on an untrained network)

Grad-CAM heatmap (10x10), ASCII-scaled 0-9:
0000000000
0000035100
0001954210
0011008721
0000000880
0000000080
0000000240
0000002000
0000000000
0000000000
```

## What's happening

`GradCAM::explain()` runs forward + backward exactly like `Saliency`, finds the *last*
`OpType::Conv` node in the graph (`ctx.graph().nodes_by_op_type(OpType::Conv)`), reads its
cached activation and gradient, averages the gradient spatially per channel to get
per-channel weights `alpha_k`, and computes `ReLU(sum_k alpha_k * A_k)`. The ASCII heatmap
above shows the resulting activation concentrated in a region roughly tracing the synthetic
circle's edge — Grad-CAM highlighting *where* the last conv layer's features that drove the
prediction were spatially located, even on this untrained network.

See also: [Ad-hoc Interpretability — Deep Learning Approaches](../../interpretability/deep-learning-approaches.md#grad-cam).
