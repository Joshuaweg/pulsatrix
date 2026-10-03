# Recipe: Grad-CAM Walkthrough

**What you'll build:** Grad-CAM's pipeline (graph wiring, activation and gradient caching,
per-channel weighting) on a small `Conv2DModule -> ReluModule -> FlattenModule -> LinearModule`
network with a synthetic "filled circle" input.

CMake target: `grad_cam_walkthrough_recipe`
(`examples/recipes/grad_cam_walkthrough.cpp`).

Run it: `./build/grad_cam_walkthrough_recipe` (Windows:
`build\Release\grad_cam_walkthrough_recipe.exe`).

!!! note
    The network is untrained (random weights) and the input is a drawn circle, not a digit. The
    recipe shows how Grad-CAM works, not what a trained model learned. For a trained MNIST
    classifier, see the [LRP on a trained MNIST classifier](mnist_lrp.md) recipe or
    `examples/mnist_training_demo.cpp`.

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
Grad-CAM recipe -- Conv(1,4,3,3)->ReLU->Flatten->Linear(400,3), SYNTHETIC/untrained

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
...
```

## What's happening

`GradCAM::explain()` runs a forward and backward pass, like `Saliency`. It then:

1. finds the *last* `OpType::Conv` node in the graph (`ctx.graph().nodes_by_op_type(OpType::Conv)`),
2. reads that node's cached activation `A_k` and gradient,
3. averages the gradient spatially per channel to get per-channel weights `alpha_k`, and
4. computes `ReLU(sum_k alpha_k * A_k)`.

In the ASCII heatmap, the activation is concentrated in a region that roughly traces the
circle's edge. Grad-CAM is showing where the last conv layer's prediction-driving features are,
even on this untrained network.

See also: [Gradient-Based Explainers](../../interpretability/deep-learning-approaches.md#grad-cam).
