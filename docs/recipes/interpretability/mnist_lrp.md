# Recipe: LRP on a Trained MNIST Classifier

**What you'll build:** a `Conv2DModule -> ReluModule -> FlattenModule -> LinearModule`
classifier (`MnistConvNet`) trained on real MNIST digits, then explained with whole-model
Layer-wise Relevance Propagation (`LRP::explain`), one held-out digit per class: why the
network predicts its class *rather than the runner-up*, under the plain epsilon rule and under
Zennit's `EpsilonPlus` composite.

CMake target: `mnist_lrp_recipe`
(`examples/recipes/mnist_lrp.cpp`).

!!! note
    Needs the real MNIST IDX files in `data/MNIST/raw/`. Run `tools/fetch_mnist.py` once
    first (the same requirement as `examples/mnist_training_demo.cpp`). Run the recipe from
    the repository root so the relative data path resolves. Training takes about 10 seconds on
    a CPU.

## Code

```cpp
MnistConvNet net(&backend);  // Conv2D(1,8,5,5) -> ReLU -> Flatten -> Linear(4608,10)
// ... net.train_step(image, label, optimizer, sink, step) over the training set ...

ExplainerContext ctx(net.modules());          // the trained layers, in forward order
Tensor x(Shape({1, 1, 28, 28}), &backend, image.to_host_vector());  // batched: output is (1, 10)

// "Why pred rather than runner_up?": target and contrast, seeded with the two logits.
const LRPTarget target{{pred}, {runner_up}};
Attribution eps = LRP().explain(ctx, x, target, &backend);                       // epsilon everywhere
Attribution eps_plus = LRP::epsilon_plus().explain(ctx, x, target, &backend);    // ZPlus on Conv2D
// eps.values has the image's shape: one relevance value per pixel.
// eps.metadata["relevance_in_sum"] vs. the logit margin shows how well relevance was conserved.
```

Full source: [`examples/recipes/mnist_lrp.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/mnist_lrp.cpp).

## Expected output

```
epoch 1 | mean train loss 0.2653 | test accuracy 95.6%
epoch 2 | mean train loss 0.1039 | test accuracy 95.4%
epoch 3 | mean train loss 0.0590 | test accuracy 96.0%

label  pred  runner-up   margin   sum(R) eps  sum(R) eps-plus
7      7     2           17.466    17.363594        17.264297
2      2     6            2.737     3.723733         3.578975
1      1     8            5.609     6.890915         7.359929
0      0     6            9.759     9.837962         9.564141
4      4     9            8.495     9.635921         9.480172
9      9     4            6.131     5.523613         5.190765
5      6     5            2.697     2.386827         2.447210   <- misclassified
6      6     0            1.210     1.283611         1.300158
3      8     5            0.648     0.826104         0.532464   <- misclassified
8      8     2           13.095    13.480010        12.657933
```

The recipe also prints the digit-0 relevance maps for both rule sets side by side as ASCII
(`+` evidence for 0, `-` evidence for the runner-up).

## What's happening

**Why a contrastive target instead of the raw logit.** LRP redistributes whatever score you seed
it with. A trained logit can be positive only because of its bias, while the epsilon rule
redistributes over the *pre-bias* sum, so seeding the raw winning logit (or a unit one-hot seed)
can flip the sign of the whole heatmap. `LRPTarget{{pred}, {runner_up}}` asks the more useful
question, "why this class rather than the next best one?". LRP is linear in the seed, so the
result is exactly explanation(pred) − explanation(runner-up), and the explained score is the
logit margin, which is positive whenever the prediction wins.

**Rule choice.** `LRP()` applies the epsilon rule on every layer. `LRP::epsilon_plus()` is
Zennit's `EpsilonPlus` composite: ZPlus (α=1, β=0) on the convolution, which keeps only
positive contributions and gives a less noisy map, and epsilon with the bias in the denominator
on the classifier.

**Conservation.** `sum(R)` over the 784 pixels tracks the margin, but not exactly, and the gap
is largest when the margin is small. Under epsilon, the classifier, ReLU and Flatten conserve
exactly. The whole gap comes from `Conv2DModule` units in the blank background around the digit:
their input patch is all zeros, so they're active only because of their bias. They still receive
relevance from the classifier, but with zero input there's nothing to redistribute it to, so it's
dropped. That dropped relevance can be positive or negative, which is why `sum(R)` lands on either
side of the margin. (Checked on the first four digits, 7, 2, 1 and 0: the gap equals the relevance on those units
to four decimals.) `EpsilonPlus` also keeps the classifier's bias in the denominator, so the bias's share
of the margin is absorbed as well.

**The misclassified digits.** The 5 the network calls a 6 has the true class as its runner-up, so
its contrastive map shows exactly which strokes tipped it from 5 to 6 (margin 2.7). The 3 called
an 8 is a near tie with 5 (margin 0.65), and its true class isn't even second; to ask "why 8 rather
than 3", pass the true label as the contrast instead of the runner-up.

See also: [Grad-CAM walkthrough](grad_cam_walkthrough.md) for the same architecture explained
with gradients instead of relevance, and the [MNIST gallery](../../visualization/index.md#mnist-gallery)
for every explainer on this model rendered as heatmaps.
