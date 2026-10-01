# Recipe: LRP on a Trained MNIST Classifier

**What you'll build:** a `Conv2DModule -> ReluModule -> FlattenModule -> LinearModule`
classifier trained on real MNIST digits, then explained with epsilon-rule Layer-wise Relevance
Propagation, one held-out digit per class.

CMake target: `mnist_lrp_recipe`
(`examples/recipes/mnist_lrp.cpp`).

!!! note
    Needs the real MNIST IDX files in `data/MNIST/raw/`. Run `tools/fetch_mnist.py` once
    first (the same requirement as `examples/mnist_training_demo.cpp`). Run the recipe from
    the repository root so the relative data path resolves. Training takes about a minute on
    a CPU.

## Code

```cpp
Conv2DModule conv(1, 8, 5, 5, &backend);
ReluModule relu(&backend);
FlattenModule flatten(&backend);
LinearModule classifier(4608, 10, &backend);
// ... train with CrossEntropyLoss + AdamOptimizer ...

// Explain "pred vs. the other nine" with a one-output contrast head over the same features.
LinearModule contrast(4608, 1, &backend);
contrast.set_weight(w_pred_minus_mean_of_others);
contrast.set_bias({b_pred - mean_of_other_biases});

Tensor score = contrast.forward(flatten.forward(relu.forward(conv.forward(image))));
LRPRuleConfig config;  // epsilon rule
Tensor relevance = conv.propagate_relevance(
    relu.propagate_relevance(
        flatten.propagate_relevance(contrast.propagate_relevance(score, config), config), config),
    config);  // shape matches the image: one relevance value per pixel
```

Full source: [`examples/recipes/mnist_lrp.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/mnist_lrp.cpp).

## Expected output

```
epoch 1 | mean train loss 0.2653 | test accuracy 95.6%
epoch 2 | mean train loss 0.1039 | test accuracy 95.4%
epoch 3 | mean train loss 0.0590 | test accuracy 96.0%

label  pred        score     sum(R)
7      7          24.907     24.546
2      2          14.834     15.665
1      1          11.406     11.755
0      0          15.525     15.489
4      4          19.002     19.659
9      9          14.423     14.207
5      6          17.469     17.607   <- misclassified
6      6          12.056     11.963
3      8           7.978      7.666   <- misclassified
8      8          22.413     21.878
```

The recipe also prints an ASCII view of the digit-0 heatmap (`+` evidence for, `-` evidence
against).

## What's happening

**Why a contrast head instead of the raw logit.** LRP redistributes whatever score you seed it
with. The winning raw logit of this trained network is often *negative*, because every logit
is shifted by the same learned bias. Seeding epsilon-LRP with a negative score flips the sign
of the whole heatmap. Explaining `logit(pred) - mean(other logits)` instead asks the more
useful question, "why this class rather than the others?", and that score is positive
whenever the prediction wins. It's built as an ordinary `LinearModule` with weights
`w_pred - mean(w_others)`, so its `propagate_relevance()` applies unchanged.

**Calling order matters.** Each module's `propagate_relevance()` reads the activations cached
by that module's *most recent* `forward()`. The recipe runs a fresh forward pass through the
contrast head for each image before propagating, so every layer's cache belongs to that image.

**Conservation.** `sum(R)` over the 784 pixels lands within about 6% of the explained score,
sometimes above it and sometimes below. Pulsatrix's epsilon rule leaves the bias out of the
denominator, so `LinearModule` and `ReluModule` conserve exactly here. The whole gap comes from
`Conv2DModule` units in the blank background around the digit: their input patch is all zeros, so
they're active only because of their bias. They still receive relevance from the classifier,
but with zero input there's nothing to redistribute it to, so it's dropped. That dropped
relevance can be positive or negative, which is why `sum(R)` can land either side of the score.

**The misclassified digits.** The 5 that the network calls a 6 is a *confident* mistake: its
score of 17.5 is in the same range as the correct digits. That's invisible in the accuracy
number. The heatmap shows which strokes the network relied on.

See also: [Grad-CAM walkthrough](grad_cam_walkthrough.md) for the same architecture explained
with gradients instead of relevance.
