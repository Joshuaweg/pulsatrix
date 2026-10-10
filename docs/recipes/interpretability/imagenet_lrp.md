# Recipe: LRP on ImageNet Models (ResNet18, VGG16)

**What you'll build:** a heatmap of why torchvision's pretrained ResNet18 or VGG16 classifies a
photo the way it does. The recipe loads the published ImageNet weights, prepares the photo as
torchvision does, prints the top five classes, and explains the top one with one of Zennit's
[composites](../../interpretability/lrp.md#composites-a-different-rule-per-layer) (a choice of LRP
rule for each kind of layer). It writes the photo as the model saw it and the heatmap: red for pixels that are
evidence for the class, blue for evidence against.

CMake target: `imagenet_lrp_recipe` (`examples/recipes/imagenet_lrp.cpp`).

```bash
./build/imagenet_lrp_recipe resnet18 resnet18.safetensors photo.jpg out/ [--composite NAME] [--device hip]
```

- `resnet18` or `vgg16`, then the converted weights, any PNG, JPEG or BMP, and an output
  directory.
- `--composite`: `epsilon_plus` (the default), `epsilon_alpha2_beta1` or `epsilon_gamma_box`.
- `--device hip` runs on an AMD GPU, in a build configured with `-DPULSATRIX_ENABLE_HIP=ON`.

!!! note
    The recipe needs torchvision's weights, converted to safetensors once (see
    [PyTorch checkpoints](../../deep-learning/index.md#pytorch-checkpoints-pt-pth)):

    ```bash
    pip install "torch>=2.6" safetensors
    curl -LO https://download.pytorch.org/models/resnet18-f37072fd.pth    # 45 MB
    curl -LO https://download.pytorch.org/models/vgg16-397923af.pth       # 528 MB
    python3 tools/convert/pickle_to_safetensors.py resnet18-f37072fd.pth resnet18.safetensors
    python3 tools/convert/pickle_to_safetensors.py vgg16-397923af.pth vgg16.safetensors
    ```

    On a CPU (Release build), ResNet18 takes under a second and VGG16 about five seconds and
    2.4 GB of memory.

For the rules and composites, see [Layer-wise Relevance Propagation](../../interpretability/lrp.md).

## Code

```cpp
TorchvisionResNet model(TorchvisionResNet::ResNet18(), &backend);  // or TorchvisionVGG::VGG16()
LoadTorchvisionWeights(model, "resnet18.safetensors");
model.set_training(false);  // eval mode: BatchNorm uses its running statistics

// The photo: ImageDecoder::DecodeFile(path, &cpu, 3) gives (1, 3, H, W) in [0, 1]. pulsatrix has
// no resize or crop, so the recipe's ResizeAndCrop() does them, then it normalizes each channel.
const Tensor x = ...;  // (1, 3, 224, 224)
Tensor logits = model.forward(x);
const int64_t top_class = ...;  // the argmax of logits (the recipe takes the top five)

// Fold each BatchNorm into its convolution while explaining. Keep `folds` alive for the whole
// explain() call: the folds undo themselves when it goes out of scope.
auto folds = model.fold_batch_norms();  // a ResNet only: VGG16 has no BatchNorm
ExplainerContext ctx(model.layers());   // the stem, each residual block, the classifier
Attribution a = LRP::epsilon_plus().explain(ctx, x, LRPTarget{{top_class}, {}, LRPSeed::OneHot}, &backend);
// a.values: (1, 3, 224, 224). Sum over the colour channels for one value per pixel.
```

For `epsilon_gamma_box`, the recipe uses one box for all channels: the lowest and highest
normalized value any channel can take,
`LRP::epsilon_gamma_box(-0.485f / 0.229f, (1.0f - 0.406f) / 0.225f)`, about −2.12 to 2.64.

Full source:
[`examples/recipes/imagenet_lrp.cpp`](https://github.com/Joshuaweg/pulsatrix/blob/master/examples/recipes/imagenet_lrp.cpp).

## Expected output

For a photo of an owl, with ResNet18:

```
top-5 ImageNet classes (index: logit):  24: 11.77  93: 10.02  21: 9.79  88: 9.57  23: 9.48
epsilon_plus relevance: total 0.3377 of the 1 it was seeded with (logit 11.7663)
wrote out/input.ppm and out/heatmap.ppm
```

Class 24 is "great grey owl". Class indices follow torchvision's ImageNet order
(`torchvision.models.ResNet18_Weights.DEFAULT.meta["categories"]` lists the names). The images
are binary PPMs, which every image viewer and converter opens. The heatmap is scaled by the
99.5th percentile of the relevance, so a few extreme pixels don't wash out the rest.

## What's happening

- **Preparing the photo.** torchvision's models expect the short side resized to 256, the
  center 224x224 cropped, and each channel normalized with ImageNet's mean and standard
  deviation. The recipe resizes bilinearly without torchvision's antialiasing, so its logits
  differ slightly from PyTorch's for the same file. The model itself matches PyTorch on the same
  tensor.
- **Folding BatchNorm.** LRP rules are defined for affine layers followed by activations.
  BatchNorm in eval mode is an affine map, so it is merged into the convolution before it. Zennit
  does the same with a *canonizer*, which rewrites a model into an equivalent one that LRP's
  rules fit. Skipping the fold isn't an error, but BatchNorm then passes relevance through
  unchanged and the heatmap differs from Zennit's.
- **Residual blocks.** At each `out = shortcut(x) + main(x)`, relevance is split between the two
  branches in proportion to what each contributed. Each branch then passes its share back
  through its own layers. Zennit does the same at the sums its ResNet canonizer inserts.
- **Composites choose a rule per layer type.** `epsilon_plus` applies ZPlus (AlphaBeta with
  alpha 1 and beta 0: only positive contributions count) to every convolution, the ones inside
  the residual blocks included, and epsilon to the classifier. `epsilon_gamma_box` applies ZBox to
  the first convolution, with the input's box (above), and Gamma to the rest. It gives sharper,
  more local heatmaps.
- **Max pooling** passes relevance to the pixel that won each window. ResNet's stem pools
  overlapping windows, so a pixel that wins several windows gets the relevance of each one.
- **Why the total is 0.34.** The
  [one-hot seed](../../interpretability/lrp.md#targets-contrasts-and-seeds) starts the explained
  class at relevance 1, not at its logit, as Zennit does. Biases and ZPlus's dropped negative
  contributions absorb about two thirds of it on the way down. The heatmap is the
  *distribution* of evidence; its overall scale isn't meaningful.

## Checked against Zennit

`tests/vision_models_test.cpp` compares these models with Zennit 1.0.0 on the same inputs:
small ResNet and VGG variants in every test run, and the published ResNet18 and VGG16 when
`PULSATRIX_TORCHVISION_DIR` points to a directory with their converted weights and the
references written by `tools/golden/make_vision_golden.py --resnet18 ... --vgg16 ...`. The
heatmaps agree to 2e-3 of the largest value, on CPU and GPU.
