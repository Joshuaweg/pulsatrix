"""Reference values for tests/vision_models_test.cpp (KS-9): torchvision ResNet and VGG, their
logits, and Zennit 1.0.0 heatmaps.

Run inside the pinned Zennit image (tools/lrp_reference.Dockerfile):

    docker run --rm -v "$PWD":/w -w /w pulsatrix-lrpref:latest \
        python3 tools/golden/make_vision_golden.py tests/fixtures/vision

writes, for the two small models the tests always run,

    tiny_resnet.safetensors  tiny_resnet_golden.safetensors
    tiny_vgg.safetensors     tiny_vgg_golden.safetensors

The weights files are torchvision state dicts, as tools/convert/pickle_to_safetensors.py writes
them. A golden file holds `input`, `logits` and one `relevance.<composite>` per Zennit composite
(EpsilonPlus, EpsilonAlpha2Beta1, EpsilonGammaBox(-3, 3)), with the attribution output the
one-hot of `targets`. ResNets are explained with Zennit's ResNetCanonizer (BatchNorm merged into
the convolutions, residual sums under the Norm rule), as Zennit's documentation does.

    --resnet18 PATH.pth --vgg16 PATH.pth

also writes goldens for the published ImageNet models (a seeded 224x224 input) from torchvision's
weight files, as resnet18_golden.safetensors and vgg16_golden.safetensors: the tests use them when
PULSATRIX_TORCHVISION_DIR holds those and the converted weights (resnet18.safetensors,
vgg16.safetensors).

The small models are torchvision's own blocks at small widths: torchvision.models.resnet's
BasicBlock in a ResNet with torchvision's attribute names (its ResNet class fixes the widths at
64 to 512), and torchvision.models.VGG on make_layers() with a smaller pool and classifier.
"""
import argparse
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file
from torchvision.models import VGG, resnet18, vgg16
from torchvision.models.resnet import BasicBlock
from torchvision.models.vgg import make_layers
from zennit.attribution import Gradient
from zennit.composites import EpsilonAlpha2Beta1, EpsilonGammaBox, EpsilonPlus
from zennit.torchvision import ResNetCanonizer

NUM_CLASSES = 5


class TinyResNet(nn.Module):
    """torchvision's ResNet.forward over len(blocks) stages, widths width * 2^stage."""

    def __init__(self, blocks, width, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        inplanes = width
        self.stages = []
        for s, n in enumerate(blocks):
            planes, stride = width << s, 1 if s == 0 else 2
            layers = []
            for i in range(n):
                down = None
                if i == 0 and (stride != 1 or inplanes != planes):
                    down = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
                layers.append(BasicBlock(inplanes, planes, stride if i == 0 else 1, down))
                inplanes = planes
            setattr(self, "layer%d" % (s + 1), nn.Sequential(*layers))
            self.stages.append("layer%d" % (s + 1))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(inplanes, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for name in self.stages:
            x = getattr(self, name)(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def tiny_vgg():
    model = VGG(make_layers([4, "M", 8, 8, "M"]), num_classes=NUM_CLASSES)
    model.avgpool = nn.AdaptiveAvgPool2d(2)
    model.classifier = nn.Sequential(nn.Linear(8 * 2 * 2, 6), nn.ReLU(True), nn.Dropout(), nn.Linear(6, 6), nn.ReLU(True),
                                     nn.Dropout(), nn.Linear(6, NUM_CLASSES))
    return model


def randomize_batch_norm(model, gen):
    """Non-trivial BatchNorm statistics and affine parameters, so folding them matters."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.copy_(torch.randn(m.num_features, generator=gen) * 0.3)
            m.running_var.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
            m.weight.data.copy_(torch.rand(m.num_features, generator=gen) + 0.5)
            m.bias.data.copy_(torch.randn(m.num_features, generator=gen) * 0.2)


def composites(x, resnet):
    canon = [ResNetCanonizer()] if resnet else []
    return {
        "EpsilonPlus": EpsilonPlus(canonizers=canon),
        "EpsilonAlpha2Beta1": EpsilonAlpha2Beta1(canonizers=canon),
        "EpsilonGammaBox": EpsilonGammaBox(low=torch.full_like(x, -3.0), high=torch.full_like(x, 3.0), canonizers=canon),
    }


def golden(model, x, targets, resnet):
    model.eval()
    with torch.no_grad():
        logits = model(x)
    out = {"input": x.contiguous(), "logits": logits.contiguous(), "targets": torch.tensor(targets, dtype=torch.int64)}
    attr_out = torch.eye(logits.shape[1])[targets]
    for name, composite in composites(x, resnet).items():
        with Gradient(model=model, composite=composite) as attributor:
            _, relevance = attributor(x.clone(), attr_out)
        out["relevance." + name] = relevance.detach().contiguous()
    return out


def weights(model):
    return {k: v.detach().contiguous() for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir")
    ap.add_argument("--resnet18", help="torchvision's resnet18-f37072fd.pth")
    ap.add_argument("--vgg16", help="torchvision's vgg16-397923af.pth")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)

    if not (args.resnet18 or args.vgg16):
        resnet = TinyResNet([1, 2], 4, NUM_CLASSES)
        randomize_batch_norm(resnet, gen)
        x = torch.randn(2, 3, 32, 32, generator=gen)
        save_file(weights(resnet), os.path.join(args.out_dir, "tiny_resnet.safetensors"))
        save_file(golden(resnet, x, [1, 3], True), os.path.join(args.out_dir, "tiny_resnet_golden.safetensors"))

        vgg = tiny_vgg()
        x = torch.randn(2, 3, 32, 32, generator=gen)
        save_file(weights(vgg), os.path.join(args.out_dir, "tiny_vgg.safetensors"))
        save_file(golden(vgg, x, [0, 4], False), os.path.join(args.out_dir, "tiny_vgg_golden.safetensors"))

    x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(2))
    for flag, build, resnet, name in [(args.resnet18, resnet18, True, "resnet18"), (args.vgg16, vgg16, False, "vgg16")]:
        if flag:
            model = build()
            model.load_state_dict(torch.load(flag, weights_only=True))
            save_file(golden(model, x, [int(model.eval()(x).argmax())], resnet),
                      os.path.join(args.out_dir, name + "_golden.safetensors"))


if __name__ == "__main__":
    main()
