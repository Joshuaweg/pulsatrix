"""Offline, one-time reference-value generator for Phase 4 Mission 2's Captum benchmark.

Not run by CTest/CI -- this is a codegen tool, not a test dependency. Run manually
(`py -3.11 tools/generate_captum_reference_values.py`) and hand-transcribe the printed
values into tests/captum_benchmark_test.cpp as hardcoded EXPECT_NEAR targets. Zero live
Python dependency at C++ test/build time.

Each model below reuses the exact architecture and hardcoded weights already used in
this project's own tests/stability_test.cpp, so the C++ side of the comparison is
running literally the same network, not a similar one.

IMPORTANT layout gotcha: exai's LinearModule stores weight as shape
(in_features, out_features), row-major -- the transpose of torch.nn.Linear's
(out_features, in_features). Every weight literal below is reshaped as
(in_features, out_features) and then .t()'d before being copied into the
torch.nn.Linear, or the numbers silently describe a different matrix (caught
the hard way: an earlier draft of this script skipped the transpose and produced
reference values that looked plausible but didn't match exai's own, independently
hand-verified gradient computation for the ReLU-nonlinear model).

Scoped to Saliency/IntegratedGradients/GradCAM only (pure-torch Captum methods).
Captum's Lime/KernelShap both transitively require sklearn, which fails to import on
this machine (missing vcomp140.dll -- a separate broken native-DLL chain from the
shap/pandas/scipy issue this script's docstring elsewhere documents working around).
Not worth chasing a second shared-environment repair for methods this project already
benchmarks against a closed-form Shapley/exact-recovery oracle in Phase 3 -- see
mission_benchmark_suite.md's Recon.
"""

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerGradCam, Saliency

torch.manual_seed(0)


def exai_linear_weight(in_features, out_features, flat):
    """Reinterprets an exai LinearModule::set_weight flat literal -- row-major over
    (in_features, out_features) -- as a torch.nn.Linear weight tensor, shape
    (out_features, in_features). See module docstring's layout gotcha."""
    return torch.tensor(flat).reshape(in_features, out_features).t().contiguous()


def model_a():
    """Single Linear(3,1). Weight {2,-3,5}, bias {100} -- matches
    KernelSHAPIsDeterministic / PDPIsDeterministic in stability_test.cpp."""
    m = nn.Linear(3, 1)
    with torch.no_grad():
        m.weight.copy_(exai_linear_weight(3, 1, [2.0, -3.0, 5.0]))
        m.bias.copy_(torch.tensor([100.0]))
    return m


def model_b():
    """Linear(3,4) -> ReLU -> Linear(4,2). Matches
    IntegratedGradientsIsDeterministic in stability_test.cpp."""
    l1 = nn.Linear(3, 4)
    l2 = nn.Linear(4, 2)
    with torch.no_grad():
        l1.weight.copy_(
            exai_linear_weight(
                3,
                4,
                [0.2, -0.4, 0.6, 0.1, -0.3, 0.5, 0.7, -0.2, 0.1, 0.4, -0.6, 0.3],
            )
        )
        l1.bias.copy_(torch.tensor([0.1, -0.1, 0.2, 0.0]))
        l2.weight.copy_(
            exai_linear_weight(
                4,
                2,
                [0.5, -0.3, 0.2, 0.4, -0.1, 0.6, 0.3, -0.5],
            )
        )
        l2.bias.copy_(torch.tensor([0.05, -0.05]))
    return nn.Sequential(l1, nn.ReLU(), l2)


class ModelC(nn.Module):
    """Conv2D(1,2,k=2,s=2) -> ReLU -> Flatten -> Linear(8,2). Matches
    GradCAMIsDeterministic in stability_test.cpp (input is 1x3x3, 'same'-less
    conv with kernel 2 stride ... see stability_test.cpp for the exact im2col
    layout this project's Conv2DModule uses; only the conv layer's output is
    consumed by LayerGradCam, so exact downstream flatten/linear shape must
    just match element count, not exactly replicate the C++ backend's stride
    convention)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1)
        self.relu = nn.ReLU()
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, 1.0]]],
                        [[[0.0, 1.0], [1.0, 0.0]]],
                    ]
                )
            )
            self.conv.bias.copy_(torch.tensor([0.0, 0.0]))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(8, 2)
        with torch.no_grad():
            self.linear.weight.copy_(
                exai_linear_weight(
                    8,
                    2,
                    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0],
                )
            )
            self.linear.bias.copy_(torch.tensor([0.0, 0.0]))

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        return self.linear(x)


def main():
    print("=== Model A: Linear(3,1) ===")
    m = model_a()
    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    sal = Saliency(m)
    print("Saliency:", sal.attribute(x, target=0, abs=False).tolist())

    print("\n=== Model B: Linear(3,4)->ReLU->Linear(4,2) ===")
    m = model_b()
    x = torch.tensor([[0.5, -0.3, 1.2]], requires_grad=True)
    baseline = torch.zeros_like(x)

    sal = Saliency(m)
    print("Saliency:", sal.attribute(x, target=0, abs=False).tolist())

    ig = IntegratedGradients(m)
    print(
        "IntegratedGradients:",
        ig.attribute(x, baselines=baseline, target=0, n_steps=50).tolist(),
    )

    print("\n=== Model C: Conv2D(1,2,2,2)->ReLU->Flatten->Linear(8,2) ===")
    m = ModelC()
    x = torch.tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 0.0, 5.0], [6.0, 7.0, 8.0]]]], requires_grad=True
    )
    gc = LayerGradCam(m, m.conv)
    attr = gc.attribute(x, target=0)
    print("GradCAM (per-channel, pre-upsample):", attr.tolist())


if __name__ == "__main__":
    main()
