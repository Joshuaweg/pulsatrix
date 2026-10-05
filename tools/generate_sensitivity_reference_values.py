"""Offline, one-time reference-value generator for tests/sensitivity_test.cpp (CFS-3): pulsatrix's
Occlusion checked against Captum's, and its percentile and standard-deviation bounds against
numpy.

Not run by CTest/CI. Run it with torch, captum and numpy installed and transcribe the printed
C++ initializer lists into tests/sensitivity_test.cpp:

    python3 tools/generate_sensitivity_reference_values.py

Versions the committed values were generated with: Python 3.12, torch 2.14.1, captum 0.9.0,
numpy 2.5.3.

Occlusion model: a fixed, non-additive function of a (2, 4, 5) instance with two outputs,
    out[0] = sum(w * tanh(x)),  out[1] = out[0] + x[0,1,2] * x[1,3,4]
explained for output 1 with a (1, 3, 3) window, strides (1, 2, 2) and baseline 0.25. Along H
the second window starts at 2 and is cropped to 2 rows, which is the case Captum's padding
handles specially. Weights and inputs use det() from the other generators.
"""

import numpy as np
import torch
from captum.attr import Occlusion


def det(n, mul, add, div):
    ints = np.array([float((i * mul + add) % 17 - 8) for i in range(n)], dtype=np.float32)
    return ints / np.float32(div)


W = torch.tensor(det(40, 7, 2, 9)).reshape(2, 4, 5)


def model(x):  # x: (batch, 2, 4, 5)
    base = (W * torch.tanh(x)).flatten(1).sum(1)
    return torch.stack([base, base + x[:, 0, 1, 2] * x[:, 1, 3, 4]], dim=1)


def cpp_float(v):
    text = "%.9g" % (float(v) + 0.0)
    return (text if any(c in text for c in ".e") else text + ".0") + "f"


def emit(name, values):
    print("const std::vector<float> k%s = {" % name)
    line = "   "
    for v in np.asarray(values).reshape(-1).tolist():
        item = " " + cpp_float(v) + ","
        if len(line) + len(item) > 116:
            print(line)
            line = "   "
        line += item
    print(line)
    print("};")


if __name__ == "__main__":
    x = torch.tensor(det(40, 5, 3, 4)).reshape(1, 2, 4, 5)
    attr = Occlusion(model).attribute(x, target=1, sliding_window_shapes=(1, 3, 3), strides=(1, 2, 2),
                                      baselines=0.25)
    emit("Occlusion", attr.detach().numpy())

    background = det(24 * 3, 5, 1, 3).reshape(24, 3)
    emit("Percentile10", np.percentile(background, 10, axis=0))
    emit("Percentile80", np.percentile(background, 80, axis=0))
    emit("Std", np.std(background, axis=0))
