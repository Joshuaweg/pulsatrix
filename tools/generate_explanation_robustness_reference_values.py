"""Offline, one-time reference-value generator for tests/explanation_robustness_test.cpp (KS-4):
pulsatrix's ExplanationSensitivity checked against numpy implementations of Quantus's
MaxSensitivity, AvgSensitivity and LocalLipschitzEstimate, on the same samples.

Not run by CTest/CI. Run it with numpy installed and transcribe the printed values into
tests/explanation_robustness_test.cpp:

    python3 tools/generate_explanation_robustness_reference_values.py

Versions the committed values were generated with: Python 3.14.4, numpy 2.5.4.

The network is tools/generate_adversarial_reference_values.py's Linear(4, 8) -> ReLU -> Linear(8, 3).
The input is that file's input 2, and the explanation is gradient x input of the predicted class's
logit. The samples are uniform in the L-infinity ball of radius 0.3 times the input's (population)
standard deviation, drawn with pulsatrix's PortableRng (SplitMix64; uniform = (bits >> 11 + 0.5) / 2^53),
one value per element in order, and rounded to float32 as pulsatrix stores them. A sample counts
only if it keeps the predicted class:

  - sensitivity = |E(x') - E(x)|_2 / |E(x)|_2, its maximum and mean (Yeh et al. 2019);
  - local Lipschitz = max |E(x') - E(x)|_2 / |x' - x|_2 (Alvarez-Melis and Jaakkola 2018);
  - adversarial sensitivity: the same at FGSM's point, x + eps sign(grad of the cross-entropy of the
    predicted class), eps = the radius, halved until the prediction stays.
"""

import numpy as np

MASK = (1 << 64) - 1


class PortableRng:
    def __init__(self, seed):
        self.state = seed

    def next(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def uniform(self):
        return ((self.next() >> 11) + 0.5) / float(1 << 53)


f32 = lambda a: np.asarray(a, dtype=np.float32).astype(np.float64)
i = np.arange(4)[:, None]
j = np.arange(8)[None, :]
W1 = f32(0.7 * np.sin(1.3 * i + 0.7 * j + 0.2))
b1 = f32(0.1 * np.cos(np.arange(8)))
W2 = f32(0.6 * np.cos(0.9 * np.arange(8)[:, None] + 1.1 * np.arange(3)[None, :] + 0.4))
b2 = f32(0.05 * np.arange(3))
x0 = f32(0.5 + 0.4 * np.sin(0.8 * 2 + 1.9 * np.arange(4) + 0.3))


def logits(x):
    return np.maximum(x @ W1 + b1, 0.0) @ W2 + b2


def explain(x, c):
    mask = (x @ W1 + b1) > 0
    return (W1 * mask[None, :]) @ W2[:, c] * x


def ce_grad(x, y):
    h_pre = x @ W1 + b1
    z = np.maximum(h_pre, 0.0) @ W2 + b2
    p = np.exp(z - z.max())
    p /= p.sum()
    p[y] -= 1.0
    return W1 @ ((W2 @ p) * (h_pre > 0))


predicted = int(np.argmax(logits(x0)))
e0 = explain(x0, predicted)
scale = np.sqrt(((x0 - x0.mean()) ** 2).mean())
radius = 0.3 * scale
rng = PortableRng(7)
sens, lips = [], []
for _ in range(12):
    x = f32(np.array([v + (2 * rng.uniform() - 1) * radius for v in x0]))
    if np.argmax(logits(x)) != predicted:
        continue
    e = explain(x, predicted)
    d = np.linalg.norm(e - e0)
    sens.append(d / np.linalg.norm(e0))
    lips.append(d / np.linalg.norm(x - x0))
eps = radius
adversarial = None
for _ in range(5):
    x = f32(x0 + eps * np.sign(ce_grad(x0, predicted)))
    if np.argmax(logits(x)) == predicted:
        adversarial = np.linalg.norm(explain(x, predicted) - e0) / np.linalg.norm(e0)
        break
    eps /= 2

print(f"predicted class {predicted}, radius {radius:.10f}, kept {len(sens)} of 12")
print(f"const double kMaxSensitivity = {max(sens):.10f};")
print(f"const double kAvgSensitivity = {np.mean(sens):.10f};")
print(f"const double kLocalLipschitz = {max(lips):.10f};")
print(f"const double kAdversarialSensitivity = {adversarial:.10f};  // at eps {eps:.10f}")
